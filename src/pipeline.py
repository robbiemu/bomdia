#!/usr/bin/env python3
"""
This file contains the core pipeline for converting a transcript to a podcast.
"""

import os
import random
import re
import shutil
import tempfile
from typing import Dict, Optional

# LangGraph + LangChain (for LLM)
from langgraph.graph import START, StateGraph
from langgraph.pregel import Pregel
from pydub import AudioSegment

# Import config
from shared.config import config

from src.components.audio_generator import (
    DiaTTS,
    chunk_to_5_10s,
)

# Import components from src/
from src.components.transcript_parser import (
    ingest_transcript,
    merge_consecutive_lines,
)
from src.components.verbal_tag_injector import (
    VerbalTagInjectorState,
    build_llm_injector,
    rule_based_injector,
)


# ---- LangGraph sub-agent (injector) ----
def build_langgraph_injector() -> Pregel:
    """
    Construct a simple LangGraph StateGraph with a single node `inject_line`.
    The node uses the LLM-based injector if configured, otherwise falls back
    to the rule-based one.
    """

    def inject_node(state: VerbalTagInjectorState) -> Dict[str, str]:
        # Decide which injector to use based on config
        if config.LLM_SPEC:
            # Use the LLM-based injector from our component
            llm_injector_func = build_llm_injector()
            return llm_injector_func(state)
        else:
            # Fallback to rule-based
            return rule_based_injector(state)

    builder = StateGraph(VerbalTagInjectorState)
    builder.add_node("inject_line", inject_node)
    builder.add_edge(START, "inject_line")
    graph = builder.compile()
    return graph


# ---- TOP-LEVEL pipeline ----
def run_pipeline(
    input_path: str,
    out_audio_path: str,
    dia_checkpoint: str = config.DIA_CHECKPOINT,
    voice_prompts: Optional[Dict] = None,
    seed: Optional[int] = config.SEED,
) -> None:
    try:
        # Ingest and parse the transcript
        try:
            lines = ingest_transcript(input_path)
            if not lines:
                raise ValueError("No content found in the transcript file") from None
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Input file not found: {input_path}") from e
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                e.encoding,
                e.object,
                e.start,
                e.end,
                f"Unable to decode file: {input_path}. Please check file encoding",
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to parse transcript file: {str(e)}") from e

        # Normalize to [S1]/[S2] and merge consecutive lines
        try:
            lines = [
                {
                    "speaker": (
                        ln["speaker"]
                        if ln["speaker"].startswith("S")
                        else ln["speaker"]
                    ),
                    "text": ln["text"],
                }
                for ln in lines
            ]
            lines = merge_consecutive_lines(lines)
        except Exception as e:
            raise RuntimeError(f"Failed to normalize transcript: {str(e)}") from e

        # The decision is now inside the graph builder
        graph = build_langgraph_injector()

        # Simple summary / topic
        try:
            convo_summary = "A short conversational summary (MVP): " + " ".join(
                [ln["text"] for ln in lines[:6]]
            )
            current_topic = lines[0]["text"].split()[0:6] if lines else "general"
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate conversation summary: {str(e)}"
            ) from e

        # Process each line through LangGraph sub-agent
        try:
            processed = []
            for idx, ln in enumerate(lines):
                prev_lines = [
                    f"[{line['speaker']}] {line['text']}"
                    for line in lines[max(0, idx - config.CONTEXT_WINDOW) : idx]
                ]
                next_lines = [
                    f"[{line['speaker']}] {line['text']}"
                    for line in lines[idx + 1 : idx + 1 + config.CONTEXT_WINDOW]
                ]
                state_in = {
                    "prev_lines": prev_lines,
                    "current_line": f"[{ln['speaker']}] {ln['text']}",
                    "next_lines": next_lines,
                    "summary": convo_summary,
                    "topic": (
                        " ".join(current_topic)
                        if isinstance(current_topic, list)
                        else str(current_topic)
                    ),
                }
                try:
                    result = graph.invoke(state_in)
                    modified = result.get("modified_line") or state_in["current_line"]
                except Exception as e:
                    print(
                        f"[pipeline] LangGraph injector failed for line {idx}: {str(e)}"
                    )
                    modified = state_in["current_line"]
                    if config.PAUSE_PLACEHOLDER in modified:
                        if not isinstance(modified, str):
                            raise TypeError(
                                f"Expected a string for the modified line, got "
                                f"{type(modified)}"
                            ) from e
                        modified = modified.replace(
                            config.PAUSE_PLACEHOLDER,
                            random.choice(config.VERBAL_TAGS),  # nosec
                        )
                if not isinstance(modified, str):
                    raise TypeError(
                        f"Expected a string for the modified line, got {type(modified)}"
                    )
                processed.append(
                    {
                        "speaker": ln["speaker"],
                        "text": re.sub(r"^\[S[12]\]\s*", "", modified),
                    }
                )
        except Exception as e:
            raise RuntimeError(f"Failed to process transcript lines: {str(e)}") from e

        # Chunk into 5-10s mini transcripts
        try:
            mini_transcripts = chunk_to_5_10s(processed)
            if not mini_transcripts:
                raise ValueError(
                    "No mini-transcripts generated from the input"
                ) from None
            print(
                f"[pipeline] Produced {len(mini_transcripts)} mini-transcript blocks "
                f"(5-10s preferred)."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to chunk transcript: {str(e)}") from e

        # TTS each block with Dia
        try:
            tts = DiaTTS(seed=seed, model_checkpoint=dia_checkpoint)
            if voice_prompts:
                tts.register_voice_prompts(voice_prompts)

            tmp_dir = tempfile.mkdtemp(prefix="dia_mvp_")
            seg_paths = []

            try:
                for i, block in enumerate(mini_transcripts):
                    out_wav = os.path.join(tmp_dir, f"block_{i:04d}.wav")
                    print(
                        f"[TTS] Generating block {i + 1}/{len(mini_transcripts)} -> "
                        f"{out_wav} ..."
                    )
                    tts.text_to_audio_file(block, out_wav)
                    seg_paths.append(out_wav)

                # Concatenate with pydub
                print("[audio] Concatenating audio segments ...")
                combined = None
                for p in seg_paths:
                    seg = AudioSegment.from_wav(p)
                    if combined is None:
                        combined = seg
                    else:
                        combined += seg

                # Export final mp3 (or wav)
                if combined is None:
                    raise RuntimeError("No audio segments produced") from None

                # Ensure output directory exists
                os.makedirs(os.path.dirname(out_audio_path) or ".", exist_ok=True)

                print(f"[audio] Exporting final to {out_audio_path} ...")
                combined.export(out_audio_path, format="mp3")
                print("[pipeline] Done.")
            except Exception as e:
                raise RuntimeError(f"Failed during TTS generation: {str(e)}") from e
            finally:
                # Cleanup
                shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize TTS: {str(e)}") from e

    except Exception as e:
        # Top-level error handling
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"\n[ERROR] Pipeline failed with {error_type}: {error_msg}")

        # Provide more context based on error type
        if "CUDA" in error_msg or "GPU" in error_msg:
            print("\nTroubleshooting tips:")
            print("- Ensure you have a CUDA-enabled GPU")
            print("- Check if you have enough VRAM (Dia model requires ~10GB)")
            print("- Verify PyTorch is installed with CUDA support")
        elif "API_KEY" in error_msg:
            print("\nTroubleshooting tips:")
            print("- Set OPENAI_API_KEY environment variable")
            print("- Check your API key is valid and has sufficient quota")
        elif "FileNotFoundError" in error_type:
            print("\nTroubleshooting tips:")
            print("- Verify the input file path is correct")
            print("- Check file permissions")
        elif "UnicodeDecodeError" in error_type:
            print("\nTroubleshooting tips:")
            print("- Check file encoding")
            print("- Try converting the file to UTF-8")

        raise  # Re-raise the exception to be caught by CLI
