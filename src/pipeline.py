#!/usr/bin/env python3
"""
This file contains the core pipeline for converting a transcript to a podcast.
"""

import logging
import os
import secrets
import shutil
import tempfile
from typing import Dict, Optional

from pydub import AudioSegment
from shared.config import config

from src.components.audio_generator import (
    DiaTTS,
    chunk_to_5_10s,
)
from src.components.transcript_parser import (
    ingest_transcript,
    merge_consecutive_lines,
)
from src.components.verbal_tag_injector.director import Director

# Initialize logger
logger = logging.getLogger(__name__)


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

        # ... after lines are ingested and merged
        if not config.LLM_SPEC:
            # Handle the case where no LLM is available (e.g., run a simplified
            #  rule-based pass or exit)
            logger.warning(
                "LLM not available. Skipping advanced transcript enhancement."
            )
            processed = lines  # Fallback to un-enhanced lines
        else:
            director = Director(transcript=lines)
            processed = director.run_rehearsal()

        # Chunk into 5-10s mini transcripts
        try:
            mini_transcripts = chunk_to_5_10s(processed)
            if not mini_transcripts:
                raise ValueError(
                    "No mini-transcripts generated from the input"
                ) from None
            logger.info(
                f"Produced {len(mini_transcripts)} mini-transcript blocks "
                f"(5-10s preferred)."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to chunk transcript: {str(e)}") from e

        # Mandatory seeding for multi-block pure TTS
        all_speakers = {ln["speaker"] for ln in lines}
        prompted_speakers = set(voice_prompts.keys()) if voice_prompts else set()
        unprompted_speakers = all_speakers - prompted_speakers

        if unprompted_speakers and len(mini_transcripts) > 1 and seed is None:
            # Generate a secure random seed if one is required but not provided
            seed = secrets.randbelow(2**32 - 1)
            logger.debug(
                "No seed provided for consistent voice generation; using "
                f"auto-generated seed: {seed}"
            )

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
                    logger.info(
                        f"Generating block {i + 1}/{len(mini_transcripts)} -> "
                        f"{out_wav} ..."
                    )
                    tts.text_to_audio_file(block, out_wav)
                    seg_paths.append(out_wav)

                # Concatenate with pydub
                logger.info("Concatenating audio segments ...")
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

                logger.info(f"Exporting final to {out_audio_path} ...")
                combined.export(out_audio_path, format="mp3")
                logger.info("Done.")
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
        logger.error(f"Pipeline failed with {error_type}: {error_msg}")

        # Provide more context based on error type
        if "CUDA" in error_msg or "GPU" in error_msg:
            logger.info("\nTroubleshooting tips:")
            logger.info("- Ensure you have a CUDA-enabled GPU")
            logger.info("- Check if you have enough VRAM (Dia model requires ~10GB)")
            logger.info("- Verify PyTorch is installed with CUDA support")
        elif "API_KEY" in error_msg:
            logger.info("\nTroubleshooting tips:")
            logger.info("- Set OPENAI_API_KEY environment variable")
            logger.info("- Check your API key is valid and has sufficient quota")
        elif "FileNotFoundError" in error_type:
            logger.info("\nTroubleshooting tips:")
            logger.info("- Verify the input file path is correct")
            logger.info("- Check file permissions")
        elif "UnicodeDecodeError" in error_type:
            logger.info("\nTroubleshooting tips:")
            logger.info("- Check file encoding")
            logger.info("- Try converting the file to UTF-8")

        raise  # Re-raise the exception to be caught by CLI
