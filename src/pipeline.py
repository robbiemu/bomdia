#!/usr/bin/env python3
"""
This file contains the core pipeline for converting a transcript to a podcast.
"""

import logging
import os
import random
import re
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
    dry_run: bool = False,
    no_rehearsals: bool = False,
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

        # No Rehearsals Logic
        if no_rehearsals:
            # Validate the lines for no rehearsals mode
            validation_errors = []

            # Check for pause placeholders (consecutive speaker lines)
            for line in lines:
                if config.PAUSE_PLACEHOLDER in line["text"]:
                    validation_errors.append(
                        "Transcript contains consecutive speaker lines that "
                        "require merging. Use the standard pipeline or manually "
                        "merge consecutive lines."
                    )
                    break

            # Check speaker tags strictly match S<number> format
            for line in lines:
                if not re.match(r"^S\d+$", line["speaker"]):
                    validation_errors.append(
                        f"Invalid speaker tag '{line['speaker']}'. "
                        "All speaker tags must strictly match the 'S<number>' "
                        "format (e.g., S1, S2)."
                    )

            if validation_errors:
                raise ValueError(
                    "Transcript validation failed for --no-rehearsals mode:\n"
                    + "\n".join(f"  - {error}" for error in validation_errors)
                )

            processed = lines
            logger.info(
                "Skipping rehearsals as requested. Using parsed transcript directly."
            )
        elif not config.LLM_SPEC:
            # Handle the case where no LLM is available (e.g., run a simplified
            #  rule-based pass or exit)
            logger.warning(
                "LLM not available. Skipping advanced transcript enhancement."
            )
            processed = lines  # Fallback to un-enhanced lines
        else:
            director = Director(transcript=lines)
            processed = director.run_rehearsal()

        # Dry Run Logic
        if dry_run:
            logger.info("--- DRY RUN MODE: Final Transcript ---")
            for line in processed:
                print(f"[{line['speaker']}] {line['text']}")
            logger.info("--- DRY RUN COMPLETE ---")
            return

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

        # Prompt Pre-computation - Combine voice prompts for batch processing
        unified_audio_prompt_path = None
        unified_transcript_prompt = None

        if voice_prompts:
            # Check if we have any high-fidelity prompts (with both path and transcript)
            high_fidelity_prompts = {
                speaker: details
                for speaker, details in voice_prompts.items()
                if details and details.get("path") and details.get("transcript")
            }

            if high_fidelity_prompts:
                logger.info(
                    "Combining high-fidelity voice prompts for batch processing"
                )

                # Initialize combined audio segment and transcript parts
                combined_audio = None
                transcript_parts = []

                # Process prompts in sorted order for deterministic results
                for speaker in sorted(high_fidelity_prompts.keys()):
                    details = high_fidelity_prompts[speaker]
                    audio_path = details["path"]
                    transcript = details["transcript"]

                    # Load and combine audio
                    try:
                        speaker_audio = AudioSegment.from_file(audio_path)
                        if combined_audio is None:
                            combined_audio = speaker_audio
                        else:
                            combined_audio += speaker_audio

                        # Format transcript with speaker tags
                        formatted_transcript = f"[{speaker}]{transcript}[{speaker}]"
                        transcript_parts.append(formatted_transcript)

                        logger.debug(f"Added {speaker} prompt: {audio_path}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to load audio prompt for {speaker}: {str(e)}"
                        )

                if combined_audio and transcript_parts:
                    # Create unified transcript prompt
                    unified_transcript_prompt = " ".join(transcript_parts)

                    # Create temporary file for combined audio
                    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio_file:
                        unified_audio_prompt_path = temp_audio_file.name

                    # Export combined audio to temporary file
                    combined_audio.export(unified_audio_prompt_path, format="wav")

                    logger.info(
                        f"Created unified audio prompt: {unified_audio_prompt_path}"
                    )
                    logger.info(
                        "Created unified transcript prompt: "
                        f"{unified_transcript_prompt}"
                    )

        # Mandatory seeding for multi-block pure TTS
        all_speakers = {ln["speaker"] for ln in lines}
        prompted_speakers = set(voice_prompts.keys()) if voice_prompts else set()
        unprompted_speakers = all_speakers - prompted_speakers

        if unprompted_speakers and len(mini_transcripts) > 1:
            # The Dia model automatically compiles functions for performance, but
            #  compiled functions can retain internal state that doesn't get reset
            config.DIA_GENERATE_PARAMS["use_torch_compile"] = False

            if seed is None:
                # Generate a secure random seed if one is required but not provided
                seed = random.randint(0, 2**32 - 1)  # nosec B311
                logger.debug(
                    "No seed provided for consistent voice generation; using "
                    f"auto-generated seed: {seed}"
                )

        # TTS generation with Dia using batch processing
        try:
            # Get the current logger level for the TTS
            log_level = logger.getEffectiveLevel()

            tts = DiaTTS(
                seed=seed, model_checkpoint=dia_checkpoint, log_level=log_level
            )
            if voice_prompts:
                tts.register_voice_prompts(voice_prompts)

            try:
                # Use batch processing for all mini-transcripts in a single call
                logger.info("Generating audio using batch processing...")
                audio_segments = tts.generate(
                    mini_transcripts,
                    unified_audio_prompt_path,
                    unified_transcript_prompt,
                )

                # Concatenate AudioSegment objects directly
                logger.info("Concatenating audio segments ...")
                combined = None
                for i, segment in enumerate(audio_segments):
                    logger.debug(f"Concatenating segment {i + 1}/{len(audio_segments)}")
                    if combined is None:
                        combined = segment
                    else:
                        combined += segment

                # Export final mp3 (or wav)
                if combined is None:
                    raise RuntimeError("No audio segments produced") from None

                # Ensure output directory exists
                os.makedirs(os.path.dirname(out_audio_path) or ".", exist_ok=True)

                logger.info(f"Exporting final to {out_audio_path} ...")
                combined.export(out_audio_path, format=config.AUDIO_OUTPUT_FORMAT)
                logger.info("Done.")
            except Exception as e:
                raise RuntimeError(f"Failed during TTS generation: {str(e)}") from e
            finally:
                # Cleanup temporary unified audio prompt file
                if unified_audio_prompt_path and os.path.exists(
                    unified_audio_prompt_path
                ):
                    try:
                        os.remove(unified_audio_prompt_path)
                        logger.debug(
                            "Cleaned up temporary audio file: "
                            f"{unified_audio_prompt_path}"
                        )
                    except OSError as e:
                        logger.warning(f"Failed to clean up temporary audio file: {e}")

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
