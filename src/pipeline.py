#!/usr/bin/env python3
"""
This file contains the core pipeline for converting a transcript to a podcast.
"""

import json
import logging
import os
import random
import re
import subprocess  # nosec
import sys
import tempfile
from pathlib import Path
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

            # Post-process chunks: strip newlines and add speaker continuity tags
            processed_mini_transcripts = []
            for chunk in mini_transcripts:
                # Strip all newlines and replace with two spaces
                processed_chunk = chunk.replace("\n", "  ")

                # Find the last speaker tag in the chunk
                last_speaker_match = re.findall(r"\b(S\d+)\b", processed_chunk)
                if last_speaker_match:
                    last_speaker = last_speaker_match[-1]
                    # Add the appropriate speaker tag at the end
                    if last_speaker == "S1":
                        processed_chunk += "  [S2]"
                    elif last_speaker == "S2":
                        processed_chunk += "  [S1]"

                processed_mini_transcripts.append(processed_chunk)

            mini_transcripts = processed_mini_transcripts

            logger.info(
                f"Produced {len(mini_transcripts)} mini-transcript blocks "
                f"(5-10s preferred)."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to chunk transcript: {str(e)}") from e

        # Synthetic Voice Prompt Generation for Unprompted Speakers
        if voice_prompts is None:
            voice_prompts = {}

        # Identify all speakers and unprompted speakers
        all_speakers = {ln["speaker"] for ln in lines}
        prompted_speakers = set(voice_prompts.keys())
        unprompted_speakers = all_speakers - prompted_speakers

        # Generate synthetic prompts if conditions are met
        if (
            config.GENERATE_SYNTHETIC_PROMPTS
            and unprompted_speakers
            and len(mini_transcripts) > 1
        ):
            logger.info(
                f"Generating synthetic voice prompts for unprompted speakers: "
                f"{sorted(unprompted_speakers)}"
            )

            # Create output directory for synthetic prompts
            output_dir = Path(config.GENERATE_PROMPT_OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Synthetic prompts output directory: {output_dir}")

            # Generate synthetic prompts for each unprompted speaker
            for speaker_id in sorted(unprompted_speakers):
                logger.debug(f"Generating synthetic prompt for {speaker_id}")

                try:
                    # Build command for worker script using module execution for
                    #  robustness
                    cmd = [
                        sys.executable,
                        "-m",
                        "generate_prompt",
                        "--speaker-id",
                        speaker_id,
                        "--output-dir",
                        str(output_dir),
                    ]

                    # Add seed if provided
                    if seed is not None:
                        cmd.extend(["--seed", str(seed)])

                    # Add verbose flag if debug logging is enabled
                    if logger.getEffectiveLevel() <= logging.DEBUG:
                        cmd.append("--verbose")

                    logger.debug(f"Running worker script: {' '.join(cmd)}")

                    # Execute worker script
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )  # nosec

                    # Parse JSON output from worker script
                    try:
                        metadata = json.loads(result.stdout.strip())
                        logger.debug(f"Worker script output: {metadata}")
                    except json.JSONDecodeError as e:
                        raise RuntimeError(
                            f"Failed to parse worker script output as JSON: {e}\n"
                            f"stdout: {result.stdout}\n"
                            f"stderr: {result.stderr}"
                        ) from e

                    # Update voice_prompts dictionary with synthetic prompt
                    voice_prompts[speaker_id] = {
                        "path": metadata["audio_path"],
                        "transcript": metadata["stdout_transcript"],
                    }

                    logger.info(
                        f"Generated synthetic voice prompt for {speaker_id}: "
                        f"{metadata['audio_path']}"
                    )

                except subprocess.CalledProcessError as e:
                    logger.error(
                        f"Worker script failed for {speaker_id}: "
                        f"return code {e.returncode}\n"
                        f"stdout: {e.stdout}\n"
                        f"stderr: {e.stderr}"
                    )
                    raise RuntimeError(
                        f"Failed to generate synthetic prompt for {speaker_id}"
                    ) from e
                except Exception as e:
                    logger.error(
                        f"Unexpected error generating synthetic prompt for {speaker_id}"
                        f": {e}"
                    )
                    raise RuntimeError(
                        f"Failed to generate synthetic prompt for {speaker_id}: {e}"
                    ) from e

            logger.info(
                f"Successfully generated {len(unprompted_speakers)} synthetic "
                "voice prompts"
            )

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

                # Smart Prompt Unification: Determine speaker order based on first chunk
                speaker_order = sorted(high_fidelity_prompts.keys())  # Default fallback

                if mini_transcripts:
                    # Inspect the first chunk to determine the starting speaker
                    first_chunk = mini_transcripts[0]
                    first_speaker_match = re.search(r"^\[(S\d+)\]", first_chunk.strip())

                    if first_speaker_match:
                        starting_speaker = first_speaker_match.group(1)
                        logger.debug(
                            f"First chunk starts with speaker: {starting_speaker}"
                        )

                        # Reorder prompts to start with the conversation's
                        #  starting speaker
                        available_speakers = list(high_fidelity_prompts.keys())
                        if starting_speaker in available_speakers:
                            # Start with the conversation starter, then add
                            #  others in sorted order
                            speaker_order = [starting_speaker]
                            remaining_speakers = [
                                s
                                for s in sorted(available_speakers)
                                if s != starting_speaker
                            ]
                            speaker_order.extend(remaining_speakers)
                            logger.info(
                                f"Smart prompt ordering: {speaker_order} "
                                f"(conversation starts with {starting_speaker})"
                            )
                        else:
                            logger.debug(
                                f"Starting speaker {starting_speaker} not in "
                                "available prompts, using sorted order"
                            )
                    else:
                        logger.debug(
                            "Could not determine starting speaker from first chunk, "
                            "using sorted order"
                        )
                else:
                    logger.debug(
                        "No mini-transcripts available for smart ordering, using "
                        "sorted order"
                    )

                # Initialize combined audio segment and transcript parts
                combined_audio = None
                transcript_parts = []

                # Process prompts in smart-determined order
                for speaker in speaker_order:
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
