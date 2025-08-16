#!/usr/bin/env python3
"""
MVP transcript -> podcast pipeline using:
 - LangGraph for the "sub-agent" verbal-tag injection
 - Dia (via transformers AutoProcessor + DiaForConditionalGeneration) for TTS

Quickstart (recommended in a new venv):
  pip install -U langgraph langchain langchain-openai transformers torch soundfile
  pydub datasets python-dotenv

Requirements / notes:
 - You will want a CUDA GPU for Dia (the model authors note ~10GB VRAM for the
   full model).
 - You should set OPENAI_API_KEY (or edit to use another init_chat_model
   provider).
 - ffmpeg is required by pydub to concatenate audio.
 - This is MVP code: sanitize inputs and improve error handling for production
   use.
"""

import argparse
import logging
import os

from pydub import AudioSegment
from shared.logging import setup_logger
from src.pipeline import run_pipeline


def validate_audio_file(file_path: str) -> bool:
    """
    Validate an audio file for voice cloning parameters.

    Args:
        file_path (str): Path to the audio file

    Returns:
        bool: True if validation passes, False otherwise
    """
    logger = logging.getLogger(__name__)

    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        return False

    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)

        # Get file properties
        duration_ms = len(audio)
        sample_rate = audio.frame_rate
        sample_width = audio.sample_width * 8  # Convert bytes to bits

        # Validate file format (WAV is preferred)
        is_wav = file_path.lower().endswith(".wav")
        if not is_wav:
            logger.warning(
                f"Audio file '{file_path}' is not WAV format. "
                f"MP3 or other compressed formats may introduce artifacts."
            )

        # Validate sample rate (16-bit, 22050 Hz or 44100 Hz recommended)
        if sample_rate not in [22050, 44100]:
            logger.warning(
                f"Audio file '{file_path}' has sample rate {sample_rate} Hz. "
                f"Use 22050 Hz or 44100 Hz for best results."
            )

        # Validate bit depth (16-bit recommended)
        if sample_width != 16:
            logger.warning(
                f"Audio file '{file_path}' has bit depth {sample_width}-bit. "
                f"Use 16-bit for best results."
            )

        # Validate audio length (ideal 5-15 seconds)
        duration_sec = duration_ms / 1000
        if duration_sec < 3:
            logger.warning(
                f"Audio file '{file_path}' is too short "
                f"({duration_sec:.2f}s). Minimum 3-4 seconds recommended."
            )
        elif duration_sec > 20:
            logger.warning(
                f"Audio file '{file_path}' is too long ({duration_sec:.2f}s). "
                f"Maximum 20 seconds recommended."
            )
        elif not (5 <= duration_sec <= 15):
            logger.info(
                f"Audio file '{file_path}' length ({duration_sec:.2f}s) "
                f"is outside ideal range (5-15s)."
            )

        return True

    except Exception as e:
        logger.error(f"Failed to load audio file '{file_path}': {str(e)}")
        return False


def validate_file_exists(file_path: str, file_description: str) -> bool:
    """
    Validate that a file exists.

    Args:
        file_path (str): Path to the file
        file_description (str): Description of the file for error messages

    Returns:
        bool: True if file exists, False otherwise
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(file_path):
        logger.error(f"{file_description} not found: {file_path}")
        return False
    return True


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a transcript to a podcast using Dia TTS."
    )
    parser.add_argument(
        "input_path", help="Path to the input transcript file (srt, txt)."
    )
    parser.add_argument(
        "output_path",
        nargs="?",  # This makes the positional argument optional
        default=None,  # Provide a default value if it's not given
        help="Path to save the final MP3 audio file. (Not required for --dry-run)",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible voice selection."
    )
    parser.add_argument(
        "--s1-voice", type=str, help="Path to an audio prompt file for Speaker 1."
    )
    parser.add_argument(
        "--s1-transcript",
        type=str,
        help="Transcript for the S1 voice prompt. Requires --s1-voice.",
    )
    parser.add_argument(
        "--s2-voice", type=str, help="Path to an audio prompt file for Speaker 2."
    )
    parser.add_argument(
        "--s2-transcript",
        type=str,
        help="Transcript for the S2 voice prompt. Requires --s2-voice.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Set logging level to INFO. Useful for seeing standard process flow.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set a specific logging level.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Execute the entire agentic rehearsal process and print the final "
            "transcript without generating audio."
        ),
    )
    parser.add_argument(
        "--no-rehearsals",
        action="store_true",
        help=(
            "Bypass the Director/Actor workflow and send the parsed "
            "transcript directly to audio generation."
        ),
    )

    args = parser.parse_args()

    # Add a new validation block right after parsing
    if not args.dry_run and args.output_path is None:
        parser.error("output_path is required unless the --dry-run flag is used.")

    # Determine logging level
    if args.verbosity != "WARNING":
        # --verbosity flag takes precedence
        log_level = getattr(logging, args.verbosity)
    elif args.verbose:
        # -v flag sets level to INFO
        log_level = logging.INFO
    else:
        # Default level
        log_level = logging.WARNING

    # Initialize logger
    setup_logger(log_level)
    logger = logging.getLogger(__name__)

    # Validate input file exists
    if not validate_file_exists(args.input_path, "Input transcript file"):
        exit(1)

    # Validate transcript and voice prompt combinations
    if args.s1_transcript and not args.s1_voice:
        logger.error("--s1-transcript requires --s1-voice.")
        exit(1)
    if args.s2_transcript and not args.s2_voice:
        logger.error("--s2-transcript requires --s2-voice.")
        exit(1)

    # Build a dictionary of voice prompts and validate them
    voice_prompts = {}
    if args.s1_voice:
        if not validate_file_exists(args.s1_voice, "Speaker 1 voice prompt file"):
            exit(1)
        if not validate_audio_file(args.s1_voice):
            logger.error("Speaker 1 voice prompt file failed validation.")
            exit(1)
        voice_prompts["S1"] = {"path": args.s1_voice, "transcript": args.s1_transcript}
    if args.s2_voice:
        if not validate_file_exists(args.s2_voice, "Speaker 2 voice prompt file"):
            exit(1)
        if not validate_audio_file(args.s2_voice):
            logger.error("Speaker 2 voice prompt file failed validation.")
            exit(1)
        voice_prompts["S2"] = {"path": args.s2_voice, "transcript": args.s2_transcript}

    # Pass the prompts to the pipeline
    try:
        run_pipeline(
            args.input_path,
            args.output_path,
            voice_prompts=voice_prompts,
            seed=args.seed,
            dry_run=args.dry_run,
            no_rehearsals=args.no_rehearsals,
        )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        exit(1)
