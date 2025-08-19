#!/usr/bin/env python3
"""
Worker script for generating synthetic voice prompts.

This script generates synthetic voice prompts for unprompted speakers using the DiaTTS
model. It runs in isolation from the main pipeline to ensure deterministic seeding
doesn't affect the main generation process.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.components.audio_generator.tts import DiaTTS


def setup_logging(level: int = logging.WARNING) -> None:
    """Setup basic logging for the worker script."""
    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(name)s - %(message)s",
        stream=sys.stderr,  # Log to stderr to keep stdout clean for JSON output
    )


def get_opposite_speaker(speaker_id: str) -> str:
    """
    Calculate the opposite speaker tag for continuity.

    Args:
        speaker_id: Current speaker ID (e.g., 'S1')

    Returns:
        Opposite speaker ID (e.g., 'S2' for 'S1')
    """
    if speaker_id == "S1":
        return "S2"
    elif speaker_id == "S2":
        return "S1"
    else:
        # For speakers beyond S1/S2, use S1 as default opposite
        return "S1"


def load_raw_text() -> str:
    """
    Load the raw text from the minimum generation asset file.

    Returns:
        Raw text content for generation

    Raises:
        FileNotFoundError: If the asset file doesn't exist
        RuntimeError: If the file is empty or cannot be read
    """
    asset_path = Path("assets/minimum_generation.one_speaker.txt")

    if not asset_path.exists():
        raise FileNotFoundError(f"Asset file not found: {asset_path}")

    try:
        content = asset_path.read_text(encoding="utf-8").strip()
        if not content:
            raise RuntimeError(f"Asset file is empty: {asset_path}")
        return content
    except Exception as e:
        raise RuntimeError(f"Failed to read asset file {asset_path}: {e}") from e


def generate_synthetic_prompt(
    speaker_id: str, seed: Optional[int], output_dir: str, verbose: bool = False
) -> dict:
    """
    Generate a synthetic voice prompt for the specified speaker.

    Args:
        speaker_id: Speaker ID (e.g., 'S1')
        seed: Random seed for deterministic generation (optional)
        output_dir: Directory to save generated files
        verbose: Enable verbose logging

    Returns:
        Dictionary containing speaker_id, audio_path, and stdout_transcript

    Raises:
        RuntimeError: If generation fails
    """
    logger = logging.getLogger(__name__)

    # Load raw text from asset file
    raw_text = load_raw_text()
    logger.debug(f"Loaded raw text: {raw_text[:50]}...")

    # Prepare transcripts according to task specification
    file_transcript = f"[{speaker_id}] {raw_text.strip()}"
    stdout_transcript = file_transcript.replace("\n", "  ")

    # Calculate opposite speaker for continuity tag
    opposite_speaker = get_opposite_speaker(speaker_id)
    generation_transcript = f"{stdout_transcript}  [{opposite_speaker}]"

    logger.debug(f"File transcript: {file_transcript[:50]}...")
    logger.debug(f"Stdout transcript: {stdout_transcript[:50]}...")
    logger.debug(f"Generation transcript: {generation_transcript[:50]}...")

    # Determine generation seed
    generation_seed = seed
    if generation_seed is None:
        # Generate temporary seed for this worker
        generation_seed = random.randint(0, 2**32 - 1)  # nosec
        logger.debug(f"Generated temporary seed: {generation_seed}")
    else:
        logger.debug(f"Using provided seed: {generation_seed}")

    # Initialize DiaTTS with deterministic seed
    logger.debug(f"Initializing DiaTTS with seed {generation_seed}")
    try:
        log_level = logging.DEBUG if verbose else logging.WARNING
        tts = DiaTTS(seed=generation_seed, log_level=log_level)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize DiaTTS: {e}") from e

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {output_path}")

    # Generate file paths using speaker_id and generation_seed with zero-padding
    audio_filename = f"{speaker_id}_seed_{generation_seed:08d}.wav"
    text_filename = f"{speaker_id}_seed_{generation_seed:08d}.txt"
    audio_path = output_path / audio_filename
    text_path = output_path / text_filename

    logger.debug(f"Audio file: {audio_path}")
    logger.debug(f"Text file: {text_path}")

    # Generate audio using batch processing (single item)
    try:
        logger.debug("Generating audio with DiaTTS...")
        audio_segments = tts.generate([generation_transcript], None, None)

        if not audio_segments:
            raise RuntimeError("No audio segments generated")

        # Export the audio segment to WAV file
        audio_segment = audio_segments[0]
        audio_segment.export(str(audio_path), format="wav")
        logger.debug(f"Saved audio to: {audio_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to generate audio: {e}") from e

    # Save the file transcript to text file
    try:
        text_path.write_text(file_transcript, encoding="utf-8")
        logger.debug(f"Saved text to: {text_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save text file: {e}") from e

    # Return metadata for orchestrator
    return {
        "speaker_id": speaker_id,
        "audio_path": str(audio_path),
        "stdout_transcript": stdout_transcript,
    }


def main() -> None:
    """Main entry point for the worker script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic voice prompts for unprompted speakers"
    )
    parser.add_argument("--speaker-id", required=True, help="Speaker ID (e.g., S1, S2)")
    parser.add_argument(
        "--seed", type=int, help="Random seed for deterministic generation"
    )
    parser.add_argument(
        "--output-dir",
        default="./",
        help="Output directory for generated files (default: ./)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.debug(f"Starting synthetic prompt generation for {args.speaker_id}")
    logger.debug(f"Seed: {args.seed}")
    logger.debug(f"Output directory: {args.output_dir}")

    try:
        # Generate synthetic prompt
        result = generate_synthetic_prompt(
            speaker_id=args.speaker_id,
            seed=args.seed,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

        # Output result as JSON to stdout
        json_output = json.dumps(result)
        print(json_output)
        logger.debug(f"Generated prompt successfully: {result}")

    except Exception as e:
        logger.error(f"Failed to generate synthetic prompt: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
