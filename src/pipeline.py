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
from typing import Dict, List, Optional, Tuple

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


class TranscriptProcessor:
    """Handles transcript ingestion, parsing, and processing."""

    @staticmethod
    def load_and_parse_transcript(input_path: str) -> List[Dict]:
        """Load and parse transcript from file."""
        try:
            lines = ingest_transcript(input_path)
            if not lines:
                raise ValueError("No content found in the transcript file")
            return lines
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

    @staticmethod
    def normalize_transcript(lines: List[Dict]) -> List[Dict]:
        """Normalize speaker tags and merge consecutive lines."""
        try:
            normalized_lines = [
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
            return merge_consecutive_lines(normalized_lines)
        except Exception as e:
            raise RuntimeError(f"Failed to normalize transcript: {str(e)}") from e

    @staticmethod
    def validate_no_rehearsals_mode(lines: List[Dict]) -> None:
        """Validate transcript for no-rehearsals mode."""
        validation_errors = []

        # Check for pause placeholders
        for line in lines:
            if config.PAUSE_PLACEHOLDER in line["text"]:
                validation_errors.append(
                    "Transcript contains consecutive speaker lines that "
                    "require merging. Use the standard pipeline or manually "
                    "merge consecutive lines."
                )
                break

        # Check speaker tags format
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

    @staticmethod
    def process_transcript(lines: List[Dict], no_rehearsals: bool) -> List[Dict]:
        """Process transcript through director or return as-is."""
        if no_rehearsals:
            TranscriptProcessor.validate_no_rehearsals_mode(lines)
            logger.info(
                "Skipping rehearsals as requested. Using parsed transcript directly."
            )
            return lines
        elif not config.LLM_SPEC:
            logger.warning(
                "LLM not available. Skipping advanced transcript enhancement."
            )
            return lines
        else:
            director = Director(transcript=lines)
            return director.run_rehearsal()


class ChunkProcessor:
    """Handles transcript chunking and processing."""

    @staticmethod
    def create_mini_transcripts(processed: List[Dict]) -> List[str]:
        """Create and process mini-transcripts from processed transcript."""
        try:
            mini_transcripts = chunk_to_5_10s(processed)
            if not mini_transcripts:
                raise ValueError("No mini-transcripts generated from the input")

            # Post-process chunks
            processed_mini_transcripts = []
            for chunk in mini_transcripts:
                processed_chunk = ChunkProcessor._process_chunk(chunk)
                processed_mini_transcripts.append(processed_chunk)

            return processed_mini_transcripts
        except Exception as e:
            raise RuntimeError(f"Failed to chunk transcript: {str(e)}") from e

    @staticmethod
    def _process_chunk(chunk: str) -> str:
        """Process individual chunk: strip newlines and add speaker continuity tags."""
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

        return processed_chunk


class VoicePromptGenerator:
    """Handles synthetic voice prompt generation."""

    @staticmethod
    def generate_synthetic_prompts(
        lines: List[Dict],
        voice_prompts: Dict,
        mini_transcripts: List[str],
        seed: Optional[int],
    ) -> Dict:
        """Generate synthetic voice prompts for unprompted speakers."""
        all_speakers = {ln["speaker"] for ln in lines}
        prompted_speakers = set(voice_prompts.keys())
        unprompted_speakers = all_speakers - prompted_speakers

        if not (
            config.GENERATE_SYNTHETIC_PROMPTS
            and unprompted_speakers
            and len(mini_transcripts) > 1
        ):
            return voice_prompts

        logger.info(
            f"Generating synthetic voice prompts for unprompted speakers: "
            f"{sorted(unprompted_speakers)}"
        )

        # Create output directory
        output_dir = Path(config.GENERATE_PROMPT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Synthetic prompts output directory: {output_dir}")

        # Generate prompts for each unprompted speaker
        for speaker_id in sorted(unprompted_speakers):
            try:
                VoicePromptGenerator._generate_speaker_prompt(
                    speaker_id, output_dir, seed, voice_prompts
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate synthetic prompt for {speaker_id}: {e}"
                ) from e

        logger.info(
            f"Successfully generated {len(unprompted_speakers)} synthetic voice prompts"
        )
        return voice_prompts

    @staticmethod
    def _generate_speaker_prompt(
        speaker_id: str, output_dir: Path, seed: Optional[int], voice_prompts: Dict
    ) -> None:
        """Generate synthetic prompt for a single speaker."""
        logger.debug(f"Generating synthetic prompt for {speaker_id}")

        # Build command for worker script
        cmd = [
            sys.executable,
            "-m",
            "generate_prompt",
            "--speaker-id",
            speaker_id,
            "--output-dir",
            str(output_dir),
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        if logger.getEffectiveLevel() <= logging.DEBUG:
            cmd.append("--verbose")

        logger.debug(f"Running worker script: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )  # nosec

            # Parse JSON output
            try:
                metadata = json.loads(result.stdout.strip())
                logger.debug(f"Worker script output: {metadata}")
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Failed to parse worker script output as JSON: {e}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                ) from e

            # Update voice_prompts dictionary
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
                f"Worker script failed for {speaker_id}: return code {e.returncode}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            )
            raise


class AudioPromptProcessor:
    """Handles audio prompt processing and per-chunk prompt variation creation."""

    @staticmethod
    def create_prompt_variations(voice_prompts: Dict) -> Dict[str, Dict[str, str]]:
        """Create pre-computed prompt variations for optimal speaker alternation."""
        high_fidelity_prompts = {
            s: d
            for s, d in voice_prompts.items()
            if d.get("path") and d.get("transcript")
        }
        if not high_fidelity_prompts or len(high_fidelity_prompts) != 2:
            logger.warning(
                "Prompt variations require exactly 2 high-fidelity prompts. Skipping."
            )
            return {}

        logger.info("Creating prompt variations for per-chunk optimization")
        prompt_variations = {}
        speakers = sorted(high_fidelity_prompts.keys())  # Ensures we have ['S1', 'S2']

        # Create S1-first variation (order S1, then S2)
        audio1, trans1 = AudioPromptProcessor._create_variation(
            high_fidelity_prompts, [speakers[0], speakers[1]]
        )
        if audio1 and trans1:
            prompt_variations["S1_first"] = {"audio": audio1, "transcript": trans1}

        # Create S2-first variation (order S2, then S1)
        audio2, trans2 = AudioPromptProcessor._create_variation(
            high_fidelity_prompts, [speakers[1], speakers[0]]
        )
        if audio2 and trans2:
            prompt_variations["S2_first"] = {"audio": audio2, "transcript": trans2}

        return prompt_variations

    @staticmethod
    def _create_variation(
        prompts: Dict, speaker_order: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Create a single prompt variation with the specified speaker order."""
        combined_audio = None
        transcript_parts = []
        for speaker_id in speaker_order:
            details = prompts[speaker_id]
            try:
                audio = AudioSegment.from_file(details["path"])
                combined_audio = (
                    audio if combined_audio is None else combined_audio + audio
                )
                transcript_parts.append(details["transcript"])
            except Exception as e:
                logger.warning(f"Failed to load audio prompt for {speaker_id}: {e}")
                return None, None

        if not combined_audio:
            return None, None

        unified_transcript = " ".join(transcript_parts)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            unified_audio_path = temp_file.name
        combined_audio.export(unified_audio_path, format="wav")
        return unified_audio_path, unified_transcript

    @staticmethod
    def build_batch_payload(
        mini_transcripts: List[str],
        prompt_variations: Dict[str, Dict[str, str]],
        high_fidelity_prompts: Dict[str, Dict[str, str]],
    ) -> Tuple[List[str], List[Optional[str]]]:
        """Build final batch payload with per-chunk prompt selection."""
        final_texts = []
        final_audio_prompts = []

        if not high_fidelity_prompts:
            return mini_transcripts, [None] * len(mini_transcripts)

        num_prompts = len(high_fidelity_prompts)

        for chunk in mini_transcripts:
            starting_speaker = AudioPromptProcessor._get_starting_speaker(chunk)
            prompt_transcript = ""
            prompt_audio_path = None

            if num_prompts == 1:
                speaker_id = list(high_fidelity_prompts.keys())[0]
                details = high_fidelity_prompts[speaker_id]
                prompt_transcript = details["transcript"]
                prompt_audio_path = details["path"]
                if starting_speaker == speaker_id:
                    opposite_speaker = "S2" if speaker_id == "S1" else "S1"
                    prompt_transcript += f"  [{opposite_speaker}]"

            elif num_prompts == 2 and prompt_variations:
                # If chunk starts with S1, we need the prompt that ends with S2
                #  (the S1-first variation)
                variation_key = "S1_first" if starting_speaker == "S1" else "S2_first"
                variation = prompt_variations.get(variation_key)
                if variation:
                    prompt_transcript = variation["transcript"]
                    prompt_audio_path = variation["audio"]

            # Prepend the selected prompt transcript to the chunk
            final_texts.append(
                f"{prompt_transcript} {chunk}" if prompt_transcript else chunk
            )
            final_audio_prompts.append(prompt_audio_path)

        return final_texts, final_audio_prompts

    @staticmethod
    def _get_starting_speaker(chunk: str) -> Optional[str]:
        """Extract the starting speaker from a chunk."""
        match = re.search(r"^\[(S\d+)\]", chunk.strip())
        return match.group(1) if match else None

    @staticmethod
    def cleanup_temp_files(prompt_variations: Dict[str, Dict[str, str]]) -> None:
        """Clean up temporary audio files from prompt variations."""
        for variation in prompt_variations.values():
            if audio_path := variation.get("audio"):
                try:
                    os.remove(audio_path)
                except OSError as e:
                    logger.warning(f"Failed to clean up temp file {audio_path}: {e}")


class AudioGenerator:
    """Handles TTS generation and audio processing."""

    @staticmethod
    def configure_tts_settings(
        lines: List[Dict],
        voice_prompts: Dict,
        mini_transcripts: List[str],
        seed: Optional[int],
    ) -> Optional[int]:
        """Configure TTS settings and return appropriate seed."""
        all_speakers = {ln["speaker"] for ln in lines}
        prompted_speakers = set(voice_prompts.keys()) if voice_prompts else set()
        unprompted_speakers = all_speakers - prompted_speakers

        if unprompted_speakers and len(mini_transcripts) > 1:
            config.DIA_GENERATE_PARAMS["use_torch_compile"] = False
            if seed is None:
                seed = random.randint(0, 2**32 - 1)  # nosec B311
                logger.debug(
                    f"Auto-generated seed for consistent voice generation: {seed}"
                )
        return seed

    @staticmethod
    def generate_audio(
        tts: DiaTTS,
        final_texts: List[str],
        final_audio_prompts: List[Optional[str]],
        out_audio_path: str,
    ) -> None:
        """
        Generates final audio using a pre-configured TTS instance.

        Args:
            tts: An initialized and registered DiaTTS instance.
            final_texts: The final list of text payloads to generate.
            final_audio_prompts: The final list of audio prompts for each payload.
            out_audio_path: The path to save the final audio file.
        """
        logger.info(
            "Generating audio using optimized batch processing with per-chunk "
            "prompts..."
        )
        audio_segments = tts.generate(
            texts=final_texts, audio_prompts=final_audio_prompts
        )

        combined = AudioGenerator._concatenate_segments(audio_segments)
        AudioGenerator._export_audio(combined, out_audio_path)

    @staticmethod
    def _concatenate_segments(audio_segments: List[AudioSegment]) -> AudioSegment:
        """Concatenate audio segments into a single AudioSegment."""
        logger.info("Concatenating audio segments ...")
        combined = None
        for segment in audio_segments:
            combined = segment if combined is None else combined + segment
        if combined is None:
            raise RuntimeError("No audio segments produced")
        return combined

    @staticmethod
    def _export_audio(combined: AudioSegment, out_audio_path: str) -> None:
        """Export audio to the specified path."""
        os.makedirs(os.path.dirname(out_audio_path) or ".", exist_ok=True)
        logger.info(f"Exporting final to {out_audio_path} ...")
        combined.export(out_audio_path, format=config.AUDIO_OUTPUT_FORMAT)
        logger.info("Done.")


class PipelineErrorHandler:
    """Handles pipeline errors and provides troubleshooting guidance."""

    @staticmethod
    def handle_error(e: Exception) -> None:
        """Handle pipeline errors with contextual troubleshooting tips."""
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"Pipeline failed with {error_type}: {error_msg}")

        # Provide context-specific troubleshooting tips
        if "CUDA" in error_msg or "GPU" in error_msg:
            PipelineErrorHandler._log_gpu_tips()
        elif "API_KEY" in error_msg:
            PipelineErrorHandler._log_api_tips()
        elif "FileNotFoundError" in error_type:
            PipelineErrorHandler._log_file_tips()
        elif "UnicodeDecodeError" in error_type:
            PipelineErrorHandler._log_encoding_tips()

        raise  # Re-raise the exception

    @staticmethod
    def _log_gpu_tips() -> None:
        """Log GPU-related troubleshooting tips."""
        logger.info("\nTroubleshooting tips:")
        logger.info("- Ensure you have a CUDA-enabled GPU")
        logger.info("- Check if you have enough VRAM (Dia model requires ~10GB)")
        logger.info("- Verify PyTorch is installed with CUDA support")

    @staticmethod
    def _log_api_tips() -> None:
        """Log API-related troubleshooting tips."""
        logger.info("\nTroubleshooting tips:")
        logger.info("- Set OPENAI_API_KEY environment variable")
        logger.info("- Check your API key is valid and has sufficient quota")

    @staticmethod
    def _log_file_tips() -> None:
        """Log file-related troubleshooting tips."""
        logger.info("\nTroubleshooting tips:")
        logger.info("- Verify the input file path is correct")
        logger.info("- Check file permissions")

    @staticmethod
    def _log_encoding_tips() -> None:
        """Log encoding-related troubleshooting tips."""
        logger.info("\nTroubleshooting tips:")
        logger.info("- Check file encoding")
        logger.info("- Try converting the file to UTF-8")


def run_pipeline(
    input_path: str,
    out_audio_path: str,
    dia_checkpoint: str = config.DIA_CHECKPOINT,
    voice_prompts: Optional[Dict] = None,
    seed: Optional[int] = config.SEED,
    dry_run: bool = False,
    no_rehearsals: bool = False,
) -> None:
    """Main pipeline function for converting transcript to podcast."""
    prompt_variations = {}
    try:
        if voice_prompts is None:
            voice_prompts = {}

        lines = TranscriptProcessor.load_and_parse_transcript(input_path)
        lines = TranscriptProcessor.normalize_transcript(lines)
        processed = TranscriptProcessor.process_transcript(lines, no_rehearsals)

        if dry_run:
            logger.info("--- DRY RUN MODE: Final Transcript ---")
            for line in processed:
                print(f"[{line['speaker']}] {line['text']}")
            logger.info("--- DRY RUN COMPLETE ---")
            return

        mini_transcripts = ChunkProcessor.create_mini_transcripts(processed)
        log_of_chunked_transcripts = ",\n".join([f'"{t}"' for t in mini_transcripts])
        logger.info(
            "Preparing to render the script as these chunks:\n"
            f"{log_of_chunked_transcripts}"
        )

        voice_prompts = VoicePromptGenerator.generate_synthetic_prompts(
            lines, voice_prompts, mini_transcripts, seed
        )

        high_fidelity_prompts = {
            s: d
            for s, d in voice_prompts.items()
            if d.get("path") and d.get("transcript")
        }

        prompt_variations = AudioPromptProcessor.create_prompt_variations(
            high_fidelity_prompts
        )

        final_texts, final_audio_prompts = AudioPromptProcessor.build_batch_payload(
            mini_transcripts, prompt_variations, high_fidelity_prompts
        )

        seed = AudioGenerator.configure_tts_settings(
            lines, voice_prompts, mini_transcripts, seed
        )

        # Instantiate, configure, and pass the TTS object
        tts = DiaTTS(
            seed=seed,
            model_checkpoint=dia_checkpoint,
            log_level=logger.getEffectiveLevel(),
        )
        if high_fidelity_prompts:
            tts.register_voice_prompts(high_fidelity_prompts)

        # Call the corrected generate_audio method with the right 4 arguments
        AudioGenerator.generate_audio(
            tts,
            final_texts,
            final_audio_prompts,
            out_audio_path,
        )

    except Exception as e:
        PipelineErrorHandler.handle_error(e)
    finally:
        # Ensure temporary prompt variations are always cleaned up
        AudioPromptProcessor.cleanup_temp_files(prompt_variations)
