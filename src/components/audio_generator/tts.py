import logging
import os
import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from dia.model import Dia
from pydub import AudioSegment
from shared.config import config

# Initialize logger
logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """
    Selects the appropriate torch device based on configuration and availability.
    Priority: Config Override > CUDA > MPS > CPU.
    """
    preferred_device = config.DIA_DEVICE

    if preferred_device != "auto":
        # User has specified a device
        if preferred_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA device specified but not available. Falling back to "
                "auto-detection."
            )
        elif preferred_device == "mps" and not torch.backends.mps.is_available():
            logger.warning(
                "MPS device specified but not available. Falling back to "
                "auto-detection."
            )
        elif preferred_device in ["cuda", "mps", "cpu"]:
            logger.debug(f"Using specified device: {preferred_device}")
            return torch.device(preferred_device)
        else:
            logger.warning(
                f"Invalid device '{preferred_device}' specified. Falling back to "
                "auto-detection."
            )

    # Auto-detection logic
    if torch.cuda.is_available():
        logger.debug("Auto-detected and using device: cuda")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.debug("Auto-detected and using device: mps")
        return torch.device("mps")

    logger.debug("Auto-detected and using device: cpu")
    return torch.device("cpu")


class DiaTTS:
    """
    Text-to-speech converter using the official Dia library wrapper.
    This approach is based on the official usage examples provided in the
    nari-labs/dia repository.
    """

    def __init__(
        self,
        seed: Optional[int],
        model_checkpoint: str = "nari-labs/Dia-1.6B-0626",
        device: str | None = None,
        log_level: int = logging.WARNING,
    ) -> None:
        """
        Initializes the TTS engine using the Dia library's native method.

        Args:
            model_checkpoint (str): The Hugging Face model identifier.
            device (str, optional): Device to run on ('cuda', 'mps', or 'cpu').
                                    If provided, overrides config settings.
            log_level (int): Logging level for verbose output.
        """
        # The device is now determined by the central helper function
        self.device = device if device is not None else _get_device().type
        self.seed = seed  # Save for per-block resets
        self.log_level = log_level

        logger.info(
            f"Loading model {model_checkpoint} via official Dia library on "
            f"device {self.device}..."
        )

        # Use configured compute dtype, with fallback to defaults
        try:
            # Ensure DIA_COMPUTE_DTYPE is a string before using getattr
            dtype_name = (
                config.DIA_COMPUTE_DTYPE
                if isinstance(config.DIA_COMPUTE_DTYPE, str)
                else "float32"
            )
            compute_dtype = getattr(torch, dtype_name)
        except (AttributeError, TypeError):
            logger.warning(
                f"Invalid compute dtype '{config.DIA_COMPUTE_DTYPE}', "
                "falling back to float32"
            )
            compute_dtype = torch.float32

        if self.seed is not None:
            logger.info(f"Setting seed to '{self.seed}' for voice selection")
            self._set_seed(self.seed)

        # The Dia library handles device placement internally.
        # We pass the device directly during creation.
        self.model = Dia.from_pretrained(
            model_checkpoint,
            compute_dtype=compute_dtype,
            device=self.device,
        )
        self._voice_prompt_details: Dict[str, Dict[str, Optional[str]]] = {}

        logger.info("Model loaded successfully.")

    def _set_seed(self, seed: int) -> None:
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        torch.use_deterministic_algorithms(True)
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Ensure deterministic behavior for cuDNN (if used)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            if os.getenv("CUBLAS_WORKSPACE_CONFIG") is None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            # Enable deterministic behavior for MPS
            os.environ["PYTORCH_MPS_DETERMINISTIC"] = "1"

        if hasattr(torch.backends.mps, "allow_higher_precision_reduce"):
            torch.backends.mps.allow_higher_precision_reduce = True

    def register_voice_prompts(
        self, voice_prompts: Dict[str, Dict[str, Optional[str]]]
    ) -> None:
        """
        Analyzes audio files to generate and store speaker embeddings and prompt info.
        Args:
            voice_prompts: A dictionary mapping speaker tags (e.g., 'S1') to
                           a dict containing 'path' and optional 'transcript'.
        """
        logger.info("Registering voice prompts...")
        self._voice_prompt_details = {}

        # Mode B: Prompts that only have a path need an embedding generated.
        prompts_to_embed = {
            speaker: details["path"]
            for speaker, details in voice_prompts.items()
            if details and not details.get("transcript")
        }

        if prompts_to_embed:
            logger.info(
                f"Generating speaker embeddings for: {list(prompts_to_embed.keys())}"
            )
            try:
                embeddings = self.model.generate_speaker_embedding(
                    list(prompts_to_embed.values())
                )
                # Map embeddings back to speakers
                for speaker, embedding in zip(
                    prompts_to_embed.keys(), embeddings, strict=True
                ):
                    self._voice_prompt_details[speaker] = {
                        "path": prompts_to_embed[speaker],
                        "transcript": None,
                        "embedding": embedding,
                    }
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate speaker embeddings: {str(e)}"
                ) from e

        # Mode A: Prompts with transcripts are stored directly.
        for speaker, details in voice_prompts.items():
            if details and details.get("transcript"):
                self._voice_prompt_details[speaker] = {
                    "path": details["path"],
                    "transcript": details["transcript"],
                    "embedding": None,  # No embedding needed for Mode A
                }
        logger.info("Voice prompts registered successfully.")

    def _generate_audio_segment(
        self,
        text: str,
        audio_prompt_paths: Optional[List[str]] = None,
        prepended_transcripts: str = "",
    ) -> AudioSegment:
        """
        Generates a single audio segment from text.
        This is the core generation logic for a single chunk.
        """
        logger.info(f"Generating single audio chunk for: '{text}'...")

        # Assemble the final text payload
        if prepended_transcripts:
            text_payload = prepended_transcripts + " " + text
        else:
            text_payload = text

        # Prepare arguments for the generate call
        generate_kwargs = {
            "text": text_payload,
            "verbose": self.log_level in (logging.INFO, logging.DEBUG),
        }
        generate_kwargs.update(config.DIA_GENERATE_PARAMS)

        if audio_prompt_paths:
            generate_kwargs["audio_prompt"] = audio_prompt_paths

        # Generate audio
        audio_output = self.model.generate(**generate_kwargs)

        # Handle case where generate returns a list with one item
        if isinstance(audio_output, list):
            audio_output = audio_output[0]

        # Convert to AudioSegment
        audio_int16 = (audio_output * 32767).astype(np.int16)
        return AudioSegment(
            audio_int16.tobytes(),
            frame_rate=config.AUDIO_SAMPLING_RATE,
            sample_width=config.AUDIO_SAMPLE_WIDTH,
            channels=config.AUDIO_CHANNELS,
        )

    def generate(
        self,
        texts: List[str],
        unified_audio_prompt: Optional[str],
        unified_transcript_prompt: Optional[str],
    ) -> List[AudioSegment]:
        """
        Converts a batch of text chunks to audio segments. If only one chunk is
        provided, it routes to a simplified single-generation method.

        Args:
            texts: List of text chunks to convert to audio.
            unified_audio_prompt: Path to a unified audio prompt file (optional).
            unified_transcript_prompt: A unified transcript prompt string (optional).

        Returns:
            A list of AudioSegment objects, one for each input text chunk.
        """
        if not texts:
            return []

        # Reset seed for consistency if any text contains pure TTS speakers
        contains_pure_tts = any(
            speaker not in self._voice_prompt_details
            for text in texts
            for speaker in set(re.findall(r"[(S\d+)]", text))
        )
        if contains_pure_tts and self.seed is not None:
            logger.debug(
                f"Resetting seed to {self.seed} for pure TTS generation in batch"
            )
            self._set_seed(self.seed)

        # Route to single generation if only one chunk is provided
        if len(texts) == 1:
            logger.info("Only one chunk detected, routing to single generation.")
            return [
                self._generate_audio_segment(
                    texts[0],
                    audio_prompt_paths=(
                        [unified_audio_prompt] if unified_audio_prompt else None
                    ),
                    prepended_transcripts=unified_transcript_prompt or "",
                )
            ]

        logger.info(f"Generating batch audio for {len(texts)} text chunks")
        chunked_transcripts = ",\n".join([f'"{t}"' for t in texts])
        logger.info(f"Chunked transcript:\n{chunked_transcripts}")

        # Prepare text payloads
        batch_text_payloads = [
            (
                (unified_transcript_prompt + " " + text)
                if unified_transcript_prompt
                else text
            )
            for text in texts
        ]

        # Prepare audio prompts
        batch_audio_prompts = (
            [unified_audio_prompt] * len(texts) if unified_audio_prompt else None
        )

        # Prepare arguments for the batch generate call
        generate_kwargs = {
            "text": batch_text_payloads,
            "verbose": self.log_level in (logging.INFO, logging.DEBUG),
        }
        generate_kwargs.update(config.DIA_GENERATE_PARAMS)

        if batch_audio_prompts:
            generate_kwargs["audio_prompt"] = batch_audio_prompts

        # Make single batch call to model
        logger.info("Making batch call to Dia model...")
        audio_outputs = self.model.generate(**generate_kwargs)

        # Convert numpy arrays to AudioSegment objects
        audio_segments = []
        for i, audio_array in enumerate(audio_outputs):
            audio_int16 = (audio_array * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=config.AUDIO_SAMPLING_RATE,
                sample_width=config.AUDIO_SAMPLE_WIDTH,
                channels=config.AUDIO_CHANNELS,
            )
            audio_segments.append(audio_segment)
            logger.debug(
                f"Converted audio chunk {i + 1}/{len(audio_outputs)} to AudioSegment"
            )

        logger.info(f"Successfully generated {len(audio_segments)} audio segments")
        return audio_segments
