import logging
import os
import random
import re
from typing import Dict, Optional

import numpy as np
import torch
from dia.model import Dia
from shared.config import config

# Initialize logger
logger = logging.getLogger(__name__)


def _get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
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
            log_level (int): Logging level for verbose output.
        """
        if device is None:
            device = _get_default_device().type
        self.device = device
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

    def text_to_audio_file(self, text: str, out_path: str) -> str:
        """
        Converts text to an audio file, routing to the correct generation mode.
        """
        logger.info(f"Generating audio for: '{text}...'")

        speakers_in_block = set(re.findall(r"\[(S\d+)\]", text))

        # Initialize variables for text payload and audio prompts
        main_text_payload = text
        prepended_transcripts = ""
        audio_prompt_paths = []
        contains_pure_tts = False

        # Determine generation mode for each speaker in the block
        for speaker in speakers_in_block:
            prompt_details = self._voice_prompt_details.get(speaker)

            if prompt_details:
                path = prompt_details.get("path")
                transcript = prompt_details.get("transcript")

                # Add audio prompt path if available
                if path and path not in audio_prompt_paths:
                    audio_prompt_paths.append(path)

                # Prepend transcript for high-fidelity cloning (Mode A)
                if transcript:
                    prepended_transcripts = (
                        f"[{speaker}]{transcript}[{speaker}] {prepended_transcripts}"
                    )
            else:  # Mode C: Pure TTS
                contains_pure_tts = True

        # Reset seed if any speaker is pure TTS to ensure consistency
        if contains_pure_tts and self.seed is not None:
            logger.debug(f"Resetting seed to {self.seed} for pure TTS generation.")
            self._set_seed(self.seed)

        # Assemble the final text payload
        text_payload = prepended_transcripts + main_text_payload

        # Prepare arguments for the generate call conditionally
        generate_kwargs = {
            "text": text_payload,
            "use_torch_compile": False,
            "verbose": self.log_level in (logging.INFO, logging.DEBUG),
        }

        # Add configured generation parameters
        generate_kwargs.update(config.DIA_GENERATE_PARAMS)

        # Only add audio_prompt argument if we have audio prompts
        if audio_prompt_paths:
            generate_kwargs["audio_prompt"] = audio_prompt_paths

        # Generate audio with the corrected keyword arguments
        audio_output = self.model.generate(**generate_kwargs)

        logger.info(f"Saving audio to {out_path}...")
        self.model.save_audio(out_path, audio_output)

        return out_path
