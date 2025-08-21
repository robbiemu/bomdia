import logging
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
        if preferred_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA specified but not available. Falling back to auto-detection."
            )
        elif preferred_device == "mps" and not torch.backends.mps.is_available():
            logger.warning(
                "MPS specified but not available. Falling back to auto-detection."
            )
        elif preferred_device in ["cuda", "mps", "cpu"]:
            logger.debug(f"Using specified device: {preferred_device}")
            return torch.device(preferred_device)
        else:
            logger.warning(
                f"Invalid device '{preferred_device}'. Falling back to auto-detection."
            )

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
    """

    def __init__(
        self,
        seed: Optional[int],
        model_checkpoint: str = "nari-labs/Dia-1.6B-0626",
        device: str | None = None,
        log_level: int = logging.WARNING,
    ) -> None:
        """Initializes the TTS engine using the Dia library."""
        self.device = device if device is not None else _get_device().type
        self.seed = seed
        self.log_level = log_level

        logger.info(f"Loading model {model_checkpoint} on device {self.device}...")

        try:
            dtype_name = (
                config.DIA_COMPUTE_DTYPE
                if isinstance(config.DIA_COMPUTE_DTYPE, str)
                else "float32"
            )
            compute_dtype = getattr(torch, dtype_name)
        except (AttributeError, TypeError):
            logger.warning(
                f"Invalid compute dtype '{config.DIA_COMPUTE_DTYPE}', falling "
                "back to float32"
            )
            compute_dtype = torch.float32

        self.model = Dia.from_pretrained(
            model_checkpoint,
            compute_dtype=compute_dtype,
            device=self.device,
        )
        self._voice_prompt_details: Dict[str, Dict[str, Optional[str]]] = {}
        logger.info("Model loaded successfully.")

    def register_voice_prompts(
        self, voice_prompts: Dict[str, Dict[str, Optional[str]]]
    ) -> None:
        """
        Analyzes audio files to generate and store speaker embeddings and prompt info.
        """
        logger.info("Registering voice prompts...")
        self._voice_prompt_details = {}

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
                for speaker, embedding in zip(
                    prompts_to_embed.keys(), embeddings, strict=True
                ):
                    self._voice_prompt_details[speaker] = {
                        "path": prompts_to_embed[speaker],
                        "transcript": None,
                        "embedding": embedding,
                    }
            except Exception as e:
                raise RuntimeError(f"Failed to generate speaker embeddings: {e}") from e

        for speaker, details in voice_prompts.items():
            if details and details.get("transcript"):
                self._voice_prompt_details[speaker] = {
                    "path": details["path"],
                    "transcript": details["transcript"],
                    "embedding": None,
                }
        logger.info("Voice prompts registered successfully.")

    def generate(
        self,
        texts: List[str],
        audio_prompts: Optional[List[Optional[str]]] = None,
    ) -> List[AudioSegment]:
        """
        Converts a batch of fully-formed text payloads to audio segments.

        Args:
            texts: A list of text strings to generate. Each string should
                   already be prepended with its corresponding transcript prompt.
            audio_prompts: A list of audio prompt file paths, one for each text
                           payload. Can contain None for chunks without prompts.

        Returns:
            A list of AudioSegment objects, one for each input text payload.
        """
        if not texts:
            return []

        logger.info(f"Generating batch audio for {len(texts)} text chunks...")
        log_of_chunked_transcripts = ",\n".join([f'"{t}"' for t in texts])
        logger.debug(f"Chunked transcript:\n{log_of_chunked_transcripts}")

        # Prepare arguments for the batch call
        generate_kwargs: Dict[str, bool | str | list[str] | int | float | None] = dict()
        if self.log_level in (logging.INFO, logging.DEBUG):
            generate_kwargs["verbose"] = True
        generate_kwargs.update(config.DIA_GENERATE_PARAMS)
        if self.seed:
            logger.info(f"Using seed {self.seed}")
            generate_kwargs["voice_seed"] = self.seed
        logger.debug(f"Generating with parameters {generate_kwargs}")

        generate_kwargs["text"] = texts[0] if len(texts) == 1 else texts
        if audio_prompts:
            logger.debug("Attaching audio prompts for generation")
            # Filter out None values to satisfy type checker
            filtered_prompts = [
                prompt for prompt in audio_prompts if prompt is not None
            ]
            generate_kwargs["audio_prompt"] = (
                filtered_prompts if filtered_prompts else None
            )

        # Make the batch call to the model
        audio_outputs = self.model.generate(**generate_kwargs)

        # Ensure audio_outputs is always a list of arrays for consistent processing.
        # The model returns a single array for a single input, and a list of
        #  arrays for multiple inputs.
        if not isinstance(audio_outputs, list):
            audio_outputs = [audio_outputs]

        # Convert numpy arrays to AudioSegment objects
        audio_segments = []
        for audio_array in audio_outputs:
            audio_int16 = (audio_array * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=config.AUDIO_SAMPLING_RATE,
                sample_width=config.AUDIO_SAMPLE_WIDTH,
                channels=config.AUDIO_CHANNELS,
            )
            audio_segments.append(audio_segment)

        logger.info(f"Successfully generated {len(audio_segments)} audio segments.")
        return audio_segments
