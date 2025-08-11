import random
from typing import Dict, Optional

import numpy as np
import torch
from dia.model import Dia


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
    ) -> None:
        """
        Initializes the TTS engine using the Dia library's native method.

        Args:
            model_checkpoint (str): The Hugging Face model identifier.
            device (str, optional): Device to run on ('cuda', 'mps', or 'cpu').
        """
        if device is None:
            device = _get_default_device().type
        self.device = device

        print(
            f"[DiaTTS] Loading model {model_checkpoint} via official Dia library on "
            f"device {self.device}..."
        )

        # Use float16 for CUDA for max performance, but force float32 for MPS to
        #  ensure stability.
        compute_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if seed is not None:
            print("[DiaTTS] Setting seed to '{seed}' for voice selection")
            self._set_seed(seed)

        # The Dia library handles device placement internally.
        # We pass the device directly during creation.
        self.model = Dia.from_pretrained(
            model_checkpoint,
            compute_dtype=compute_dtype,
            device=self.device,
        )

        print("[DiaTTS] Model loaded successfully.")

    def _set_seed(self, seed: int) -> None:
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN (if used)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def register_voice_prompts(self, voice_prompts: Dict[str, str]) -> None:
        """
        Analyzes audio files to generate and store speaker embeddings.
        Args:
            voice_prompts: A dictionary mapping speaker tags (e.g., 'S1')
                           to audio file paths.
        """
        print("[DiaTTS] Generating speaker embeddings from audio prompts...")
        # The Dia library's `generate_speaker_embedding` method is perfect for this.
        self._speaker_embeddings = self.model.generate_speaker_embedding(
            list(voice_prompts.values())
        )

        # We need to map them back to the speaker tags
        prompt_speakers = list(voice_prompts.keys())
        self._speaker_embeddings = {
            speaker: embedding
            for speaker, embedding in zip(
                prompt_speakers, self._speaker_embeddings, strict=True
            )
        }
        print("[DiaTTS] Speaker embeddings registered successfully.")

    def text_to_audio_file(self, text: str, out_path: str) -> str:
        """
        Converts text to an audio file using the intended two-step process:
        1. Generate the audio data in memory.
        2. Save the audio data to a file.

        Args:
            text (str): Text to convert to speech.
            out_path (str): Path to save the output audio file (e.g., 'output.mp3').

        Returns:
            str: The path to the output audio file.
        """
        print(f"[DiaTTS] Generating audio for: '{text}...'")

        # The `use_torch_compile` flag is crucial for compatibility, especially
        #  on non-Linux systems.
        audio_output = self.model.generate(text, use_torch_compile=False, verbose=True)

        print(f"[DiaTTS] Saving audio to {out_path}...")
        self.model.save_audio(out_path, audio_output)

        return out_path
