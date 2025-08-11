"""Audio generator component."""

from .chunker import chunk_to_5_10s, estimate_seconds_for_text
from .tts import DiaTTS

__all__ = ["chunk_to_5_10s", "estimate_seconds_for_text", "DiaTTS"]
