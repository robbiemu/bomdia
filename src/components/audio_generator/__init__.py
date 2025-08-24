"""Audio generator component."""

from .chunker import chunk_lines, estimate_seconds_for_text
from .tts import DiaTTS

__all__ = ["chunk_lines", "estimate_seconds_for_text", "DiaTTS"]
