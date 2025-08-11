"""Utilities for merging consecutive transcript lines."""

from typing import Dict, List

from shared.config import config
from shared.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def merge_consecutive_lines(lines: List[Dict]) -> List[Dict]:
    """Merge consecutive lines from the same speaker with a pause placeholder."""
    out: List[Dict] = []
    warnings = []
    for ln in lines:
        if out and ln["speaker"] == out[-1]["speaker"]:
            # warn & merge with placeholder
            warnings.append(
                f"Consecutive lines from {ln['speaker']}, merging with placeholder."
            )
            out[-1]["text"] = (
                out[-1]["text"].rstrip()
                + " "
                + config.PAUSE_PLACEHOLDER
                + " "
                + ln["text"].lstrip()
            )
        else:
            out.append(ln.copy())
    if warnings:
        logger.warning("WARN: %s", "\n  - ".join(warnings))
    return out
