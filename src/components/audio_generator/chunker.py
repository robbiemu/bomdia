"""Audio generation utilities."""

import re
from typing import List

from shared.config import config


def estimate_seconds_for_text(text: str) -> float:
    """Estimate the time in seconds for a given text."""
    words = len(text.split())
    return words / config.AVG_WPS


def chunk_to_5_10s(lines: List[dict]) -> List[str]:
    """
    Combine line strings into blocks of 5-10 seconds with edge case handling.

    Each block is a newline-separated transcript with speaker tags preserved.

    Edge case handling:
    - Lines longer than 10 seconds are split at sentence boundaries (.?! punctuation)
    - Lines shorter than 5 seconds are combined with subsequent lines when possible
    - Very short leftover blocks at the end are merged with the previous block
    - Buffer exceeding 10 seconds tries to maintain speaker continuity by combining

    Args:
        lines (List[dict]): List of dictionaries with 'speaker' and 'text' keys.

    Returns:
        List[str]: List of transcript blocks as strings, each 5-10 seconds long.
    """
    blocks = []
    buf = []
    buf_seconds = 0.0

    i = 0
    while i < len(lines):
        ln = lines[i]
        sline = (
            f"[{ln['speaker']}] {ln['text']}"
            if not ln["text"].startswith(f"[{ln['speaker']}]")
            else ln["text"]
        )
        sec = estimate_seconds_for_text(ln["text"])
        # if single line longer than 10s: split crude by sentence punctuation
        if sec > 10:
            # naive split
            parts = re.split(r"(?<=[.?!])\\s+", ln["text"])
            parts = [p.strip() for p in parts if p.strip()]
            for p in parts:
                s = estimate_seconds_for_text(p)
                if buf_seconds + s <= 10:
                    buf.append(f"[{ln['speaker']}] {p}")
                    buf_seconds += s
                else:
                    if buf and buf_seconds >= 5:
                        blocks.append("\n".join(buf))
                        buf = []
                        buf_seconds = 0.0
                    # put the long chunk alone if >10s after splitting: accept
                    #  it (edge-case)
                    blocks.append(f"[{ln['speaker']}] {p}")
            i += 1
            continue

        # normal flow: append until we are at least 5s but no more than 10s
        if buf_seconds + sec <= 10:
            buf.append(sline)
            buf_seconds += sec
            # if we reached >=5s, finalize a block (but try to accumulate a bit more
            #  to favor upper bound)
            if buf_seconds >= 5:
                blocks.append("\n".join(buf))
                buf = []
                buf_seconds = 0.0
            i += 1
        else:
            # buffer would exceed 10s if we add this line: flush buffer if >=5s, else
            #  put the line alone (if buffer <5s)
            if buf_seconds >= 5:
                blocks.append("\n".join(buf))
                buf = []
                buf_seconds = 0.0
            else:
                # combine buf + this line even though >10s; prefer to keep continuity
                #  (rare)
                buf.append(sline)
                blocks.append("\n".join(buf))
                buf = []
                buf_seconds = 0.0
                i += 1

    # leftover
    if buf:
        # try to merge with previous block if too short
        if blocks and estimate_seconds_for_text("\n".join(buf)) < 5:
            blocks[-1] = blocks[-1] + "\n" + "\n".join(buf)
        else:
            blocks.append("\n".join(buf))
    return blocks
