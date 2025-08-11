"""Transcript parsing utilities."""

import re
from typing import Dict, List


def parse_simple_txt(path: str) -> List[Dict]:
    """
    Parse simple transcript text files with basic speaker tagging.

    Supports three line formats:
    1. [S1] text... - Explicit speaker tags in brackets
    2. Name: text... - Name-based speaker identification
    3. bare text - Assumed to be continuation of previous speaker or S1 if first line

    Example input:
    ```
    [S1] Hello there
    John: How are you?
    [S2] I'm doing well
    Nice to hear
    ```

    Example output:
    [
      {'speaker': 'S1', 'text': 'Hello there'},
      {'speaker': 'S2', 'text': "How are you? I'm doing well Nice to hear"}
    ]

    Args:
        path (str): Path to the text file to parse.

    Returns:
        List[Dict]: List of dictionaries with 'speaker' and 'text' keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    parsed = []
    speakers_map = {}  # name -> S1/S2
    next_speaker_id = 1

    for ln in raw_lines:
        # bracketed form
        m = re.match(r"^\[?(S\d+)\]?\s*(.*)$", ln)
        if m:
            speaker = m.group(1)
            text = m.group(2).strip()
            parsed.append({"speaker": speaker, "text": text})
            continue
        # "Name: ... " form
        m2 = re.match(r"^([A-Za-z0-9 _\-]+?):\s*(.*)$", ln)
        if m2:
            name = m2.group(1).strip()
            text = m2.group(2).strip()
            if name not in speakers_map:
                tag = f"S{next_speaker_id}"
                speakers_map[name] = tag
                next_speaker_id += 1
            parsed.append({"speaker": speakers_map[name], "text": text})
            continue
        # fallback: treat as continuation of previous speaker if any, else S1
        if parsed:
            parsed[-1]["text"] += " " + ln
        else:
            parsed.append({"speaker": "S1", "text": ln})
    return parsed


def parse_srt(path: str) -> List[Dict]:
    """
    Extremely-primitive .srt parser that extracts text content while ignoring timestamps
    and speaker information.

    This parser is intentionally simple and does not handle:
    - Timestamp information from SRT files
    - Speaker identification or attribution
    - Complex subtitle formatting

    Args:
        path (str): Path to the .srt file to parse.

    Returns:
        List[Dict]: List of dictionaries with 'speaker' (default 'S1') and 'text' keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # split numbered blocks
    blocks = re.split(r"\n\s*\n", content)
    parsed = []
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        # drop index and timestamps if present
        if re.match(r"^\d+$", lines[0]) and len(lines) >= 2:
            candidate = (
                " ".join(lines[2:]) if "-->" in lines[1] else " ".join(lines[1:])
            )
        else:
            candidate = " ".join(lines)
        # no speaker info in srt - push as S1 by default
        parsed.append({"speaker": "S1", "text": candidate})
    return parsed


def ingest_transcript(path: str) -> List[Dict]:
    if path.lower().endswith(".srt"):
        return parse_srt(path)
    else:
        return parse_simple_txt(path)
