"""Transcript parsing utilities - ACTUALLY FIXED VERSION."""

import re
from typing import Dict, List

from shared.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def parse_simple_txt(path: str) -> List[Dict[str, str]]:
    """
    Parse simple transcript text files with basic speaker tagging.

    Supports three line formats:
    1. [S1] text... - Explicit speaker tags in brackets (speaker_name = None)
    2. [S1: Name] text... - Explicit speaker tags with names (speaker_name = Name)
    3. Name: text... - Name-based speaker identification
    4. bare text - Assumed to be continuation of previous speaker or S1 if first line

    Example input:
    ```
    [S1] Hello there
    John: How are you?
    [S2] I'm doing well
    Nice to hear
    ```

    Example output:
    [
      {'speaker': 'S1', 'speaker_name': None, 'text': 'Hello there'},
      {
        'speaker': 'S2',
        'speaker_name': 'John',
        'text': 'How are you? I\'m doing well Nice to hear'
      }
    ]

    Args:
        path (str): Path to the text file to parse.

    Returns:
        List[Dict]: List of dictionaries with 'speaker', 'speaker_name', and
        'text' keys.
    """
    logger.debug(f"Parsing simple TXT transcript: {path}")
    parsed = []
    speakers_map = {}  # name -> S1/S2
    next_speaker_id = 1

    with open(path, "r", encoding="utf-8") as f:
        line_number = 0
        for line in f:
            line_number += 1
            ln = line.rstrip("\r\n")  # Remove both \r and \n
            logger.debug(f"Processing line {line_number}: '{ln}'")
            if not ln.strip():
                logger.debug(f"Skipping empty line {line_number}")
                continue

            # bracketed form: [S1] or [S1: Name]
            m = re.match(r"^\s*\[([Ss]([0-9]+))(?::\s*([^]]+))?\]\s*(.*)$", ln)
            if m:
                speaker = m.group(1)
                name = m.group(3)  # Will be None if no name provided
                text = m.group(4).rstrip()
                logger.debug(
                    f"Matched bracketed form: speaker={speaker}, name={name}, "
                    f"text='{text}'"
                )
                # FIXED: Don't fall back to speaker ID if name is None
                parsed.append({"speaker": speaker, "speaker_name": name, "text": text})
                continue

            # "Name: ... " form - using the corrected regex (remove the ? to make
            #  it greedy)
            m2 = re.match(r"^([A-Za-z0-9 _\-\u2014\u2013]+):\s*(.*)$", ln)
            if m2:
                name = m2.group(1).strip()
                text = m2.group(2).rstrip()
                logger.debug(f"Matched name form: name={name}, text='{text}'")
                if name not in speakers_map:
                    tag = f"S{next_speaker_id}"
                    speakers_map[name] = tag
                    next_speaker_id += 1
                parsed.append(
                    {
                        "speaker": speakers_map[name],
                        "speaker_name": name,
                        "text": text,
                    }
                )
                continue

            # fallback: treat as continuation of previous speaker if any, else S1
            logger.debug("Treating as continuation or first line")
            if parsed:
                parsed[-1]["text"] += " " + ln
            else:
                parsed.append({"speaker": "S1", "speaker_name": "S1", "text": ln})

    logger.info(f"Successfully parsed {len(parsed)} lines from TXT transcript")
    for i, parsed_line in enumerate(parsed):
        logger.debug(f"Parsed line {i + 1}: {parsed_line}")
    return parsed


def parse_srt(path: str) -> List[Dict[str, str]]:
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
    logger.debug(f"Parsing SRT transcript: {path}")
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
        parsed.append({"speaker": "S1", "speaker_name": "S1", "text": candidate})

    logger.info(f"Successfully parsed {len(parsed)} blocks from SRT transcript")
    return parsed


def ingest_transcript(path: str) -> List[Dict[str, str]]:
    logger.info(f"Ingesting transcript from: {path}")
    if path.lower().endswith(".srt"):
        return parse_srt(path)
    else:
        return parse_simple_txt(path)
