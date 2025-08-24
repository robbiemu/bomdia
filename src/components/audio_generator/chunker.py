from typing import List, Union

import regex as re
from shared.config import config
from shared.logging import get_logger

logger = get_logger(__name__)


_HYPHENS = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]+"


def count_words_in_text(text: str) -> int:
    """
    Rules:
    - Each digit counts as a word (e.g., '52' -> 2).
    - Each contiguous run of letters counts as a word ('McFlurry' -> 1).
    - Hyphenated words split ('cul-de-sac' -> 3).
    - Speaker tags [S1], [S2], ... are removed.
    - Punctuation does not count as words.
    """
    # Remove speaker tags like [S1], [S2]
    clean_text = re.sub(r"\[S\d+\]\s*", "", text)
    if not clean_text.strip():
        return 0

    total = 0
    for token in clean_text.split():
        # 1) Count all digits (each digit is a word)
        total += len(re.findall(r"[0-9]", token))

        # 2) Count letter runs, keeping hyphens as separators (digits replaced by
        #  spaces)
        letters_masked = re.sub(r"[0-9]", " ", token)
        # keep letters and hyphens; turn everything else into spaces
        letters_masked = re.sub(
            r"[^\p{L}\-\u2010\u2011\u2012\u2013\u2014\u2212]+", " ", letters_masked
        )

        # split on whitespace to get letter runs possibly containing hyphens
        for run in letters_masked.split():
            # split hyphenated runs into separate spoken words
            parts = re.split(_HYPHENS, run)
            total += sum(1 for p in parts if p)

    return total


def protect_numbers(src: str) -> tuple[str, list[str]]:
    r"""
    Protect number units matching the pattern /\d+(\D\d+)*/
    (digit group followed by zero or more non-digit+digit-group pairs).
    """
    print(f"\n---protect_numbers processing: {src}---")
    n = len(src)
    i = 0
    out = []
    placeholders: list[str] = []

    while i < n:
        ch = src[i]
        if ch.isdigit():
            j = i

            # Consume initial digits
            while j < n and src[j].isdigit():
                j += 1

            # Now look for pattern: non-digit followed by digits, repeating
            while j < n:
                # Check if we have a non-digit character
                if j < n and not src[j].isdigit():
                    non_digit_start = j
                    # Skip the non-digit character
                    j += 1

                    # Check if followed by digits
                    if j < n and src[j].isdigit():
                        # Consume the following digits
                        while j < n and src[j].isdigit():
                            j += 1
                        # Continue the loop to look for more non-digit+digit patterns
                        continue
                    else:
                        # Non-digit not followed by digit, so backtrack
                        j = non_digit_start
                        break
                else:
                    # No more non-digit characters, we're done with this number unit
                    break

            unit = src[i:j]
            ph = f"__NUMBER_UNIT_{len(placeholders)}__"
            placeholders.append(unit)
            out.append(ph)
            i = j
        else:
            out.append(ch)
            i += 1

    return "".join(out), placeholders


def split_by_punctuation(text: str) -> List[str]:
    """
    Split text into phrases based on punctuation, dash-like marks, and speaker tags.

    Rules implemented:
      - Speaker tags [S1] / [S2] start a new phrase:
        "[S1] a [S2] b" -> ["[S1] a", "[S2] b"].
      - Group consecutive punctuation as a single unit:
        "...", "--", "!?" etc are 1 PUNCT.
      - Dash-like punctuation (hyphen '-', minus '−', en-dash '–', em-dash '—'):
          "a PUNCT b" -> ["a", "PUNCT b"].
        Exception: a single hyphen between word-chars is part of the word ("dead-end").
      - Typical punctuation (.,?!,) attaches left:
          "a PUNCT b" -> ["a PUNCT", "b"].
      - Ellipsis: unicode '…' -> '...' and 3+ periods collapse to '...'.
      - Numbers: digit groups separated by exactly one char are a unit.
          - Allows mixing of space and dot within a unit (e.g., "5 000.000").
          - If a comma is used anywhere, a subsequent space breaks the number unit:
            "5,000 000.000" -> ["5,000", "000.000"].
    """
    # 1) Normalize ellipsis and hyphen runs
    s = re.sub(r"…", "...", text)
    s = re.sub(r"\.{3,}", "...", s)  # collapse 3+ '.' to '...'
    s = re.sub(r"-{2,}", "--", s)  # collapse '---' -> '--'

    # 2) Protect number units using a scanner that enforces the comma+space split rule
    s, number_placeholders = protect_numbers(s)

    # 3) Tokenize and segment according to punctuation + speaker tag rules
    DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2212"  # -, ‐, ‑, ‒, –, —, −
    PUNCT_CHARS = ".,!?," + DASH_CHARS  # grouped punctuation handled here

    def is_word_char(c: str) -> bool:
        return bool(re.match(r"\w", c))

    def is_dash_like_char(c: str) -> bool:
        return c in DASH_CHARS

    def is_punct_char(c: str) -> bool:
        return c in PUNCT_CHARS

    def match_tag(buf: str, pos: int) -> Union[str, None]:
        m = re.match(r"S[12]", buf[pos:])
        return m.group(0) if m else None

    segments: List[str] = []
    buf: List[str] = []

    i = 0
    n = len(s)
    while i < n:
        # Speaker tag starts a new segment
        tag = match_tag(s, i)
        if tag:
            part = "".join(buf).strip()
            if part:
                segments.append(part)
            buf = [tag]
            i += len(tag)
            continue

        c = s[i]

        # Group punctuation (while respecting the hyphen-in-a-word exception)
        if is_punct_char(c):
            # single hyphen inside a word is NOT punctuation
            if (
                c == "-"
                and 0 < i < n - 1
                and is_word_char(s[i - 1])
                and is_word_char(s[i + 1])
            ):
                buf.append(c)
                i += 1
                continue

            # collect a cluster of consecutive punctuation
            j = i
            while j < n:
                cj = s[j]
                if is_punct_char(cj):
                    # don't absorb a hyphen that's between word chars
                    if (
                        cj == "-"
                        and 0 < j < n - 1
                        and is_word_char(s[j - 1])
                        and is_word_char(s[j + 1])
                    ):
                        break
                    j += 1
                else:
                    break
            cluster = s[i:j]

            # dash-like cluster attaches to the right ("a", "— b")
            dash_like = is_dash_like_char(cluster[0]) or cluster.startswith("--")
            if dash_like:
                part = "".join(buf).strip()
                if part:
                    segments.append(part)
                buf = [cluster]
            else:
                # typical punctuation attaches to the left ("a...","b")
                buf.append(cluster)
                part = "".join(buf).strip()
                if part:
                    segments.append(part)
                buf = []

            i = j
            continue

        # regular char
        buf.append(c)
        i += 1

    last = "".join(buf).strip()
    if last:
        segments.append(last)

    # 4) Restore number placeholders
    def restore_numbers(chunk: str) -> str:
        return re.sub(
            r"__NUMBER_UNIT_(\d+)__",
            lambda m: number_placeholders[int(m.group(1))],
            chunk,
        )

    return [restore_numbers(seg) for seg in segments]


def soft_cost(duration: float, min_d: float, max_d: float) -> float:
    """Quadratic penalty outside [min_d, max_d]."""
    if duration < min_d:
        return (min_d - duration) ** 2
    elif duration > max_d:
        return (duration - max_d) ** 2
    return 0.0


def dp_segment(
    phrases: List[str], wps: float, min_d: Union[float, str], max_d: Union[float, str]
) -> List[str]:
    """Segment phrases using DP with soft-constraint cost."""
    # Convert min_d and max_d to float if they are strings
    if isinstance(min_d, str):
        min_d = float(min_d)
    if isinstance(max_d, str):
        max_d = float(max_d)
    n = len(phrases)
    # Precompute word counts
    word_counts = [count_words_in_text(p) for p in phrases]
    prefix_words = [0]
    for wc in word_counts:
        prefix_words.append(prefix_words[-1] + wc)

    # DP arrays
    dp = [float("inf")] * (n + 1)
    back = [-1] * (n + 1)
    dp[0] = 0.0

    for i in range(1, n + 1):
        for j in range(max(0, i - 50), i):  # limit lookback for speed
            words = prefix_words[i] - prefix_words[j]
            duration = words / wps
            cost = dp[j] + soft_cost(duration, min_d, max_d)
            if cost < dp[i]:
                dp[i] = cost
                back[i] = j

    # Reconstruct segmentation
    segments = []
    idx = n
    while idx > 0:
        j = back[idx]
        seg = " ".join(phrases[j:idx])
        segments.append(seg)
        idx = j

    return list(reversed(segments))


def chunk_lines(lines: List[dict]) -> List[str]:
    """Chunk transcript lines into segments via DP soft-constraint."""
    # Flatten transcript into phrases
    all_text_parts = []
    for line in lines:
        speaker = line["speaker"]
        text = line["text"]
        # The f-string correctly adds the initial speaker tag for the first
        #  phrase of a line
        parts = split_by_punctuation(f"[{speaker}] {text}")
        all_text_parts.extend(parts)

    logger.debug(f"Total phrases: {len(all_text_parts)}")

    min_duration = config.MIN_CHUNK_DURATION
    max_duration = config.MAX_CHUNK_DURATION

    # Initial segmentation based on duration cost
    segments = dp_segment(all_text_parts, config.AVG_WPS, min_duration, max_duration)

    # Post-processing pass to correct speaker tags
    corrected_segments = []
    last_speaker_tag = None
    speaker_tag_pattern = re.compile(r"(\[S[0-9]+\])")

    for segment in segments:
        current_segment = segment
        # Check if the segment starts with a speaker tag
        if not speaker_tag_pattern.match(current_segment) and last_speaker_tag:
            # Prepend the last known speaker tag if missing
            current_segment = f"{last_speaker_tag} {current_segment}"
            logger.debug(f"Prepended speaker tag '{last_speaker_tag}' to a segment.")

        # Find the last speaker tag in the current segment to carry over to the next
        all_tags_in_segment = speaker_tag_pattern.findall(current_segment)
        if all_tags_in_segment:
            last_speaker_tag = all_tags_in_segment[-1]

        corrected_segments.append(current_segment)

    # Final logging and duration validation pass
    for i, seg in enumerate(corrected_segments):
        duration = estimate_seconds_for_text(seg)
        # Log a debug message for every segment
        logger.debug(f"Seg {i}: {repr(seg)} ({duration:.1f}s)")
        # Log a warning for segments outside the desired duration
        if not (float(min_duration) <= duration <= float(max_duration)):
            logger.warning(
                f"Segment {i} is outside the desired duration range: {duration:.1f}s. "
                f"Allowed: [{min_duration:.1f}s - {max_duration:.1f}s]."
            )

    logger.info(f"Produced {len(corrected_segments)} DP-optimized transcript blocks")

    return corrected_segments


def estimate_seconds_for_text(text: str) -> float:
    """Estimate the time in seconds for a given text."""
    words = count_words_in_text(text)
    # Prevent division by zero for empty strings
    if config.AVG_WPS is None or config.AVG_WPS == 0:
        return 0.0
    return words / config.AVG_WPS
