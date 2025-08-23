from typing import List, Union

import regex as re
from shared.config import config
from shared.logging import get_logger

logger = get_logger(__name__)


def count_words_in_text(text: str) -> int:
    """
    Count words in text, treating each digit as a separate word and handling
    number units.
    """
    clean_text = re.sub(r"\[S\d+\]\s*", "", text)
    number_unit_pattern = r"\d+(?:[.,\s]?\d+)*"
    placeholders = []

    def replace_number_unit(match: re.Match) -> str:
        unit = match.group(0)
        placeholders.append(unit)
        return f"__NUMBER_UNIT_{len(placeholders) - 1}__"

    processed_text = re.sub(number_unit_pattern, replace_number_unit, clean_text)
    words = processed_text.split()
    total_words = 0

    for word in words:
        if word.startswith("__NUMBER_UNIT_"):
            # Safely extract the index from the placeholder, ignoring any
            # attached punctuation (e.g., from "__NUMBER_UNIT_0__,")
            match = re.search(r"__NUMBER_UNIT_(\d+)__", word)
            if match:
                idx = int(match.group(1))
                num_unit = placeholders[idx]
                # Count each digit in the original number unit as a word
                total_words += len(re.findall(r"\d", num_unit))
            else:
                # Fallback for malformed placeholders: treat as a single word
                total_words += 1
        else:
            clean_word = re.sub(r"[^\w\'-]", "", word)
            if clean_word:
                total_words += 1

    return total_words


def split_by_punctuation(text: str) -> List[str]:
    """
    Split text into phrases based on punctuation and speaker tags using a robust
    zero-width split.
    """
    number_unit_pattern = r"\d+(?:[.,\s]?\d+)*"
    placeholders = []

    def replace_with_placeholder(match: re.Match) -> str:
        unit = match.group(0)
        placeholders.append(unit)
        return f"__NUMBER_UNIT_{len(placeholders) - 1}__"

    processed = re.sub(number_unit_pattern, replace_with_placeholder, text)
    processed = re.sub(r"\.{3,}", "...", processed)
    processed = re.sub(r"-{2,}", "--", processed)

    # We now split on the zero-width boundary *next to* the punctuation,
    # not on the whitespace itself. This is far more robust.
    # Punctuation that ends a phrase. Added em (—) and en (–) dashes.
    preceding_punct = r"\.\.\.|[?!.,—–]"
    # Punctuation/tags that start a new phrase.
    # The hyphen (-) logic is now more complex: it splits only if the hyphen
    # is NOT adjecent to whitespace characters on either side. This
    # preserves hyphenated words like "natural-sounding" and "cul-de-sac".
    # (?<!\S)- : hyphen not preceded by a non-space (e.g., "word -")
    # -(?!\S)  : hyphen not followed by a non-space (e.g., "- word")
    following_punct = r"--|(?<!\S)-|-(?!\S)|\[S[0-9]+\]"

    # The pattern now looks for the boundary itself.
    split_pattern = re.compile(f"(?<={preceding_punct})|(?={following_punct})")

    # re.split on a zero-width pattern can create empty strings, which we filter out.
    parts = re.split(split_pattern, processed)

    restored_parts = []
    for part in parts:
        # The .strip() here is now essential to remove whitespace that was
        # not consumed by the split.
        stripped_part = part.strip()
        if not stripped_part:
            continue

        def restore_from_placeholder(match: re.Match) -> str:
            idx = int(match.group(1))
            return str(placeholders[idx])

        restored = re.sub(
            r"__NUMBER_UNIT_(\d+)__", restore_from_placeholder, stripped_part
        )
        if restored:
            restored_parts.append(restored)

    return restored_parts


def soft_cost(duration: float, min_d: float, max_d: float) -> float:
    """Quadratic penalty outside [min_d, max_d]."""
    if duration < min_d:
        return (min_d - duration) ** 2
    elif duration > max_d:
        return (duration - max_d) ** 2
    return 0.0


def dp_segment(phrases: List[str], wps: float, min_d: Union[float, str], max_d: Union[float, str]) -> List[str]:
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
