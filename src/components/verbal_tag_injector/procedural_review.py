"""Procedural review logic for the Director's Final Cut.

This module provides a deterministic, rule-based approach to enforcing
tag budgets in the agentic workflow. When the Actor's performance exceeds
the allowed tag budget, this module removes newly added tags (preserving
original tags) until the budget is met.
"""

import re
from typing import Any, Dict, List, Tuple

# Regular expression to match parenthetical verbal tags
TAG_RX = re.compile(r"\(.*?\)")


def procedural_final_cut(
    original_by_line: Dict[int, str],
    performed_by_line: Dict[int, str],
    tag_budget: int,
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Apply procedural tag budget enforcement to an Actor's performance.

    This function identifies newly added tags by comparing the Actor's performance
    to the original script, then removes the last N new tags if over budget.
    Original tags are never touched.

    Args:
        original_by_line: Dictionary mapping line numbers to original text
        performed_by_line: Dictionary mapping line numbers to Actor's performed text
        tag_budget: Maximum number of new tags allowed for this moment

    Returns:
        Tuple of (final_by_line, metrics) where:
        - final_by_line: Dictionary mapping line numbers to approved final text
        - metrics: Dictionary with 'removed', 'added', and 'overage' counts
    """
    # Track new tag spans (line_no, start_pos, end_pos) for removal
    new_tag_spans: List[Tuple[int, int, int]] = []
    total_new_tags = 0

    # Analyze each performed line to identify new tags
    for line_num, performed_text in sorted(performed_by_line.items()):
        original_text = original_by_line.get(line_num, "")

        # Extract tags from both versions
        original_tags = TAG_RX.findall(original_text)
        performed_tags = list(TAG_RX.finditer(performed_text))

        # Use multiset subtraction to identify new tags
        # Create a copy of original tags that we can modify
        remaining_original_tags = original_tags.copy()

        for tag_match in performed_tags:
            tag_token = tag_match.group(0)
            if tag_token in remaining_original_tags:
                # This tag was in the original, remove it from our tracking
                remaining_original_tags.remove(tag_token)
            else:
                # This is a new tag added by the Actor
                total_new_tags += 1
                new_tag_spans.append((line_num, tag_match.start(), tag_match.end()))

    # If within budget, no changes needed
    if total_new_tags <= tag_budget:
        return (
            performed_by_line,
            {"removed": 0, "added": total_new_tags, "overage": 0},
        )

    # Calculate how many tags to remove
    overage = total_new_tags - tag_budget

    # Start with a copy of the performed text
    final_by_line = {ln: txt for ln, txt in performed_by_line.items()}
    removed = 0

    # Remove tags from the end (last in reading order) first
    for line_num, start_pos, end_pos in reversed(new_tag_spans):
        if removed >= overage:
            break

        current_text = final_by_line[line_num]
        # Remove the tag and clean up any multiple spaces
        new_text = current_text[:start_pos] + current_text[end_pos:]
        # Clean up multiple spaces and trim
        while "  " in new_text:
            new_text = new_text.replace("  ", " ")
        new_text = new_text.strip()
        # Ensure we don't have space before period
        new_text = new_text.replace(" .", ".")
        final_by_line[line_num] = new_text
        removed += 1

    return (
        final_by_line,
        {"removed": removed, "added": total_new_tags, "overage": overage},
    )
