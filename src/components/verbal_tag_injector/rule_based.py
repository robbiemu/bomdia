"""Rule-based verbal tag injector."""

import random
import re
from typing import Dict

from shared.config import config

from . import VerbalTagInjectorState


def rule_based_injector(state: VerbalTagInjectorState) -> Dict[str, str]:
    """Inject verbal tags using rule-based logic."""
    state.get("prev_lines", [])
    cur = state["current_line"]
    state.get("next_lines", [])

    if cur is None:
        raise KeyError("Configuration error: current_line not set in app.toml")

    # replace placeholder with random varied tag
    if config.PAUSE_PLACEHOLDER in cur:
        tag = random.choice(config.LINE_COMBINERS)  # nosec
        cur = cur.replace(config.PAUSE_PLACEHOLDER, tag)
    else:
        # maybe insert a sparing tag at start (approx MAX_TAG_RATE)
        if random.random() < config.MAX_TAG_RATE:  # nosec
            tag = random.choice(config.VERBAL_TAGS)  # nosec
            # ensure format like: [S1] (gasps) rest...
            cur = re.sub(r"^(\[S[12]\])\s*", r"\1 " + tag + " ", cur)
    return {"modified_line": cur}
