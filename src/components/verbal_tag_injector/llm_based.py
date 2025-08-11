"""LLM-based verbal tag injector."""

from typing import Callable, Dict

import litellm  # Import litellm
from shared.config import config

from . import VerbalTagInjectorState


def build_llm_injector() -> (
    Callable[[VerbalTagInjectorState], Dict[str, str]]
):  # No longer needs llm param
    """
    Build an LLM-based injector function that uses LiteLLM.
    """

    def llm_injector(state: VerbalTagInjectorState) -> Dict[str, str]:
        """
        Process a line of transcript with contextual awareness using LiteLLM.
        """
        # build a prompt for the LLM
        prev_lines = state.get("prev_lines", [])
        cur = state["current_line"]
        next_lines = state.get("next_lines", [])
        summary = state.get("summary", "")
        topic = state.get("topic", "")

        # Use standard dicts for messages
        messages = [
            {"role": "system", "content": config.VERBAL_TAG_INJECTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": config.VERBAL_TAG_INJECTOR_HUMAN_PROMPT_TEMPLATE.format(
                    prev_lines="\\n".join(prev_lines),
                    current_line=cur,
                    next_lines="\\n".join(next_lines),
                    summary=summary,
                    topic=topic,
                    verbal_tags=config.VERBAL_TAGS,
                ),
            },
        ]

        try:
            # Call LiteLLM completion
            response = litellm.completion(
                model=config.LLM_SPEC,
                messages=messages,
                **config.LLM_PARAMETERS,  # Pass configured parameters
            )
            # Extract content from the LiteLLM response object
            content = response.choices[0].message.content
        except Exception as e:
            print(
                f"[LiteLLM] Error during completion: {e}. Falling back to original "
                "line."
            )
            content = cur

        # take only first non-empty line (safety)
        for line in content.splitlines():
            if line.strip():
                modified = line.strip()
                break
        else:
            modified = cur
        return {"modified_line": modified}

    return llm_injector
