"""LLM-based verbal tag injector."""

from typing import Any, Callable, Dict

from shared.config import config


def build_llm_injector(llm: Any) -> Callable[[Dict[str, Any]], Dict[str, str]]:
    """
    Build an LLM-based injector function using the provided LLM
    """

    def llm_injector(state: Dict[str, Any]) -> Dict[str, str]:
        """
        Process a line of transcript with contextual awareness using an LLM.

        Takes the current line and surrounding context (previous/next lines, summary,
        topic)
        and generates a modified line with appropriate verbal tags inserted. Uses only
        the
        first non-empty line from the LLM response for safety.

        Args:
            state (Dict[str, Any]): Contains 'prev_lines', 'current_line', 'next_lines',
              'summary', and 'topic' keys for contextual processing.

        Returns:
            Dict[str, str]: Dictionary with 'modified_line' key containing the processed
            text.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # build a prompt for the LLM
        prev_lines = state.get("prev_lines", [])
        cur = state["current_line"]
        next_lines = state.get("next_lines", [])
        summary = state.get("summary", "")
        topic = state.get("topic", "")

        system = SystemMessage(content=config.VERBAL_TAG_INJECTOR_SYSTEM_PROMPT)

        human_prompt = config.VERBAL_TAG_INJECTOR_HUMAN_PROMPT_TEMPLATE.format(
            prev_lines=chr(10).join(prev_lines),
            current_line=cur,
            next_lines=chr(10).join(next_lines),
            summary=summary,
            topic=topic,
            verbal_tags=config.VERBAL_TAGS,
        )

        resp = llm.invoke([system, HumanMessage(content=human_prompt)])
        # resp is an AIMessage — extract content
        content = getattr(resp, "content", None)
        if content is None:
            # fallback: convert to str
            content = str(resp)
        # take only first non-empty line (safety)
        for line in content.splitlines():
            if line.strip():
                modified = line.strip()
                break
        else:
            modified = cur
        return {"modified_line": modified}

    return llm_injector
