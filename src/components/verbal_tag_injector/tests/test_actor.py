from unittest.mock import MagicMock

import pytest
from src.components.verbal_tag_injector.actor import get_actor_suggestion


@pytest.fixture
def mock_llm_invoker():
    return MagicMock()


@pytest.fixture
def sample_briefing_packet():
    return {
        "task_directive_template": "Perform this line: {current_line}",
        "global_summary": "A global summary.",
        "local_context": "Some local context.",
        "moment_summary": "A moment summary.",
        "directors_notes": "A director's note.",
        "current_line": "The current line.",
    }


def test_get_actor_suggestion(sample_briefing_packet, mock_llm_invoker):
    mock_llm_invoker.invoke.return_value.content = "An actor suggestion."
    suggestion = get_actor_suggestion(sample_briefing_packet, mock_llm_invoker)

    mock_llm_invoker.invoke.assert_called_once()
    call_args = mock_llm_invoker.invoke.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0]["role"] == "user"
    assert "Perform this line: The current line." in call_args[0]["content"]
    assert suggestion == "An actor suggestion."
