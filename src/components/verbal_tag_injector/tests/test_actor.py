from unittest.mock import MagicMock, patch

import pytest
from src.components.verbal_tag_injector.actor import Actor


@pytest.fixture
def mock_llm_invoker():
    return MagicMock()


@pytest.fixture
def sample_lines():
    return [
        {"speaker": "S1", "text": "Hello there.", "global_line_number": 0},
        {"speaker": "S2", "text": "General Kenobi.", "global_line_number": 1},
    ]


@patch("src.components.verbal_tag_injector.actor.config")
def test_actor_perform_moment(mock_config, mock_llm_invoker, sample_lines):
    mock_config.actor_agent = {
        "moment_task_directive_template": (
            "{moment_text}\n{global_summary}\n{token_budget}\n{constraints_text}\n"
            "{available_verbal_tags}\n{available_line_combiners}"
        )
    }
    mock_llm_invoker.invoke.return_value.content = (
        '{"line_0": "Hello there.", "line_1": "General Kenobi."}'
    )
    actor = Actor(mock_llm_invoker)

    result = actor.perform_moment(
        moment_id="moment_0",
        lines=sample_lines,
        token_budget=10.0,
        constraints={},
        global_summary="A global summary.",
    )

    mock_llm_invoker.invoke.assert_called_once()
    call_args = mock_llm_invoker.invoke.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0]["role"] == "user"
    assert "Hello there." in call_args[0]["content"]
    assert "General Kenobi." in call_args[0]["content"]
    assert "A global summary." in call_args[0]["content"]

    # Check that we got results for both lines
    assert 0 in result
    assert 1 in result
    assert result[0]["text"] == "Hello there."
    assert result[1]["text"] == "General Kenobi."


@patch("src.components.verbal_tag_injector.actor.config")
def test_actor_perform_moment_with_constraints(
    mock_config, mock_llm_invoker, sample_lines
):
    mock_config.actor_agent = {
        "moment_task_directive_template": (
            "{moment_text}\n{global_summary}\n{token_budget}\n{constraints_text}\n"
            "{available_verbal_tags}\n{available_line_combiners}"
        )
    }
    mock_llm_invoker.invoke.return_value.content = (
        '{"line_0": "Hello there.", "line_1": "General Kenobi."}'
    )
    actor = Actor(mock_llm_invoker)

    constraints = {0: "This line is locked"}

    result = actor.perform_moment(
        moment_id="moment_0",
        lines=sample_lines,
        token_budget=10.0,
        constraints=constraints,
        global_summary="A global summary.",
    )

    mock_llm_invoker.invoke.assert_called_once()
    call_args = mock_llm_invoker.invoke.call_args[0][0]
    assert len(call_args) == 1
    assert call_args[0]["role"] == "user"
    assert "CONSTRAINTS:" in call_args[0]["content"]
    assert "Line 0: This line is locked" in call_args[0]["content"]
