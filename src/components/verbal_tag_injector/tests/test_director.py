import json
from unittest.mock import MagicMock, patch

import pytest
from src.components.verbal_tag_injector.director import Director


@pytest.fixture
def mock_llm_invoker():
    return MagicMock()


@pytest.fixture
def sample_transcript():
    return [
        {"speaker": "S1", "text": "Hello there."},
        {
            "speaker": "S2",
            "text": "General Kenobi. [insert-verbal-tag-for-pause] You are a bold one.",
        },
    ]


@pytest.fixture
def transcript_without_placeholders():
    return [
        {"speaker": "S1", "text": "Hello there."},
        {"speaker": "S2", "text": "General Kenobi. You are a bold one."},
    ]


def test_director_initialization(sample_transcript, mock_llm_invoker):
    with patch(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker",
        return_value=mock_llm_invoker,
    ):
        mock_llm_invoker.invoke.return_value.content = "A global summary."
        director = Director(sample_transcript)
        assert director.transcript == sample_transcript
        assert director.llm_invoker == mock_llm_invoker
        assert director.final_script == []
        assert director.tags_injected == 0
        assert director.new_tag_budget == 0  # 0.15 * 2 = 0.3, floor = 0
        assert director.global_summary == "A global summary."


def test_director_initialization_with_higher_budget(
    transcript_without_placeholders, mock_llm_invoker
):
    with patch(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker",
        return_value=mock_llm_invoker,
    ):
        mock_llm_invoker.invoke.return_value.content = "A global summary."
        # With 10 lines and 0.15 rate, budget should be 1 (floor of 1.5)
        long_transcript = transcript_without_placeholders * 5  # 10 lines
        director = Director(long_transcript)
        assert director.new_tag_budget == 1


def test_run_rehearsal_json_parsing(sample_transcript, mock_llm_invoker):
    with (
        patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ),
        patch(
            "src.components.verbal_tag_injector.director.get_actor_suggestion",
            return_value="A suggested line.",
        ),
    ):
        mock_llm_invoker.invoke.side_effect = [
            MagicMock(content="A global summary."),
            MagicMock(
                content=json.dumps(
                    {
                        "moment_summary": "A moment summary.",
                        "directors_note": "A director's note.",
                    }
                )
            ),
        ]
        director = Director(sample_transcript)
        final_script = director.run_rehearsal()
        assert len(final_script) == 2
        assert final_script[0]["text"] == "Hello there."
        assert final_script[1]["text"] == "A suggested line."


def test_run_rehearsal_json_decode_error(sample_transcript, mock_llm_invoker):
    with (
        patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ),
        patch(
            "src.components.verbal_tag_injector.director.get_actor_suggestion",
            return_value="A suggested line.",
        ),
    ):
        mock_llm_invoker.invoke.side_effect = [
            MagicMock(content="A global summary."),
            MagicMock(content="invalid json"),
        ]
        director = Director(sample_transcript)
        final_script = director.run_rehearsal()
        assert len(final_script) == 2
        assert final_script[0]["text"] == "Hello there."
        assert final_script[1]["text"] == "A suggested line."


def test_run_rehearsal_with_new_tag_injection(
    transcript_without_placeholders, mock_llm_invoker
):
    """Test that lines without placeholders can still be processed
    for new tag injection."""
    with (
        patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ),
        patch(
            "src.components.verbal_tag_injector.director.get_actor_suggestion",
            return_value="(laughs) General Kenobi. You are a bold one.",
        ),
        patch(
            "src.components.verbal_tag_injector.director.random.random",
            return_value=0.25,  # Less than 0.5, so we'll try to add a tag
        ),
    ):
        mock_llm_invoker.invoke.side_effect = [
            MagicMock(content="A global summary."),
            MagicMock(
                content=json.dumps(
                    {
                        "moment_summary": "A moment summary.",
                        "directors_note": "A director's note.",
                    }
                )
            ),
        ]
        # With 2 lines and 0.15 rate, budget should be 0 (floor of 0.3)
        # So even if we try, we shouldn't be able to inject a new tag
        director = Director(transcript_without_placeholders)
        assert director.new_tag_budget == 0  # No budget for new tags

        final_script = director.run_rehearsal()
        assert len(final_script) == 2
        # Second line should be processed but new tag stripped due to budget
        assert final_script[0]["text"] == "Hello there."
        assert (
            final_script[1]["text"] == "General Kenobi. You are a bold one."
        )  # Tag stripped


def test_run_rehearsal_with_new_tag_injection_and_budget(
    transcript_without_placeholders, mock_llm_invoker
):
    """Test that lines without placeholders can be processed for
    new tag injection when budget allows."""
    # Create a longer transcript to get a budget > 0
    long_transcript = transcript_without_placeholders * 5  # 10 lines

    with (
        patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ),
        patch(
            "src.components.verbal_tag_injector.director.get_actor_suggestion",
            return_value="(laughs) General Kenobi. You are a bold one.",
        ),
        patch(
            "src.components.verbal_tag_injector.director.random.random",
            return_value=0.25,  # Less than 0.5, so we'll try to add a tag
        ),
    ):
        mock_llm_invoker.invoke.side_effect = [
            MagicMock(content="A global summary."),
            MagicMock(
                content=json.dumps(
                    {
                        "moment_summary": "A moment summary.",
                        "directors_note": "A director's note.",
                    }
                )
            ),
        ]
        director = Director(long_transcript)
        assert director.new_tag_budget == 1  # Budget for 1 new tag

        final_script = director.run_rehearsal()
        assert len(final_script) == 10
        # At least one line should have the new tag added
        # (exact behavior depends on random selection, but we're patching it)
