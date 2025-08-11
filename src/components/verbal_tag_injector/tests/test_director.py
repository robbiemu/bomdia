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
        assert director.global_summary == "A global summary."


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
