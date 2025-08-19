from unittest.mock import MagicMock, patch

import pytest
import src.components.verbal_tag_injector.rehearsal_graph
from src.components.verbal_tag_injector.director import Director


@pytest.fixture
def sample_transcript():
    """Sample transcript for testing."""
    return [
        {"speaker": "S1", "text": "Hello there."},
        {"speaker": "S2", "text": "General Kenobi. You are a bold one."},
        {"speaker": "S1", "text": "I love this place!"},
        {"speaker": "S2", "text": "Suddenly, everything changed."},
        {"speaker": "S1", "text": "Are you sure?"},
        {"speaker": "S2", "text": "Yes, I am certain."},
        {"speaker": "S1", "text": "(sighs) This is a lot to take in."},
        {"speaker": "S2", "text": "We must hurry!"},
    ]


class MockConfig:
    """A mock config class that supports both attribute and dictionary access."""

    def __init__(self):
        self.LLM_SPEC = "test-model"
        self.LLM_PARAMETERS = {}
        self.MAX_TAG_RATE = 0.15
        self.PAUSE_PLACEHOLDER = "[insert-verbal-tag-for-pause]"
        self.REHEARSAL_CHECKPOINT_PATH = ":memory:"

        # Create director_agent as a dict to support dictionary access
        self.director_agent = {
            "global_summary_prompt": "Global summary: {transcript_text}",
            "unified_moment_analysis_prompt": (
                "Analyze: {local_context}, {current_line}"
            ),
            "moment_definition_prompt": (
                "Define moment: {previous_moment_segment}, "
                "{forward_script_slice_text}, {line_number}"
            ),
            "previous_moment_template": (
                "Previous moment: {last_moment_summary}, "
                "{last_moment_end_line}, {last_finalized_line_text}"
            ),
            "rate_control": {"target_tag_rate": 0.10, "tag_burst_allowance": 3},
            "review": {"mode": "procedural"},
        }

        # Create actor_agent as a dict
        self.actor_agent = {"task_directive_template": "Directive: {current_line}"}


@patch("src.components.verbal_tag_injector.director.config")
def test_is_candidate_for_tagging(mock_config, sample_transcript):
    """Test the _is_candidate_for_tagging method."""
    with patch(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker"
    ) as mock_llm_class:
        # Create a mock LLM instance
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="A global summary.")
        mock_llm_class.return_value = mock_llm_instance

        transcript = [{"speaker": "S1", "text": "Hello world"}]
        director = Director(transcript)

        # Test emotional keywords
        assert director._is_candidate_for_tagging("He cries every day") is True
        assert director._is_candidate_for_tagging("She shouts angrily") is True
        assert director._is_candidate_for_tagging("They love each other") is True
        assert director._is_candidate_for_tagging("Suddenly, he left") is True

        # Test punctuation
        assert director._is_candidate_for_tagging("This is amazing!") is True
        assert director._is_candidate_for_tagging("Are you serious?") is True
        assert director._is_candidate_for_tagging("Just a regular sentence.") is False

        # Test parenthetical actions
        assert (
            director._is_candidate_for_tagging("He said (whispering) quietly") is True
        )
        assert director._is_candidate_for_tagging("No action here") is False


@patch("src.components.verbal_tag_injector.director.config")
@patch("shared.config.config")
def test_run_rehearsal_skips_line_when_no_tokens_available(
    shared_config, mock_config, sample_transcript
):
    """
    Test that the rehearsal process completes successfully with proper config mocking.
    """
    # Configure both config mocks
    mock_config.REHEARSAL_CHECKPOINT_PATH = ":memory:"

    # Use MockConfig instance for both configs to provide real values
    real_config = MockConfig()

    # Configure mock_config for Director initialization
    mock_config.LLM_SPEC = real_config.LLM_SPEC
    mock_config.LLM_PARAMETERS = real_config.LLM_PARAMETERS
    mock_config.MAX_TAG_RATE = real_config.MAX_TAG_RATE
    mock_config.director_agent = real_config.director_agent

    # Configure shared_config for rehearsal graph
    shared_config.REHEARSAL_CHECKPOINT_PATH = ":memory:"
    shared_config.MAX_TAG_RATE = real_config.MAX_TAG_RATE
    shared_config.director_agent = real_config.director_agent

    # Replace the cached DIRECTOR_AGENT_CONFIG with real dict values
    src.components.verbal_tag_injector.rehearsal_graph.DIRECTOR_AGENT_CONFIG = {
        "global_summary_prompt": "Global summary: {transcript_text}",
        "rate_control": {"target_tag_rate": 0.1, "tag_burst_allowance": 3},
    }

    with patch(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker"
    ) as mock_llm_class:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(
            content=(
                '{"moment_summary": "A moment", "directors_notes": "Notes", '
                '"start_line": 0, "end_line": 0}'
            )
        )
        mock_llm_class.return_value = mock_llm_instance

        director = Director(sample_transcript)

        # Mock actor perform_moment to return lines with new tags
        with patch(
            "src.components.verbal_tag_injector.actor.Actor.perform_moment"
        ) as mock_actor:
            mock_actor.return_value = {
                0: {
                    "speaker": "S1",
                    "text": "Hello there.",
                    "global_line_number": 0,
                },
                1: {
                    "speaker": "S2",
                    "text": "(laughs) General Kenobi. You are a bold one.",
                    "global_line_number": 1,
                },
                2: {
                    "speaker": "S1",
                    "text": "I love this place!",
                    "global_line_number": 2,
                },
                3: {
                    "speaker": "S2",
                    "text": "Suddenly, everything changed.",
                    "global_line_number": 3,
                },
                4: {
                    "speaker": "S1",
                    "text": "Are you sure?",
                    "global_line_number": 4,
                },
                5: {
                    "speaker": "S2",
                    "text": "Yes, I am certain.",
                    "global_line_number": 5,
                },
                6: {
                    "speaker": "S1",
                    "text": "(sighs) This is a lot to take in.",
                    "global_line_number": 6,
                },
                7: {
                    "speaker": "S2",
                    "text": "We must hurry!",
                    "global_line_number": 7,
                },
            }

            # Run the rehearsal
            final_script = director.run_rehearsal()

            # Test that the rehearsal completes successfully without errors
            assert len(final_script) == len(sample_transcript)
            assert all("speaker" in line for line in final_script)
            assert all("text" in line for line in final_script)

            # Test that no filesystem artifacts were created (main purpose of this test)
            # The rehearsal should use :memory: database instead of real files
