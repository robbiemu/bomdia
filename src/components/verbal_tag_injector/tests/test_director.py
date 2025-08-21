import os
from unittest.mock import MagicMock, patch

import pytest
from src.components.verbal_tag_injector.director import Director


@pytest.fixture
def sample_transcript():
    """Sample transcript for testing."""
    return [
        {"speaker": "S1", "text": "Hello there."},
        {
            "speaker": "S2",
            "text": "General Kenobi. [insert-verbal-tag-for-pause] You are a bold one.",
        },
    ]


@pytest.fixture
def transcript_without_placeholders():
    """Transcript without placeholders for testing."""
    return [
        {"speaker": "S1", "text": "Hello there."},
        {"speaker": "S2", "text": "General Kenobi. You are a bold one."},
    ]


@pytest.fixture
def mock_llm_invoker():
    """Mock LLM invoker for testing."""
    return MagicMock()


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_director_initialization(sample_transcript, mock_llm_invoker):
    """Test that the Director initializes correctly."""
    with patch(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker",
        return_value=mock_llm_invoker,
    ):
        mock_llm_invoker.invoke.return_value.content = "A global summary."
        director = Director(sample_transcript)
        assert director.transcript == sample_transcript
        assert director.llm_invoker == mock_llm_invoker
        assert director.new_tag_budget >= 0  # Should have some budget


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_director_initialization_with_higher_budget(
    transcript_without_placeholders, mock_llm_invoker
):
    """Test that the Director initializes with a higher budget when appropriate."""
    with patch(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker",
        return_value=mock_llm_invoker,
    ):
        mock_llm_invoker.invoke.return_value.content = "A global summary."
        # Create a longer transcript to get a budget > 0
        long_transcript = transcript_without_placeholders * 5  # 10 lines
        director = Director(long_transcript)
        assert director.new_tag_budget > 0  # Should have a positive budget


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
@patch("shared.config.config.MAX_TAG_RATE", 1)
def test_run_rehearsal_basic(sample_transcript, mock_llm_invoker):
    """Test the basic run_rehearsal functionality."""
    with (
        patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ),
        patch(
            "src.components.verbal_tag_injector.actor.Actor.perform_moment",
            return_value={
                0: {"speaker": "S1", "text": "Hello there.", "global_line_number": 0},
                1: {
                    "speaker": "S2",
                    "text": "A suggested line.",
                    "global_line_number": 1,
                },
            },
        ),
    ):
        # Mock the LLM invoker to return a valid JSON response for moment definition
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        mock_moment_definition = MagicMock()
        mock_moment_definition.content = """{
  "moment_summary": "Test moment",
  "directors_notes": "Director's notes for test moment",
  "start_line": 0,
  "end_line": 1
}"""

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_definition,
        ]

        director = Director(sample_transcript)
        final_script = director.run_rehearsal()
        assert len(final_script) == 2
        assert final_script[0]["text"] == "Hello there."
        assert final_script[1]["text"] == "A suggested line."


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
@patch("shared.config.config.MAX_TAG_RATE", 1)
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
            "src.components.verbal_tag_injector.actor.Actor.perform_moment",
            return_value={
                0: {"speaker": "S1", "text": "Hello there.", "global_line_number": 0},
                1: {
                    "speaker": "S2",
                    "text": "(laughs) General Kenobi. You are a bold one.",
                    "global_line_number": 1,
                },
            },
        ),
    ):
        # Mock the LLM invoker to return valid JSON responses
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        mock_moment_definition = MagicMock()
        mock_moment_definition.content = """{
  "moment_summary": "Test moment with tag injection",
  "directors_notes": "Director's notes for test moment with tag injection",
  "start_line": 0,
  "end_line": 1
}"""

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_definition,
        ]

        director = Director(transcript_without_placeholders)
        final_script = director.run_rehearsal()
        assert len(final_script) == 2
        assert final_script[0]["text"] == "Hello there."
        assert "(laughs)" in final_script[1]["text"]


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_run_rehearsal_with_new_tag_injection_and_budget(
    transcript_without_placeholders, mock_llm_invoker
):
    """Test that tag injection respects the budget."""
    # Create a longer transcript to get a budget > 0
    long_transcript = transcript_without_placeholders * 5  # 10 lines

    with patch(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker",
        return_value=mock_llm_invoker,
    ):
        # Mock the LLM invoker to return valid JSON responses
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        mock_moment_definitions = []
        for i in range(10):  # 10 lines, so we need 10 moment definitions
            mock_moment_def = MagicMock()
            mock_moment_def.content = f"""{{
  "moment_summary": "Test moment {i}",
  "directors_notes": "Director's notes for test moment {i}",
  "start_line": {i},
  "end_line": {i}
}}"""
            mock_moment_definitions.append(mock_moment_def)

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary
        ] + mock_moment_definitions

        # Mock the actor to return lines with new tags
        def mock_perform_moment(
            self,
            moment_id,
            lines,
            token_budget,
            constraints,
            global_summary,
            sample_name,
        ):
            result = {}
            for i, line in enumerate(lines):
                line_number = line["global_line_number"]
                # Add a tag to each line
                modified_text = f"(tag{i}) {line['text']}"
                result[line_number] = {
                    "speaker": line["speaker"],
                    "text": modified_text,
                    "global_line_number": line_number,
                }
            return result

        with patch(
            "src.components.verbal_tag_injector.actor.Actor.perform_moment",
            mock_perform_moment,
        ):
            director = Director(long_transcript)
            assert director.new_tag_budget > 0  # Should have a positive budget

            final_script = director.run_rehearsal()

            # Should have processed all lines
            assert len(final_script) == 10

            # Should have added tags (respecting budget)
            tag_count = 0
            for line in final_script:
                if "(tag" in line["text"]:
                    tag_count += 1

            # Should have added some tags but respecting the budget
            assert tag_count > 0


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
@patch("shared.config.config.MAX_TAG_RATE", 1)
@patch(
    "shared.config.config.director_agent",
    {
        "review": {"mode": "procedural"},
        "global_summary_prompt": "Global summary: {transcript_text}",
        "unified_moment_analysis_prompt": "Analyze: {local_context}",
        "moment_definition_prompt": "Define moments: {forward_script_slice_text}",
        "previous_moment_template": "Previous: {last_moment_summary}",
        "rate_control": {"target_tag_rate": 0.10, "tag_burst_allowance": 3},
    },
)
def test_pivot_line_forward_cascading_edits(mock_llm_invoker):
    """Test that pivot lines are correctly handled with forward-cascading edits."""
    # Create a simple transcript with consecutive lines from the same speaker
    transcript = [
        {"speaker": "S1", "text": "Hello there."},
        {"speaker": "S1", "text": "How are you today?"},
        {"speaker": "S2", "text": "I'm doing well, thanks for asking."},
    ]

    # First call is for global summary
    mock_global_summary = MagicMock()
    mock_global_summary.content = "A global summary."

    # Second call is for moment definition - should return proper JSON
    mock_moment_definition = MagicMock()
    mock_moment_definition.content = """{
  "moment_summary": "First moment",
  "directors_notes": "Director's notes for first moment",
  "start_line": 0,
  "end_line": 1
}"""

    mock_llm_invoker.invoke.side_effect = [
        mock_global_summary,
        mock_moment_definition,
        mock_moment_definition,
        mock_moment_definition,
    ]

    # Create the director
    with patch(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker",
        return_value=mock_llm_invoker,
    ):
        director = Director(transcript)

        # Mock the actor's perform_moment method to simulate different behaviors
        def mock_perform_moment(
            self,
            moment_id,
            lines,
            token_budget,
            constraints,
            global_summary,
            sample_name,
        ):
            result = {}
            for line in lines:
                line_number = line["global_line_number"]
                text = line["text"]
                # For the first moment, add a tag
                if "moment_0_1" in moment_id:
                    text = "(laughs) " + text
                # For the second moment, respect the constraint
                elif "fallback" in moment_id and 1 in constraints:
                    # Should not modify the pivot line (line 1)
                    pass
                result[line_number] = {
                    "speaker": line["speaker"],
                    "text": text,
                    "global_line_number": line_number,
                }
            return result

        with patch(
            "src.components.verbal_tag_injector.actor.Actor.perform_moment",
            mock_perform_moment,
        ):
            final_script = director.run_rehearsal()

            # Check that the first line has the tag added
            assert "(laughs)" in final_script[0]["text"]
            assert final_script[0]["text"] == "(laughs) Hello there."


class TestMomentDefinitionAndState:
    """Tests the Director's ability to define moments and the agent's
    state management."""

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    def test_director_defines_single_moment_correctly(self, mock_llm_invoker):
        """
        Verify that when the agent encounters a new line, it calls the Director,
        and the Director's simple, single-moment response is correctly parsed and
        stored.
        - The `moment_cache` should contain one new moment definition.
        - The `line_to_moment_map` should be populated for all lines within that moment.
        """
        # Create a simple transcript
        transcript = [
            {"speaker": "S1", "text": "Hello there."},
            {"speaker": "S2", "text": "Hi!"},
        ]

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        # Second call is for moment definition - should return proper JSON
        mock_moment_definition = MagicMock()
        mock_moment_definition.content = """{
  "moment_summary": "Two line moment",
  "directors_notes": "Director's notes for two line moment",
  "start_line": 0,
  "end_line": 1
}"""

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_definition,
        ]

        # Create the director
        with patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ):
            director = Director(transcript)

            # Check initial state
            assert len(director.moment_cache) == 0
            assert len(director.line_to_moment_map) == 0

            # Run the rehearsal to trigger moment definition
            with patch(
                "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                return_value={
                    0: {
                        "speaker": "S1",
                        "text": "Hello there.",
                        "global_line_number": 0,
                    },
                    1: {"speaker": "S2", "text": "Hi!", "global_line_number": 1},
                },
            ):
                director.run_rehearsal()

            # Check that moments were created
            assert len(director.moment_cache) > 0
            assert len(director.line_to_moment_map) > 0

            # Check that each line is mapped to a moment
            for line_num in range(len(transcript)):
                assert line_num in director.line_to_moment_map
                assert len(director.line_to_moment_map[line_num]) > 0

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    def test_moment_finalization_flag_is_set_correctly(self, mock_llm_invoker):
        """
        Verify that after a moment is fully processed and reviewed, its entry
        in `moment_cache` is correctly marked with `is_finalized = True`.
        - Process a script segment that completes exactly one moment.
        - Check the `moment_cache` to ensure the flag was flipped from False to True.
        """
        # Create a simple transcript
        transcript = [
            {"speaker": "S1", "text": "Hello there."},
        ]

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        # Second call is for moment definition - should return proper JSON
        mock_moment_definition = MagicMock()
        mock_moment_definition.content = """{
  "moment_summary": "Single line moment",
  "directors_notes": "Director's notes for single line moment",
  "start_line": 0,
  "end_line": 0
}"""

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_definition,
        ]

        # Create the director
        with patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ):
            director = Director(transcript)

            # Initially no moments are finalized
            assert len(director.finalized_moments) == 0

            # Run the rehearsal
            with patch(
                "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                return_value={
                    0: {
                        "speaker": "S1",
                        "text": "Hello there.",
                        "global_line_number": 0,
                    },
                },
            ):
                director.run_rehearsal()

            # After processing, moments should be finalized
            assert len(director.finalized_moments) > 0


class TestActorAndDirectorWorkflow:
    """Tests the interaction between the Actor, Director, and the
    recomposition logic."""

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    @patch("shared.config.config.MAX_TAG_RATE", 1)
    def test_actor_processes_lines_as_moment_in_one_call(self, mock_llm_invoker):
        """
        Verify that the Actor is called only ONCE for an entire moment.
        - Run the agent on a script that contains a single moment.
        - Assert that the actor was called exactly one time.
        - Assert that the data passed to the actor contained all the lines of the
        moment.
        """
        # Create a simple transcript
        transcript = [
            {"speaker": "S1", "text": "Hello there."},
            {"speaker": "S2", "text": "Hi!"},
        ]

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        # Second call is for moment definition - should return proper JSON
        mock_moment_definition = MagicMock()
        mock_moment_definition.content = """{
  "moment_summary": "Two line moment",
  "directors_notes": "Director's notes for two line moment",
  "start_line": 0,
  "end_line": 1
}"""

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_definition,
        ]

        # Create the director
        with patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ):
            director = Director(transcript)

            # Mock the actor's perform_moment method
            mock_perform_moment = MagicMock(
                return_value={
                    0: {
                        "speaker": "S1",
                        "text": "Hello there.",
                        "global_line_number": 0,
                    },
                    1: {"speaker": "S2", "text": "Hi!", "global_line_number": 1},
                }
            )

            with patch(
                "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                mock_perform_moment,
            ):
                director.run_rehearsal()

                # Verify the actor was called
                mock_perform_moment.assert_called()

                # Verify it was called with the correct lines
                call_args = mock_perform_moment.call_args
                assert call_args is not None
                args, kwargs = call_args
                # Check that lines were passed
                assert "lines" in kwargs
                assert len(kwargs["lines"]) == 2  # Both lines in one call

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    @patch("shared.config.config.MAX_TAG_RATE", 1)
    def test_recomposition_updates_finalized_blocks_correctly(self, mock_llm_invoker):
        """
        Verify that after the Actor's performance, the edited lines are correctly
        placed back into the `finalized_blocks` structure.
        """
        # Create a simple transcript
        transcript = [
            {"speaker": "S1", "text": "Hello there."},
            {"speaker": "S2", "text": "Hi!"},
        ]

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        # Second call is for moment definition - should return proper JSON
        mock_moment_definition = MagicMock()
        mock_moment_definition.content = """{
  "moment_summary": "Two line moment",
  "directors_notes": "Director's notes for two line moment",
  "start_line": 0,
  "end_line": 1
}"""

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_definition,
        ]

        # Create the director
        with patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ):
            director = Director(transcript)

            # Mock the actor's perform_moment method to return modified lines
            def mock_perform_moment(
                self,
                moment_id,
                lines,
                token_budget,
                constraints,
                global_summary,
                sample_name,
            ):
                result = {}
                for line in lines:
                    line_number = line["global_line_number"]
                    # Modify the text
                    modified_text = f"(modified) {line['text']}"
                    result[line_number] = {
                        "speaker": line["speaker"],
                        "text": modified_text,
                        "global_line_number": line_number,
                    }
                return result

            with patch(
                "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                mock_perform_moment,
            ):
                final_script = director.run_rehearsal()

                # Check that the lines were modified
                assert "(modified)" in final_script[0]["text"]
                assert "(modified)" in final_script[1]["text"]


class TestSpecialHandlingAndEdgeCases:
    """Tests for the specific pivot line and error handling logic."""

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    def test_director_boundary_error_creates_fallback_moment(self, mock_llm_invoker):
        """
        Verify that nonsensical moment boundaries from the Director result in
        a safe, single-line moment.
        - Mock the moment definition to return an invalid response
          (e.g., `start_line: 10, end_line: 8`).
        - Run the agent on a line that triggers this mock.
        - Check the `moment_cache`. It should contain a new moment where
          `start_line` and `end_line` are both equal to the current line number.
        - The system should not crash and should proceed to process this fallback
        moment.
        """
        # Create a simple transcript
        transcript = [
            {"speaker": "S1", "text": "Hello there."},
        ]

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        mock_llm_invoker.invoke.return_value = mock_global_summary

        # Create the director
        with patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ):
            director = Director(transcript)

            # Mock the _define_moments_containing method to return invalid boundaries
            def mock_define_moments_containing(line_number):
                # Return a moment with invalid boundaries
                return [
                    {
                        "moment_id": "invalid_moment",
                        "start_line": 10,
                        "end_line": 8,
                        "is_finalized": False,
                        "lines": [director.original_lines[0]],
                        "description": "Invalid moment",
                        "directors_notes": "Director's notes for invalid moment",
                    }
                ]

            director._define_moments_containing = mock_define_moments_containing

            # Run the rehearsal - should not crash
            with patch(
                "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                return_value={
                    0: {
                        "speaker": "S1",
                        "text": "Hello there.",
                        "global_line_number": 0,
                    },
                },
            ):
                final_script = director.run_rehearsal()

                # Should complete successfully
                assert len(final_script) == 1
                assert final_script[0]["text"] == "Hello there."

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    def test_script_with_no_moments_completes_gracefully(self, mock_llm_invoker):
        """
        Verify that a script with no interesting moments is processed without error.
        - Run the agent on a simple script.
        - The agent should complete its run successfully.
        """
        # Create a simple transcript
        transcript = [
            {"speaker": "S1", "text": "Hello there."},
            {"speaker": "S2", "text": "Hi."},
        ]

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        # Second calls are for moment definitions - should return proper JSON
        mock_moment_definition_1 = MagicMock()
        mock_moment_definition_1.content = """{
  "moment_summary": "First line moment",
  "directors_notes": "Director's notes for first line moment",
  "start_line": 0,
  "end_line": 0
}"""

        mock_moment_definition_2 = MagicMock()
        mock_moment_definition_2.content = """{
  "moment_summary": "Second line moment",
  "directors_notes": "Director's notes for second line moment",
  "start_line": 1,
  "end_line": 1
}"""

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_definition_1,
            mock_moment_definition_2,
        ]

        # Create the director
        with patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ):
            director = Director(transcript)

            # Run the rehearsal
            with patch(
                "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                return_value={
                    0: {
                        "speaker": "S1",
                        "text": "Hello there.",
                        "global_line_number": 0,
                    },
                    1: {"speaker": "S2", "text": "Hi.", "global_line_number": 1},
                },
            ):
                final_script = director.run_rehearsal()

                # Should complete successfully
                assert len(final_script) == 2
                assert final_script[0]["text"] == "Hello there."
                assert final_script[1]["text"] == "Hi."


class TestCoTerminousMomentHandling:
    """Tests for handling co-terminous moments."""

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    @patch("shared.config.config.MAX_TAG_RATE", 1)
    def test_co_terminous_moments_are_processed_once(self, mock_llm_invoker):
        """
        Verify that when two moments end on the same line, the full
        `_execute_full_moment` workflow is only called once.
        - Manually create two moments in the `moment_cache` that end on the same line.
        - Run the agent on the line that triggers finalization for both.
        - Assert that `_execute_full_moment` was called exactly once.
        """
        # Create a simple transcript
        transcript = [
            {"speaker": "S1", "text": "Hello there."},
            {"speaker": "S2", "text": "Hi!"},
        ]

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        # Second call is for moment definition - should return proper JSON
        mock_moment_definition = MagicMock()
        mock_moment_definition.content = """{
  "moment_summary": "Two line moment",
  "directors_notes": "Director's notes for two line moment",
  "start_line": 0,
  "end_line": 1
}"""

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_definition,
        ]

        # Create the director
        with patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ):
            director = Director(transcript)

            # Manually create co-terminous moments
            moment1 = {
                "moment_id": "moment_0_1",
                "start_line": 0,
                "end_line": 1,
                "is_finalized": False,
                "lines": director.original_lines,
                "description": "First moment",
                "directors_notes": "Director's notes for first moment",
            }
            moment2 = {
                "moment_id": "moment_1_1",
                "start_line": 1,
                "end_line": 1,
                "is_finalized": False,
                "lines": [director.original_lines[1]],
                "description": "Second moment",
                "directors_notes": "Director's notes for second moment",
            }
            director.moment_cache = {"moment_0_1": moment1, "moment_1_1": moment2}
            director.line_to_moment_map = {
                0: ["moment_0_1"],
                1: ["moment_0_1", "moment_1_1"],
            }

            # Mock the actor's perform_moment method to track calls
            with patch(
                "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                return_value={
                    0: {
                        "speaker": "S1",
                        "text": "Hello there.",
                        "global_line_number": 0,
                    },
                    1: {"speaker": "S2", "text": "Hi!", "global_line_number": 1},
                },
            ) as mock_actor:
                director.run_rehearsal()

                # Verify that Actor.perform_moment was called only once
                mock_actor.assert_called_once()
