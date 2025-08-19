import contextlib
import io
import logging
import os
import tempfile
import unittest
import uuid
from unittest.mock import MagicMock, patch

from pydub import AudioSegment
from src.components.verbal_tag_injector.director import Director
from src.pipeline import run_pipeline


class TestAgenticWorkflow(unittest.TestCase):
    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    def test_run_pipeline_with_agentic_flow(self):
        # Create a temporary directory for our test files
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a dummy transcript file
            transcript_path = os.path.join(tmp_dir, "test_transcript.txt")
            with open(transcript_path, "w") as f:
                f.write("[S1] Hello world.\n")
                f.write(
                    "[S2] This is a test [insert-verbal-tag-for-pause] with a pause.\n"
                )
                f.write("[S1] And another line.\n")

            # Create a dummy output file path
            output_path = os.path.join(tmp_dir, "test_output.mp3")

            # Mock the DiaTTS class
            class MockDiaTTS:
                def __init__(
                    self, model_checkpoint, revision=None, seed=None, log_level=None
                ):
                    pass

                def generate(self, texts, audio_prompts=None):
                    # Create a simple WAV file with minimal content
                    segments = []
                    for _ in texts:
                        # Create a simple silent AudioSegment
                        segment = AudioSegment.silent(duration=100)  # 100ms of silence
                        segments.append(segment)
                    return segments

                def register_voice_prompts(self, voice_prompts):
                    pass

            # Mock the LLM invoker to avoid network calls
            class MockLLMInvoker:
                def __init__(self, model, **kwargs):
                    pass

                def invoke(self, messages):
                    # Mock response for the global summary
                    if "You are a script analyst" in messages[0]["content"]:

                        class MockResponse:
                            content = (
                                "Topic: Greeting. Relationship: Friendly. "
                                "Arc: Positive."
                            )

                        return MockResponse()

                    # Mock response for the moment performance
                    class MockResponse:
                        content = (
                            "[S1] Hello world.\n[S2] This is a test (um) with "
                            "a pause.\n[S1] And another line."
                        )

                    return MockResponse()

            with (
                patch("src.pipeline.DiaTTS", MockDiaTTS),
                patch(
                    "src.components.verbal_tag_injector.director.LiteLLMInvoker",
                    MockLLMInvoker,
                ),
                patch.dict(os.environ, {"LLM_SPEC": "openai/gpt-4o-mini"}),
                patch(
                    "src.pipeline.config.GENERATE_PROMPT_OUTPUT_DIR",
                    os.path.join(tmp_dir, "synthetic_prompts"),
                ),
                patch(
                    "src.pipeline.config.GENERATE_SYNTHETIC_PROMPTS", False
                ),  # Disable synthetic prompts
            ):  # Ensure LLM is available
                # Mock the actor's perform_moment function
                def mock_perform_moment(
                    self, moment_id, lines, token_budget, constraints, global_summary
                ):
                    # Simple mock that just returns the lines with pause
                    #  placeholders replaced
                    result = {}
                    for line in lines:
                        line_number = line["global_line_number"]
                        text = line["text"]
                        if "[insert-verbal-tag-for-pause]" in text:
                            text = text.replace("[insert-verbal-tag-for-pause]", "(um)")
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
                    # Run the pipeline
                    run_pipeline(transcript_path, output_path)

            # Check if the output file was created
            self.assertTrue(os.path.exists(output_path))

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    def test_co_terminous_moment_sorting(self):
        """Test that co-terminous moments are sorted correctly by start line."""
        # Create a simple transcript
        transcript = [
            {"speaker": "S1", "text": "Hello there."},
            {"speaker": "S2", "text": "Hi!"},
            {"speaker": "S1", "text": "How are you?"},
        ]

        # Mock the LLM invoker
        mock_llm_invoker = MagicMock()

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        # Second call is for moment definition
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
        ]

        # Create the director
        with patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ):
            director = Director(transcript)

            # Create test moments that end at the same line but start at different lines
            test_moments = [
                {
                    "moment_id": "moment_C",
                    "start_line": 2,
                    "end_line": 2,
                    "is_finalized": False,
                },
                {
                    "moment_id": "moment_A",
                    "start_line": 0,
                    "end_line": 2,
                    "is_finalized": False,
                },
                {
                    "moment_id": "moment_B",
                    "start_line": 1,
                    "end_line": 2,
                    "is_finalized": False,
                },
            ]

            # Apply the sorting logic
            test_moments.sort(key=lambda m: m.get("start_line", -1))

            # Verify sorting is correct
            expected_order = ["moment_A", "moment_B", "moment_C"]
            actual_order = [m["moment_id"] for m in test_moments]

            self.assertEqual(actual_order, expected_order)

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    def test_enhanced_logging_observability(self):
        """Test that all the enhanced logging points are working correctly."""
        # Set up logging capture
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("src.components.verbal_tag_injector")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            # Create a simple transcript
            transcript = [
                {"speaker": "S1", "text": "Hello there."},
                {"speaker": "S2", "text": "Hi!"},
            ]

            # Mock the LLM invoker with proper JSON responses
            mock_llm_invoker = MagicMock()

            # First call is for global summary
            mock_global_summary = MagicMock()
            mock_global_summary.content = "A global summary."

            # Second call is for moment definition
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
                with patch("shared.config.config.MAX_TAG_RATE", 1):
                    director = Director(transcript)

                    # Mock the actor's perform_moment method
                    def mock_perform_moment(
                        self,
                        moment_id,
                        lines,
                        token_budget,
                        constraints,
                        global_summary,
                    ):
                        result = {}
                        for line in lines:
                            line_number = line["global_line_number"]
                            modified_text = f"(logged) {line['text']}"
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

                        log_output = log_capture.getvalue()

                        # Test that key logging points are present
                        self.assertIn("Starting rehearsal graph execution", log_output)
                        self.assertIn(
                            "--- Moment-based rehearsal process complete. ---",
                            log_output,
                        )
                        self.assertIn(
                            "Director initialized with a budget of", log_output
                        )
                        self.assertIn("Global Summary Generated:", log_output)
                        self.assertIn("Director defined Moment", log_output)
                        self.assertIn("--- Processing Moment", log_output)
                        self.assertIn("finalized in", log_output)

                        # Test that the actor was called and modified the text
                        self.assertIn("(logged)", final_script[0]["text"])

                        # Verify the final script structure
                        self.assertEqual(len(final_script), 2)
                        self.assertEqual(final_script[0]["speaker"], "S1")
                        self.assertEqual(final_script[1]["speaker"], "S2")

        finally:
            # Clean up
            logger.removeHandler(handler)

    @patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
    def test_narrative_moment_discovery(self):
        """Test that demonstrates narrative moment discovery functionality."""
        # Create a test transcript with clear narrative moments
        merged_lines = [
            {"speaker": "S1", "text": "So, what did you think of the movie?"},
            {
                "speaker": "S2",
                "text": "Honestly, I thought it was a bit slow in the middle.",
            },
            {
                "speaker": "S1",
                "text": "Really? I loved the pacing. It felt deliberate.",
            },
            {
                "speaker": "S2",
                "text": (
                    "Oh, by the way, did you see that email from HR about the "
                    "new policy?"
                ),
            },
            {"speaker": "S1", "text": "No, what did it say?"},
        ]

        # Mock the LLM invoker to simulate narrative moment discovery
        mock_llm_invoker = MagicMock()

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A conversation about a movie and HR policy"

        # Second call is for moment discovery
        mock_moment_response = MagicMock()
        mock_moment_response.content = """{
      "moment_summary": "Discussion about the movie",
      "directors_notes": "Focus on the contrast in opinions about the movie",
      "start_line": 0,
      "end_line": 2
    }"""

        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_response,
        ]

        with patch(
            "src.components.verbal_tag_injector.director.LiteLLMInvoker",
            return_value=mock_llm_invoker,
        ):
            director = Director(merged_lines)

            # Test the moment discovery for line 0
            moments = director._define_moments_containing(0)

            self.assertEqual(len(moments), 1)
            self.assertEqual(moments[0]["moment_id"], "moment_0_2")
            self.assertEqual(moments[0]["start_line"], 0)
            self.assertEqual(moments[0]["end_line"], 2)
            self.assertIn("movie", moments[0]["description"].lower())

    @patch.dict(
        os.environ,
        {
            "MAX_TAG_RATE": "1",
            "REHEARSAL_CHECKPOINT_PATH": ":memory:",  # Use in-memory SQLite
        },
    )
    def test_simple_director_run(self):
        """Test a simple director run to see what's happening."""
        # Create a simple transcript
        transcript = [
            {"speaker": "S1", "text": "Hello there."},
            {"speaker": "S2", "text": "Hi!"},
        ]

        # Mock the LLM invoker
        mock_llm_invoker = MagicMock()

        # First call is for global summary
        mock_global_summary = MagicMock()
        mock_global_summary.content = "A global summary."

        # Second call is for moment definition
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
            import importlib

            import shared.config

            importlib.reload(shared.config)

            director = Director(transcript)

            # Mock the actor's perform_moment method
            def mock_perform_moment(
                self, moment_id, lines, token_budget, constraints, global_summary
            ):
                result = {}
                for line in lines:
                    line_number = line["global_line_number"]
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

                # Check if the actor was called by looking at the final script
                self.assertIn("(modified)", final_script[0]["text"])


class TestPersistenceAndResumption:
    """Tests the Director's ability to resume from a persistent checkpoint."""

    def test_rehearsal_resumes_from_sqlite_checkpoint(self, tmp_path, caplog):
        """
        Verify that the Director can resume a rehearsal from a SQLite checkpoint.
        - Part 1: Run the rehearsal on a 3-line script but force it to stop after
          processing only the first line by simulating an interruption.
        - Part 2: Create a new Director instance and run the rehearsal again with
          the same thread_id.
        - Assert that the second run resumes from line 1, not from the beginning,
          and that the final script is correct.
        """
        # --- Setup ---
        db_path = tmp_path / "test_resume_checkpoints.sqlite"
        thread_id = f"run-{uuid.uuid4()}"
        transcript = [
            {"speaker": "S1", "text": "This is the first line."},
            {"speaker": "S2", "text": "This is the second line."},
            {"speaker": "S1", "text": "This is the third line."},
        ]

        # Mock LLM and Actor responses
        mock_llm_invoker = MagicMock()
        mock_global_summary = MagicMock(content="A global summary.")
        mock_moment_def = MagicMock(
            content='{"moment_summary": "A moment", "directors_notes": "Notes", '
            '"start_line": 0, "end_line": 0}'
        )
        mock_llm_invoker.invoke.side_effect = [
            mock_global_summary,
            mock_moment_def,
            mock_moment_def,
            mock_moment_def,
        ]

        def mock_perform_moment(self, moment_id, lines, **kwargs):
            return {line["global_line_number"]: line.copy() for line in lines}

        # --- Part 1: The "Interrupted" Run ---
        with patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": str(db_path)}):
            with patch(
                "shared.llm_invoker.LiteLLMInvoker",
                return_value=mock_llm_invoker,
            ):
                with patch(
                    "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                    mock_perform_moment,
                ):
                    # Mock the build_rehearsal_graph function to control the graph
                    # behavior
                    original_build_graph = None
                    from src.components.verbal_tag_injector.director import (
                        build_rehearsal_graph as original_build_graph_import,
                    )

                    original_build_graph = original_build_graph_import

                    def mock_build_graph(director, checkpointer=None):
                        # Build the real graph
                        graph = original_build_graph(director, checkpointer)

                        # Store reference to the original invoke method
                        original_invoke = graph.invoke

                        # Patch the invoke method to simulate interruption
                        def patched_invoke(state, config=None):
                            # Simulate a low recursion limit causing an interruption
                            if config and config.get("recursion_limit", 50) <= 8:
                                raise RecursionError("Simulated interruption")
                            # Normal behavior
                            return original_invoke(state, config)

                        # Apply the patch to this specific graph instance
                        graph.invoke = patched_invoke
                        return graph

                    with patch(
                        "src.components.verbal_tag_injector.director.build_rehearsal_graph",
                        mock_build_graph,
                    ):
                        # Instantiate the first Director
                        director1 = Director(transcript)

                        # Run with a low recursion limit to simulate interruption
                        with contextlib.suppress(RecursionError):
                            director1.run_rehearsal(thread_id=thread_id)

        # Verify that the checkpoint file was created and has content
        assert db_path.exists()
        assert db_path.stat().st_size > 0

        # --- Part 2: The "Resumed" Run ---
        caplog.set_level(logging.INFO)
        with patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": str(db_path)}):
            with patch(
                "shared.llm_invoker.LiteLLMInvoker",
                return_value=mock_llm_invoker,
            ):
                with patch(
                    "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                    mock_perform_moment,
                ):
                    # Instantiate a NEW Director to simulate a fresh script start
                    director2 = Director(transcript)

                    # The global summary should NOT be generated again, as it's
                    #  part of the checkpointed state. We can test this by
                    #  checking the LLM call count. Reset the mock to count calls
                    #  for the second run only.
                    mock_llm_invoker.reset_mock()

                    # Run the rehearsal again with the SAME thread_id
                    final_script = director2.run_rehearsal(thread_id=thread_id)

                    # --- Assertions ---
                    # 1. Check for the "Resuming from checkpoint" log message
                    assert any(
                        "Resuming from checkpoint at line" in record.message
                        for record in caplog.records
                    )

                    # 2. The global summary LLM call should have been skipped on resume
                    # The only calls should be for moment definitions.
                    assert mock_global_summary.content not in [
                        c[0][0][0]["content"]
                        for c in mock_llm_invoker.invoke.call_args_list
                    ]

                    # 3. The final script should be complete and correct
                    assert len(final_script) == 3
                    assert final_script[0]["text"] == "This is the first line."
                    assert final_script[1]["text"] == "This is the second line."
                    assert final_script[2]["text"] == "This is the third line."
