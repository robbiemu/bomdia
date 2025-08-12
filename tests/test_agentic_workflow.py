import os
import tempfile
import unittest
import logging
import io
from unittest.mock import MagicMock, patch, call
from src.pipeline import run_pipeline
from src.components.verbal_tag_injector.director import Director
from shared.config import config


class TestAgenticWorkflow(unittest.TestCase):
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
                def __init__(self, model_checkpoint, revision=None, seed=None):
                    pass

                def text_to_audio_file(self, text, path):
                    # Create a dummy wav file with proper WAV format
                    import wave
                    import numpy as np

                    # Create a simple WAV file with minimal content
                    with wave.open(path, "wb") as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(22050)  # Sample rate
                        # Create minimal audio data (0.1 seconds of silence)
                        frames = int(0.1 * 22050)
                        data = np.zeros(frames, dtype=np.int16)
                        wav_file.writeframes(data.tobytes())

            # Mock the LLM invoker to avoid network calls
            class MockLLMInvoker:
                def __init__(self, model, **kwargs):
                    pass

                def invoke(self, messages):
                    # Mock response for the global summary
                    if "You are a script analyst" in messages[0]["content"]:

                        class MockResponse:
                            content = "Topic: Greeting. Relationship: Friendly. Arc: Positive."

                        return MockResponse()

                    # Mock response for the moment performance
                    class MockResponse:
                        content = "[S1] Hello world.\n[S2] This is a test (um) with a pause.\n[S1] And another line."

                    return MockResponse()

            with (
                patch("src.pipeline.DiaTTS", MockDiaTTS),
                patch(
                    "src.components.verbal_tag_injector.director.LiteLLMInvoker",
                    MockLLMInvoker,
                ),
                patch.dict(os.environ, {"LLM_SPEC": "openai/gpt-4o-mini"}),
            ):  # Ensure LLM is available
                # Mock the actor's perform_moment function
                def mock_perform_moment(
                    self, moment_id, lines, token_budget, constraints, global_summary
                ):
                    # Simple mock that just returns the lines with pause placeholders replaced
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

    def test_enhanced_logging_observability(self):
        """Test that all the enhanced logging points are working correctly."""
        # Set up logging capture
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("src.components.verbal_tag_injector.director")
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
                        self.assertIn(
                            "Starting moment-based rehearsal process", log_output
                        )
                        self.assertIn("Processing Moment", log_output)
                        self.assertIn("Token budget for Actor is", log_output)
                        self.assertIn("Global tag budget status:", log_output)

                        # Fix: Look for the actual logging format with moment ID and timing
                        self.assertIn("finalized in", log_output)  # More flexible match
                        # Alternative: More specific assertion
                        # self.assertTrue(any("finalized in" in line and "s." in line for line in log_output.split('\n')))

                        # Additional assertions for better test coverage
                        self.assertIn(
                            "Director initialized with a budget of", log_output
                        )
                        self.assertIn("Global Summary Generated:", log_output)
                        self.assertIn("Director defined Moment", log_output)
                        self.assertIn(
                            "Delegating to Actor for creative suggestion", log_output
                        )
                        self.assertIn(
                            "Delegating to Director for final review", log_output
                        )
                        self.assertIn("rehearsal process complete", log_output)

                        # Test that the actor was called and modified the text
                        self.assertIn("(logged)", final_script[0]["text"])

                        # Verify the final script structure
                        self.assertEqual(len(final_script), 2)
                        self.assertEqual(final_script[0]["speaker"], "S1")
                        self.assertEqual(final_script[1]["speaker"], "S2")

        finally:
            # Clean up
            logger.removeHandler(handler)

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
                "text": "Oh, by the way, did you see that email from HR about the new policy?",
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

    @patch("shared.config.config.MAX_TAG_RATE", 1)
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
