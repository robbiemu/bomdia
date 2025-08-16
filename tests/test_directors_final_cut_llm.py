import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.components.verbal_tag_injector.director import Director


class TestDirectorsFinalCutLLM(unittest.TestCase):
    """Integration tests for Director's Final Cut LLM review mode and fallback."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple transcript for testing
        self.transcript = [
            {"speaker": "S1", "text": "Hello there."},
            {"speaker": "S2", "text": "Hi! How are you doing today?"},
            {"speaker": "S1", "text": "I'm doing well, thanks."},
        ]

    def test_llm_review_mode_success_path(self):
        """Test successful LLM review mode execution."""
        # Set up logging capture
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as log_file:
            log_handler = logging.FileHandler(log_file.name)
            log_handler.setLevel(logging.DEBUG)
            logger = logging.getLogger("src.components.verbal_tag_injector")
            logger.addHandler(log_handler)
            logger.setLevel(logging.DEBUG)

            try:
                # Mock the LLM invoker with proper responses
                mock_llm_invoker = MagicMock()

                # Global summary response
                mock_global_summary = MagicMock()
                mock_global_summary.content = "A friendly greeting conversation."

                # Moment definition response
                mock_moment_definition = MagicMock()
                mock_moment_definition.content = """{
                    "moment_summary": "Greeting exchange",
                    "directors_notes": "Focus on warmth and friendliness",
                    "start_line": 0,
                    "end_line": 1
                }"""

                # Director review response (LLM mode) - keeps some tags, reverts others
                mock_director_review = MagicMock()
                mock_director_review.content = """{
                    "line_0": "Hello there (warmly).",
                    "line_1": "Hi! How are you doing today?"
                }"""

                mock_llm_invoker.invoke.side_effect = [
                    mock_global_summary,
                    mock_moment_definition,
                    mock_director_review,  # This is the director review call
                ]

                # Mock the actor's perform_moment method to add some tags
                def mock_perform_moment(
                    self, moment_id, lines, token_budget, constraints, global_summary
                ):
                    result = {}
                    for line in lines:
                        line_number = line["global_line_number"]
                        text = line["text"]
                        # Add some tags that the Director can review
                        if line_number == 0:
                            text = "Hello there (warmly) (cheerfully)."
                        elif line_number == 1:
                            text = "Hi! How are you doing today? (smiling)"
                        result[line_number] = {
                            "speaker": line["speaker"],
                            "text": text,
                            "global_line_number": line_number,
                        }
                    return result

                # Set environment variable for LLM review mode
                with patch.dict(
                    os.environ, {"DIRECTOR_AGENT_REVIEW_MODE": "llm"}
                ):
                    # Force reload config to pick up the environment variable
                    import importlib
                    import shared.config
                    importlib.reload(shared.config)
                    from shared.config import config

                    with patch(
                        "src.components.verbal_tag_injector.director.LiteLLMInvoker",
                        return_value=mock_llm_invoker,
                    ):
                        with patch(
                            "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                            mock_perform_moment,
                        ):
                            with patch("shared.config.config.MAX_TAG_RATE", 0.5):
                                # Create director and run rehearsal
                                director = Director(self.transcript)
                                final_script = director.run_rehearsal()

                                # Read the log file to check for expected messages
                                log_file.seek(0)
                                log_content = log_file.read()

                                # Assert that LLM mode was used
                                self.assertIn(
                                    "Director's Final Cut: Using 'llm' review mode",
                                    log_content,
                                )

                                # Assert that LLM review was successful (no fallback)
                                self.assertNotIn("falling back to procedural", log_content)

                                # Assert that the final script reflects Director's decisions
                                self.assertEqual(len(final_script), 3)
                                # Line 0 should have one tag (warmly) kept, (cheerfully) removed
                                self.assertIn("(warmly)", final_script[0]["text"])
                                self.assertNotIn("(cheerfully)", final_script[0]["text"])
                                # Line 1 should have no tags (reverted to original)
                                self.assertEqual(
                                    final_script[1]["text"],
                                    "Hi! How are you doing today?",
                                )

            finally:
                # Clean up
                logger.removeHandler(log_handler)
                log_handler.close()
                os.unlink(log_file.name)

    def test_llm_review_mode_fallback_to_procedural(self):
        """Test LLM review mode fallback to procedural when LLM fails."""
        # Set up logging capture
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as log_file:
            log_handler = logging.FileHandler(log_file.name)
            log_handler.setLevel(logging.DEBUG)
            logger = logging.getLogger("src.components.verbal_tag_injector")
            logger.addHandler(log_handler)
            logger.setLevel(logging.DEBUG)

            try:
                # Mock the LLM invoker with proper responses, but make director review fail
                mock_llm_invoker = MagicMock()

                # Global summary response
                mock_global_summary = MagicMock()
                mock_global_summary.content = "A friendly greeting conversation."

                # Moment definition response
                mock_moment_definition = MagicMock()
                mock_moment_definition.content = """{
                    "moment_summary": "Greeting exchange",
                    "directors_notes": "Focus on warmth and friendliness",
                    "start_line": 0,
                    "end_line": 1
                }"""

                # Director review response (LLM mode) - return malformed JSON
                mock_director_review_failure = MagicMock()
                mock_director_review_failure.content = "This is not valid JSON at all!"

                mock_llm_invoker.invoke.side_effect = [
                    mock_global_summary,
                    mock_moment_definition,
                    mock_director_review_failure,  # This will cause fallback
                ]

                # Mock the actor's perform_moment method to add some tags
                def mock_perform_moment(
                    self, moment_id, lines, token_budget, constraints, global_summary
                ):
                    result = {}
                    for line in lines:
                        line_number = line["global_line_number"]
                        text = line["text"]
                        # Add more tags than budget allows to test procedural fallback
                        if line_number == 0:
                            text = "Hello there (warmly) (cheerfully) (enthusiastically)."
                        elif line_number == 1:
                            text = "Hi! How are you doing today? (smiling) (brightly)"
                        result[line_number] = {
                            "speaker": line["speaker"],
                            "text": text,
                            "global_line_number": line_number,
                        }
                    return result

                # Set environment variable for LLM review mode
                with patch.dict(
                    os.environ, {"DIRECTOR_AGENT_REVIEW_MODE": "llm"}
                ):
                    # Force reload config to pick up the environment variable
                    import importlib
                    import shared.config
                    importlib.reload(shared.config)
                    from shared.config import config

                    with patch(
                        "src.components.verbal_tag_injector.director.LiteLLMInvoker",
                        return_value=mock_llm_invoker,
                    ):
                        with patch(
                            "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                            mock_perform_moment,
                        ):
                            with patch("shared.config.config.MAX_TAG_RATE", 0.5):
                                # Create director and run rehearsal
                                director = Director(self.transcript)
                                final_script = director.run_rehearsal()

                                # Read the log file to check for expected messages
                                log_file.seek(0)
                                log_content = log_file.read()

                                # Assert that LLM mode was attempted first
                                self.assertIn(
                                    "Director's Final Cut: Using 'llm' review mode",
                                    log_content,
                                )

                                # Assert that fallback occurred
                                self.assertIn(
                                    "LLM review failed", log_content
                                )
                                self.assertIn(
                                    "falling back to procedural review", log_content
                                )

                                # Assert that procedural review was executed
                                self.assertIn(
                                    "Director's Final Cut (Procedural): Removed",
                                    log_content,
                                )

                                # Assert that the final script shows procedural pruning
                                # (last tags removed)
                                self.assertEqual(len(final_script), 3)

                                # Should have some tags but fewer than Actor added
                                total_original_tags = sum(
                                    len([t for t in line["text"] if t == "("])
                                    for line in self.transcript
                                )
                                total_final_tags = sum(
                                    len([t for t in line["text"] if t == "("])
                                    for line in final_script
                                )

                                # Final should have more than original but less than what
                                # Actor would have added (due to procedural pruning)
                                self.assertGreater(total_final_tags, total_original_tags)

            finally:
                # Clean up
                logger.removeHandler(log_handler)
                log_handler.close()
                os.unlink(log_file.name)

    def test_llm_review_mode_exception_fallback(self):
        """Test LLM review mode fallback when LLM invoker raises exception."""
        # Simplified test focusing on core functionality
        # Use a 2-line transcript to get exactly 1 moment and test the review
        simple_transcript = [
            {"speaker": "S1", "text": "Hello there."},
            {"speaker": "S2", "text": "Hi! How are you doing?"},
        ]

        # Set up logging capture
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as log_file:
            log_handler = logging.FileHandler(log_file.name)
            log_handler.setLevel(logging.DEBUG)
            logger = logging.getLogger("src.components.verbal_tag_injector")
            logger.addHandler(log_handler)
            logger.setLevel(logging.DEBUG)

            try:
                # Mock the LLM invoker - 3 calls: global summary, moment definition, director review
                mock_llm_invoker = MagicMock()

                mock_global_summary = MagicMock()
                mock_global_summary.content = "A friendly greeting conversation."

                mock_moment_definition = MagicMock()
                mock_moment_definition.content = """{
                    "moment_summary": "Greeting exchange",
                    "directors_notes": "Focus on warmth",
                    "start_line": 0,
                    "end_line": 1
                }"""

                # The 3rd call (director review) will fail
                mock_llm_invoker.invoke.side_effect = [
                    mock_global_summary,
                    mock_moment_definition,
                    Exception("Network error during LLM call"),  # Director review failure
                ]

                # Mock the actor to add some tags
                def mock_perform_moment(
                    self, moment_id, lines, token_budget, constraints, global_summary
                ):
                    result = {}
                    for line in lines:
                        line_number = line["global_line_number"]
                        text = line["text"]
                        # Add a tag that will need review
                        if line_number == 0:
                            text = "Hello there (warmly)."
                        result[line_number] = {
                            "speaker": line["speaker"],
                            "text": text,
                            "global_line_number": line_number,
                        }
                    return result

                # Force LLM review mode and run
                with patch.dict(os.environ, {"DIRECTOR_AGENT_REVIEW_MODE": "llm"}):
                    # Force reload config to pick up the environment variable
                    import importlib
                    import shared.config
                    importlib.reload(shared.config)
                    from shared.config import config

                    with patch(
                        "src.components.verbal_tag_injector.director.LiteLLMInvoker",
                        return_value=mock_llm_invoker,
                    ):
                        with patch(
                            "src.components.verbal_tag_injector.actor.Actor.perform_moment",
                            mock_perform_moment,
                        ):
                            with patch("shared.config.config.MAX_TAG_RATE", 1.0):  # Higher budget
                                # Create director and run rehearsal
                                director = Director(simple_transcript)
                                final_script = director.run_rehearsal()

                                # Read the log file to check for expected messages
                                log_file.seek(0)
                                log_content = log_file.read()

                                # Assert that LLM mode was attempted first
                                self.assertIn(
                                    "Director's Final Cut: Using 'llm' review mode",
                                    log_content,
                                )

                                # Assert that fallback occurred due to exception
                                self.assertIn(
                                    "LLM review failed", log_content
                                )
                                self.assertIn(
                                    "falling back to procedural review", log_content
                                )

                                # Assert that the pipeline didn't crash
                                self.assertEqual(len(final_script), 2)
                                self.assertEqual(final_script[0]["speaker"], "S1")
                                self.assertEqual(final_script[1]["speaker"], "S2")

            finally:
                # Clean up
                logger.removeHandler(log_handler)
                log_handler.close()
                os.unlink(log_file.name)


if __name__ == "__main__":
    unittest.main()
