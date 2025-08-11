import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from src.pipeline import run_pipeline

class TestAgenticFlowIntegration(unittest.TestCase):
    def test_run_pipeline_with_agentic_flow(self):
        # Create a temporary directory for our test files
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a dummy transcript file
            transcript_path = os.path.join(tmp_dir, "test_transcript.txt")
            with open(transcript_path, "w") as f:
                f.write("[S1] Hello world.\n")
                f.write("[S2] This is a test [insert-verbal-tag-for-pause] with a pause.\n")
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

                    # Mock response for the unified moment analysis
                    class MockResponse:
                        content = '{"moment_summary": "Testing pause", "directors_note": "Add a pause"}'
                    return MockResponse()

            with patch("src.pipeline.DiaTTS", MockDiaTTS), \
                 patch("src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker), \
                 patch.dict(os.environ, {"LLM_SPEC": "openai/gpt-4o-mini"}):  # Ensure LLM is available

                # Mock the actor's get_actor_suggestion function
                def mock_get_actor_suggestion(briefing_packet, llm_invoker):
                    # Simple mock that just returns the current line with a verbal tag
                    current_line = briefing_packet["current_line"]
                    if "[insert-verbal-tag-for-pause]" in current_line:
                        return current_line.replace("[insert-verbal-tag-for-pause]", "(um)")
                    return current_line

                with patch("src.components.verbal_tag_injector.director.get_actor_suggestion", mock_get_actor_suggestion):
                    # Run the pipeline
                    run_pipeline(transcript_path, output_path)

            # Check if the output file was created
            self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()
