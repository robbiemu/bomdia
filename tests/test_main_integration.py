"""Integration test for the main pipeline."""

import os

from src.pipeline import run_pipeline


def test_main_pipeline_integration(tmp_path, monkeypatch):
    """Test the main pipeline with a simple transcript."""
    # Create a temporary transcript file
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """[S1] Hello there
[S2] Hi, how are you?
[S1] I'm doing well, thanks for asking.
[S1] It's a beautiful day today.
[S2] Yes, perfect for a walk in the park."""
    )

    # Create a temporary output file path
    output_path = tmp_path / "output.mp3"

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

    monkeypatch.setattr("src.pipeline.DiaTTS", MockDiaTTS)

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
                content = '{"moment_summary": "Friendly greeting", "directors_note": "Be warm and welcoming"}'
            return MockResponse()

    monkeypatch.setattr("src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker)

    # Mock the actor's get_actor_suggestion function
    def mock_get_actor_suggestion(briefing_packet, llm_invoker):
        # Simple mock that just returns the current line with a verbal tag
        current_line = briefing_packet["current_line"]
        if "[insert-verbal-tag-for-pause]" in current_line:
            return current_line.replace("[insert-verbal-tag-for-pause]", "(um)")
        return current_line

    monkeypatch.setattr("src.components.verbal_tag_injector.director.get_actor_suggestion", mock_get_actor_suggestion)

    # Run the pipeline
    run_pipeline(str(transcript_path), str(output_path))

    # Check if the output file was created
    assert os.path.exists(output_path)
