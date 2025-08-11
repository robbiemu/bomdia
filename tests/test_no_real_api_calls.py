"""Test to verify no real API calls are made."""

import pytest
from unittest.mock import patch, MagicMock

from shared.config import config
from src.components.verbal_tag_injector.llm_based import build_llm_injector
from src.pipeline import run_pipeline


def test_no_real_api_calls(tmp_path, monkeypatch):
    """Test that verifies no real API calls are made during testing."""

    # Create a temporary transcript file
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """Speaker 1: Hello there
Speaker 2: Hi, how are you?"""
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

    # Mock litellm.completion to raise an exception if it's ever called
    def fail_if_called(*args, **kwargs):
        raise Exception("Real API call detected! This should not happen in tests.")

    with patch("litellm.completion", side_effect=fail_if_called):
        # This should not raise an exception because no real API calls should be made
        run_pipeline(str(transcript_path), str(output_path))

        # Check if the output file was created
        assert (output_path).exists()

        # Also test the LLM injector directly with proper mocking
        config.LLM_SPEC = "test/model"
        try:
            mock_choice = MagicMock()
            mock_choice.message.content = "[S1] Hello there (laughs)"
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]

            with patch("litellm.completion", return_value=mock_response) as mock_completion:
                injector = build_llm_injector()
                state = {"current_line": "[S1] Hello there"}
                result = injector(state)
                assert result["modified_line"] == "[S1] Hello there (laughs)"
                # Verify the mock was called
                mock_completion.assert_called_once()
        finally:
            # Reset config
            config.LLM_SPEC = None
