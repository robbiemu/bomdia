"""Integration test for the main pipeline."""

import os

from src.pipeline import run_pipeline


def test_main_pipeline_integration(tmp_path, monkeypatch):
    """Test the main pipeline with a simple transcript."""
    # Create a temporary transcript file
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """Speaker 1: Hello there
Speaker 2: Hi, how are you?
Speaker 1: I'm doing well, thanks for asking.
Speaker 1: It's a beautiful day today.
Speaker 2: Yes, perfect for a walk in the park."""
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

    # Run the pipeline
    run_pipeline(str(transcript_path), str(output_path))

    # Check if the output file was created
    assert os.path.exists(output_path)
