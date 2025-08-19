"""Integration test for the main pipeline."""

import os
from unittest.mock import patch

from src.pipeline import run_pipeline


def test_main_pipeline_integration(tmp_path, monkeypatch):
    """Test the main pipeline with a simple transcript."""
    # Create a temporary transcript file
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """
[S1] Hello there
[S2] Hi, how are you?
[S1] I'm doing well, thanks for asking.
[S1] It's a beautiful day today.
[S2] Yes, perfect for a walk in the park."""
    )

    # Create a temporary output file path
    output_path = tmp_path / "output.mp3"

    # Mock the DiaTTS class
    class MockDiaTTS:
        def __init__(self, model_checkpoint, revision=None, seed=None, log_level=None):
            pass

        def generate(self, texts, unified_audio_prompt, unified_transcript_prompt):
            # Create mock AudioSegment objects for each text
            from pydub import AudioSegment
            import numpy as np

            audio_segments = []
            for text in texts:
                # Create minimal audio data (0.1 seconds of silence)
                frames = int(0.1 * 22050)
                data = np.zeros(frames, dtype=np.int16)

                # Create AudioSegment from raw bytes
                segment = AudioSegment(
                    data.tobytes(),
                    frame_rate=22050,
                    sample_width=2,  # 16-bit = 2 bytes
                    channels=1       # Mono
                )
                audio_segments.append(segment)

            return audio_segments

        def register_voice_prompts(self, voice_prompts):
            pass

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

            # Mock response for the moment performance
            class MockResponse:
                content = "[S1] Hello there\n[S2] Hi, how are you?"

            return MockResponse()

    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )

    # Mock the actor's perform_moment function
    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
    ):
        # Simple mock that just returns the lines as they are
        result = {}
        for line in lines:
            line_number = line["global_line_number"]
            result[line_number] = line
        return result

    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )

    # Run the pipeline
    run_pipeline(str(transcript_path), str(output_path))

    # Check if the output file was created
    assert os.path.exists(output_path)

def test_main_pipeline_integration_seeding(tmp_path, monkeypatch):
    """Test the main pipeline with a simple transcript and no seed."""
    # Create a temporary transcript file
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """[S1] Hello there
[S2] Hi, how are you?
[S1] I'm doing well, thanks for asking.
[S1] It's a beautiful day today.
[S2] Yes, perfect for a walk in the park.
[S1] I could walk for hours on a day like this.
[S2] Me too. It's so refreshing.
[S1] We should do this more often.
[S2] I agree completely."""
    )

    # Create a temporary output file path
    output_path = tmp_path / "output.mp3"

    # Mock the DiaTTS class
    class MockDiaTTS:
        def __init__(self, model_checkpoint, revision=None, seed=None, log_level=None):
            self.seed = seed

        def generate(self, texts, unified_audio_prompt, unified_transcript_prompt):
            # Create mock AudioSegment objects for each text
            from pydub import AudioSegment
            import numpy as np

            audio_segments = []
            for text in texts:
                # Create minimal audio data (0.1 seconds of silence)
                frames = int(0.1 * 22050)
                data = np.zeros(frames, dtype=np.int16)

                # Create AudioSegment from raw bytes
                segment = AudioSegment(
                    data.tobytes(),
                    frame_rate=22050,
                    sample_width=2,  # 16-bit = 2 bytes
                    channels=1       # Mono
                )
                audio_segments.append(segment)

            return audio_segments

        def register_voice_prompts(self, voice_prompts):
            pass

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

            # Mock response for the moment performance
            class MockResponse:
                content = "[S1] Hello there\n[S2] Hi, how are you?"

            return MockResponse()

    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )

    # Mock the actor's perform_moment function
    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
    ):
        # Simple mock that just returns the lines as they are
        result = {}
        for line in lines:
            line_number = line["global_line_number"]
            result[line_number] = line
        return result

    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )

    # Run the pipeline
    with patch("src.pipeline.random.randint") as mock_randint:
        mock_randint.return_value = 12345
        run_pipeline(str(transcript_path), str(output_path), seed=None)

    # Check if the output file was created
    assert os.path.exists(output_path)

    # Check if randint was called
    mock_randint.assert_called_once()
