"""Integration test for the main pipeline."""

import os
from unittest.mock import patch

from src.pipeline import run_pipeline


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
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

        def generate(self, texts, audio_prompts=None):
            # Create mock AudioSegment objects for each text
            import numpy as np
            from pydub import AudioSegment

            audio_segments = []
            for _text in texts:
                # Create minimal audio data (0.1 seconds of silence)
                frames = int(0.1 * 22050)
                data = np.zeros(frames, dtype=np.int16)

                # Create AudioSegment from raw bytes
                segment = AudioSegment(
                    data.tobytes(),
                    frame_rate=22050,
                    sample_width=2,  # 16-bit = 2 bytes
                    channels=1,  # Mono
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
                content = """{
                    "moment_summary": "A moment",
                    "directors_notes": "Some notes",
                    "start_line": 0,
                    "end_line": 1
                }"""

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

    # Mock subprocess.run to prevent calling actual worker script
    def mock_subprocess_run(*args, **kwargs):
        # Mock successful worker script output
        class MockResult:
            returncode = 0
            stdout = (
                '{"speaker_id": "S2", "audio_path": "/tmp/synthetic_prompts/'
                'S2_seed_00012345.wav", "stdout_transcript": "[S2] Test synthetic '
                'prompt  [S1]"}'
            )
            stderr = ""

        return MockResult()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    # Mock config to prevent synthetic prompt generation from creating real folders
    with (
        patch("src.pipeline.config.GENERATE_SYNTHETIC_PROMPTS", False),
        patch(
            "src.pipeline.config.GENERATE_PROMPT_OUTPUT_DIR",
            os.path.join(tmp_path, "synthetic_prompts"),
        ),
    ):
        # Run the pipeline
        run_pipeline(str(transcript_path), str(output_path))

    # Check if the output file was created
    assert os.path.exists(output_path)


# The lost test 'test_main_pipeline_integration_seeding' was not restored because
# it tests behavior that contradicts the task specification. According to
# docs/tasks/simulated_voice_cloning.task.md, when no seed is provided, the
# orchestrator should proceed without generating one (not call random.randint).
# The synthetic prompt workers generate their own temporary seeds as needed.

# Removed the complex test_pipeline_with_synthetic_prompts test as it was
# overly complex and the synthetic prompts functionality is already thoroughly
# tested in tests/test_synthetic_prompts_integration.py. The main concern was
# preventing filesystem artifacts during tests, which has been resolved.
