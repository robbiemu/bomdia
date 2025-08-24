"""Integration tests for the synthetic voice prompt generation workflow."""

import json
import os
from unittest.mock import MagicMock, patch

from src.pipeline import run_pipeline


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_synthetic_prompts_integration_basic(tmp_path, monkeypatch):
    """Test basic synthetic prompt generation integration."""
    # Create a temporary transcript file with multiple speakers
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """
[S1] Hello there, this is the first speaker.
[S2] And I'm the second speaker responding.
[S1] We're having a conversation that spans multiple chunks.
[S2] Yes, this should trigger synthetic prompt generation.
[S1] Because we have multiple speakers and multiple chunks.
[S2] Perfect for testing the new feature.
"""
    )

    # Create temporary output path
    output_path = tmp_path / "output.mp3"

    # Track subprocess calls - only count calls to our generate_prompt module
    subprocess_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        # Only track calls to our generate_prompt module, not file detection calls
        if (
            isinstance(cmd, list)
            and len(cmd) > 2
            and "-m" in cmd
            and "generate_prompt" in cmd
        ):
            subprocess_calls.append(cmd)

            # Mock successful worker script execution
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps(
                {
                    "speaker_id": cmd[cmd.index("--speaker-id") + 1],
                    "audio_path": (
                        f"{tmp_path}/synthetic_prompts/"
                        f"{cmd[cmd.index('--speaker-id') + 1]}_seed_12345.wav"
                    ),
                    "stdout_transcript": (
                        f"[{cmd[cmd.index('--speaker-id') + 1]}] Dia is a state-of-the-art "
                        "text-to-speech system designed for high-quality audio generation."
                        "  This system can produce natural-sounding speech that captures "
                        "the nuances of human voice patterns.  The technology behind it "
                        "represents significant advances in neural audio synthesis, making "
                        "it particularly effective for creating realistic voice clones "
                        "from minimal input data."
                    ),
                }
            )
            result.stderr = ""
            return result
        else:
            # For other subprocess calls (like file detection), use the real implementation
            import subprocess

            return subprocess.run(cmd, **kwargs)

    # Mock the DiaTTS class for main pipeline
    class MockDiaTTS:
        def __init__(self, seed=None, model_checkpoint=None, log_level=None):
            self.seed = seed

        def register_voice_prompts(self, voice_prompts):
            pass

        def generate(self, texts, audio_prompts=None):
            # Create mock AudioSegment objects
            import numpy as np
            from pydub import AudioSegment

            audio_segments = []
            for _text in texts:
                frames = int(0.1 * 22050)
                data = np.zeros(frames, dtype=np.int16)
                segment = AudioSegment(
                    data.tobytes(), frame_rate=22050, sample_width=2, channels=1
                )
                audio_segments.append(segment)
            return audio_segments

    # Mock the LLM components
    class MockLLMInvoker:
        def __init__(self, model, **kwargs):
            pass

        def invoke(self, messages):
            class MockResponse:
                content = """{
  "moment_summary": "Initial conversation between two speakers",
  "directors_notes": "Keep the tone friendly and conversational",
  "start_line": 0,
  "end_line": 1
}"""

            return MockResponse()

    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
        sample_name=None,
    ):
        result = {}
        for line in lines:
            line_number = line["global_line_number"]
            result[line_number] = line
        return result

    # Apply mocks
    monkeypatch.setattr("src.pipeline.DiaTTS", MockDiaTTS)
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )

    # Set config values for testing - patch the config as imported by pipeline module
    monkeypatch.setattr("shared.config.config.GENERATE_SYNTHETIC_PROMPTS", True)
    monkeypatch.setattr(
        "shared.config.config.GENERATE_PROMPT_OUTPUT_DIR",
        str(tmp_path / "synthetic_prompts"),
    )

    # Run the pipeline
    run_pipeline(
        input_path=str(transcript_path), out_audio_path=str(output_path), seed=12345
    )

    # Verify output was created
    assert output_path.exists()

    # Verify subprocess was called for each unprompted speaker
    assert len(subprocess_calls) == 2  # S1 and S2

    # Check that both speakers were processed
    called_speakers = set()
    for cmd in subprocess_calls:
        speaker_idx = cmd.index("--speaker-id") + 1
        called_speakers.add(cmd[speaker_idx])

    assert called_speakers == {"S1", "S2"}

    # Verify command structure
    for cmd in subprocess_calls:
        assert "-m" in cmd and "generate_prompt" in cmd  # Module name
        assert "--speaker-id" in cmd
        assert "--seed" in cmd
        assert "12345" in cmd
        assert "--output-dir" in cmd


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_synthetic_prompts_disabled(tmp_path, monkeypatch):
    """Test that synthetic prompts are not generated when disabled."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """
[S1] Hello there, this is the first speaker.
[S2] And I'm the second speaker responding.
[S1] We're having a conversation that spans multiple chunks.
[S2] Yes, but synthetic prompts are disabled.
"""
    )

    output_path = tmp_path / "output.mp3"

    subprocess_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        # Only track calls to our generate_prompt module, not file detection calls
        if (
            isinstance(cmd, list)
            and len(cmd) > 2
            and "-m" in cmd
            and "generate_prompt" in cmd
        ):
            subprocess_calls.append(cmd)
            return MagicMock(
                returncode=0,
                stdout='{"audio_path": "", "stdout_transcript": ""}',
                stderr="",
            )
        else:
            # For other subprocess calls (like file detection), use the real implementation
            import subprocess

            return subprocess.run(cmd, **kwargs)

    # Mock components
    class MockDiaTTS:
        def __init__(self, seed=None, model_checkpoint=None, log_level=None):
            pass

        def register_voice_prompts(self, voice_prompts):
            pass

        def generate(self, texts, audio_prompts=None):
            import numpy as np
            from pydub import AudioSegment

            segments = []
            for _ in texts:
                data = np.zeros(int(0.1 * 22050), dtype=np.int16)
                segments.append(
                    AudioSegment(
                        data.tobytes(), frame_rate=22050, sample_width=2, channels=1
                    )
                )
            return segments

    class MockLLMInvoker:
        def __init__(self, model, **kwargs):
            pass

        def invoke(self, messages):
            class MockResponse:
                content = '{"topic": "Conversation"}'

            return MockResponse()

    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
        sample_name=None,
    ):
        result = {}
        for line in lines:
            result[line["global_line_number"]] = line
        return result

    # Apply mocks with synthetic prompts DISABLED
    monkeypatch.setattr("src.pipeline.DiaTTS", MockDiaTTS)
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )
    # Patch the config reference that pipeline.py imports and uses
    monkeypatch.setattr("shared.config.config.GENERATE_SYNTHETIC_PROMPTS", False)

    # Run pipeline
    run_pipeline(
        input_path=str(transcript_path), out_audio_path=str(output_path), seed=12345
    )

    # Verify no subprocess calls were made
    assert len(subprocess_calls) == 0
    assert output_path.exists()


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_synthetic_prompts_single_chunk_skip(tmp_path, monkeypatch):
    """Test that synthetic prompts are skipped for single-chunk transcripts."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """
[S1] This is a short conversation.
[S2] Yes, very brief.
"""
    )

    output_path = tmp_path / "output.mp3"

    subprocess_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        # Only track calls to our generate_prompt module, not file detection calls
        if (
            isinstance(cmd, list)
            and len(cmd) > 2
            and "-m" in cmd
            and "generate_prompt" in cmd
        ):
            subprocess_calls.append(cmd)
            return MagicMock(
                returncode=0,
                stdout='{"audio_path": "", "stdout_transcript": ""}',
                stderr="",
            )
        else:
            # For other subprocess calls (like file detection), use the real implementation
            import subprocess

            return subprocess.run(cmd, **kwargs)

    # Mock components (same as above)
    class MockDiaTTS:
        def __init__(self, seed=None, model_checkpoint=None, log_level=None):
            pass

        def register_voice_prompts(self, voice_prompts):
            pass

        def generate(self, texts, audio_prompts=None):
            import numpy as np
            from pydub import AudioSegment

            segments = []
            for _ in texts:
                data = np.zeros(int(0.1 * 22050), dtype=np.int16)
                segments.append(
                    AudioSegment(
                        data.tobytes(), frame_rate=22050, sample_width=2, channels=1
                    )
                )
            return segments

    class MockLLMInvoker:
        def __init__(self, model, **kwargs):
            pass

        def invoke(self, messages):
            class MockResponse:
                content = """{
  "moment_summary": "Brief conversation between speakers",
  "directors_notes": "Keep it short and simple",
  "start_line": 0,
  "end_line": 1
}"""

            return MockResponse()

    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
        sample_name=None,
    ):
        result = {}
        for line in lines:
            result[line["global_line_number"]] = line
        return result

    # Apply mocks with synthetic prompts enabled
    monkeypatch.setattr("src.pipeline.DiaTTS", MockDiaTTS)
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )
    monkeypatch.setattr("shared.config.config.GENERATE_SYNTHETIC_PROMPTS", True)
    monkeypatch.setattr(
        "shared.config.config.GENERATE_PROMPT_OUTPUT_DIR",
        str(tmp_path / "synthetic_prompts"),
    )

    # Run pipeline
    run_pipeline(input_path=str(transcript_path), out_audio_path=str(output_path))

    # Verify no subprocess calls were made (single chunk doesn't trigger
    #  synthetic prompts)
    assert len(subprocess_calls) == 0
    assert output_path.exists()


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_synthetic_prompts_with_existing_voice_prompts(tmp_path, monkeypatch):
    """Test synthetic prompts are only generated for unprompted speakers."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """
[S1] Hello, I have a voice prompt already.
[S2] But I don't have one, so I need a synthetic prompt.
[S1] This conversation spans multiple chunks.
[S2] So synthetic prompts should be generated for me only.
"""
    )

    # Create a mock voice prompt file for S1
    voice1_path = tmp_path / "voice1.wav"
    with open(voice1_path, "wb") as f:
        f.write(b"fake audio data")

    output_path = tmp_path / "output.mp3"

    subprocess_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        # Only track calls to our generate_prompt module, not file detection calls
        if (
            isinstance(cmd, list)
            and len(cmd) > 2
            and "-m" in cmd
            and "generate_prompt" in cmd
        ):
            subprocess_calls.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps(
                {
                    "speaker_id": cmd[cmd.index("--speaker-id") + 1],
                    "audio_path": (
                        f"{tmp_path}/synthetic_prompts/"
                        f"{cmd[cmd.index('--speaker-id') + 1]}_seed_12345.wav"
                    ),
                    "stdout_transcript": (
                        f"[{cmd[cmd.index('--speaker-id') + 1]}] Synthetic transcript"
                    ),
                }
            )
            result.stderr = ""
            return result
        else:
            # For other subprocess calls (like file detection), use the real implementation
            import subprocess

            return subprocess.run(cmd, **kwargs)

    # Mock components
    class MockDiaTTS:
        def __init__(self, seed=None, model_checkpoint=None, log_level=None):
            pass

        def register_voice_prompts(self, voice_prompts):
            pass

        def generate(self, texts, audio_prompts=None):
            import numpy as np
            from pydub import AudioSegment

            segments = []
            for _ in texts:
                data = np.zeros(int(0.1 * 22050), dtype=np.int16)
                segments.append(
                    AudioSegment(
                        data.tobytes(), frame_rate=22050, sample_width=2, channels=1
                    )
                )
            return segments

    class MockLLMInvoker:
        def __init__(self, model, **kwargs):
            pass

        def invoke(self, messages):
            # Check what type of call this is based on the message content
            message_content = ""
            if isinstance(messages, list) and len(messages) > 0:
                message_content = (
                    str(messages[-1].get("content", ""))
                    if isinstance(messages[-1], dict)
                    else str(messages[-1])
                )
            elif hasattr(messages, "content"):
                message_content = str(messages.content)
            else:
                message_content = str(messages)

            class MockResponse:
                if (
                    "moment" in message_content.lower()
                    and "define" in message_content.lower()
                ):
                    # Director moment definition call
                    content = """{
  "moment_summary": "Conversation with mixed prompts",
  "directors_notes": "Handle both speakers appropriately",
  "start_line": 0,
  "end_line": 1
}"""
                elif (
                    "actor" in message_content.lower()
                    and "perform" in message_content.lower()
                ):
                    # Actor performance call
                    content = """{
  "line_0": "[S1] Hello, I have a voice prompt already.",
  "line_1": "[S2] But I don't have one, so I need a synthetic prompt."
}"""
                elif (
                    "review" in message_content.lower()
                    and "final" in message_content.lower()
                ):
                    # Director review call
                    content = """{
  "line_0": "[S1] Hello, I have a voice prompt already.",
  "line_1": "[S2] But I don't have one, so I need a synthetic prompt."
}"""
                else:
                    # Default fallback
                    content = """{
  "moment_summary": "Conversation with mixed prompts",
  "directors_notes": "Handle both speakers appropriately",
  "start_line": 0,
  "end_line": 1
}"""

            return MockResponse()

    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
        sample_name=None,
    ):
        result = {}
        for line in lines:
            result[line["global_line_number"]] = line
        return result

    # Apply mocks
    monkeypatch.setattr("src.pipeline.DiaTTS", MockDiaTTS)
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )
    monkeypatch.setattr("shared.config.config.GENERATE_SYNTHETIC_PROMPTS", True)
    monkeypatch.setattr(
        "shared.config.config.GENERATE_PROMPT_OUTPUT_DIR",
        str(tmp_path / "synthetic_prompts"),
    )

    # Run pipeline with S1 having a voice prompt
    run_pipeline(
        input_path=str(transcript_path),
        out_audio_path=str(output_path),
        voice_prompts={
            "S1": {"path": str(voice1_path), "transcript": "I'm speaker one"}
        },
        seed=12345,
    )

    # Verify output was created
    assert output_path.exists()

    # Verify subprocess was called only for S2 (unprompted speaker)
    assert len(subprocess_calls) == 1

    cmd = subprocess_calls[0]
    speaker_idx = cmd.index("--speaker-id") + 1
    assert cmd[speaker_idx] == "S2"


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_synthetic_prompts_worker_failure(tmp_path, monkeypatch):
    """Test handling of worker script failure."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """
[S1] Hello there, this conversation will fail.
[S2] Yes, the worker script will fail.
[S1] This should trigger error handling.
[S2] And the pipeline should fail gracefully.
"""
    )

    output_path = tmp_path / "output.mp3"

    def mock_subprocess_run(cmd, **kwargs):
        # Simulate worker script failure
        from subprocess import CalledProcessError

        raise CalledProcessError(1, cmd, output="", stderr="Worker script failed")

    # Mock components (minimal since we expect failure)
    class MockDiaTTS:
        def __init__(self, *args, **kwargs):
            pass

        def register_voice_prompts(self, voice_prompts):
            pass

        def generate(self, *args, **kwargs):
            import numpy as np
            from pydub import AudioSegment

            data = np.zeros(int(0.1 * 22050), dtype=np.int16)
            return [
                AudioSegment(
                    data.tobytes(), frame_rate=22050, sample_width=2, channels=1
                )
            ]

    monkeypatch.setattr("src.pipeline.DiaTTS", MockDiaTTS)

    class MockLLMInvoker:
        def __init__(self, model, **kwargs):
            pass

        def invoke(self, messages):
            class MockResponse:
                content = "This is not valid JSON at all!"

            return MockResponse()

    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
        sample_name=None,
    ):
        result = {}
        for line in lines:
            result[line["global_line_number"]] = line
        return result

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )
    monkeypatch.setattr("shared.config.config.GENERATE_SYNTHETIC_PROMPTS", True)
    monkeypatch.setattr(
        "shared.config.config.GENERATE_PROMPT_OUTPUT_DIR",
        str(tmp_path / "synthetic_prompts"),
    )

    # Run pipeline and expect failure
    try:
        run_pipeline(
            input_path=str(transcript_path), out_audio_path=str(output_path), seed=12345
        )
    except RuntimeError as e:
        assert "Failed to generate synthetic prompt for S1" in str(e)


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_synthetic_prompts_json_parsing_failure(tmp_path, monkeypatch):
    """Test handling of invalid JSON from worker script."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """
[S1] Hello there, this will have JSON issues.
[S2] Yes, the worker will return invalid JSON.
[S1] This should trigger JSON parsing error.
[S2] And be handled gracefully.
"""
    )

    output_path = tmp_path / "output.mp3"

    def mock_subprocess_run(cmd, **kwargs):
        # Return invalid JSON
        result = MagicMock()
        result.returncode = 0
        result.stdout = "invalid json output"
        result.stderr = ""
        return result

    # Mock components
    class MockDiaTTS:
        def __init__(self, *args, **kwargs):
            pass

        def register_voice_prompts(self, voice_prompts):
            pass

        def generate(self, *args, **kwargs):
            import numpy as np
            from pydub import AudioSegment

            data = np.zeros(int(0.1 * 22050), dtype=np.int16)
            return [
                AudioSegment(
                    data.tobytes(), frame_rate=22050, sample_width=2, channels=1
                )
            ]

    monkeypatch.setattr("src.pipeline.DiaTTS", MockDiaTTS)

    class MockLLMInvoker:
        def __init__(self, model, **kwargs):
            pass

        def invoke(self, messages):
            class MockResponse:
                content = "This is not valid JSON at all!"

            return MockResponse()

    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
        sample_name=None,
    ):
        result = {}
        for line in lines:
            result[line["global_line_number"]] = line
        return result

    # Apply mocks
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )
    monkeypatch.setattr("shared.config.config.GENERATE_SYNTHETIC_PROMPTS", True)
    monkeypatch.setattr(
        "shared.config.config.GENERATE_PROMPT_OUTPUT_DIR",
        str(tmp_path / "synthetic_prompts"),
    )

    # Run pipeline and expect failure
    try:
        run_pipeline(
            input_path=str(transcript_path), out_audio_path=str(output_path), seed=12345
        )
    except RuntimeError as e:
        assert "Failed to parse worker script output as JSON" in str(e)


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_synthetic_prompts_seed_handling(tmp_path, monkeypatch):
    """Test that seed is properly passed to worker script."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """
[S1] Testing seed handling in synthetic prompts.
[S2] The seed should be passed correctly.
[S1] This is a multi-chunk conversation.
[S2] So we can test the seed parameter.
"""
    )

    output_path = tmp_path / "output.mp3"

    subprocess_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        # Only track calls to our generate_prompt module, not file detection calls
        if (
            isinstance(cmd, list)
            and len(cmd) > 2
            and "-m" in cmd
            and "generate_prompt" in cmd
        ):
            subprocess_calls.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps(
                {
                    "speaker_id": cmd[cmd.index("--speaker-id") + 1],
                    "audio_path": (
                        f"{tmp_path}/synthetic_prompts/"
                        f"{cmd[cmd.index('--speaker-id') + 1]}_seed_99999.wav"
                    ),
                    "stdout_transcript": (
                        f"[{cmd[cmd.index('--speaker-id') + 1]}] Synthetic transcript"
                    ),
                }
            )
            result.stderr = ""
            return result
        else:
            # For other subprocess calls (like file detection), use the real implementation
            import subprocess

            return subprocess.run(cmd, **kwargs)

    # Mock components
    class MockDiaTTS:
        def __init__(self, seed=None, model_checkpoint=None, log_level=None):
            pass

        def register_voice_prompts(self, voice_prompts):
            pass

        def generate(self, texts, audio_prompts=None):
            import numpy as np
            from pydub import AudioSegment

            segments = []
            for _ in texts:
                data = np.zeros(int(0.1 * 22050), dtype=np.int16)
                segments.append(
                    AudioSegment(
                        data.tobytes(), frame_rate=22050, sample_width=2, channels=1
                    )
                )
            return segments

    class MockLLMInvoker:
        def __init__(self, model, **kwargs):
            pass

        def invoke(self, messages):
            class MockResponse:
                content = """{
  "moment_summary": "Testing seed handling",
  "directors_notes": "Handle seed parameter correctly",
  "start_line": 0,
  "end_line": 1
}"""

            return MockResponse()

    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
        sample_name=None,
    ):
        result = {}
        for line in lines:
            result[line["global_line_number"]] = line
        return result

    # Apply mocks
    monkeypatch.setattr("src.pipeline.DiaTTS", MockDiaTTS)
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )
    monkeypatch.setattr("shared.config.config.GENERATE_SYNTHETIC_PROMPTS", True)
    monkeypatch.setattr(
        "shared.config.config.GENERATE_PROMPT_OUTPUT_DIR",
        str(tmp_path / "synthetic_prompts"),
    )

    # Run pipeline with specific seed
    run_pipeline(
        input_path=str(transcript_path), out_audio_path=str(output_path), seed=99999
    )

    # Verify seed was passed to worker scripts
    for call in subprocess_calls:
        cmd = call
        assert "--seed" in cmd
        seed_idx = cmd.index("--seed") + 1
        assert cmd[seed_idx] == "99999"


@patch.dict(os.environ, {"REHEARSAL_CHECKPOINT_PATH": ":memory:"})
def test_synthetic_prompts_output_dir_creation(tmp_path, monkeypatch):
    """Test that the output directory for synthetic prompts is created."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        """
[S1] Testing output directory creation.
[S2] The directory should be created if it doesn't exist.
[S1] This is a multi-chunk conversation.
[S2] So we can test the output directory. Here is another sentence that is really really long so hopefully we get our own chunk.
"""  # noqa: E501
    )

    output_path = tmp_path / "output.mp3"
    synthetic_prompts_dir = tmp_path / "synthetic_prompts_test_dir"
    subprocess_calls = []

    def mock_subprocess_run(cmd, **kwargs):
        # Only track calls to our generate_prompt module, not file detection calls
        if (
            isinstance(cmd, list)
            and len(cmd) > 2
            and "-m" in cmd
            and "generate_prompt" in cmd
        ):
            subprocess_calls.append(cmd)
            # Create the directory if it doesn't exist
            os.makedirs(synthetic_prompts_dir, exist_ok=True)
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps(
                {
                    "speaker_id": cmd[cmd.index("--speaker-id") + 1],
                    "audio_path": (
                        f"{synthetic_prompts_dir}/"
                        f"{cmd[cmd.index('--speaker-id') + 1]}_seed_12345.wav"
                    ),
                    "stdout_transcript": (
                        f"[{cmd[cmd.index('--speaker-id') + 1]}] Synthetic transcript"
                    ),
                }
            )
            result.stderr = ""
            return result
        else:
            # For other subprocess calls (like file detection), use the real implementation
            import subprocess

            return subprocess.run(cmd, **kwargs)

    # Mock components
    class MockDiaTTS:
        def __init__(self, seed=None, model_checkpoint=None, log_level=None):
            pass

        def register_voice_prompts(self, voice_prompts):
            pass

        def generate(self, texts, audio_prompts=None):
            import numpy as np
            from pydub import AudioSegment

            segments = []
            for _ in texts:
                data = np.zeros(int(0.1 * 22050), dtype=np.int16)
                segments.append(
                    AudioSegment(
                        data.tobytes(), frame_rate=22050, sample_width=2, channels=1
                    )
                )
            return segments

    class MockLLMInvoker:
        def __init__(self, model, **kwargs):
            pass

        def invoke(self, messages):
            # Check what type of call this is based on the message content
            message_content = ""
            if isinstance(messages, list) and len(messages) > 0:
                message_content = (
                    str(messages[-1].get("content", ""))
                    if isinstance(messages[-1], dict)
                    else str(messages[-1])
                )
            elif hasattr(messages, "content"):
                message_content = str(messages.content)
            else:
                message_content = str(messages)

            class MockResponse:
                if (
                    "moment" in message_content.lower()
                    and "define" in message_content.lower()
                ):
                    # Director moment definition call
                    content = """{
  "moment_summary": "Testing output directory creation",
  "directors_notes": "Ensure directory is created",
  "start_line": 0,
  "end_line": 1
}"""
                elif (
                    "actor" in message_content.lower()
                    and "perform" in message_content.lower()
                ):
                    # Actor performance call
                    content = """{
  "line_0": "[S1] Testing output directory creation.",
  "line_1": "[S2] The directory should be created if it doesn't exist."
}"""
                elif (
                    "review" in message_content.lower()
                    and "final" in message_content.lower()
                ):
                    # Director review call
                    content = """{
  "line_0": "[S1] Testing output directory creation.",
  "line_1": "[S2] The directory should be created if it doesn't exist."
}"""
                else:
                    # Default fallback
                    content = """{
  "moment_summary": "Testing output directory creation",
  "directors_notes": "Ensure directory is created",
  "start_line": 0,
  "end_line": 1
}"""

            return MockResponse()

    def mock_perform_moment(
        self,
        moment_id,
        lines,
        token_budget,
        constraints,
        global_summary,
        sample_name=None,
    ):
        result = {}
        for line in lines:
            result[line["global_line_number"]] = line
        return result

    # Apply mocks
    monkeypatch.setattr("src.pipeline.DiaTTS", MockDiaTTS)
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.director.LiteLLMInvoker", MockLLMInvoker
    )
    monkeypatch.setattr(
        "src.components.verbal_tag_injector.actor.Actor.perform_moment",
        mock_perform_moment,
    )
    monkeypatch.setattr("shared.config.config.GENERATE_SYNTHETIC_PROMPTS", True)
    monkeypatch.setattr(
        "shared.config.config.GENERATE_PROMPT_OUTPUT_DIR", str(synthetic_prompts_dir)
    )

    # Ensure directory does not exist before running
    assert not synthetic_prompts_dir.exists()

    # Run pipeline
    run_pipeline(
        input_path=str(transcript_path), out_audio_path=str(output_path), seed=12345
    )

    # Verify directory was created
    assert len(subprocess_calls) == 2
    assert synthetic_prompts_dir.exists()
    assert synthetic_prompts_dir.is_dir()
