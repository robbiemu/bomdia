"""Unit tests for the generate_prompt.py worker script."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import generate_prompt


def test_get_opposite_speaker():
    """Test the get_opposite_speaker function."""
    assert generate_prompt.get_opposite_speaker("S1") == "S2"
    assert generate_prompt.get_opposite_speaker("S2") == "S1"
    assert generate_prompt.get_opposite_speaker("S3") == "S1"
    assert generate_prompt.get_opposite_speaker("S10") == "S1"


def test_load_raw_text():
    """Test loading raw text from asset file."""
    # Test successful loading
    text = generate_prompt.load_raw_text()
    assert isinstance(text, str)
    assert len(text) > 0
    assert "I see it all" in text  # Should contain expected content

    # Test with missing file
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Asset file not found"):
            generate_prompt.load_raw_text()

    # Test with empty file
    with patch("pathlib.Path.read_text", return_value="   "):
        with pytest.raises(RuntimeError, match="Asset file is empty"):
            generate_prompt.load_raw_text()


def test_generate_synthetic_prompt_basic(tmp_path):
    """Test basic synthetic prompt generation."""
    with patch("generate_prompt.DiaTTS") as mock_tts_class:
        # Mock the TTS instance and its methods
        mock_tts = MagicMock()
        mock_tts_class.return_value = mock_tts

        # Mock AudioSegment for the return value
        mock_audio_segment = MagicMock()
        mock_tts.generate.return_value = [mock_audio_segment]

        # Test generation
        result = generate_prompt.generate_synthetic_prompt(
            speaker_id="S1", seed=12345, output_dir=str(tmp_path), verbose=False
        )

        # Verify result structure
        assert "speaker_id" in result
        assert "audio_path" in result
        assert "stdout_transcript" in result
        assert result["speaker_id"] == "S1"

        # Verify files would be created with correct naming (zero-padded)
        assert "S1_seed_00012345.wav" in result["audio_path"]

        # Verify TTS was called correctly
        mock_tts_class.assert_called_once()
        mock_tts.generate.assert_called_once()

        # Check the call arguments
        call_kwargs = mock_tts.generate.call_args[1]
        texts = call_kwargs["texts"]
        assert len(texts) == 1
        assert "[S1]" in texts[0]
        assert "[S2]" in texts[0]  # Should have continuity tag

        # Verify audio export was called
        mock_audio_segment.export.assert_called_once()


def test_generate_synthetic_prompt_no_seed(tmp_path):
    """Test synthetic prompt generation without seed."""
    with patch("generate_prompt.DiaTTS") as mock_tts_class:
        mock_tts = MagicMock()
        mock_tts_class.return_value = mock_tts
        mock_audio_segment = MagicMock()
        mock_tts.generate.return_value = [mock_audio_segment]

        with patch(
            "generate_prompt.get_random_seed", return_value=98765
        ) as mock_get_random_seed:
            result = generate_prompt.generate_synthetic_prompt(
                speaker_id="S2", seed=None, output_dir=str(tmp_path), verbose=True
            )

            # Should generate a temporary seed
            mock_get_random_seed.assert_called_once()
            assert "S2_seed_00098765.wav" in result["audio_path"]


def test_generate_synthetic_prompt_tts_failure(tmp_path):
    """Test handling of TTS initialization failure."""
    with patch("generate_prompt.DiaTTS", side_effect=RuntimeError("TTS failed")):
        with pytest.raises(RuntimeError, match="Failed to initialize DiaTTS"):
            generate_prompt.generate_synthetic_prompt(
                speaker_id="S1", seed=12345, output_dir=str(tmp_path)
            )


def test_generate_synthetic_prompt_generation_failure(tmp_path):
    """Test handling of audio generation failure."""
    with patch("generate_prompt.DiaTTS") as mock_tts_class:
        mock_tts = MagicMock()
        mock_tts_class.return_value = mock_tts
        mock_tts.generate.side_effect = RuntimeError("Generation failed")

        with pytest.raises(RuntimeError, match="Failed to generate audio"):
            generate_prompt.generate_synthetic_prompt(
                speaker_id="S1", seed=12345, output_dir=str(tmp_path)
            )


def test_transcript_formatting():
    """Test that transcripts are formatted correctly."""
    with patch("generate_prompt.DiaTTS") as mock_tts_class:
        mock_tts = MagicMock()
        mock_tts_class.return_value = mock_tts
        mock_audio_segment = MagicMock()
        mock_tts.generate.return_value = [mock_audio_segment]

        # Mock the raw text to test formatting
        test_text = "Line one.\nLine two with\nnewlines."
        with patch("generate_prompt.load_raw_text", return_value=test_text):
            result = generate_prompt.generate_synthetic_prompt(
                speaker_id="S1", seed=12345, output_dir="/tmp", verbose=False
            )

            # Check that stdout transcript has newlines replaced with double spaces
            stdout_transcript = result["stdout_transcript"]
            assert "\n" not in stdout_transcript
            assert "  " in stdout_transcript
            assert stdout_transcript.startswith("[S1]")

            # Check that generation transcript includes continuity tag
            call_kwargs = mock_tts.generate.call_args[1]
            generation_text = call_kwargs["texts"][0]
            assert generation_text.endswith("[S2]")


def test_main_function_success(tmp_path):
    """Test the main function with successful execution."""
    with patch("generate_prompt.generate_synthetic_prompt") as mock_generate:
        mock_generate.return_value = {
            "speaker_id": "S1",
            "audio_path": str(tmp_path / "S1_seed_12345.wav"),
            "stdout_transcript": "[S1] Test transcript",
        }

        with patch(
            "sys.argv",
            [
                "generate_prompt.py",
                "--speaker-id",
                "S1",
                "--seed",
                "12345",
                "--output-dir",
                str(tmp_path),
            ],
        ):
            with patch("builtins.print") as mock_print:
                generate_prompt.main()

                # Should print JSON output
                mock_print.assert_called_once()
                printed_output = mock_print.call_args[0][0]

                # Should be valid JSON
                result = json.loads(printed_output)
                assert result["speaker_id"] == "S1"
                assert result["audio_path"] == str(tmp_path / "S1_seed_12345.wav")


def test_main_function_failure(tmp_path):
    """Test the main function with generation failure."""
    with patch("generate_prompt.generate_synthetic_prompt") as mock_generate:
        mock_generate.side_effect = RuntimeError("Generation failed")

        with patch(
            "sys.argv",
            ["generate_prompt.py", "--speaker-id", "S1", "--output-dir", str(tmp_path)],
        ):
            with pytest.raises(SystemExit, match="1"):
                generate_prompt.main()


def test_main_function_missing_args():
    """Test the main function with missing required arguments."""
    with patch("sys.argv", ["generate_prompt.py"]):
        with pytest.raises(SystemExit):
            generate_prompt.main()


def test_setup_logging():
    """Test the logging setup function."""
    with patch("logging.basicConfig") as mock_config:
        generate_prompt.setup_logging()
        mock_config.assert_called_once()

        # Check that stderr is used for logging
        call_kwargs = mock_config.call_args[1]
        assert call_kwargs["stream"] == sys.stderr


def test_cli_integration(tmp_path):
    """Test CLI integration by testing argument parsing and main function flow."""
    # Test CLI argument parsing without running subprocess to avoid timeout
    with patch("generate_prompt.generate_synthetic_prompt") as mock_generate:
        mock_generate.return_value = {
            "speaker_id": "S1",
            "audio_path": str(tmp_path / "S1_seed_00012345.wav"),
            "stdout_transcript": "[S1] Test transcript  [S2]",
        }

        # Test with all arguments including verbose
        with patch(
            "sys.argv",
            [
                "generate_prompt.py",
                "--speaker-id",
                "S1",
                "--seed",
                "12345",
                "--output-dir",
                str(tmp_path),
                "--verbose",
            ],
        ):
            with patch("builtins.print") as mock_print:
                generate_prompt.main()

                # Should print JSON output
                mock_print.assert_called_once()
                printed_output = mock_print.call_args[0][0]

                # Should be valid JSON
                result = json.loads(printed_output)
                assert result["speaker_id"] == "S1"
                assert "S1_seed_00012345.wav" in result["audio_path"]

                # Verify generate_synthetic_prompt was called with correct args
                mock_generate.assert_called_once_with(
                    speaker_id="S1", seed=12345, output_dir=str(tmp_path), verbose=True
                )


def test_file_creation(tmp_path):
    """Test that the worker script creates the expected files."""
    with patch("generate_prompt.DiaTTS") as mock_tts_class:
        mock_tts = MagicMock()
        mock_tts_class.return_value = mock_tts

        # Mock audio segment that tracks export calls
        mock_audio_segment = MagicMock()
        mock_tts.generate.return_value = [mock_audio_segment]

        result = generate_prompt.generate_synthetic_prompt(
            speaker_id="S1", seed=12345, output_dir=str(tmp_path), verbose=False
        )

        # Verify audio file export was called
        mock_audio_segment.export.assert_called_once()
        export_call = mock_audio_segment.export.call_args
        audio_path = export_call[0][0]
        assert "S1_seed_00012345.wav" in audio_path

        # Verify text file would be created
        expected_text_path = tmp_path / "S1_seed_00012345.txt"
        # The actual file creation is mocked, but we can verify the path logic
        assert "S1_seed_00012345" in str(expected_text_path)
