"""Coverage tests for the audio generator TTS module."""

import numpy as np
from unittest.mock import MagicMock, patch

from src.components.audio_generator.tts import DiaTTS
from pydub import AudioSegment


def test_diatts_init():
    """Test DiaTTS initialization."""
    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_dia.from_pretrained.return_value = MagicMock()

            # Test initialization with default parameters
            tts = DiaTTS(seed=12345)

            # Verify the mock was called
            mock_dia.from_pretrained.assert_called()
            assert tts is not None


def test_diatts_generate_single():
    """Test the generate method for a single chunk."""
    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.generate.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {}
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1

                tts = DiaTTS(seed=12345)

                # Test generate method
                with patch("builtins.print"):  # Suppress print statements
                    result = tts.generate(
                        ["Hello world"], None, None
                    )

                # Verify the methods were called
                mock_model.generate.assert_called()
                assert isinstance(result[0], AudioSegment)


def test_diatts_register_voice_prompts():
    """Test the register_voice_prompts method."""
    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.generate_speaker_embedding.return_value = [
                "embedding1",
                "embedding2",
            ]

            tts = DiaTTS(seed=12345)

            # Test register_voice_prompts method
            with patch("builtins.print"):  # Suppress print statements
                tts.register_voice_prompts({"S1": {"path": "path1", "transcript": None}, "S2": {"path": "path2", "transcript": None}})

            # Verify the method was called
            mock_model.generate_speaker_embedding.assert_called()
            assert hasattr(tts, "_voice_prompt_details")


def test_diatts_register_voice_prompts_hifi():
    """Test the register_voice_prompts method for high-fidelity cloning."""
    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model

            tts = DiaTTS(seed=12345)

            # Test register_voice_prompts method
            with patch("builtins.print"):  # Suppress print statements
                tts.register_voice_prompts({"S1": {"path": "path1", "transcript": "transcript1"}})

            # Verify the method was called
            mock_model.generate_speaker_embedding.assert_not_called()
            assert "S1" in tts._voice_prompt_details
            assert tts._voice_prompt_details["S1"]["transcript"] == "transcript1"


def test_diatts_generate_hifi():
    """Test the generate method for high-fidelity cloning."""
    import logging
    # Set logging level to INFO to trigger verbose=True
    logging.getLogger().setLevel(logging.INFO)

    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.generate.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {
                    "max_tokens": 3072,
                    "cfg_scale": 3.0,
                    "temperature": 1.2,
                    "top_p": 0.95,
                    "cfg_filter_top_k": 45,
                    "use_cfg_filter": False,
                }
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1

                tts = DiaTTS(seed=12345, log_level=logging.INFO)
                tts.register_voice_prompts({"S1": {"path": "path1", "transcript": "transcript1"}})

                # Test generate method
                with patch("builtins.print"):  # Suppress print statements
                    tts.generate(["[S1] Hello world"], "path1", "[S1]transcript1[S1]")

                # Verify the methods were called with all expected parameters
                call_args, call_kwargs = mock_model.generate.call_args
                assert call_kwargs['text'] == '[S1]transcript1[S1] [S1] Hello world'
                assert call_kwargs['audio_prompt'] == ['path1']
                assert call_kwargs['verbose'] == True
                assert call_kwargs['max_tokens'] == 3072
                assert call_kwargs['cfg_scale'] == 3.0
                assert call_kwargs['temperature'] == 1.2
                assert call_kwargs['top_p'] == 0.95
                assert call_kwargs['cfg_filter_top_k'] == 45
                assert call_kwargs['use_cfg_filter'] == False


def test_diatts_generate_mixed_modes():
    """Test the generate method for mixed cloning modes."""
    import logging
    # Set logging level to INFO to trigger verbose=True
    logging.getLogger().setLevel(logging.INFO)

    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.generate_speaker_embedding.return_value = ["embedding2"]
            mock_model.generate.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {
                    "max_tokens": 3072,
                    "cfg_scale": 3.0,
                    "temperature": 1.2,
                    "top_p": 0.95,
                    "cfg_filter_top_k": 45,
                    "use_cfg_filter": False,
                }
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1

                tts = DiaTTS(seed=12345, log_level=logging.INFO)
                tts.register_voice_prompts({
                    "S1": {"path": "path1", "transcript": "transcript1"},
                    "S2": {"path": "path2"}
                })

                with patch("builtins.print"):  # Suppress print statements
                    tts.generate(["[S1] Hello [S2] world"], "path1", "[S1]transcript1[S1]")

                # Verify generate was called with correct arguments
                call_args, call_kwargs = mock_model.generate.call_args
                assert call_kwargs['text'] == '[S1]transcript1[S1] [S1] Hello [S2] world'
                assert call_kwargs['verbose'] == True
                assert call_kwargs['max_tokens'] == 3072
                assert call_kwargs['cfg_scale'] == 3.0
                assert call_kwargs['temperature'] == 1.2
                assert call_kwargs['top_p'] == 0.95
                assert call_kwargs['cfg_filter_top_k'] == 45
                assert call_kwargs['use_cfg_filter'] == False

def test_diatts_generate_pure_tts():
    """Test the generate method for pure TTS."""
    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.generate.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {}
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1

                tts = DiaTTS(seed=12345)

                # Test generate method
                with patch("builtins.print"):  # Suppress print statements
                    result = tts.generate(["[S1] Hello world"], None, None)

                # Verify voice_seed is passed to model.generate
                call_args, call_kwargs = mock_model.generate.call_args
                assert call_kwargs.get("voice_seed") == 12345

                # Verify result is returned correctly
                assert len(result) == 1


def test_diatts_voice_seed_parameter():
    """Test that voice_seed parameter is passed correctly to model.generate."""
    import numpy as np

    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model

            # Mock the model.generate to return audio
            mock_model.generate.return_value = np.array([0.1, 0.2], dtype=np.float32)

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {}
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1

                # Test that voice_seed is passed when seed is provided
                tts = DiaTTS(seed=54321)
                tts.generate(["[S1] Hello world"], None, None)

                # Verify voice_seed parameter was passed to model.generate
                call_args, call_kwargs = mock_model.generate.call_args
                assert call_kwargs.get("voice_seed") == 54321


def test_diatts_generate_batch():
    """Test the generate method for batch processing."""
    import numpy as np

    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.sampling_rate = 22050

            # Mock batch audio outputs (list of numpy arrays)
            mock_audio_outputs = [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.4, 0.5, 0.6], dtype=np.float32),
            ]
            mock_model.generate.return_value = mock_audio_outputs

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {
                    "max_tokens": 3072,
                    "cfg_scale": 3.0,
                }
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1

                tts = DiaTTS(seed=12345)

                # Test generate method
                texts = ["[S1] Hello world", "[S2] How are you?"]
                unified_audio_prompt = "/path/to/unified.wav"
                unified_transcript_prompt = "[S1]Sample transcript[S1] [S2]Another sample[S2]"

                with patch("builtins.print"):  # Suppress print statements
                    audio_segments = tts.generate(
                        texts, unified_audio_prompt, unified_transcript_prompt
                    )

                # Verify the model was called with correct arguments
                call_args, call_kwargs = mock_model.generate.call_args

                # Check batch text payloads include unified transcript prompt
                expected_text_payloads = [
                    f"{unified_transcript_prompt} [S1] Hello world",
                    f"{unified_transcript_prompt} [S2] How are you?",
                ]
                assert call_kwargs["text"] == expected_text_payloads

                # Check batch audio prompts
                expected_audio_prompts = [unified_audio_prompt, unified_audio_prompt]
                assert call_kwargs["audio_prompt"] == expected_audio_prompts

                # Check other parameters
                assert call_kwargs['max_tokens'] == 3072
                assert call_kwargs['cfg_scale'] == 3.0

                # Verify returned audio segments
                assert len(audio_segments) == 2
                # Each segment should be a pydub AudioSegment
                for segment in audio_segments:
                    assert hasattr(segment, 'frame_rate')
                    assert hasattr(segment, 'channels')
                    assert hasattr(segment, 'sample_width')


def test_diatts_generate_no_prompts():
    """Test generate method with no unified prompts."""
    import numpy as np

    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.sampling_rate = 22050

            # Mock batch audio outputs
            mock_audio_outputs = np.array([0.1, 0.2], dtype=np.float32)
            mock_model.generate.return_value = mock_audio_outputs

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {}
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1

                tts = DiaTTS(seed=12345)

                # Test generate method without prompts
                texts = ["[S1] Hello world"]

                with patch("builtins.print"):  # Suppress print statements
                    audio_segments = tts.generate(texts, None, None)

                # Verify the model was called with correct arguments
                call_args, call_kwargs = mock_model.generate.call_args

                # Check text payloads are unmodified
                assert call_kwargs["text"] == texts[0]

                # Check no audio prompt is passed
                assert "audio_prompt" not in call_kwargs

                # Verify returned audio segments
                assert len(audio_segments) == 1


def test_diatts_generate_voice_seed_consistency():
    """Test generate method passes voice_seed consistently."""
    import numpy as np

    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.sampling_rate = 22050

            # Mock batch audio outputs
            mock_audio_outputs = np.array([0.1, 0.2], dtype=np.float32)
            mock_model.generate.return_value = mock_audio_outputs

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {}
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1

                tts = DiaTTS(seed=12345)

                # Test with text that requires voice_seed parameter
                texts = ["[S3] Hello world"]

                with patch("builtins.print"):  # Suppress print statements
                    tts.generate(texts, None, None)

                # Verify voice_seed was passed to model.generate
                call_args, call_kwargs = mock_model.generate.call_args
                assert call_kwargs.get("voice_seed") == 12345


def test_diatts_generate_empty_input():
    """Test generate method with empty input list."""
    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model

            tts = DiaTTS(seed=12345)

            # Test with empty list of texts
            audio_segments = tts.generate([], None, None)

            # Verify no call was made to the model
            mock_model.generate.assert_not_called()

            # Verify an empty list is returned
            assert audio_segments == []
