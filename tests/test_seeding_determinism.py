"""Tests for seeding and determinism functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.components.audio_generator.tts import DiaTTS


def test_voice_seed_parameter_passed_to_model():
    """Test that voice_seed parameter is correctly passed to model.generate."""
    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model

            # Mock the model.generate to return audio as numpy array
            mock_model.generate.return_value = np.array([0.1, 0.2], dtype=np.float32)

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {}
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1
                mock_config.FULLY_DETERMINISTIC = False

                # Test that voice_seed is passed when seed is provided
                tts = DiaTTS(seed=54321)
                tts.generate(texts=["[S1] Hello world"], audio_prompts=None)

                # Verify voice_seed parameter was passed to model.generate
                call_args, call_kwargs = mock_model.generate.call_args
                assert call_kwargs.get("voice_seed") == 54321


def test_fully_deterministic_flag_triggers_set_seed():
    """Test that FULLY_DETERMINISTIC flag triggers _set_seed method."""
    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model

            # Mock the model.generate to return audio as numpy array
            mock_model.generate.return_value = np.array([0.1, 0.2], dtype=np.float32)

            with patch("src.components.audio_generator.tts.config") as mock_config:
                mock_config.DIA_GENERATE_PARAMS = {}
                mock_config.AUDIO_SAMPLING_RATE = 22050
                mock_config.AUDIO_SAMPLE_WIDTH = 2
                mock_config.AUDIO_CHANNELS = 1
                mock_config.FULLY_DETERMINISTIC = True

                with patch.object(DiaTTS, "_set_seed") as mock_set_seed:
                    tts = DiaTTS(seed=12345)
                    tts.generate(texts=["[S1] Hello world"], audio_prompts=None)

                    # Verify _set_seed was called with the correct seed
                    mock_set_seed.assert_called_once_with(12345)

                    # Also verify voice_seed was passed to model.generate
                    call_args, call_kwargs = mock_model.generate.call_args
                    assert call_kwargs.get("voice_seed") == 12345


def test_set_seed_calls_torch_manual_seed():
    """Test that _set_seed correctly calls torch.manual_seed and related functions."""
    with patch("src.components.audio_generator.tts._get_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_dia.from_pretrained.return_value = MagicMock()

            with patch("src.components.audio_generator.tts.random") as mock_random:
                with patch("src.components.audio_generator.tts.np") as mock_np:
                    with patch(
                        "src.components.audio_generator.tts.torch"
                    ) as mock_torch:
                        with patch("src.components.audio_generator.tts.os") as mock_os:
                            mock_os.getenv.return_value = None

                            tts = DiaTTS(seed=99999)
                            tts.device = "cpu"
                            tts._set_seed(99999)

                            # Verify all the seed functions were called
                            mock_random.seed.assert_called_once_with(99999)
                            mock_np.random.seed.assert_called_once_with(99999)
                            mock_torch.manual_seed.assert_called_once_with(99999)
                            mock_torch.use_deterministic_algorithms.assert_called_once_with(
                                True
                            )
