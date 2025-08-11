"""Coverage tests for the audio generator TTS module."""

import pytest
from unittest.mock import patch, MagicMock

from src.components.audio_generator.tts import DiaTTS


def test_diatts_init():
    """Test DiaTTS initialization."""
    with patch('src.components.audio_generator.tts._get_default_device') as mock_device:
        mock_device.return_value = MagicMock(type='cpu')
        with patch('src.components.audio_generator.tts.Dia') as mock_dia:
            mock_dia.from_pretrained.return_value = MagicMock()

            # Test initialization with default parameters
            tts = DiaTTS(seed=12345)

            # Verify the mock was called
            mock_dia.from_pretrained.assert_called()
            assert tts is not None


def test_diatts_text_to_audio_file():
    """Test the text_to_audio_file method."""
    with patch('src.components.audio_generator.tts._get_default_device') as mock_device:
        mock_device.return_value = MagicMock(type='cpu')
        with patch('src.components.audio_generator.tts.Dia') as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model

            tts = DiaTTS(seed=12345)

            # Test text_to_audio_file method
            with patch('builtins.print'):  # Suppress print statements
                result = tts.text_to_audio_file("Hello world", "/tmp/test.wav")

            # Verify the methods were called
            mock_model.generate.assert_called()
            mock_model.save_audio.assert_called()
            assert result == "/tmp/test.wav"


def test_diatts_register_voice_prompts():
    """Test the register_voice_prompts method."""
    with patch('src.components.audio_generator.tts._get_default_device') as mock_device:
        mock_device.return_value = MagicMock(type='cpu')
        with patch('src.components.audio_generator.tts.Dia') as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.generate_speaker_embedding.return_value = ["embedding1", "embedding2"]

            tts = DiaTTS(seed=12345)

            # Test register_voice_prompts method
            with patch('builtins.print'):  # Suppress print statements
                tts.register_voice_prompts({"S1": "path1", "S2": "path2"})

            # Verify the method was called
            mock_model.generate_speaker_embedding.assert_called()
            assert hasattr(tts, '_speaker_embeddings')


def test_diatts_set_seed():
    """Test the _set_seed method."""
    with patch('src.components.audio_generator.tts._get_default_device') as mock_device:
        mock_device.return_value = MagicMock(type='cpu')
        with patch('src.components.audio_generator.tts.Dia') as mock_dia:
            mock_dia.from_pretrained.return_value = MagicMock()

            tts = DiaTTS(seed=12345)

            # Test _set_seed method directly
            with patch('random.seed') as mock_random, \
                 patch('numpy.random.seed') as mock_numpy, \
                 patch('torch.manual_seed') as mock_torch, \
                 patch('torch.cuda.manual_seed') as mock_cuda, \
                 patch('torch.cuda.manual_seed_all') as mock_cuda_all:

                tts._set_seed(54321)

                # Verify the methods were called
                mock_random.assert_called_with(54321)
                mock_numpy.assert_called_with(54321)
                mock_torch.assert_called_with(54321)
