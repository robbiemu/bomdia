"""Coverage tests for the audio generator TTS module."""

from unittest.mock import MagicMock, patch

from src.components.audio_generator.tts import DiaTTS


def test_diatts_init():
    """Test DiaTTS initialization."""
    with patch("src.components.audio_generator.tts._get_default_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_dia.from_pretrained.return_value = MagicMock()

            # Test initialization with default parameters
            tts = DiaTTS(seed=12345)

            # Verify the mock was called
            mock_dia.from_pretrained.assert_called()
            assert tts is not None


def test_diatts_text_to_audio_file():
    """Test the text_to_audio_file method."""
    with patch("src.components.audio_generator.tts._get_default_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model

            tts = DiaTTS(seed=12345)

            # Test text_to_audio_file method
            with patch("builtins.print"):  # Suppress print statements
                result = tts.text_to_audio_file(
                    "Hello world", "/tmp/test.wav"
                )  # nosec B108

            # Verify the methods were called
            mock_model.generate.assert_called()
            mock_model.save_audio.assert_called()
            assert result == "/tmp/test.wav"  # nosec B108


def test_diatts_register_voice_prompts():
    """Test the register_voice_prompts method."""
    with patch("src.components.audio_generator.tts._get_default_device") as mock_device:
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
    with patch("src.components.audio_generator.tts._get_default_device") as mock_device:
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


def test_diatts_text_to_audio_file_hifi():
    """Test the text_to_audio_file method for high-fidelity cloning."""
    import logging
    # Set logging level to INFO to trigger verbose=True
    logging.getLogger().setLevel(logging.INFO)

    with patch("src.components.audio_generator.tts._get_default_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model

            tts = DiaTTS(seed=12345, log_level=logging.INFO)
            tts.register_voice_prompts({"S1": {"path": "path1", "transcript": "transcript1"}})

            # Test text_to_audio_file method
            with patch("builtins.print"):  # Suppress print statements
                tts.text_to_audio_file("[S1] Hello world", "/tmp/test.wav")  # nosec B108

            # Verify the methods were called with all expected parameters
            call_args, call_kwargs = mock_model.generate.call_args
            assert call_kwargs['text'] == '[S1]transcript1[S1] [S1] Hello world'
            assert call_kwargs['audio_prompt'] == ['path1']
            assert call_kwargs['use_torch_compile'] == False
            assert call_kwargs['verbose'] == True
            assert call_kwargs['max_tokens'] == 3072
            assert call_kwargs['cfg_scale'] == 3.0
            assert call_kwargs['temperature'] == 1.2
            assert call_kwargs['top_p'] == 0.95
            assert call_kwargs['cfg_filter_top_k'] == 45
            assert call_kwargs['use_cfg_filter'] == False


def test_diatts_text_to_audio_file_mixed_modes():
    """Test the text_to_audio_file method for mixed cloning modes."""
    import logging
    # Set logging level to INFO to trigger verbose=True
    logging.getLogger().setLevel(logging.INFO)

    with patch("src.components.audio_generator.tts._get_default_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model
            mock_model.generate_speaker_embedding.return_value = ["embedding2"]

            tts = DiaTTS(seed=12345, log_level=logging.INFO)
            tts.register_voice_prompts({
                "S1": {"path": "path1", "transcript": "transcript1"},
                "S2": {"path": "path2"}
            })

            with patch("builtins.print"):  # Suppress print statements
                tts.text_to_audio_file("[S1] Hello [S2] world", "/tmp/test.wav")  # nosec B108

            # Verify generate was called with correct arguments
            call_args, call_kwargs = mock_model.generate.call_args
            assert call_kwargs['text'] == '[S1]transcript1[S1] [S1] Hello [S2] world'
            assert call_kwargs['use_torch_compile'] == False
            assert call_kwargs['verbose'] == True
            assert call_kwargs['max_tokens'] == 3072
            assert call_kwargs['cfg_scale'] == 3.0
            assert call_kwargs['temperature'] == 1.2
            assert call_kwargs['top_p'] == 0.95
            assert call_kwargs['cfg_filter_top_k'] == 45
            assert call_kwargs['use_cfg_filter'] == False

def test_diatts_text_to_audio_file_pure_tts():
    """Test the text_to_audio_file method for pure TTS."""
    with patch("src.components.audio_generator.tts._get_default_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_model = MagicMock()
            mock_dia.from_pretrained.return_value = mock_model

            tts = DiaTTS(seed=12345)

            with patch.object(tts, '_set_seed') as mock_set_seed:
                # Test text_to_audio_file method
                with patch("builtins.print"):  # Suppress print statements
                    tts.text_to_audio_file("[S1] Hello world", "/tmp/test.wav")  # nosec B108

                # Verify the methods were called
                mock_set_seed.assert_called_with(12345)


def test_diatts_set_seed():
    """Test the _set_seed method."""
    with patch("src.components.audio_generator.tts._get_default_device") as mock_device:
        mock_device.return_value = MagicMock(type="cpu")
        with patch("src.components.audio_generator.tts.Dia") as mock_dia:
            mock_dia.from_pretrained.return_value = MagicMock()

            tts = DiaTTS(seed=12345)

            # Test _set_seed method directly
            with (
                patch("random.seed") as mock_random,
                patch("numpy.random.seed") as mock_numpy,
                patch("torch.manual_seed") as mock_torch,
                patch("torch.cuda.manual_seed") as mock_cuda,
                patch("torch.cuda.manual_seed_all") as mock_cuda_all,
            ):
                tts._set_seed(54321)

                # Verify the methods were called
                mock_random.assert_called_with(54321)
                mock_numpy.assert_called_with(54321)
                mock_torch.assert_called_with(54321)
