"""Tests for device selection logic in the TTS component."""

import os
import unittest
from unittest.mock import patch

from shared.config import Config
from src.components.audio_generator.tts import _get_device


class TestDeviceSelection(unittest.TestCase):
    """Test cases for device selection logic."""

    def setUp(self):
        """Set up test environment."""
        # Clear any existing environment variable
        if "BOMDIA_DEVICE" in os.environ:
            del os.environ["BOMDIA_DEVICE"]

    def tearDown(self):
        """Clean up test environment."""
        # Clear any existing environment variable
        if "BOMDIA_DEVICE" in os.environ:
            del os.environ["BOMDIA_DEVICE"]

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_auto_detection_cuda_first(self, mock_mps_available, mock_cuda_available):
        """Test auto-detection prioritizes CUDA when available."""
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = True

        # Create a config instance with device="auto"
        config = Config.__new__(Config)
        config.DIA_DEVICE = "auto"

        with patch("src.components.audio_generator.tts.config", config):
            device = _get_device()
            self.assertEqual(device.type, "cuda")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_auto_detection_mps_second(self, mock_mps_available, mock_cuda_available):
        """Test auto-detection falls back to MPS when CUDA is not available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True

        # Create a config instance with device="auto"
        config = Config.__new__(Config)
        config.DIA_DEVICE = "auto"

        with patch("src.components.audio_generator.tts.config", config):
            device = _get_device()
            self.assertEqual(device.type, "mps")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_auto_detection_cpu_fallback(self, mock_mps_available, mock_cuda_available):
        """Test auto-detection falls back to CPU when no accelerator is available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False

        # Create a config instance with device="auto"
        config = Config.__new__(Config)
        config.DIA_DEVICE = "auto"

        with patch("src.components.audio_generator.tts.config", config):
            device = _get_device()
            self.assertEqual(device.type, "cpu")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_config_override_cuda(self, mock_mps_available, mock_cuda_available):
        """Test config override with CUDA device."""
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = True

        # Create a config instance with device="cuda"
        config = Config.__new__(Config)
        config.DIA_DEVICE = "cuda"

        with patch("src.components.audio_generator.tts.config", config):
            device = _get_device()
            self.assertEqual(device.type, "cuda")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_config_override_cuda_unavailable(
        self, mock_mps_available, mock_cuda_available
    ):
        """
        Test config override with CUDA device when not available falls back to auto.
        """
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False  # Also disable MPS to force CPU

        # Capture logs to verify warning is issued
        with self.assertLogs(
            "src.components.audio_generator.tts", level="WARNING"
        ) as log:
            # Create a config instance with device="cuda"
            config = Config.__new__(Config)
            config.DIA_DEVICE = "cuda"

            with patch("src.components.audio_generator.tts.config", config):
                device = _get_device()
                # Should fall back to CPU (the only available device in our mock)
                self.assertEqual(device.type, "cpu")
                # Check that warning was logged
                self.assertIn("CUDA specified but not available", log.output[0])

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_config_override_mps(self, mock_mps_available, mock_cuda_available):
        """Test config override with MPS device."""
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = True

        # Create a config instance with device="mps"
        config = Config.__new__(Config)
        config.DIA_DEVICE = "mps"

        with patch("src.components.audio_generator.tts.config", config):
            device = _get_device()
            self.assertEqual(device.type, "mps")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_config_override_mps_unavailable(
        self, mock_mps_available, mock_cuda_available
    ):
        """
        Test config override with MPS device when not available falls back to auto.
        """
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False  # Also disable CUDA to force CPU

        # Capture logs to verify warning is issued
        with self.assertLogs(
            "src.components.audio_generator.tts", level="WARNING"
        ) as log:
            # Create a config instance with device="mps"
            config = Config.__new__(Config)
            config.DIA_DEVICE = "mps"

            with patch("src.components.audio_generator.tts.config", config):
                device = _get_device()
                # Should fall back to CPU (the only available device in our mock)
                self.assertEqual(device.type, "cpu")
                # Check that warning was logged
                self.assertIn("MPS specified but not available", log.output[0])

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_config_override_cpu(self, mock_mps_available, mock_cuda_available):
        """Test config override with CPU device."""
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = True

        # Create a config instance with device="cpu"
        config = Config.__new__(Config)
        config.DIA_DEVICE = "cpu"

        with patch("src.components.audio_generator.tts.config", config):
            device = _get_device()
            self.assertEqual(device.type, "cpu")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_invalid_device_fallback(self, mock_mps_available, mock_cuda_available):
        """Test invalid device string falls back to auto-detection."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False  # Force CPU fallback

        # Capture logs to verify warning is issued
        with self.assertLogs(
            "src.components.audio_generator.tts", level="WARNING"
        ) as log:
            # Create a config instance with an invalid device
            config = Config.__new__(Config)
            config.DIA_DEVICE = "invalid"

            with patch("src.components.audio_generator.tts.config", config):
                device = _get_device()
                # Should fall back to CPU
                self.assertEqual(device.type, "cpu")
                # Check that warning was logged
                self.assertIn(
                    "Invalid device 'invalid'. Falling back to auto-detection.",
                    log.output[0],
                )

    @patch.dict(os.environ, {"BOMDIA_DEVICE": "cpu"})
    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_environment_variable_override(
        self, mock_mps_available, mock_cuda_available
    ):
        """Test environment variable overrides config file."""
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = True

        # Create a config instance
        config = Config()
        # The environment variable should override the config file value

        with patch("src.components.audio_generator.tts.config", config):
            device = _get_device()
            self.assertEqual(device.type, "cpu")

    @patch.dict(os.environ, {"BOMDIA_DEVICE": "invalid"})
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_invalid_environment_variable_fallback(
        self, mock_mps_available, mock_cuda_available
    ):
        """Test invalid environment variable falls back to auto-detection."""
        # Capture logs to verify warning is issued
        with self.assertLogs(
            "src.components.audio_generator.tts", level="WARNING"
        ) as log:
            # Create a config instance
            config = Config()

            with patch("src.components.audio_generator.tts.config", config):
                device = _get_device()
                # Should fall back to CPU (the only available device in our mock)
                self.assertEqual(device.type, "cpu")
                # Check that warning was logged
                self.assertIn(
                    "Invalid device 'invalid'. Falling back to auto-detection.",
                    log.output[0],
                )


if __name__ == "__main__":
    unittest.main()
