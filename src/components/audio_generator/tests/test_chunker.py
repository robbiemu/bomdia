"""Unit tests for the audio generator component."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_config():
    """Mock config object."""
    config = MagicMock()
    config.AVG_WPS = 2.5
    return config


def test_estimate_seconds_for_text(mock_config):
    """Test the estimate_seconds_for_text function."""
    # Import the function after mocking the config
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("shared.config.config", mock_config)
        from src.components.audio_generator.chunker import estimate_seconds_for_text

        # Test with a known text
        text = "This is a test sentence with seven words."
        # Count the actual words
        word_count = len(text.split())
        expected_seconds = word_count / mock_config.AVG_WPS
        actual_seconds = estimate_seconds_for_text(text)
        # Use approximate equality due to floating point precision
        assert abs(actual_seconds - expected_seconds) < 0.0001

        # Test with empty text
        assert estimate_seconds_for_text("") == 0.0

        # Test with whitespace-only text
        assert estimate_seconds_for_text("   ") == 0.0
