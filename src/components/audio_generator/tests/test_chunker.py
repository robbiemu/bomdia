"""Unit tests for the audio generator component."""

from shared.config import config
from src.components.audio_generator.chunker import estimate_seconds_for_text


def test_estimate_seconds_for_text():
    """Test the estimate_seconds_for_text function."""
    # Test with a known text
    text = "This is a test sentence with seven words."
    # Count the actual words
    word_count = len(text.split())
    expected_seconds = word_count / config.AVG_WPS
    actual_seconds = estimate_seconds_for_text(text)
    # Use approximate equality due to floating point precision
    assert abs(actual_seconds - expected_seconds) < 0.0001

    # Test with empty text
    assert estimate_seconds_for_text("") == 0.0

    # Test with whitespace-only text
    assert estimate_seconds_for_text("   ") == 0.0
