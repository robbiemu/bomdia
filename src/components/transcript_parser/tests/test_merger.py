"""Unit tests for the transcript merger component."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_config():
    """Mock config object."""
    config = MagicMock()
    config.PAUSE_PLACEHOLDER = "[insert-verbal-tag-for-pause]"
    return config


def test_merge_consecutive_lines(mock_config):
    """Test merging consecutive lines from the same speaker."""
    # Import the function after mocking the config
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr("shared.config.config", mock_config)
        from src.components.transcript_parser.merger import merge_consecutive_lines

        # Test with consecutive lines from same speaker
        consecutive_lines = [
            {"speaker": "S1", "text": "First line"},
            {"speaker": "S1", "text": "Second line"},
            {"speaker": "S2", "text": "Response"},
        ]
        merged = merge_consecutive_lines(consecutive_lines)
        assert len(merged) == 2
        assert mock_config.PAUSE_PLACEHOLDER in merged[0]["text"]
        assert (
            merged[0]["text"]
            == f"First line {mock_config.PAUSE_PLACEHOLDER} Second line"
        )
        assert merged[1]["text"] == "Response"

        # Test with no consecutive lines
        non_consecutive_lines = [
            {"speaker": "S1", "text": "First line"},
            {"speaker": "S2", "text": "Response"},
            {"speaker": "S1", "text": "Another line"},
        ]
        merged = merge_consecutive_lines(non_consecutive_lines)
        assert len(merged) == 3
        assert merged[0]["text"] == "First line"
        assert merged[1]["text"] == "Response"
        assert merged[2]["text"] == "Another line"
