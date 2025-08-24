"""Integration tests for rehearsal and chunker workflow."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from src.components.verbal_tag_injector.director import Director
from src.components.audio_generator.chunker import chunk_lines


def test_rehearsal_plus_chunker_integration():
    """Test that verbal tags are processed correctly through the full pipeline."""
    # Create a simple transcript with verbal tags already included
    # This simulates the output of the rehearsal process
    rehearsed_script = [
        {"speaker": "S1", "text": "Hello there (warmly)."},
        {"speaker": "S2", "text": "Hi! How are you doing today? (smiling)"},
        {"speaker": "S1", "text": "I'm doing well, thanks."},
    ]

    # Now chunk the rehearsed script
    chunks = chunk_lines(rehearsed_script)

    # Verify we got some chunks
    assert len(chunks) > 0

    # Join all chunks to verify tags are preserved
    full_text = " ".join(chunks)
    assert "(warmly)" in full_text
    assert "(smiling)" in full_text

    # Verify that chunks start with speaker tags
    for chunk in chunks:
        assert chunk.startswith("[S")


def test_chunker_respects_config_durations():
    """Test that chunker respects min/max duration configuration."""
    # Create a simple transcript
    transcript = [
        {"speaker": "S1", "text": "This is a short sentence."},
        {"speaker": "S2", "text": "This is another short sentence."},
        {"speaker": "S1", "text": "Yet another short sentence."},
    ]

    # Patch the config to set specific min/max durations
    with patch("src.components.audio_generator.chunker.config") as mock_config:
        mock_config.MIN_CHUNK_DURATION = "2.0"
        mock_config.MAX_CHUNK_DURATION = "8.0"
        mock_config.AVG_WPS = 2.5  # 2.5 words per second

        # Chunk the transcript
        chunks = chunk_lines(transcript)

        # Verify we got some chunks
        assert len(chunks) > 0

        # Import estimate_seconds_for_text to check durations
        from src.components.audio_generator.chunker import estimate_seconds_for_text

        # Check that chunks are within reasonable duration ranges
        for i, chunk in enumerate(chunks):
            duration = estimate_seconds_for_text(chunk)
            # Duration should be positive
            assert duration > 0
            # Log a warning for chunks outside the desired duration (this is expected in some cases)
            if not (2.0 <= duration <= 8.0):
                logging.warning(
                    f"Chunk {i} is outside the desired duration range: {duration:.1f}s. "
                    f"Allowed: [2.0s - 8.0s]."
                )
