"""Tests for pipeline components."""

import re
from unittest.mock import patch

import pytest
from src.pipeline import AudioPromptProcessor


def test_get_starting_speaker():
    """Test the _get_starting_speaker function with various inputs."""
    # Test normal case with speaker tag at start
    assert AudioPromptProcessor._get_starting_speaker("[S1] Hello world") == "S1"

    # Test with leading whitespace
    assert AudioPromptProcessor._get_starting_speaker(" [S2] How are you?") == "S2"

    # Test without speaker tag
    assert AudioPromptProcessor._get_starting_speaker("Hello world") is None

    # Test with malformed speaker tag
    assert AudioPromptProcessor._get_starting_speaker("S1] Hello world") is None
    assert AudioPromptProcessor._get_starting_speaker("[S1 Hello world") is None

    # Test edge case with just speaker tag
    assert AudioPromptProcessor._get_starting_speaker("[S1]") == "S1"

    # Test with numbers in speaker ID
    assert AudioPromptProcessor._get_starting_speaker("[S123] Hello") == "S123"


def test_get_starting_speaker_regex_behavior():
    """Test that the regex correctly matches speaker tags."""
    # The regex should match at the start of the string after stripping
    pattern = re.compile(r"^\[(S\d+)\]")

    # Test that it matches correctly
    match = pattern.search("[S1] Hello")
    assert match is not None
    assert match.group(1) == "S1"

    # Test that it doesn't match in the middle
    match = pattern.search("Hello [S1] world")
    assert match is None
