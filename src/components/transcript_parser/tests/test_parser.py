"""Unit tests for the transcript parser component."""

from src.components.transcript_parser.parser import (
    ingest_transcript,
    parse_simple_txt,
    parse_srt,
)


def test_parse_simple_txt(tmp_path):
    """Test parsing of simple text transcripts."""
    transcript_path = tmp_path / "test.txt"
    transcript_path.write_text(
        "Speaker 1: Hello there\nSpeaker 2: Hi, how are you?\n"
        "[S1] Direct tag\nName With Space: Another line"
    )

    lines = parse_simple_txt(str(transcript_path))
    assert len(lines) == 4
    assert lines[0]["speaker"] == "S1"
    assert lines[0]["text"] == "Hello there"
    assert lines[1]["speaker"] == "S2"
    assert lines[1]["text"] == "Hi, how are you?"
    assert lines[2]["speaker"] == "S1"
    assert lines[2]["text"] == "Direct tag"
    assert lines[3]["speaker"] == "S3"
    assert lines[3]["text"] == "Another line"


def test_parse_srt(tmp_path):
    """Test parsing of SRT transcripts."""
    srt_path = tmp_path / "test.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nHello there\n\n"
        "2\n00:00:05,000 --> 00:00:08,000\nHi, how are you?"
    )

    lines = parse_srt(str(srt_path))
    assert len(lines) == 2
    assert lines[0]["speaker"] == "S1"
    assert lines[0]["text"] == "Hello there"
    assert lines[1]["speaker"] == "S1"
    assert lines[1]["text"] == "Hi, how are you?"


def test_ingest_transcript(tmp_path):
    """Test the ingest_transcript function."""
    # Test with TXT file
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("Test line")

    lines = ingest_transcript(str(txt_path))
    assert len(lines) == 1
    assert lines[0]["speaker"] == "S1"
    assert lines[0]["text"] == "Test line"

    # Test with SRT file
    srt_path = tmp_path / "test.srt"
    srt_path.write_text("1\n00:00:01,000 --> 00:00:04,000\nTest line")

    lines = ingest_transcript(str(srt_path))
    assert len(lines) == 1
    assert lines[0]["speaker"] == "S1"
    assert lines[0]["text"] == "Test line"
