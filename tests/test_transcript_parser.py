import os
import tempfile
import unittest

from src.pipeline import TranscriptProcessor


class TestTranscriptParser(unittest.TestCase):
    def run_test_with_lines(self, lines, expected):
        with tempfile.TemporaryDirectory() as tmp_dir:
            transcript_path = os.path.join(tmp_dir, "test_transcript.txt")
            with open(transcript_path, "w") as f:
                for line in lines:
                    f.write(line + "\n")
            self.assertEqual(
                TranscriptProcessor.load_and_parse_transcript(transcript_path), expected
            )

    def test_parse_transcript_with_speaker_names(self):
        lines = [
            "S1: Speaker One - Hello world.",
            "S2: Speaker Two - This is a test.",
        ]
        expected = [
            {
                "speaker": "S1",
                "speaker_name": "S1",
                "text": "Speaker One - Hello world.",
            },
            {
                "speaker": "S2",
                "speaker_name": "S2",
                "text": "Speaker Two - This is a test.",
            },
        ]
        self.run_test_with_lines(lines, expected)

    def test_parse_transcript_without_speaker_names(self):
        lines = [
            "[S1] Hello world.",
            "[S2] This is a test.",
        ]
        expected = [
            {"speaker": "S1", "speaker_name": None, "text": "Hello world."},
            {"speaker": "S2", "speaker_name": None, "text": "This is a test."},
        ]
        self.run_test_with_lines(lines, expected)

    def test_parse_transcript_with_mixed_formats(self):
        lines = [
            "S1: Speaker One - Hello world.",
            "[S2] This is a test.",
        ]
        expected = [
            {
                "speaker": "S1",
                "speaker_name": "S1",
                "text": "Speaker One - Hello world.",
            },
            {"speaker": "S2", "speaker_name": None, "text": "This is a test."},
        ]
        self.run_test_with_lines(lines, expected)

    def test_parse_transcript_with_empty_lines(self):
        lines = [
            "S1: Speaker One - Hello world.",
            "",
            "[S2] This is a test.",
        ]
        expected = [
            {
                "speaker": "S1",
                "speaker_name": "S1",
                "text": "Speaker One - Hello world.",
            },
            {"speaker": "S2", "speaker_name": None, "text": "This is a test."},
        ]
        self.run_test_with_lines(lines, expected)

    def test_parse_transcript_with_continuation(self):
        lines = [
            "S1: Speaker One - Hello world.",
            "This is a continuation.",
            "[S2] This is a test.",
        ]
        expected = [
            {
                "speaker": "S1",
                "speaker_name": "S1",
                "text": "Speaker One - Hello world. This is a continuation.",
            },
            {"speaker": "S2", "speaker_name": None, "text": "This is a test."},
        ]
        self.run_test_with_lines(lines, expected)
