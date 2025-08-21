import os
import tempfile
import unittest

from src.pipeline import TranscriptProcessor


class SimpleParserTest(unittest.TestCase):
    def run_test_with_lines(self, lines, expected):
        with tempfile.TemporaryDirectory() as tmp_dir:
            transcript_path = os.path.join(tmp_dir, "test_transcript.txt")
            with open(transcript_path, "w") as f:
                for line in lines:
                    f.write(line + "\n")
            self.assertEqual(
                TranscriptProcessor.load_and_parse_transcript(transcript_path), expected
            )

    def test_single_line_with_speaker_name(self):
        lines = ["S1: Speaker One - Hello world."]
        expected = [
            {
                "speaker": "S1",
                "speaker_name": "S1",
                "text": "Speaker One - Hello world.",
            }
        ]
        self.run_test_with_lines(lines, expected)
