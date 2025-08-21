import os
import tempfile
import unittest

from src.pipeline import TranscriptProcessor


class ParserDebugTest(unittest.TestCase):
    def test_simple_parse(self):
        lines = ["S1: Speaker One - Hello world."]
        with tempfile.TemporaryDirectory() as tmp_dir:
            transcript_path = os.path.join(tmp_dir, "test_transcript.txt")
            with open(transcript_path, "w") as f:
                for line in lines:
                    f.write(line + "\n")
            parsed = TranscriptProcessor.load_and_parse_transcript(transcript_path)
            print(f"Parsed output: {parsed}")
