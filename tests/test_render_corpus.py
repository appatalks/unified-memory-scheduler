#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BINARY = Path(sys.argv.pop(1)).resolve()


class RenderCorpusInputs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus.jsonl"
        self.corpus.write_text("{}\n")
        self.output = self.root / "tokens.json"

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self):
        return subprocess.run([str(BINARY), "--model", str(self.root / "missing.gguf"),
            "--corpus", str(self.corpus), "--output", str(self.output)], capture_output=True, text=True, timeout=10)

    def rejected(self, result, message):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("llama_model_loader", result.stderr)

    def test_required_paths_fail_before_model_loading(self):
        result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=10)
        self.rejected(result, "All paths are required")

    def test_public_parent_is_rejected_before_model_loading(self):
        self.root.chmod(0o755)
        self.rejected(self.invoke(), "owned and private")
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.write_text("preserve")
        self.rejected(self.invoke(), "overwrite")
        self.assertEqual(self.output.read_text(), "preserve")

    def test_oversized_corpus_is_rejected_before_model_loading(self):
        self.corpus.write_bytes(b"x" * (2 * 1024**2 + 1))
        self.rejected(self.invoke(), "over2MiB")
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
