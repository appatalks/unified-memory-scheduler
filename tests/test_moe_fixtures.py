#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(sys.argv.pop(1)).resolve()


class FixtureInputTests(unittest.TestCase):
    def test_help_requires_no_gguf_numpy_or_gpu(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn("untrained", result.stdout)

    def test_rejects_unbounded_or_misaligned_sizes_before_importing_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture.gguf"
            common = [sys.executable, str(SCRIPT), "--llama-source", "/missing/source",
                      "--ggml-base", "/missing/libggml-base.so", "--architecture", "llama", "--output", str(output)]
            for extra in (["--active", "9"], ["--experts", "1"], ["--experts", "17"],
                          ["--layers", "1"], ["--layers", "5"], ["--embedding", "63"],
                          ["--embedding", "1024"], ["--feed-forward", "4096"],
                          ["--expert-type", "Q4_K"], ["--seed", "-1"]):
                result = subprocess.run([*common, *extra], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Require bounded fixture", result.stderr)
                self.assertFalse(output.exists())
                self.assertFalse(Path(str(output) + ".part").exists())

    def test_never_replaces_an_existing_final_or_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture.gguf"
            for path in (output, Path(str(output) + ".part")):
                path.write_text("preserve")
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "--llama-source", "/missing/source",
                     "--ggml-base", "/missing/library", "--architecture", "qwen2moe", "--output", str(output)],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Refusing to overwrite", result.stderr)
                self.assertEqual(path.read_text(), "preserve")
                path.unlink()


if __name__ == "__main__":
    unittest.main()
