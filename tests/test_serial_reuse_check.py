import importlib.util
from pathlib import Path
import tempfile
import unittest


path = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-serial-reuse-check.py"
spec = importlib.util.spec_from_file_location("serial_reuse_check", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(values):
    return {"completion_probabilities": [{"top_logprobs": [
        {"id": index, "logprob": value} for index, value in enumerate(values)]}]}


class SerialReuseTests(unittest.TestCase):
    def test_identical_and_altered_full_vocabulary(self):
        baseline = event([-1.0, -2.0, -3.0])
        self.assertTrue(module.compare(baseline, baseline, 3)["passes"])
        self.assertFalse(module.compare(baseline, event([-1.0, -2.1, -3.0]), 3)["passes"])

    def test_rejects_missing_duplicate_or_nonfinite_entries(self):
        baseline = event([-1.0, -2.0, -3.0])
        duplicate = event([-1.0, -2.0, -3.0])
        duplicate["completion_probabilities"][0]["top_logprobs"][1]["id"] = 0
        for bad in (event([-1.0, -2.0]), duplicate, event([-1.0, float("nan"), -3.0])):
            with self.assertRaises(ValueError):
                module.compare(baseline, bad, 3)

    def test_requires_all_libraries_from_the_bound_directory(self):
        names = ("libllama-server-impl.so", "libllama-common.so.0", "libllama.so.0", "libmtmd.so.0",
                 "libggml.so.0", "libggml-base.so.0", "libggml-cpu.so.0", "libggml-cuda.so.0")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in names:
                (root / name).touch()
            maps = "\n".join(f"0-1 r-xp 0 00:00 0 {root / name}" for name in names)
            module.verify_local_libraries(maps, root)
            with self.assertRaises(ValueError):
                module.verify_local_libraries(maps.rsplit("\n", 1)[0], root)
            other = root / "elsewhere"
            other.mkdir()
            (other / names[-1]).touch()
            with self.assertRaises(ValueError):
                module.verify_local_libraries(maps + f"\n0-1 r-xp 0 00:00 0 {other / names[-1]}", root)


if __name__ == "__main__":
    unittest.main()
