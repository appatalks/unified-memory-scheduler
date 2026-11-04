import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-synapse-critical-path.py"
spec = importlib.util.spec_from_file_location("synapse_critical_path", SCRIPT)
assert spec and spec.loader
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@unittest.skipUnless(os.name == "posix", "Private Linux diagnostic tooling")
class CriticalPathTests(unittest.TestCase):
    def test_private_output_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            path.chmod(0o700)
            study.validate_directory(path)
            path.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "private directory"):
                study.validate_directory(path)
            path.chmod(0o700)

    def test_failed_or_unclean_diagnostics_cannot_claim_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.gguf", "profile.json", "cpu.so", "profiler.so"):
                (root / name).write_bytes(b"fixture")
            args = SimpleNamespace(model=root / "model.gguf", profile=root / "profile.json",
                                   context=131072, cache_mib=4096, variant="classic", category="coding",
                                   tokens=64, backend=root / "backend", cpu_library=root / "cpu.so",
                                   cpu_profiler=root / "profiler.so", synapse_source=root)
            runtime = mock.Mock(logs=["backend stopped"], key="not-recorded")
            app = mock.Mock()
            app.AppError = type("AppError", (Exception,), {})
            app.Runtime.return_value = runtime
            app.read_hybrid_profile.return_value = {}
            for cleanup_error in (None, OSError("cleanup failed")):
                runtime.stop.side_effect = cleanup_error
                with mock.patch.object(study.baseline, "sha256", return_value="fixture"), \
                        mock.patch.object(study.baseline, "wait_ready", side_effect=RuntimeError("load failed")):
                    report = study.capture(args, app)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["samples"], [])
                if cleanup_error:
                    self.assertIn("cleanup_error", report)
                    self.assertNotIn("owned_backend_stopped", report)
                self.assertEqual(app.Runtime.call_args.kwargs["hybrid_context"], 131072)
                self.assertEqual(runtime.load.call_args.args[0]["context"], 131072)
                self.assertFalse(runtime.load.call_args.kwargs["remember"])


if __name__ == "__main__":
    unittest.main()
