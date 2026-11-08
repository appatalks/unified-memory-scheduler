import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


path = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-stage-causal-backend.py"
spec = importlib.util.spec_from_file_location("stage_causal_backend", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CausalStagingTests(unittest.TestCase):
    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing"
            output.mkdir()
            with mock.patch.object(module.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "must not already exist"):
                    module.stage(root, root / "libllama.so", output, root / "patchelf")
                run.assert_not_called()

    def test_unknown_components_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "staged"
            with mock.patch.object(module.subprocess, "run") as run:
                run.return_value.stdout = "patchelf 0.19.3\n"
                with self.assertRaisesRegex(ValueError, "Unqualified source component"):
                    module.stage(root, root / "libllama.so", output, root / "patchelf")
            self.assertFalse(output.exists())

    def test_wrong_relocation_tool_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(module.subprocess, "run") as run:
                run.return_value.stdout = "patchelf 0.18.0\n"
                with self.assertRaisesRegex(ValueError, "patchelf version"):
                    module.stage(root, root / "libllama.so", root / "staged", root / "patchelf")


if __name__ == "__main__":
    unittest.main()
