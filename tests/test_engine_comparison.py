#!/usr/bin/env python3

import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("engine_comparison_under_test",
    Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-engine-comparison.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class EngineComparisonTests(unittest.TestCase):
    def engine(self):
        engine = module.NativeEngine.__new__(module.NativeEngine)
        engine.url, engine.key = "http://127.0.0.1:1", "test-key"
        engine.ignore_eos = True
        return engine

    def stream(self, positions):
        events = [{"stop": False, "tokens": [token], "tokens_predicted": position + 1}
                  for position, token in positions]
        events.append({"stop": True, "tokens": [], "tokens_predicted": 3, "tokens_evaluated": 2,
                       "timings": {"prompt_ms": 10, "predicted_ms": 20}})
        return io.BytesIO(b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events))

    def test_utf8_stream_gap_retains_authoritative_count_and_positions(self):
        engine = self.engine()
        with mock.patch.object(module.urllib.request, "urlopen", return_value=self.stream([(0, 11), (2, 33)])):
            self.assertEqual(list(engine.generate([4, 5], 3)), [11, 33])
        self.assertEqual(engine.stream_positions, [0, 2])
        self.assertEqual(engine.last["generated"], 3)

    def test_duplicate_stream_position_is_rejected(self):
        engine = self.engine()
        with mock.patch.object(module.urllib.request, "urlopen", return_value=self.stream([(0, 11), (0, 11)])):
            with self.assertRaisesRegex(RuntimeError, "positions"):
                list(engine.generate([4, 5], 3))

    def test_full_audit_checks_sparse_stream_positions(self):
        result = {"tokens": [11, 22, 33], "tokens_predicted": 3, "tokens_evaluated": 2}
        with mock.patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(result).encode())):
            tokens, elapsed = self.engine().audit([4, 5], 3, [11, 33], [0, 2])
        self.assertEqual(tokens, [11, 22, 33])
        self.assertGreaterEqual(elapsed, 0)

    def test_audit_divergence_is_not_a_success(self):
        result = {"tokens": [11, 22, 44], "tokens_predicted": 3, "tokens_evaluated": 2}
        with mock.patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(result).encode())):
            with self.assertRaisesRegex(RuntimeError, "differs"):
                self.engine().audit([4, 5], 3, [11, 33], [0, 2])

    def test_audit_count_mismatch_is_rejected(self):
        result = {"tokens": [11, 22], "tokens_predicted": 3, "tokens_evaluated": 2}
        with mock.patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(result).encode())):
            with self.assertRaisesRegex(RuntimeError, "count mismatch"):
                self.engine().audit([4, 5], 3, [11, 33], [0, 2])

    def test_child_environment_does_not_inherit_experimental_controls(self):
        with mock.patch.dict(os.environ, {"LD_PRELOAD": "unsafe", "VRAMBRIDGE_HYBRID_ADAPT": "1",
                "STRATA_ROUTE_TAIL_SKIP": "1", "GGML_CUDA_GRAPH_OPT": "1", "OMP_WAIT_POLICY": "PASSIVE"}):
            before = dict(os.environ)
            cleaned = module.clean_environment()
            self.assertFalse(any(k.startswith(("VRAMBRIDGE_", "STRATA_", "GGML_", "OMP_")) for k in cleaned))
            self.assertNotIn("LD_PRELOAD", cleaned)
            self.assertEqual(dict(os.environ), before)

    def test_failed_startup_residency_acknowledgement_stops_owned_native_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("binary", "plugin", "plan", "model"):
                (root / name).write_text("fixture")
            args = SimpleNamespace(mode="scalar", natural_eos=False, experts=2, cache_mib=6144, attention_window=None,
                managed_memory=False, cpu_repack=False, exact_packed_cpu=None, kv_cold_tier=None,
                native_binary=root / "binary", cuda_plugin=root / "plugin", native_plan=root / "plan", model=root / "model")
            process = mock.Mock()
            process.poll.return_value = None
            response = io.BytesIO(b"")
            response.status = 200
            with mock.patch.object(module.subprocess, "Popen", return_value=process) as popen, \
                    mock.patch.object(module.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "acknowledge"):
                    module.NativeEngine(args, root / "log")
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
            self.assertTrue(popen.call_args.kwargs["stdout"].closed)

    def test_cleanup_failure_cannot_leave_a_complete_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, output = root / "corpus.json", root / "report.json"
            corpus.write_text(json.dumps({"records": [
                {"id": "fixture", "category": "coding", "split": "test", "tokens": [4, 5]}]}))
            engine = mock.Mock(pid=os.getpid(), load_seconds=.1, info={},
                last={"generated": 16, "prompt_tokens": 2, "finish": "length"},
                stream_positions=list(range(16)))
            engine.generate.return_value = iter([11] * 16)
            engine.audit.return_value = ([11] * 16, .1)
            engine.close.side_effect = RuntimeError("cleanup incomplete")
            previous = os.umask(0o077)
            try:
                with mock.patch.object(module.sys, "argv", ["comparison", "--engine", "native", "--tokens", "16",
                        "--category", "coding", "--model", str(root / "model"),
                        "--token-corpus", str(corpus), "--output", str(output)]), \
                        mock.patch.object(module, "NativeEngine", return_value=engine), mock.patch("builtins.print"):
                    with self.assertRaisesRegex(RuntimeError, "cleanup incomplete"):
                        module.main()
            finally:
                os.umask(previous)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["cleanup_error"], "cleanup incomplete")
            self.assertNotIn("owned_backend_stopped", report)

    def test_selective_kv_does_not_enable_global_managed_allocations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("binary", "plugin", "plan", "model", "tier"):
                (root / name).write_text("fixture")
            args = SimpleNamespace(mode="scalar", natural_eos=False, experts=2, cache_mib=6144,
                attention_window=None, managed_memory=False, cpu_repack=False, exact_packed_cpu=None,
                kv_cold_tier=root / "tier", native_binary=root / "binary", cuda_plugin=root / "plugin",
                native_plan=root / "plan", model=root / "model")
            process = mock.Mock()
            process.poll.return_value = None
            response = io.BytesIO(b"")
            response.status = 200
            with mock.patch.object(module.subprocess, "Popen", return_value=process) as popen, \
                    mock.patch.object(module.urllib.request, "urlopen", return_value=response), \
                    mock.patch.dict(os.environ, {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}):
                with self.assertRaisesRegex(RuntimeError, "acknowledge"):
                    module.NativeEngine(args, root / "log")
            env = popen.call_args.kwargs["env"]
            self.assertNotIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", env)
            self.assertEqual(env["VRAMBRIDGE_KV_COLD_MODE"], "stage")
            self.assertEqual(env["LD_PRELOAD"], str(root / "tier"))
            process.terminate.assert_called_once()

    def test_selective_kv_rejects_global_managed_or_non_native_engine(self):
        for controls in (["--engine", "native", "--managed-memory"], ["--engine", "strata"]):
            with self.subTest(controls=controls), \
                    mock.patch.object(module.sys, "argv", ["comparison", "--model", "unused",
                        "--token-corpus", "unused", "--output", "nonexistent-kv-test-output.json",
                        "--kv-cold-tier", "unused", *controls]), \
                    mock.patch.object(module.sys, "stderr", io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("Selective cold KV", stderr.getvalue())

    def test_expert_product_ablation_requires_window_prototype(self):
        with mock.patch.object(module.sys, "argv", ["comparison", "--model", "unused",
                "--token-corpus", "unused", "--output", "nonexistent-expert-test-output.json",
                "--engine", "native", "--expert-products", "native"]), \
                mock.patch.object(module.sys, "stderr", io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                module.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Expert-product ablation", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
