#!/usr/bin/env python3

import http.client
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from urllib.parse import urlsplit


APP_PATH = Path(sys.argv.pop(1)).resolve()
spec = importlib.util.spec_from_file_location("vrambridge_app", APP_PATH)
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

HYBRID_METADATA = {
    "general.architecture": "gpt-oss", "general.file_type": 38,
    "gpt-oss.block_count": 36, "gpt-oss.embedding_length": 2880,
    "gpt-oss.feed_forward_length": 2880, "gpt-oss.expert_count": 128,
    "gpt-oss.expert_used_count": 4,
}


def write_metadata(path, metadata):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    entries = []
    for key, value in metadata.items():
        kind, raw = (8, string(value)) if isinstance(value, str) else (4, struct.pack("<I", value))
        entries.append(string(key) + struct.pack("<I", kind) + raw)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + b"".join(entries))


class LocalAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.backend = self.root / "llama-server"
        shutil.copyfile(Path(__file__).with_name("fake_llama_server.py"), self.backend)
        self.backend.chmod(0o755)
        self.model = self.root / "model with spaces.gguf"
        self.model.write_bytes(b"GGUF-test-model")
        self.runtime = app.Runtime(str(self.backend))
        self.server = app.AppServer(("127.0.0.1", 0), self.runtime)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.runtime.stop(closing=True)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            supplied = {"Content-Type": "application/json"}
            supplied.update(headers or {})
            body = json.dumps(data) if data is not None else None
            connection.request(method, path, body=body, headers=supplied)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def wait_state(self, wanted):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = self.runtime.status()
            if status["state"] == wanted:
                return status
            time.sleep(0.03)
        self.fail(f"Expected {wanted}: {self.runtime.status()}")

    def load(self, model=None, **extra):
        code, _, body = self.request("POST", "/api/v1/model/load",
                                    {"model": str(model or self.model), **extra})
        self.assertEqual(code, 202, body)
        self.assertEqual(json.loads(body)["state"], "loading")
        return self.wait_state("ready")

    def test_static_assets_status_and_unknown_routes(self):
        for path, expected in (("/", b"VRAMBridge"), ("/app.js", b"textContent"),
                               ("/chat-stream.mjs", b"class ChatStream"),
                               ("/style.css", b"@media")):
            code, headers, body = self.request("GET", path)
            self.assertEqual(code, 200)
            self.assertIn(expected, body)
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertEqual(headers["Cache-Control"], "no-store")
        code, _, body = self.request("GET", "/api/v1/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["state"], "stopped")
        for path in ("/../app.py", "/%2e%2e/app.py", "/api/v2/status"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_missing_backend_is_explicit(self):
        self.runtime.executable = None
        self.assertFalse(self.runtime.status()["backend_available"])
        code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model)})
        self.assertEqual(code, 503)
        self.assertIn(b"--llama-server", body)

    def test_load_models_api_and_unload(self):
        status = self.load()
        self.assertEqual(status["model"], str(self.model))
        self.assertEqual(status["placement"], "auto")
        code, _, body = self.request("GET", "/v1/models")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "local-model")
        self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": str(self.model)})[0], 409)
        process = self.runtime.process
        self.assertIsNotNone(process)
        self.assertEqual(self.request("POST", "/api/v1/model/unload", {})[0], 200)
        self.assertIsNotNone(process.poll())
        self.assertEqual(self.runtime.status()["state"], "stopped")
        self.assertEqual(self.request("GET", "/v1/models")[0], 503)

    def test_cpu_flags_and_environment_isolation(self):
        with mock.patch.dict(os.environ, {"LLAMA_ARG_MODEL": "bad",
                                         "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
                                         "VRAMBRIDGE_HYBRID_PLAN": "/unexpected/plan",
                                         "VRAMBRIDGE_HYBRID_DIRECT": "1",
                                         "GGML_CUDA_REGISTER_HOST": "1"}):
            status = self.load(placement="cpu")
        self.assertIn('"gpu_layers": "0"', "\n".join(status["logs"]))
        self.assertIn('"inherited_override": false', "\n".join(status["logs"]))
        self.assertIn('"managed_memory": false', "\n".join(status["logs"]))
        self.assertNotIn(self.runtime.key, "\n".join(status["logs"]))
        flags = json.loads(status["logs"][0])
        self.assertIsNone(flags["hybrid_plan"])
        self.assertIsNone(flags["hybrid_direct"])
        self.assertIsNone(flags["register_host"])
        self.assertEqual(status["execution_profile"], "standard")
        self.assertIsNone(status["execution_warning"])

    def test_auto_flags(self):
        status = self.load()
        self.assertIn('"gpu_layers": "auto"', "\n".join(status["logs"]))
        self.assertIn('"fit_target": "1024"', "\n".join(status["logs"]))

    def hybrid_runtime(self, **extra):
        self.plan = self.root / "hot experts.txt"
        self.plan.write_text("# training-only test plan\n0 7\n35 127\n")
        self.plugin = self.root / "libggml-cuda.so"
        self.plugin.write_bytes(b"test-plugin")
        write_metadata(self.model, HYBRID_METADATA)
        self.runtime = app.Runtime(str(self.backend), hybrid_plan=self.plan,
                                   hybrid_cuda_plugin=self.plugin, **extra)
        self.server.runtime = self.runtime

    def test_hybrid_profile_flags_environment_and_readiness(self):
        self.hybrid_runtime()
        self.assertEqual(self.runtime.status()["context"], 16384)
        self.assertFalse(self.runtime.status()["hybrid_verified"])
        with mock.patch.dict(os.environ, {"VRAMBRIDGE_HYBRID_BATCH": "32",
                                         "VRAMBRIDGE_HYBRID_REPACK": "1",
                                         "VRAMBRIDGE_HYBRID_PLAN": "/unexpected/plan",
                                         "GGML_BACKEND_PATH": "/unexpected/plugin",
                                         "GGML_CUDA_REGISTER_HOST": "0"}):
            status = self.load()
        flags = json.loads(status["logs"][0])
        self.assertEqual(status["execution_profile"], "experimental-hybrid-direct")
        self.assertTrue(status["hybrid_verified"])
        self.assertFalse(status["fit_enabled"])
        self.assertFalse(status["speculation_enabled"])
        self.assertIn("remain failed", status["execution_warning"])
        self.assertEqual(status["context"], 16384)
        self.assertEqual(flags["fit"], "off")
        self.assertEqual(flags["gpu_layers"], "all")
        self.assertIsNone(flags["fit_target"])
        self.assertTrue(flags["cpu_moe"])
        self.assertTrue(flags["no_host"])
        self.assertTrue(flags["no_repack"])
        self.assertTrue(flags["no_op_offload"])
        self.assertEqual(flags["expert_cache_mib"], "10240")
        self.assertEqual(flags["hybrid_plan"], str(self.plan))
        self.assertEqual(flags["cuda_plugin"], str(self.plugin))
        for field in ("hybrid_adapt", "hybrid_async", "hybrid_direct", "register_host"):
            self.assertEqual(flags[field], "1")
        self.assertIsNone(flags["hybrid_batch"])
        self.assertIsNone(flags["hybrid_repack"])
        self.assertIsNone(flags["draft_model"])
        process = self.runtime.process
        self.runtime.stop()
        self.assertIsNotNone(process.poll())
        self.assertFalse(self.runtime.status()["hybrid_verified"])
        self.assertTrue(self.load()["hybrid_verified"])

    def test_hybrid_cannot_report_ready_without_verified_execution(self):
        self.hybrid_runtime(load_timeout=0.4)
        for name in ("hybrid-no-markers", "hybrid-wrong-context"):
            model = self.root / f"{name}.gguf"
            write_metadata(model, HYBRID_METADATA)
            code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(model)})
            self.assertEqual(code, 202)
            status = self.wait_state("error")
            self.assertIn("hybrid readiness timed out", status["error"])
            self.assertFalse(status["hybrid_verified"])
            self.assertIsNotNone(self.runtime.process.poll())

    def test_hybrid_rejects_conflicting_settings_before_spawning(self):
        self.hybrid_runtime()
        for extra in ({"placement": "cpu"}, {"context": 8192}, {"load_mode": "none"},
                      {"expert_cache_mib": 0}, {"host_op_offload": True},
                      {"fit_target_mib": 512}, {"op_offload_min_batch": 128},
                      {"draft_model": str(self.model)}):
            with self.subTest(extra=extra):
                code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model), **extra})
                self.assertEqual(code, 400, body)
                self.assertIsNone(self.runtime.process)

    def test_automatic_profile_selects_supported_layout_and_standard_for_other_models(self):
                patched = self.root / "patched-server"
                shutil.copyfile(self.backend, patched)
                patched.chmod(0o755)
                self.hybrid_runtime(execution_mode="auto", hybrid_executable=str(patched))
                status = self.load()
                self.assertEqual(status["execution_mode"], "auto")
                self.assertEqual(status["backend"], str(patched))
                self.assertEqual(status["execution_profile"], "experimental-hybrid-direct")
                self.assertTrue(status["hybrid_verified"])
                self.runtime.stop()
                write_metadata(self.model, {"general.architecture": "gemma3", "general.file_type": 15})
                status = self.load()
                self.assertEqual(status["execution_profile"], "standard")
                self.assertEqual(status["backend"], str(self.backend))
                self.assertFalse(status["hybrid_verified"])
                self.assertIn("not supported", status["selection_reason"])
                self.assertIsNone(status["execution_warning"])
                flags = json.loads(status["logs"][0])
                self.assertIsNone(flags["hybrid_plan"])
                self.assertIsNone(flags["register_host"])
                self.runtime.stop()
                code, _, body = self.request("POST", "/api/v1/model/load",
                                            {"model": str(self.model), "execution_mode": "hybrid"})
                self.assertEqual(code, 400, body)
                self.assertIsNone(self.runtime.process)
                write_metadata(self.model, HYBRID_METADATA)
                self.assertTrue(self.load()["hybrid_verified"])

    def test_standard_override_uses_original_backend_even_for_a_supported_model(self):
                self.hybrid_runtime(execution_mode="auto")
                status = self.load(execution_mode="standard")
                self.assertEqual(status["execution_profile"], "standard")
                self.assertFalse(status["hybrid_verified"])
                self.assertIsNone(json.loads(status["logs"][0])["hybrid_plan"])
                self.runtime.stop()
                self.assertEqual(self.load()["execution_profile"], "experimental-hybrid-direct")

    def test_unsupported_quantization_and_layout_are_not_treated_as_hybrid(self):
                self.hybrid_runtime(execution_mode="auto")
                for field, value in (("general.file_type", 7), ("gpt-oss.block_count", 24),
                                     ("gpt-oss.expert_used_count", 8), ("gpt-oss.feed_forward_length", 5760)):
                    write_metadata(self.model, {**HYBRID_METADATA, field: value})
                    status = self.load()
                    self.assertEqual(status["execution_profile"], "standard")
                    self.runtime.stop()
                self.assertFalse(app.supports_hybrid({**HYBRID_METADATA, "gpt-oss.expert_used_count": True}))

    def test_bad_metadata_is_not_a_silent_standard_fallback(self):
                self.hybrid_runtime(execution_mode="auto")
                for data in (b"GGUF", b"GGUF" + struct.pack("<IQQ", 3, 0, 100001),
                             b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 65537)):
                    self.model.write_bytes(data)
                    code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model)})
                    self.assertEqual(code, 400, body)
                    self.assertIn(b"Cannot select", body)
                    self.assertIsNone(self.runtime.process)

    def test_metadata_reader_skips_vocabulary_arrays_and_bounds_allocations(self):
        def string(value):
                    raw = value.encode()
                    return struct.pack("<Q", len(raw)) + raw
        architecture = string("general.architecture") + struct.pack("<I", 8) + string("qwen3")
        vocabulary = (string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) +
                              string("hello") + string("world"))
        self.model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + architecture + vocabulary)
        self.assertEqual(app.model_metadata(self.model), {"general.architecture": "qwen3"})
        for suffix in (struct.pack("<I", 99), struct.pack("<IIQ", 9, 4, 1000001),
                               struct.pack("<IIQ", 9, 4, 999999), struct.pack("<I", 8) + struct.pack("<Q", 2**40)):
                    self.model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) +
                                           architecture + string("unused.key") + suffix)
                    with self.assertRaises(ValueError):
                        app.model_metadata(self.model)
        self.model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + architecture * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
                    app.model_metadata(self.model)

    def test_invalid_installed_profile_fails_before_binding_or_loading(self):
        path = self.root / "invalid-profile.json"
        path.write_text('{"schema_version": 99}')
        result = subprocess.run([sys.executable, str(APP_PATH), "--port", "0", "--no-browser",
                                         "--hybrid-profile", str(path)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Invalid pinned hybrid installation profile", result.stderr)
        self.assertNotIn("VRAMBridge: http", result.stdout)

    def test_persistent_profile_requires_pinned_identity_and_absolute_paths(self):
                self.hybrid_runtime()
                path = self.root / "profile.json"
                profile = {"schema_version": 1, "backend_commit": app.PINNED_LLAMA_COMMIT,
                           "executable": str(self.backend), "plan": str(self.plan), "cuda_plugin": str(self.plugin)}
                path.write_text(json.dumps(profile))
                options = app.read_hybrid_profile(path)
                self.assertEqual(options["hybrid_plan"], self.plan)
                for changes in ({"schema_version": True}, {"backend_commit": "wrong"},
                                {"executable": "relative"}, {"plan": None}, {"unexpected": 1}):
                    path.write_text(json.dumps({**profile, **changes}))
                    with self.assertRaises(ValueError):
                        app.read_hybrid_profile(path)

    def test_hybrid_validates_launch_paths_and_plan(self):
        self.hybrid_runtime()
        for extra in ({"hybrid_plan": self.plan}, {"hybrid_cuda_plugin": self.plugin},
                      {"hybrid_plan": self.plan, "hybrid_cuda_plugin": self.plugin,
                       "backend_kind": "ik_llama.cpp"},
                      {"hybrid_plan": self.plan, "hybrid_cuda_plugin": self.plugin,
                       "enable_speculation": True}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    app.Runtime(str(self.backend), **extra)
        for content in ("", "0 7\n0 7\n", "36 0\n", "0 128\n", "not a plan\n",
                        "".join(f"{i // 128} {i % 128}\n" for i in range(801))):
            self.plan.write_text(content)
            with self.assertRaises(ValueError):
                app.Runtime(str(self.backend), hybrid_plan=self.plan, hybrid_cuda_plugin=self.plugin)
        code, _, body = self.request("POST", "/api/v1/model/load", {"model": str(self.model)})
        self.assertEqual(code, 400, body)
        self.assertIsNone(self.runtime.process)

    def test_host_execution_controls_are_explicit(self):
        status = self.load(host_op_offload=False, cpu_poll=0, op_offload_min_batch=128)
        self.assertFalse(status["host_op_offload"])
        self.assertEqual(status["cpu_poll"], 0)
        flags = json.loads(status["logs"][0])
        self.assertTrue(flags["no_op_offload"])
        self.assertEqual(flags["poll"], "0")
        self.assertEqual(status["op_offload_min_batch"], 128)
        self.assertEqual(flags["op_offload_min_batch"], "128")

    def test_host_execution_defaults_preserve_backend_behavior(self):
        status = self.load()
        flags = json.loads(status["logs"][0])
        self.assertFalse(flags["no_op_offload"])
        self.assertEqual(flags["poll"], "50")
        self.assertEqual(flags["op_offload_min_batch"], "32")

    def test_offload_threshold_cannot_be_overridden_by_shell(self):
        with mock.patch.dict(os.environ, {"GGML_OP_OFFLOAD_MIN_BATCH": "1"}):
            status = self.load()
        self.assertEqual(json.loads(status["logs"][0])["op_offload_min_batch"], "32")

    def test_invalid_host_execution_controls_are_rejected(self):
        for extra in ({"host_op_offload": 0}, {"host_op_offload": "false"},
                      {"cpu_poll": True}, {"cpu_poll": -1}, {"cpu_poll": 101},
                      {"op_offload_min_batch": True}, {"op_offload_min_batch": 0},
                      {"op_offload_min_batch": 8193}):
            code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(self.model), **extra})
            self.assertEqual(code, 400)
        self.runtime.backend_kind = "ik_llama.cpp"
        for extra in ({"host_op_offload": False}, {"cpu_poll": 0}, {"op_offload_min_batch": 128}):
            code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(self.model), **extra})
            self.assertEqual(code, 400)
        self.assertIsNone(self.runtime.process)

    def test_explicit_execution_settings_and_cache_flags(self):
        threads = min(2, os.cpu_count() or 1)
        status = self.load(cpu_threads=threads, batch_threads=1, load_mode="none",
                           expert_cache_mib=8192)
        self.assertEqual(status["cpu_threads"], threads)
        self.assertEqual(status["batch_threads"], 1)
        self.assertEqual(status["load_mode"], "none")
        self.assertEqual(status["expert_cache_mib"], 8192)
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["threads"], str(threads))
        self.assertEqual(flags["batch_threads"], "1")
        self.assertEqual(flags["load_mode"], "none")
        self.assertEqual(flags["expert_cache_mib"], "8192")
        self.assertEqual(flags["gpu_layers"], "all")
        self.assertTrue(flags["cpu_moe"])

    def test_invalid_execution_settings(self):
        for extra in ({"cpu_threads": True}, {"cpu_threads": 0},
                      {"cpu_threads": (os.cpu_count() or 1) + 1},
                      {"batch_threads": 0}, {"batch_threads": "4"},
                      {"load_mode": "unrestricted"}, {"expert_cache_mib": -1},
                      {"expert_cache_mib": True}, {"expert_cache_mib": 65537},
                      {"expert_cache_mib": 1024, "placement": "cpu"}):
            code, _, body = self.request("POST", "/api/v1/model/load",
                                        {"model": str(self.model), **extra})
            self.assertEqual(code, 400, body)
        self.assertIsNone(self.runtime.process)

    def test_default_threads_count_physical_cores_within_affinity(self):
        def topology(path, *args, **kwargs):
            cpu = int(path.parent.parent.name.removeprefix("cpu"))
            return "0" if path.name == "physical_package_id" else str(cpu % 2)
        with mock.patch.object(app.os, "sched_getaffinity", return_value={0, 1, 2, 3}):
            with mock.patch.object(Path, "read_text", autospec=True, side_effect=topology):
                self.assertEqual(app.default_cpu_threads(), 2)

    def test_explicit_experimental_backend_adapter(self):
        self.runtime.backend_kind = "ik_llama.cpp"
        status = self.load(load_mode="none")
        self.assertEqual(status["backend_kind"], "ik_llama.cpp")
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["gpu_layers"], "999")
        self.assertEqual(flags["fit"], "enabled")
        self.assertEqual(flags["fit_margin"], "1024")
        self.assertTrue(flags["no_mmap"])
        self.assertIsNone(flags["fit_target"])

    def test_experimental_backend_rejects_unverified_modes(self):
        self.runtime.backend_kind = "ik_llama.cpp"
        for extra in ({"placement": "cpu"}, {"expert_cache_mib": 1024}):
            code, _, body = self.request("POST", "/api/v1/model/load",
                                        {"model": str(self.model), **extra})
            self.assertEqual(code, 400, body)
            self.assertIn(b"experimental ik_llama.cpp", body)
        self.assertIsNone(self.runtime.process)

    def test_dflash_flags_are_explicit_and_backend_specific(self):
        self.runtime.enable_speculation = True
        draft = self.root / "draft.gguf"
        draft.write_bytes(b"GGUF")
        status = self.load(draft_model=str(draft), draft_tokens=5, fit_target_mib=4096)
        flags = json.loads(status["logs"][0])
        self.assertEqual(status["draft_model"], str(draft))
        self.assertEqual(status["fit_target_mib"], 4096)
        self.assertEqual(flags["spec_type"], "draft-dflash")
        self.assertEqual(flags["draft_tokens"], "5")
        self.assertEqual(flags["draft_gpu_layers"], "all")
        self.assertEqual(flags["fit_target"], "4096")
        self.runtime.stop()
        self.runtime.backend_kind = "ik_llama.cpp"
        status = self.load(draft_model=str(draft), draft_tokens=9, fit_target_mib=4096)
        flags = json.loads(status["logs"][0])
        self.assertEqual(flags["spec_type"], "dflash:n_max=9,p_min=0.0")
        self.assertEqual(flags["draft_gpu_layers"], "999")
        self.assertEqual(flags["fit_margin"], "4096")

    def test_invalid_dflash_configuration(self):
        self.runtime.enable_speculation = True
        draft = self.root / "draft.gguf"
        draft.write_bytes(b"GGUF")
        for extra in ({"draft_model": "/missing.gguf"}, {"draft_model": str(self.model)},
                      {"draft_tokens": True}, {"draft_tokens": 0}, {"draft_tokens": 33},
                      {"fit_target_mib": True}, {"fit_target_mib": 127},
                      {"draft_model": str(draft), "placement": "cpu"},
                      {"draft_model": str(draft), "expert_cache_mib": 1024}):
            code, _, body = self.request("POST", "/api/v1/model/load",
                                        {"model": str(self.model), **extra})
            self.assertEqual(code, 400, body)
        self.assertIsNone(self.runtime.process)

    def test_dflash_is_disabled_by_default(self):
        draft = self.root / "draft.gguf"
        draft.write_bytes(b"GGUF")
        code, _, body = self.request("POST", "/api/v1/model/load",
                                    {"model": str(self.model), "draft_model": str(draft)})
        self.assertEqual(code, 400)
        self.assertIn(b"--experimental-speculation", body)
        self.assertFalse(self.runtime.status()["speculation_enabled"])
        self.assertIsNone(self.runtime.process)

    def test_process_resource_usage(self):
        self.load()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = self.runtime.status()
            resources = status["resources"]
            if resources and resources["cpu_logical_cores"] is not None:
                break
            time.sleep(0.05)
        self.assertIsNotNone(resources)
        self.assertGreater(resources["rss_bytes"], 0)
        self.assertGreaterEqual(resources["anonymous_bytes"], 0)
        self.assertGreaterEqual(resources["file_backed_bytes"], 0)
        self.assertGreater(resources["system_available_ram_bytes"], 0)
        self.assertGreaterEqual(resources["cpu_logical_cores"], 0)
        self.assertIsNone(status["resource_error"])
        self.runtime.stop()
        self.assertIsNone(self.runtime.status()["resources"])

    def test_resource_sampling_failure_is_visible(self):
        self.load()
        with mock.patch.object(Path, "read_text", side_effect=PermissionError("test denied")):
            self.runtime._sample_resources(self.runtime.process)
            self.assertIn("test denied", self.runtime.status()["resource_error"])
            self.assertIsNone(self.runtime.status()["resources"])
            self.assertEqual(self.runtime.status()["state"], "ready")

    def test_invalid_load_inputs(self):
        for values in ({}, {"model": 5}, {"model": "/missing.gguf"},
                       {"model": str(self.model), "context": True},
                       {"model": str(self.model), "context": 511},
                       {"model": str(self.model), "context": 65537},
                       {"model": str(self.model), "placement": "magic"}):
            self.assertEqual(self.request("POST", "/api/v1/model/load", values)[0], 400)
        self.model.write_bytes(b"not-a-model")
        self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": str(self.model)})[0], 400)

    def test_backend_failure_and_recovery(self):
        failed = self.root / "fail.gguf"
        failed.write_bytes(b"GGUF")
        self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": str(failed)})[0], 202)
        status = self.wait_state("error")
        self.assertIn("code 7", status["error"])
        self.load()
        self.assertIsNone(self.runtime.status()["error"])

    def test_loading_is_cancellable(self):
        slow = self.root / "slow.gguf"
        slow.write_bytes(b"GGUF")
        self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": str(slow)})[0], 202)
        process = self.runtime.process
        self.request("POST", "/api/v1/model/unload", {})
        self.assertIsNotNone(process.poll())
        time.sleep(0.3)
        self.assertEqual(self.runtime.status()["state"], "stopped")

    def test_loading_timeout(self):
        slow = self.root / "slow.gguf"
        slow.write_bytes(b"GGUF")
        self.runtime.load_timeout = 0.1
        self.request("POST", "/api/v1/model/load", {"model": str(slow)})
        status = self.wait_state("error")
        self.assertIn("timed out", status["error"])
        self.assertIsNotNone(self.runtime.process.poll())

    def test_backend_crash_after_ready(self):
        self.load()
        self.runtime.process.terminate()
        self.runtime.process.wait(timeout=5)
        self.assertIn("exited", self.wait_state("error")["error"])
        self.assertEqual(self.request("GET", "/v1/models")[0], 503)

    def test_gpu_init_failure_is_not_a_successful_cpu_fallback(self):
        model = self.root / "gpu-fail.gguf"
        model.write_bytes(b"GGUF")
        code, _, _ = self.request("POST", "/api/v1/model/load", {"model": str(model)})
        self.assertEqual(code, 202)
        status = self.wait_state("error")
        self.assertIn("CUDA initialization failed", status["error"])
        self.assertIn("CPU-only", status["error"])
        self.assertEqual(self.request("GET", "/v1/models")[0], 503)
        self.runtime.process.wait(timeout=6)
        self.assertIsNotNone(self.runtime.process.poll())
        self.assertEqual(self.runtime.status()["state"], "error")

    def test_explicit_cpu_mode_does_not_require_cuda(self):
        model = self.root / "gpu-fail.gguf"
        model.write_bytes(b"GGUF")
        status = self.load(model=model, placement="cpu")
        self.assertEqual(status["state"], "ready")
        self.assertIsNone(status["error"])

    def test_chat_and_backend_errors(self):
        self.load()
        payload = {"messages": [{"role": "user", "content": "hello"}], "model": "anything"}
        code, _, body = self.request("POST", "/v1/chat/completions", payload)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["model"], "local-model")
        payload["messages"][0]["content"] = "backend-error"
        code, _, body = self.request("POST", "/v1/chat/completions", payload)
        self.assertEqual(code, 400)
        self.assertIn(b"Context limit", body)

    def test_stream_and_single_request_admission(self):
        self.load()
        payload = {"messages": [{"role": "user", "content": "hello"}], "stream": True}
        code, headers, body = self.request("POST", "/v1/chat/completions", payload)
        self.assertEqual(code, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        self.assertIn(b"data: [DONE]", body)
        self.assertIn(b"<script>not executable</script>", body)
        self.runtime.chat_lock.acquire()
        try:
            self.assertEqual(self.request("POST", "/v1/chat/completions", payload)[0], 409)
        finally:
            self.runtime.chat_lock.release()

    def test_disconnect_cancels_stream(self):
        self.load()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = {"messages": [{"role": "user", "content": "wait"}], "stream": True}
        connection.request("POST", "/v1/chat/completions", json.dumps(payload),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(self.runtime.chat_lock.locked())
        response.close()
        connection.close()
        deadline = time.monotonic() + 2
        while self.runtime.chat_lock.locked() and time.monotonic() < deadline:
            time.sleep(0.03)
        self.assertFalse(self.runtime.chat_lock.locked())

    def test_reasoning_usage_and_timings_are_forwarded(self):
        self.load()
        for prompt in ("reasoning", "reasoning-limit"):
            payload = {
                "messages": [{"role": "user", "content": prompt}], "stream": True,
                "max_tokens": 8192, "stream_options": {"include_usage": True},
                "timings_per_token": True,
            }
            code, _, body = self.request("POST", "/v1/chat/completions", payload)
            self.assertEqual(code, 200)
            events = [json.loads(line[6:]) for line in body.splitlines()
                      if line.startswith(b"data: ") and line != b"data: [DONE]"]
            self.assertEqual(events[0]["choices"][0]["delta"]["reasoning_content"],
                             "Working on the answer.")
            self.assertEqual(events[-1]["usage"]["completion_tokens"], 730)
            self.assertEqual(events[-2]["timings"]["predicted_per_second"], 123.5)
            self.assertEqual(events[-2]["choices"][0]["finish_reason"],
                             "length" if prompt == "reasoning-limit" else "stop")

    def test_host_and_origin_restrictions(self):
        for headers in ({"Host": "attacker.example"},
                        {"Origin": "https://attacker.example"},
                        {"Origin": "null"},
                        {"Sec-Fetch-Site": "cross-site"}):
            self.assertEqual(self.request("GET", "/api/v1/status", headers=headers)[0], 403)
            self.assertEqual(self.request("POST", "/api/v1/model/unload", {}, headers)[0], 403)
        self.assertEqual(self.request("GET", "/api/v1/status",
                                     headers={"Origin": f"http://127.0.0.1:{self.port}"})[0], 200)

    def test_request_validation(self):
        self.assertEqual(self.request("POST", "/api/v1/model/load", [])[0], 400)
        self.assertEqual(self.request("POST", "/api/v1/model/load", {},
                                     {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/v1/model/load", {},
                                     {"Transfer-Encoding": "chunked"})[0], 400)
        self.assertEqual(self.request("POST", "/api/v1/model/load", {"model": "x" * app.BODY_LIMIT})[0], 413)
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"messages": []})[0], 400)
        self.assertEqual(self.request("POST", "/v1/chat/completions",
                                     {"messages": [{}], "stream": "yes"})[0], 400)

    def test_launcher_shuts_down_on_sigterm(self):
        process = subprocess.Popen(
            [sys.executable, str(APP_PATH), "--port", "0", "--no-browser",
             "--llama-server", str(self.backend), "--execution-mode", "standard"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            line = process.stdout.readline()
            self.assertIn("http://127.0.0.1:", line)
            port = urlsplit(line.split(" ", 1)[1].strip()).port
            backend_pid = None
            client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                client.request("POST", "/api/v1/model/load",
                               json.dumps({"model": str(self.model)}),
                               {"Content-Type": "application/json"})
                response = client.getresponse()
                self.assertEqual(response.status, 202)
                response.read()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    client.request("GET", "/api/v1/status")
                    response = client.getresponse()
                    status = json.loads(response.read())
                    if status["state"] == "ready":
                        backend_pid = json.loads(status["logs"][0])["pid"]
                        break
                    time.sleep(0.03)
                self.assertIsNotNone(backend_pid)
            finally:
                client.close()
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stdout + stderr)
            with self.assertRaises(ProcessLookupError):
                os.kill(backend_pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
