#!/usr/bin/env python3

import ctypes
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest


BINARY, PROFILE, VNNI, CPU, HINT = (Path(sys.argv.pop(1)).resolve() for _ in range(5))


class CpuResearchTests(unittest.TestCase):
    def test_integer_replacement_preserves_native_dot_bits(self):
        native = ctypes.CDLL(str(CPU))
        native.ggml_cpu_init()
        candidate = ctypes.CDLL(str(VNNI))
        signature = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                     ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        functions = [native.ggml_vec_dot_mxfp4_q8_0, candidate.ggml_vec_dot_mxfp4_q8_0]
        for function in functions:
            function.argtypes, function.restype = signature, None
        rng = random.Random(4217)
        for width in (64, 128, 2880):
            for sample in range(80):
                weights, activation = bytearray(), bytearray()
                for block in range(width // 32):
                    weights += bytes([rng.choice([0, 1, 2, 110, 120, 126, 127, 128, 130])])
                    weights += rng.randbytes(16)
                    scale = rng.choice([0, 1, 0x0400, 0x1000, 0x3c00, 0x3c01, 0x5000, 0x8000])
                    activation += struct.pack("<H", scale)
                    activation += bytes(rng.randrange(256) for _ in range(32)) if sample else bytes(32)
                x, y = ctypes.create_string_buffer(bytes(weights)), ctypes.create_string_buffer(bytes(activation))
                results = []
                for function in functions:
                    output = ctypes.c_float()
                    function(width, ctypes.byref(output), 0, x, 0, y, 0, 1)
                    self.assertTrue(math.isfinite(output.value))
                    results.append(bytes(output))
                self.assertEqual(results[0], results[1], (width, sample))

    def test_native_cpu_trait_really_binds_preloaded_candidate(self):
        script = """
import ctypes,sys
class Trait(ctypes.Structure):
    _fields_=[('from_float',ctypes.c_void_p),('vec_dot',ctypes.c_void_p)]
cpu=ctypes.CDLL(sys.argv[1])
cpu.ggml_get_type_traits_cpu.argtypes=[ctypes.c_int]
cpu.ggml_get_type_traits_cpu.restype=ctypes.POINTER(Trait)
actual=cpu.ggml_get_type_traits_cpu(39).contents.vec_dot
candidate=ctypes.cast(ctypes.CDLL(None).ggml_vec_dot_mxfp4_q8_0,ctypes.c_void_p).value
original=ctypes.cast(cpu.ggml_vec_dot_mxfp4_q8_0,ctypes.c_void_p).value
assert actual==candidate and actual!=original
print('native trait bound to candidate')
"""
        result = subprocess.run([sys.executable, "-c", script, str(CPU)],
                                env={**os.environ, "LD_PRELOAD": str(VNNI)},
                                capture_output=True, text=True, check=True, timeout=20)
        self.assertIn("bound to candidate", result.stdout)

    def check_graph_profile(self, expert_ids):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory).chmod(0o700)
            output = Path(directory) / "profile.jsonl"
            result = subprocess.run(
                [str(BINARY), "--self-test", "--threads", "2", "--rows", "1", "--samples", "1", "--repetitions", "1"],
                env={**os.environ, "LD_PRELOAD": str(PROFILE), "VRAMBRIDGE_CPU_PROFILE_LIBRARY": str(CPU),
                     "VRAMBRIDGE_CPU_PROFILE_OUT": str(output),
                     "VRAMBRIDGE_CPU_PROFILE_EXPERT_IDS": "1" if expert_ids else "0"},
                capture_output=True, text=True, check=True, timeout=30)
            self.assertTrue(json.loads(result.stdout)["all_checks_passed"])
            self.assertIn("profiler active", result.stderr)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertGreater(len(records), 0)
            self.assertTrue(any(r["matmuls"] for r in records))
            for record in records:
                self.assertEqual(record["schema_version"], 3)
                self.assertEqual(record["contains_sensitive_routing"], expert_ids)
                self.assertEqual(record["status"], 0)
                self.assertGreater(record["elapsed_ns"], 0)
                self.assertEqual(record["end_ns"] - record["start_ns"], record["elapsed_ns"])
                for matrix in record["matmuls"]:
                    self.assertNotIn("ids", matrix)
                    self.assertNotIn("values", matrix)
                    if matrix["cold_experts"] is not None:
                        self.assertLessEqual(matrix["cold_experts"], matrix["active_slots"])
                        self.assertGreaterEqual(matrix["logical_weight_bytes"], 0)
                        if expert_ids:
                            self.assertEqual(len(matrix["cold_ids"]), matrix["cold_experts"])
                            self.assertTrue(all(0 <= identifier < matrix["expert_bank_size"]
                                                for identifier in matrix["cold_ids"]))
                            self.assertTrue(all(type(key) is int and key > 0 for key in matrix["id_source_keys"]))
                        else:
                            self.assertNotIn("cold_ids", matrix)

    def test_graph_profiler_observes_without_changing_graph(self):
        self.check_graph_profile(False)

    def test_private_expert_ids_preserve_math_and_accounting(self):
        self.check_graph_profile(True)

    def test_dispatcher_hint_is_captured_before_admission_mutates_map(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory).chmod(0o700)
            output = Path(directory) / "profile.jsonl"
            subprocess.run([str(HINT)], env={**os.environ, "LD_PRELOAD": str(PROFILE),
                "VRAMBRIDGE_CPU_PROFILE_LIBRARY": str(CPU), "VRAMBRIDGE_CPU_PROFILE_OUT": str(output),
                "VRAMBRIDGE_CPU_PROFILE_EXPERT_IDS": "1"},
                capture_output=True, text=True, check=True, timeout=30)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            hints = [node["early_route"] for record in records for node in record["nodes"]
                     if "early_route" in node]
            self.assertEqual(len(hints), 1)
            self.assertTrue(hints[0]["observable"])
            self.assertEqual(hints[0]["selected_ids"], [0, 2])
            self.assertEqual(hints[0]["pre_admission_misses"], [2])


if __name__ == "__main__":
    unittest.main()
