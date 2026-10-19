#!/usr/bin/env python3

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BINARY, CUDA, LIBRARY, BASE = (str(Path(sys.argv.pop(1)).resolve()) for _ in range(4))


class ColdKvGpuTests(unittest.TestCase):
    def test_roundtrip_selectivity_and_owner_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "kv.json"
            env = {k: v for k, v in os.environ.items()
                   if k != "LD_PRELOAD" and not k.startswith(("GGML_", "VRAMBRIDGE_"))}
            env.update(LD_PRELOAD=LIBRARY, VRAMBRIDGE_KV_COLD_MODE="stage",
                       VRAMBRIDGE_KV_COLD_BASE_LIBRARY=BASE, VRAMBRIDGE_KV_COLD_OUT=str(report))
            result = subprocess.run([BINARY, CUDA], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = json.loads(result.stdout)
            metrics = json.loads(report.read_text())
            self.assertTrue(observed["roundtrip"] and observed["weights_remained_device"])
            self.assertTrue(metrics["complete"])
            self.assertEqual(metrics["kv_allocations"], 1)
            self.assertEqual(metrics["managed_allocation_calls"], 1)
            self.assertEqual(metrics["logical_buffer_bytes"], observed["kv_buffer_bytes"])
            self.assertEqual(metrics["cpu_stage_requests"], 1)
            self.assertEqual(metrics["buffers_still_tracked"], 0)


if __name__ == "__main__":
    unittest.main()
