#!/usr/bin/env python3

import json
import subprocess
import sys


completed = subprocess.run(
    [sys.argv[1], "--json", "--no-bench"],
    check=True,
    capture_output=True,
    text=True,
)
report = json.loads(completed.stdout)
assert report["schema_version"] == 1
assert report["system"]["logical_cpu_count"] > 0
assert report["system"]["ram_total_bytes"] > 0
assert isinstance(report["cuda"]["devices"], list)
assert report["benchmarks"] == []