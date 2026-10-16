#!/usr/bin/env python3

import json
import math
import os
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

for threads in sorted({1, min(2, os.cpu_count() or 1)}):
    completed = subprocess.run(
        [sys.argv[1], "--json", "--size-mib", "1", "--iterations", "2", "--ram-threads", str(threads)],
        check=True, capture_output=True, text=True,
    )
    measurements = json.loads(completed.stdout)["benchmarks"]
    ram = {item["name"]: item for item in measurements if item["name"].startswith("ram_")}
    assert ram["ram_sequential_read"]["cpu_threads"] == 1
    assert ram["ram_sequential_write"]["cpu_threads"] == 1
    assert ("ram_parallel_read" in ram) == (threads > 1)
    if threads > 1:
        measurement = ram["ram_parallel_read"]
        assert measurement["cpu_threads"] == threads
        assert measurement["bytes"] == 1024**2
        assert measurement["iterations"] == 2
        assert measurement["available"] is True
        assert measurement["seconds"] > 0
        assert math.isclose(measurement["gb_per_second"],
                            measurement["bytes"] * measurement["iterations"] / measurement["seconds"] / 1e9,
                            rel_tol=1e-8)

for value in ("0", "-1", "1.5", "2garbage", str((os.cpu_count() or 1) + 1)):
    invalid = subprocess.run([sys.argv[1], "--json", "--size-mib", "1", "--ram-threads", value],
                             capture_output=True, text=True)
    assert invalid.returncode == 2
    assert "RAM threads" in invalid.stderr