#!/usr/bin/env python3

import json
import os
import pathlib
import subprocess
import sys
import tempfile


with tempfile.TemporaryDirectory() as temporary_directory:
    root = pathlib.Path(temporary_directory)
    model = root / "model.gguf"
    model.write_bytes(b"GGUF-test-model")
    fake_bench = root / "llama-bench"
    fake_bench.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "json.dump([{'args': sys.argv[1:], 'unified': os.getenv('GGML_CUDA_ENABLE_UNIFIED_MEMORY')}], sys.stdout)\n",
        encoding="utf-8",
    )
    fake_bench.chmod(0o755)
    output_dir = root / "results"

    completed = subprocess.run(
        [
            sys.executable,
            sys.argv[1],
            str(model),
            "--llama-bench", str(fake_bench),
            "--output-dir", str(output_dir),
            "--upstream-commit", "deadbeef",
            "--no-telemetry",
            "--config", "cpu",
            "--config", "auto-fit",
            "--config", "unified-memory",
            "--config", "tensor-placement",
            "--tensor-override", "blk\\.0\\..*=CPU",
            "--config", "cpu-moe",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result_path = pathlib.Path(completed.stdout.strip())
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["backend"]["llama_cpp_commit"] == "deadbeef"
    assert [run["name"] for run in report["runs"]] == [
        "cpu", "auto-fit", "unified-memory", "tensor-placement", "cpu-moe"
    ]
    assert all(run["status"] == "ok" for run in report["runs"])
    assert report["runs"][0]["measurements"][0]["args"][-4:] == ["-ngl", "0", "-dev", "none"]
    assert report["runs"][2]["measurements"][0]["args"][-2:] == ["-ngl", "-1"]
    assert report["runs"][2]["measurements"][0]["unified"] == "1"
    assert report["runs"][3]["measurements"][0]["args"][-4:] == [
        "-ngl", "-1", "-ot", "blk\\.0\\..*=CPU"
    ]
    assert report["runs"][4]["measurements"][0]["args"][-4:] == ["-ngl", "-1", "-ncmoe", "999"]