import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


path = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-row-stream-study.py"
spec = importlib.util.spec_from_file_location("row_stream_study", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class RowStreamStudyTests(unittest.TestCase):
    def test_fixture_reference_matches_cross_page_and_non_repeating_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "rows.bin"
            study.write_fixture(fixture, 16 * study.PAGE, study.FIXTURE_SEED)
            data = fixture.read_bytes()
            for row in (0, 1, 25, 26, 397):
                begin = study.TENSOR_OFFSET + row * study.ROW_BYTES
                self.assertEqual(study.expected_row(row, study.FIXTURE_SEED),
                                 data[begin:begin + study.ROW_BYTES])
            self.assertNotEqual(data[:study.PAGE], data[study.PAGE:2 * study.PAGE])
            with self.assertRaises(FileExistsError):
                study.write_fixture(fixture, 16 * study.PAGE, study.FIXTURE_SEED)

    def test_reserve_failure_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "rows.bin"
            usage = mock.Mock(free=study.RESERVE_BYTES)
            with mock.patch.object(study.shutil, "disk_usage", return_value=usage):
                with self.assertRaisesRegex(RuntimeError, "16 GiB"):
                    study.write_fixture(fixture, 16 * study.PAGE, study.FIXTURE_SEED)
            self.assertFalse(fixture.exists())

    def test_request_stream_is_matched_with_explicit_warmup_and_reuse(self):
        first = study.request_stream(4096, 4, 17)
        self.assertEqual(first, study.request_stream(4096, 4, 17))
        self.assertEqual(len(first), 16)
        self.assertEqual(sum(not warm for _, warm, _ in first), 8)
        for index in range(0, len(first), 2):
            ids, warmup, reused = first[index]
            self.assertEqual(len(set(ids)), 48)
            self.assertFalse(reused)
            self.assertEqual(first[index + 1], (ids, warmup, True))

    def test_mmap_and_upstream_reference_validate_exact_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "rows.bin"
            study.write_fixture(fixture, 64 * study.PAGE, study.FIXTURE_SEED)
            for method in ("mmap", "upstream-madvise"):
                result = study.run_case(fixture, method, 4, 20261102, 2, 64)
                self.assertTrue(result["all_rows_byte_equal"])
                self.assertEqual(len(result["samples"]), 8)
                self.assertEqual(result["novel_ids"]["logical_row_bytes"], 4 * 48 * 144)
                self.assertEqual(result["immediate_reuse"]["requests"], 4)

    def test_missing_physical_reads_invalidates_direct_control(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "rows.bin"
            study.write_fixture(fixture, 64 * study.PAGE, study.FIXTURE_SEED)
            interval = {"physical_read_bytes": 0, "minor_faults": 0, "major_faults": 0}
            with mock.patch.object(study.io, "interval", return_value=interval):
                with self.assertRaisesRegex(RuntimeError, "No physical I/O"):
                    study.run_case(fixture, "direct-concurrent", 4, 20261102, 2, 64)

    def test_direct_methods_request_direct_io_and_preserve_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "rows.bin"
            study.write_fixture(fixture, 64 * study.PAGE, study.FIXTURE_SEED)
            real_store = study.rows.RowStore
            interval = {"physical_read_bytes": study.PAGE, "minor_faults": 0, "major_faults": 0}
            for method in ("direct-serial", "direct-concurrent", "direct-cache"):
                with self.subTest(method=method), \
                        mock.patch.object(study.io, "interval", return_value=interval), \
                        mock.patch.object(study.rows, "RowStore", wraps=real_store) as constructor:
                    result = study.run_case(fixture, method, 4, 20261102, 2, 64)
                    self.assertIs(constructor.call_args.kwargs["direct"], True)
                    self.assertTrue(result["all_rows_byte_equal"])
                    self.assertGreater(result["physical_reads_including_warmup"], 0)
                    if method == "direct-cache":
                        self.assertEqual(result["immediate_reuse"]["submitted_page_bytes"], 0)
                        self.assertEqual(result["immediate_reuse"]["explicit_cache_page_hit_fraction"], 1)

    def test_cli_invalid_bounds_do_not_write_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            for extra in (["--fixture-mib", "512"], ["--requests", "0"], ["--workers", "0"],
                          ["--trials", "6"], ["--cache-pages", "0"]):
                result = subprocess.run(
                    [sys.executable, str(path), "--fixture-dir", directory, "--output", str(output), *extra],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Invalid bounded", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
