import importlib.util
import mmap
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


path = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-row-stream.py"
spec = importlib.util.spec_from_file_location("row_stream", path)
rows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rows)


class RowStreamTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "rows.bin"
        self.data = bytes(index % 251 for index in range(65536))
        self.path.write_bytes(self.data)

    def tearDown(self):
        self.directory.cleanup()

    def store(self, **options):
        return rows.RowStore(self.path, row_bytes=144, tensor_offset=416, tensor_rows=400, **options)

    def test_cross_page_duplicate_order_and_exact_bytes(self):
        requests = [26, 1, 26, 397, 0]
        with self.store() as store:
            request = store.submit(requests)
            expected = [self.data[416 + row * 144:416 + (row + 1) * 144] for row in requests]
            self.assertEqual(request.gather(), expected)
            self.assertLessEqual(request.miss_pages, 2 * len(set(requests)))
            with self.assertRaises(RuntimeError):
                request.gather()
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_request_and_payload_bounds(self):
        with self.store(cache_pages=1, max_rows=2, max_pending_pages=2) as store:
            with self.assertRaises(ValueError):
                store.submit([0, 1, 2])
            first = store.submit([1])
            with self.assertRaises(RuntimeError):
                store.submit([2])
            first.gather()
            store.submit([100]).gather()
            self.assertEqual(len(store.cache), 1)
            self.assertEqual(store.payload_memory_bound, (1 + 2 + 8) * mmap.PAGESIZE + 4 * 144)

    def test_hits_do_not_submit_more_io(self):
        with self.store() as store:
            store.submit([0, 1, 26]).gather()
            count = store.submitted_bytes
            pending = store.submit([26, 0])
            self.assertGreater(pending.hit_pages, 0)
            self.assertEqual(pending.miss_pages, 0)
            pending.gather()
            self.assertEqual(store.submitted_bytes, count)

    def test_failed_read_surfaces_and_releases_request(self):
        with self.store() as store, mock.patch.object(store, "_read_page", side_effect=OSError("I/O failure")):
            with self.assertRaisesRegex(OSError, "I/O failure"):
                store.submit([0]).gather()
            self.assertIsNone(store.pending)
            self.assertEqual(len(store.cache), 0)

    def test_short_read_is_not_a_zero_filled_success(self):
        with self.store() as store, mock.patch.object(rows.os, "pread", return_value=b"x"):
            with self.assertRaisesRegex(RuntimeError, "Short positional read"):
                store.submit([0]).gather()

    def test_invalid_rows_and_changed_file(self):
        with self.store() as store:
            for ids in ([-1], [400], [False], [1.5]):
                with self.assertRaises(ValueError):
                    store.submit(ids)
            self.path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                store.submit([0])

    def test_out_of_file_tensor_is_rejected(self):
        with self.assertRaises(ValueError):
            rows.RowStore(self.path, row_bytes=4096, tensor_offset=0, tensor_rows=100)

    def test_page_merge(self):
        self.assertEqual(rows.merge_pages([8192, 0, 4096, 0, 16384], 4096), [(0, 12288), (16384, 20480)])

    def test_close_drains_and_invalidates_pending_work(self):
        store = self.store()
        pending = store.submit([0, 26, 397])
        store.close()
        self.assertTrue(store.closed)
        self.assertTrue(pending.finished)
        self.assertEqual(pending.pages, {})
        with self.assertRaises(RuntimeError):
            pending.gather()
        with self.assertRaises(RuntimeError):
            store.submit([0])
        store.close()

    @unittest.skipUnless(hasattr(os, "O_DIRECT"), "Linux direct-I/O test")
    def test_aligned_direct_reads_match_original_rows(self):
        with self.store(direct=True) as store:
            request = [0, 26, 397]
            self.assertEqual(store.submit(request).gather(),
                             [self.data[416 + row * 144:416 + (row + 1) * 144] for row in request])


if __name__ == "__main__":
    unittest.main()
