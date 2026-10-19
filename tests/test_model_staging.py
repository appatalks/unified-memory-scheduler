#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(sys.argv.pop(1)).resolve()
spec = importlib.util.spec_from_file_location("model_staging", SCRIPT)
assert spec and spec.loader
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


def fixture(path: Path, count: int = 1) -> None:
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded
    metadata = string("general.architecture") + struct.pack("<I", 8) + string("test")
    metadata += string("split.count") + struct.pack("<II", 4, count)
    path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 0, 2) + metadata + b"fixture" * 64)


class ModelStagingTests(unittest.TestCase):
    def setUp(self):
        self.space = mock.patch.object(
            staging.shutil, "disk_usage", return_value=type("Usage", (), {"free": 64 * 1024 ** 3})())
        self.space.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.home = Path(self.temporary.name)
        self.root = self.home / "stage"
        self.archive = self.home / "archive"
        self.archive.mkdir()
        self.first = self.archive / "tiny-00001-of-00002.gguf"
        self.second = self.archive / "tiny-00002-of-00002.gguf"
        fixture(self.first, 2)
        fixture(self.second, 2)
        staging.initialize(self.root)

    def tearDown(self):
        self.temporary.cleanup()
        self.space.stop()

    def copied(self):
        with mock.patch.object(staging, "same_filesystem", return_value=False):
            return staging.stage(self.root, self.first)

    def test_complete_verified_copy_reuse_and_eviction_preserve_originals(self):
        report = self.copied()
        cache_id = report["cache_id"]
        self.assertEqual(report["status"], "staged")
        self.assertEqual(staging.verify(self.root, cache_id)["status"], "verified")
        with mock.patch.object(staging, "same_filesystem", return_value=False):
            self.assertEqual(staging.stage(self.root, self.first)["status"], "verified")
        with mock.patch.object(staging.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
            self.assertEqual(staging.evict(self.root, cache_id, True)["status"], "evicted")
        self.assertTrue(self.first.is_file())
        self.assertTrue(self.second.is_file())
        self.assertFalse((self.root / cache_id).exists())

    def test_existing_root_filesystem_copy_is_not_duplicated(self):
        result = staging.stage(self.root, self.first)
        self.assertEqual(result["status"], "reuse_existing_copy")
        self.assertEqual(result["copied_bytes"], 0)
        self.assertEqual(staging.listing(self.root)["entries"], [])

    @unittest.skipUnless(shutil.which("fuser"), "requires the Linux fuser utility")
    def test_real_open_file_blocks_eviction(self):
        report = self.copied()
        with Path(report["model_path"]).open("rb"):
            with self.assertRaisesRegex(RuntimeError, "in use"):
                staging.evict(self.root, report["cache_id"], True)
        self.assertEqual(staging.evict(self.root, report["cache_id"], True)["status"], "evicted")
        self.assertTrue(self.first.is_file())
        self.assertTrue(self.second.is_file())

    def test_missing_or_nonfirst_shard_is_rejected(self):
        with self.assertRaises((ValueError, staging.AppError)):
            staging.plan(self.root, self.second)
        self.second.unlink()
        with self.assertRaises(staging.AppError):
            staging.plan(self.root, self.first)

    def test_space_failure_creates_no_partial_copy(self):
        required = self.first.stat().st_size + self.second.stat().st_size
        boundary = staging.RESERVE + staging.COPY_OVERHEAD + required
        with mock.patch.object(staging, "same_filesystem", return_value=False), \
                mock.patch.object(staging.shutil, "disk_usage",
                                  return_value=type("Usage", (), {"free": boundary - 1})()):
            with self.assertRaisesRegex(ValueError, "16 GiB"):
                staging.stage(self.root, self.first)
        self.assertEqual(staging.listing(self.root)["entries"], [])
        self.assertFalse(list(self.root.glob(".partial-*")))
        with mock.patch.object(staging, "same_filesystem", return_value=False), \
                mock.patch.object(staging.shutil, "disk_usage",
                                  return_value=type("Usage", (), {"free": boundary})()):
            self.assertEqual(staging.stage(self.root, self.first)["status"], "staged")

    def test_failed_readback_or_changed_source_cleans_only_partial_copy(self):
        for failure in ("readback", "source"):
            with self.subTest(failure=failure), mock.patch.object(staging, "same_filesystem", return_value=False):
                if failure == "readback":
                    patch = mock.patch.object(staging, "sha256", return_value="0" * 64)
                else:
                    before = staging.fingerprint(self.first)
                    patch = mock.patch.object(staging, "fingerprint", side_effect=[before, (*before[:3], before[3] + 1)])
                with patch, self.assertRaises(ValueError):
                    staging.stage(self.root, self.first)
                self.assertFalse(list(self.root.glob(".partial-*")))
                self.assertTrue(self.first.is_file())
                self.assertTrue(self.second.is_file())

    def test_changed_archive_or_staged_symlink_blocks_eviction(self):
        report = self.copied()
        self.first.write_bytes(self.first.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Archive original"):
            staging.evict(self.root, report["cache_id"], True)
        target = Path(report["model_path"])
        target.unlink()
        target.symlink_to(self.first)
        with self.assertRaisesRegex(ValueError, "symlink"):
            staging.load_entry(self.root, report["cache_id"])
        self.assertTrue(self.first.is_file())

    def test_earlier_source_change_during_later_copy_blocks_publication(self):
        original_hash = staging.sha256

        def change_earlier(path):
            if path.name == self.second.name:
                self.first.write_bytes(self.first.read_bytes() + b"changed")
            return original_hash(path)

        with mock.patch.object(staging, "same_filesystem", return_value=False), \
                mock.patch.object(staging, "sha256", side_effect=change_earlier):
            with self.assertRaisesRegex(ValueError, "earlier archive shard"):
                staging.stage(self.root, self.first)
        self.assertEqual(staging.listing(self.root)["entries"], [])
        self.assertFalse(list(self.root.glob(".partial-*")))
        self.assertTrue(self.first.is_file())
        self.assertTrue(self.second.is_file())

    def test_confirmation_busy_files_and_unknown_entries_are_not_deleted(self):
        report = self.copied()
        with self.assertRaisesRegex(ValueError, "confirm-unloaded"):
            staging.evict(self.root, report["cache_id"], False)
        with mock.patch.object(staging.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "123", "")) as fuser:
            with self.assertRaisesRegex(RuntimeError, "in use"):
                staging.evict(self.root, report["cache_id"], True)
        self.assertEqual(fuser.call_args.args[0][:2], ["fuser", "-I"])
        extra = self.root / report["cache_id"] / "not-owned.txt"
        extra.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            staging.evict(self.root, report["cache_id"], True)
        self.assertEqual(extra.read_text(), "preserve")

    def test_unmarked_directory_and_manifest_path_escape_are_rejected(self):
        foreign = self.home / "foreign"
        foreign.mkdir()
        (foreign / "keep").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            staging.initialize(foreign)
        report = self.copied()
        manifest_path = self.root / report["cache_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][0]["name"] = "../escape.gguf"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            staging.load_entry(self.root, report["cache_id"])


if __name__ == "__main__":
    unittest.main()
