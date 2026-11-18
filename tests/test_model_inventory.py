import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "tools/benchmark/vrambridge-model-inventory.py"
spec = importlib.util.spec_from_file_location("model_inventory", SCRIPT)
inventory = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(inventory)


def fixture(path: Path, count: int = 1) -> None:
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    entries = [("general.architecture", 8, string("qwen3")), ("general.name", 8, string("Fixture")),
               ("general.author", 8, string("Example Author")), ("general.license", 8, string("example-license")),
               ("split.count", 4, struct.pack("<I", count)), ("qwen3.context_length", 4, struct.pack("<I", 4096)),
               ("qwen3.expert_count", 4, struct.pack("<I", 16))]
    data = b"".join(string(k) + struct.pack("<I", kind) + value for k, kind, value in entries)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + data)


def source(name, path, mode="gguf", priority=0):
    return {"id": name, "path": str(path), "mount": None, "uuid": None, "mode": mode,
            "priority": priority, "ephemeral": False}


class ModelInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.a, self.b = self.root / "a", self.root / "b"
        self.a.mkdir(); self.b.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_explicit_sources_are_validated_and_normalized(self):
        config = self.root / "sources.json"
        spec = {"id": "archive", "path": str(self.a)}
        config.write_text(json.dumps([spec]))
        extra, = [item for item in inventory.sources(config) if item["id"] == "archive"]
        self.assertEqual(extra, {**source("archive", self.a), "priority": 50})
        for value in ([spec, spec], [{**spec, "path": "relative"}],
                      [{**spec, "mount": str(self.b), "uuid": "ABCD-1234"}],
                      [{**spec, "mount": str(self.root), "uuid": "../invalid"}]):
            with self.subTest(value=value):
                config.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    inventory.sources(config)

    def shards(self, root, *, complete=True):
        fixture(root / "Example-Q4_K_M-00001-of-00002.gguf", 2)
        if complete:
            fixture(root / "Example-Q4_K_M-00002-of-00002.gguf", 2)

    def test_grouped_shards_include_model_metadata_and_copy_caveat(self):
        self.shards(self.a); self.shards(self.b)
        data = inventory.scan([source("preferred", self.a), source("archive", self.b, priority=10)])
        self.assertEqual(len(data["models"]), 1)
        model = data["models"][0]
        self.assertEqual(model["quantization_hint"], "Q4_K_M")
        self.assertEqual(model["preferred_path"], str(self.a / "Example-Q4_K_M-00001-of-00002.gguf"))
        self.assertIn("not assumed", model["identity_basis"])
        self.assertEqual(len(model["locations"]), 2)
        self.assertEqual(model["locations"][0]["metadata"]["context_length"], 4096)
        self.assertEqual(model["locations"][0]["metadata"]["author"], "Example Author")
        self.assertEqual(model["locations"][0]["metadata"]["license"], "example-license")
        self.assertEqual(model["locations"][0]["metadata"]["expert_count"], 16)
        self.assertEqual(model["locations"][0]["expected_shards"], 2)

    def test_incomplete_or_corrupt_shards_are_not_available(self):
        self.shards(self.a, complete=False)
        (self.b / "bad.gguf").write_bytes(b"invalid")
        data = inventory.scan([source("a", self.a), source("b", self.b)])
        by_name = {m["name"]: m for m in data["models"]}
        self.assertEqual(by_name["Example-Q4_K_M"]["locations"][0]["missing_shards"], [2])
        self.assertIsNone(by_name["Example-Q4_K_M"]["preferred_path"])
        self.assertEqual(by_name["bad.gguf"]["locations"][0]["availability"], "incomplete_or_unreadable")

    def test_hardlinked_models_are_reported_as_shared_data(self):
        first = self.a / "first.gguf"
        second = self.b / "second.gguf"
        fixture(first)
        os.link(first, second)
        data = inventory.scan([source("a", self.a), source("b", self.b)])
        self.assertEqual(len(data["hardlinks"]), 1)
        self.assertEqual(set(data["hardlinks"][0]["paths"]), {str(first), str(second)})
        self.assertIn("in-place write", data["hardlinks"][0]["warning"])

    def test_offline_sources_and_user_notes_are_preserved(self):
        self.shards(self.a)
        spec_a = source("a", self.a)
        first = inventory.scan([spec_a])
        first["notes"] = "my library"
        first["user_metadata"] = {"owner": "me"}
        first["models"][0]["notes"] = "important"
        first["models"][0]["user_metadata"] = {"rating": 5}
        with mock.patch.object(inventory, "mount_available", return_value=False):
            second = inventory.scan([spec_a], first)
        self.assertEqual(second["notes"], "my library")
        self.assertEqual(second["user_metadata"], {"owner": "me"})
        self.assertEqual(second["models"][0]["notes"], "important")
        self.assertEqual(second["models"][0]["user_metadata"], {"rating": 5})
        self.assertEqual(second["models"][0]["locations"][0]["availability"], "offline")
        self.assertFalse(second["models"][0]["locations"][0]["files"][0]["available"])
        self.assertIsNone(second["models"][0]["preferred_path"])

    def test_removed_source_is_retained_as_untracked(self):
        self.shards(self.a)
        first = inventory.scan([source("a", self.a)])
        first["models"][0]["notes"] = "keep"
        second = inventory.scan([], first)
        self.assertEqual(second["sources"][0]["status"], "untracked")
        self.assertEqual(second["models"][0]["locations"][0]["availability"], "untracked")
        self.assertEqual(second["models"][0]["notes"], "keep")

    def test_gguf_symlink_outside_online_roots_is_not_followed(self):
        outside = self.b / "outside.gguf"
        fixture(outside)
        (self.a / "escape.gguf").symlink_to(outside)
        data = inventory.scan([source("a", self.a)])
        self.assertEqual(data["models"], [])
        self.assertEqual(data["sources"][0]["status"], "partial")
        self.assertIn("outside known", data["sources"][0]["warnings"][0])

    def test_annotations_for_replaced_model_are_not_discarded(self):
        path = self.a / "model.gguf"
        fixture(path)
        first = inventory.scan([source("a", self.a)])
        first["models"][0]["notes"] = "old variant"
        with path.open("ab") as stream:
            stream.write(b"changed size")
        second = inventory.scan([source("a", self.a)], first)
        self.assertEqual(len(second["models"]), 2)
        old = next(m for m in second["models"] if m.get("notes") == "old variant")
        self.assertEqual(old["locations"][0]["availability"], "replaced")
        self.assertIsNone(old["preferred_path"])

    def test_huggingface_cache_does_not_follow_external_symlinks(self):
        repo = self.a / "models--owner--model"
        blobs, snapshot = repo / "blobs", repo / "snapshots" / "revision"
        blobs.mkdir(parents=True); snapshot.mkdir(parents=True)
        (blobs / "abc").write_bytes(b"weights")
        (blobs / "config").write_text(json.dumps({"model_type": "whisper", "hidden_size": 512}))
        (snapshot / "model.safetensors").symlink_to("../../blobs/abc")
        (snapshot / "config.json").symlink_to("../../blobs/config")
        outside = self.b / "outside.bin"; outside.write_bytes(b"outside")
        (snapshot / "escape.bin").symlink_to(outside)
        data = inventory.scan([source("cache", self.a, "hf")])
        self.assertEqual(len(data["models"]), 1)
        files = data["models"][0]["locations"][0]["files"]
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0]["is_symlink"])
        self.assertEqual(data["models"][0]["locations"][0]["metadata"]["model_type"], "whisper")
        self.assertEqual(data["sources"][0]["status"], "partial")

    def test_colibri_reports_metadata_but_not_unverified_completeness(self):
        (self.a / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa", "num_hidden_layers": 78}))
        (self.a / "out-00000.safetensors").write_bytes(b"weights")
        data = inventory.scan([source("colibri", self.a, "colibri")])
        meta = data["models"][0]["locations"][0]["metadata"]
        self.assertEqual(meta["model_type"], "glm_moe_dsa")
        self.assertEqual(meta["weight_completeness"], "not independently validated")

    def test_safetensors_model_directory_is_discovered_without_gguf(self):
        folder = self.a / "new-model"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"_name_or_path": "owner/new-model", "model_type": "llama"}))
        (folder / "model-00001-of-00002.safetensors").write_bytes(b"first")
        (folder / "model-00002-of-00002.safetensors").write_bytes(b"second")
        data = inventory.scan([source("a", self.a)])
        model = data["models"][0]
        self.assertEqual(model["name"], "owner/new-model")
        self.assertEqual(model["locations"][0]["metadata"]["model_type"], "llama")
        self.assertEqual(len(model["locations"][0]["files"]), 2)
        self.assertEqual(model["locations"][0]["availability"], "present_unverified")

    def test_manifest_hash_is_attributed_only_when_size_and_mtime_match(self):
        fixture(self.a / "model.gguf")
        file = self.a / "model.gguf"
        manifests = self.root / "manifests"; manifests.mkdir()
        data = {"kind": "verified_model_consolidation", "models": [{"source": str(file), "destination": str(file),
            "verified_log_mtime_ns": 1_700_000_000_000_000_000, "files": [{"relative_path": "model.gguf",
            "sha256": "a" * 64, "bytes": file.stat().st_size, "source_mtime_ns": file.stat().st_mtime_ns}]}]}
        (manifests / "consolidation-test.json").write_text(json.dumps(data))
        hashes, warnings = inventory.known_hashes(manifests)
        self.assertFalse(warnings)
        self.assertEqual(inventory.entry(file, hashes)["sha256"], "a" * 64)
        os.utime(file, ns=(file.stat().st_atime_ns, file.stat().st_mtime_ns + 100000))
        self.assertNotIn("sha256", inventory.entry(file, hashes))

    def test_refuses_unmanaged_files_and_concurrent_changes(self):
        output = self.root / "available_local_models.json"
        output.write_text('{"user":"preserve"}')
        with self.assertRaisesRegex(ValueError, "unmanaged"):
            inventory.read_previous(output)
        self.assertEqual(json.loads(output.read_text()), {"user": "preserve"})
        output.unlink()
        with mock.patch.object(inventory, "NVME_LIBRARY_ROOT", self.root), \
                mock.patch.object(inventory, "require_nvme_mount"), mock.patch.object(inventory, "RESERVE", 0), \
                mock.patch.object(inventory, "HEADROOM", 0):
            data = inventory.scan([])
            inventory.publish(output, data, None)
            previous, fingerprint = inventory.read_previous(output)
            output.write_text(output.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "changed during"):
                inventory.publish(output, previous, fingerprint)
            self.assertTrue(output.read_text().endswith(" "))

    def test_initial_publication_never_replaces_existing_file(self):
        output = self.root / "available_local_models.json"
        output.write_text("preserve")
        with mock.patch.object(inventory, "NVME_LIBRARY_ROOT", self.root), \
                mock.patch.object(inventory, "require_nvme_mount"), mock.patch.object(inventory, "RESERVE", 0), \
                mock.patch.object(inventory, "HEADROOM", 0):
            with self.assertRaises(FileExistsError):
                inventory.publish(output, inventory.scan([]), None)
        self.assertEqual(output.read_text(), "preserve")
        self.assertFalse(list(self.root.glob(".available-models-*.tmp")))


if __name__ == "__main__":
    unittest.main()
