#!/usr/bin/env python3
"""Refresh a bounded local model inventory without moving or downloading weights."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrambridge.app import model_metadata
from vrambridge.model_storage import (
    DEVICE_UUID, NVME_LIBRARY_ROOT, NVME_MODELS_ROOT, NVME_MOUNT, NVME_STAGING_ROOT,
    require_nvme_mount,
)

MANAGED_BY = "vrambridge-local-model-inventory"
RESERVE = 16 * 1024**3
HEADROOM = 16 * 1024**2
MAX_JSON = 16 * 1024**2
MAX_ENTRIES = 200000
MAX_DEPTH = 64
SPLIT = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
QUANT = re.compile(r"(?:^|[-_])((?:UD-)?(?:IQ[1-9](?:_[A-Z0-9]+)+|Q[1-9](?:_[A-Z0-9]+)*|MXFP\d+|BF16|F16))(?=\.gguf$|-[0-9]{5}-of-)", re.IGNORECASE)
WEIGHT_EXTENSIONS = (".gguf", ".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".onnx")
MIN_ASSET_BYTES = 1
GGUF_INFO_KEYS = {"general.author", "general.organization", "general.license", "general.size_label",
                  "general.basename", "general.finetune", "general.description"}

def sources(config: Path | None = None) -> list[dict]:
    result = [
        {"id": "nvme_models", "path": str(NVME_MODELS_ROOT), "mode": "gguf", "priority": 0, "ephemeral": False},
        {"id": "nvme_staging", "path": str(NVME_STAGING_ROOT), "mode": "gguf", "priority": 5, "ephemeral": True},
    ]
    if config is None:
        return result
    with config.open("rb") as stream:
        encoded = stream.read(1024**2 + 1)
    if len(encoded) > 1024**2:
        raise ValueError("Inventory source configuration exceeds 1 MiB.")
    extra = json.loads(encoded)
    if not isinstance(extra, list) or len(extra) > 64:
        raise ValueError("Inventory source configuration must be a list of at most 64 sources.")
    seen = {spec["id"] for spec in result}
    for spec in extra:
        if not isinstance(spec, dict) or set(spec) - {"id", "path", "mode", "mount", "uuid", "priority", "ephemeral"}:
            raise ValueError("Invalid inventory source fields.")
        name, path = spec.get("id"), spec.get("path")
        mode, priority, ephemeral = spec.get("mode", "gguf"), spec.get("priority", 50), spec.get("ephemeral", False)
        mount, uuid = spec.get("mount"), spec.get("uuid")
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_-]{1,64}", name) or name in seen or
                not isinstance(path, str) or not Path(path).is_absolute() or "\0" in path or
                mode not in ("gguf", "colibri", "media", "hf", "strata") or type(priority) is not int or
                not 0 <= priority <= 10000 or type(ephemeral) is not bool):
            raise ValueError("Invalid or duplicate inventory source.")
        if ((mount is None) != (uuid is None) or (mount is not None and
                (not isinstance(mount, str) or not Path(mount).is_absolute() or "\0" in mount or
                 not isinstance(uuid, str) or not DEVICE_UUID.fullmatch(uuid) or
                 not Path(os.path.abspath(path)).is_relative_to(mount)))):
            raise ValueError("A mounted inventory source requires an absolute mount, a UUID, and a path under the mount.")
        result.append({"id": name, "path": path, "mode": mode, "priority": priority, "ephemeral": ephemeral,
                       "mount": mount, "uuid": uuid})
        seen.add(name)
    return result


def stable_id(kind: str, *parts: object) -> str:
    data = json.dumps([kind, *parts], ensure_ascii=False, separators=(",", ":")).encode()
    return kind + "-" + hashlib.sha256(data).hexdigest()[:20]


def mount_available(spec: dict) -> bool:
    if spec.get("mount") is None:
        if spec["id"] in {"nvme_models", "nvme_staging"}:
            require_nvme_mount()
            source = Path(spec["path"]).resolve(strict=False)
            return (source.is_relative_to(NVME_LIBRARY_ROOT.resolve(strict=True)) and
                    (not source.exists() or source.stat().st_dev == NVME_MOUNT.stat().st_dev))
        return True
    mount = Path(spec["mount"])
    try:
        device = (Path("/dev/disk/by-uuid") / spec["uuid"]).stat()
        source = Path(spec["path"]).resolve(strict=False)
        return (mount.is_mount() and stat.S_ISBLK(device.st_mode) and mount.stat().st_dev == device.st_rdev and
                source.is_relative_to(mount.resolve(strict=True)) and
                (not source.exists() or source.stat().st_dev == device.st_rdev))
    except OSError:
        return False


def walk_files(root: Path, skipped: set[str], warnings: list[str]):
    seen = 0
    errors = []
    for base, directories, files in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
        parent = Path(base)
        seen += len(directories)
        if seen > MAX_ENTRIES:
            warnings.append(f"Entry limit reached: {root}")
            return
        if len(parent.relative_to(root).parts) >= MAX_DEPTH:
            warnings.append(f"Depth limit reached: {parent}")
            directories[:] = []
        else:
            safe = []
            for name in sorted(directories):
                path = parent / name
                if name in skipped:
                    continue
                if path.is_symlink():
                    if not path.exists():
                        warnings.append(f"Broken directory symlink: {path}")
                    continue
                safe.append(name)
            directories[:] = safe
        for name in sorted(files):
            seen += 1
            if seen > MAX_ENTRIES:
                warnings.append(f"Entry limit reached: {root}")
                return
            path = parent / name
            if path.is_symlink() and not path.exists():
                warnings.append(f"Broken symlink: {path}")
            yield path
    warnings.extend(errors)


def entry(path: Path, hashes: dict) -> dict:
    result = {"path": str(path), "is_symlink": path.is_symlink()}
    try:
        resolved = path.resolve(strict=True)
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Not a regular file")
        result.update(size_bytes=info.st_size, modified_ns=info.st_mtime_ns,
                      device_id=info.st_dev, inode=info.st_ino, hardlink_count=info.st_nlink, available=info.st_size > 0)
        if result["is_symlink"]:
            result["resolved_path"] = str(resolved)
        recorded = hashes.get(str(path)) or hashes.get(str(resolved))
        if recorded and info.st_size == recorded["size_bytes"] and abs(info.st_mtime_ns - recorded["mtime_ns"]) < 1000:
            result["sha256"] = recorded["sha256"]
            result["sha256_status"] = "verified_at_copy; size and mtime match, not rehashed"
            result["sha256_verified_at"] = recorded["verified_at"]
    except (OSError, RuntimeError, ValueError) as exc:
        result.update(available=False, error=str(exc))
    return result


def known_hashes(manifest_dir: Path) -> tuple[dict, list[str]]:
    result, warnings = {}, []
    if not manifest_dir.is_dir() or manifest_dir.is_symlink():
        return result, warnings
    manifests = sorted(manifest_dir.glob("consolidation-*.json"))
    if len(manifests) > 100:
        warnings.append("More than 100 consolidation manifests; only the first 100 were inspected.")
    for path in manifests[:100]:
        try:
            if path.is_symlink() or path.stat().st_size > MAX_JSON:
                raise ValueError("Unsafe or oversized manifest")
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("kind") != "verified_model_consolidation" or not isinstance(data.get("models"), list):
                continue
            for model in data["models"]:
                source, destination = Path(model["source"]), Path(model["destination"])
                checked = datetime.fromtimestamp(model["verified_log_mtime_ns"] / 1e9, timezone.utc).isoformat()
                for f in model["files"]:
                    rel = Path(f["relative_path"])
                    if rel.is_absolute() or ".." in rel.parts or not re.fullmatch(r"[a-f0-9]{64}", f["sha256"]):
                        raise ValueError("Unsafe manifest file entry")
                    record = {"sha256": f["sha256"], "size_bytes": f["bytes"],
                              "mtime_ns": f["source_mtime_ns"], "verified_at": checked}
                    for root in (source, destination):
                        candidate = root / rel if root.is_dir() else root
                        result[str(candidate)] = record
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, OverflowError) as exc:
            warnings.append(f"Could not use consolidation manifest {path}: {exc}")
    return result, warnings


def normalized_metadata(meta: dict) -> dict:
    arch = meta.get("general.architecture")
    result = {"general_name": meta.get("general.name"), "architecture": arch,
              "file_type_code": meta.get("general.file_type"), "declared_shards": meta.get("split.count", 1)}
    for key in GGUF_INFO_KEYS:
        value = meta.get(key)
        if isinstance(value, str):
            result[key.removeprefix("general.")] = value
    if isinstance(arch, str):
        for key in ("context_length", "block_count", "embedding_length", "feed_forward_length",
                    "expert_feed_forward_length", "expert_count", "expert_used_count"):
            if arch + "." + key in meta:
                result[key] = meta[arch + "." + key]
    result = {key: value for key, value in result.items() if value is not None}
    for value in result.values():
        values = value if isinstance(value, list) else [value]
        if any(isinstance(v, float) and not math.isfinite(v) for v in values):
            raise ValueError("GGUF metadata contains a nonfinite value.")
    return result


def kind_from_name(name: str) -> str:
    lower = name.lower()
    if lower.startswith("mmproj") or "projector" in lower:
        return "vision_projector"
    if "mtp" in lower or "dflash" in lower or "draft" in lower:
        return "draft_or_mtp"
    return "model"


def scan_gguf(spec: dict, hashes: dict, now: str, warnings: list[str]) -> list[dict]:
    root = Path(spec["path"])
    grouped = defaultdict(dict)
    tensor_groups = defaultdict(list)
    media_roots = {"easy-diffusion", "hunyuan-video-1.5-480p-i2v", "hunyuan-video-1.5-480p-i2v-step-distilled",
                   "sdxl-1.0-inpainting", "ltx-2.5", "ltx-video"}
    skipped = {".git", ".cache"}
    if not spec.get("ephemeral"):
        skipped.add("model-staging")
    for path in walk_files(root, skipped, warnings):
        excluded_media = spec["id"] == "xfs_models" and path.relative_to(root).parts[0] in media_roots
        if not excluded_media and not path.is_symlink() and (path.suffix.lower() == ".safetensors" or
                (path.suffix.lower() == ".bin" and path.name.lower().startswith(("model", "pytorch_model"))
                 and (path.parent / "config.json").is_file())):
            tensor_groups[path.parent].append(path)
        if path.suffix.lower() != ".gguf":
            continue
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
                if not any(resolved.is_relative_to(Path(p)) for p in spec.get("allowed_link_roots", ())):
                    warnings.append(f"Skipped GGUF symlink outside known online model roots: {path}")
                    continue
            except (OSError, RuntimeError) as exc:
                warnings.append(f"GGUF symlink unavailable {path}: {exc}")
                continue
        match = SPLIT.fullmatch(path.name)
        if match:
            stem, number, expected = match.group(1), int(match.group(2)), int(match.group(3))
            if not 1 <= number <= expected <= 1000:
                warnings.append(f"Invalid GGUF shard numbering: {path}")
                continue
        else:
            stem, number, expected = path.name, 1, 1
        grouped[(path.parent, stem, expected)][number] = path
    records = []
    for (parent, stem, expected), parts in sorted(grouped.items(), key=lambda row: (str(row[0][0]), row[0][1].casefold())):
        files = [entry(parts[i], hashes) for i in sorted(parts)]
        missing = [i for i in range(1, expected + 1) if i not in parts or not entry(parts[i], hashes).get("available")]
        first = parts.get(1)
        metadata, error = {}, None
        if first and first.exists():
            try:
                metadata = normalized_metadata(model_metadata(first, extra_keys=GGUF_INFO_KEYS))
                if metadata.get("declared_shards", 1) != expected:
                    error = "GGUF header shard count does not match filenames."
            except (OSError, ValueError, TypeError, UnicodeError) as exc:
                error = str(exc)
        elif not missing:
            error = "First shard is unavailable."
        sizes = [parts[i].stat().st_size if i in parts and parts[i].exists() else None for i in range(1, expected + 1)]
        key = stable_id("gguf", stem.casefold(), expected, sizes)
        hint = QUANT.search(first.name if first else stem + ".gguf")
        location = {"source_id": spec["id"], "path": str(first or parent),
                    "availability": "available" if not missing and not error else "incomplete_or_unreadable",
                    "expected_shards": expected, "present_shards": expected - len(missing),
                    "missing_shards": missing, "size_bytes": sum(x.get("size_bytes", 0) for x in files),
                    "files": files, "metadata": metadata, "last_seen": now,
                    "ephemeral": spec.get("ephemeral", False), "priority": spec.get("priority", 50)}
        if error:
            location["metadata_error"] = error
        records.append({"id": key, "name": stem, "kind": kind_from_name(stem), "format": "GGUF",
                        "quantization_hint": hint.group(1).upper() if hint else None,
                        "identity_basis": "matching filename and shard sizes only; content equivalence is not assumed",
                        "locations": [location]})
    for parent, paths in sorted(tensor_groups.items(), key=lambda row: str(row[0])):
        try:
            config = bounded_json(parent / "config.json")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Cannot read model config {parent}: {exc}")
            config = {}
        metadata = {k: config[k] for k in ("model_type", "architectures", "hidden_size", "num_hidden_layers",
                                            "num_attention_heads", "vocab_size", "max_position_embeddings")
                    if isinstance(config.get(k), (str, int)) or
                    (isinstance(config.get(k), list) and all(isinstance(x, str) for x in config[k]))}
        metadata["weight_completeness"] = "not independently validated"
        name = config.get("_name_or_path") if isinstance(config.get("_name_or_path"), str) and config["_name_or_path"] else parent.name
        records.append(collection_record(spec, name, parent, "model_collection", "safetensors or PyTorch weights",
                                         sorted(paths), hashes, now, metadata))
    return records


def bounded_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def collection_record(spec: dict, name: str, path: Path, kind: str, fmt: str,
                      files: list[Path], hashes: dict, now: str, metadata: dict | None = None) -> dict:
    entries = [entry(p, hashes) for p in files]
    present = bool(entries) and all(f.get("available") for f in entries)
    return {"id": stable_id(kind, str(path)), "name": name, "kind": kind, "format": fmt,
            "identity_basis": "specific directory or file; no cross-location identity assumed",
            "locations": [{"source_id": spec["id"], "path": str(path), "availability": "present_unverified" if present else "incomplete_or_unreadable",
                           "size_bytes": sum(x.get("size_bytes", 0) for x in entries), "files": entries,
                           "metadata": metadata or {}, "last_seen": now,
                           "ephemeral": spec.get("ephemeral", False), "priority": spec.get("priority", 50)}]}


def scan_colibri(spec: dict, hashes: dict, now: str, warnings: list[str]) -> list[dict]:
    root = Path(spec["path"])
    try:
        config = bounded_json(root / "config.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"Cannot read Colibri config {root}: {exc}")
        config = {}
    fields = ("model_type", "architectures", "num_hidden_layers", "hidden_size", "n_routed_experts",
              "num_experts_per_tok", "vocab_size", "max_position_embeddings")
    metadata = {k: config[k] for k in fields if k in config}
    metadata["weight_completeness"] = "not independently validated"
    files = sorted(p for p in root.glob("*.safetensors") if p.is_file() and not p.is_symlink())
    if not files:
        warnings.append(f"No Colibri weight files found: {root}")
        return []
    metadata["safetensor_files"] = len(files)
    is_glm52 = config.get("model_type") == "glm_moe_dsa" and config.get("num_hidden_layers") == 78
    name = "GLM-5.2 Colibri (config-based hint)" if is_glm52 else "Colibri local model"
    metadata["model_identity"] = "config-based hint; checkpoint identity not independently verified"
    return [collection_record(spec, name, root, "model_collection", "Colibri safetensors",
                              files, hashes, now, metadata)]


def scan_media(spec: dict, hashes: dict, now: str, warnings: list[str]) -> list[dict]:
    root = Path(spec["path"])
    collections = ("hunyuan-video-1.5-480p-i2v", "hunyuan-video-1.5-480p-i2v-step-distilled",
                   "sdxl-1.0-inpainting", "ltx-2.5")
    records = []
    for name in collections:
        path = root / name
        if not path.is_dir() or path.is_symlink():
            continue
        files = [p for p in walk_files(path, {".git", ".cache"}, warnings)
                 if not p.is_symlink() and p.suffix.lower() in WEIGHT_EXTENSIONS and p.stat().st_size >= MIN_ASSET_BYTES]
        if not files:
            continue
        try:
            index = bounded_json(path / "model_index.json")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Cannot read model index {path}: {exc}")
            index = {}
        metadata = {k: index[k] for k in ("_class_name", "_name_or_path") if isinstance(index.get(k), str)}
        metadata["weight_completeness"] = "not independently validated"
        records.append(collection_record(spec, name, path, "model_collection", "diffusion weights", files, hashes, now, metadata))
    files = []
    for path in root.glob("*.ckpt"):
        if path.is_file() and not path.is_symlink() and path.stat().st_size >= MIN_ASSET_BYTES:
            files.append(path)
    easy = root / "easy-diffusion"
    if easy.is_dir() and not easy.is_symlink():
        files += [p for p in walk_files(easy, {".git", ".cache"}, warnings)
                  if not p.is_symlink() and p.suffix.lower() in (".ckpt", ".safetensors", ".pth", ".pt")
                  and p.stat().st_size >= MIN_ASSET_BYTES]
    for path in sorted(files):
        kind = "auxiliary" if path.parent.name in ("vae", "gfpgan", "realesrgan") or path.suffix.lower() in (".pth", ".pt") else "model"
        records.append(collection_record(spec, path.stem, path, kind, path.suffix.lstrip(".").upper(), [path], hashes, now))
    return records


def scan_hf(spec: dict, hashes: dict, now: str, warnings: list[str]) -> list[dict]:
    root = Path(spec["path"])
    records = []
    for repo in sorted(root.glob("models--*")):
        if not repo.is_dir() or repo.is_symlink():
            continue
        snapshots = repo / "snapshots"
        if not snapshots.is_dir() or snapshots.is_symlink():
            continue
        for snapshot in sorted(snapshots.iterdir()):
            if not snapshot.is_dir() or snapshot.is_symlink():
                continue
            files = []
            for path in walk_files(snapshot, {".git", ".cache"}, warnings):
                if path.suffix.lower() not in WEIGHT_EXTENSIONS:
                    continue
                try:
                    resolved = path.resolve(strict=True)
                    if not resolved.is_relative_to(repo.resolve(strict=True)) or not resolved.is_file():
                        warnings.append(f"Cache weight escapes its repository: {path}")
                        continue
                    if resolved.stat().st_size >= MIN_ASSET_BYTES:
                        files.append(path)
                except (OSError, RuntimeError) as exc:
                    warnings.append(f"Cache weight unavailable {path}: {exc}")
            if files:
                name = repo.name.removeprefix("models--").replace("--", "/")
                metadata = {"revision": snapshot.name, "weight_completeness": "not independently validated"}
                config = snapshot / "config.json"
                try:
                    if config.exists():
                        resolved = config.resolve(strict=True)
                        if not resolved.is_relative_to(repo.resolve(strict=True)):
                            raise ValueError("Config escapes its cache repository.")
                        fields = bounded_json(resolved)
                        for key in ("model_type", "architectures", "hidden_size", "num_hidden_layers",
                                    "num_attention_heads", "vocab_size", "max_position_embeddings", "torch_dtype"):
                            value = fields.get(key)
                            if isinstance(value, (str, int)) or (isinstance(value, list) and all(isinstance(x, str) for x in value)):
                                metadata[key] = value
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    warnings.append(f"Cache config unavailable {config}: {exc}")
                records.append(collection_record(spec, name, snapshot, "model_collection", "Hugging Face cache",
                                                 files, hashes, now, metadata))
    return records


def scan_strata(spec: dict, hashes: dict, now: str, warnings: list[str]) -> list[dict]:
    root = Path(spec["path"])
    records = []
    for relative in ("mtp/tensors", "mtp/rt", "packs/iq3_s"):
        path = root / relative
        if not path.is_dir() or path.is_symlink():
            continue
        files = [p for p in walk_files(path, {".git", ".cache"}, warnings)
                 if not p.is_symlink() and p.suffix.lower() in WEIGHT_EXTENSIONS and p.stat().st_size >= MIN_ASSET_BYTES]
        if files:
            records.append(collection_record(spec, "Strata " + relative, path, "auxiliary", "raw tensor files",
                                             files, hashes, now, {"weight_completeness": "not independently validated"}))
    return records


def merge_records(records: list[dict], source_status: dict, previous: dict | None) -> list[dict]:
    merged = {}
    seen_locations = set()
    for record in records:
        item = merged.setdefault(record["id"], {**record, "locations": []})
        for location in record["locations"]:
            key = (location["source_id"], location["path"])
            if key not in seen_locations:
                item["locations"].append(location)
                seen_locations.add(key)
    for old in (previous or {}).get("models", []):
        if not isinstance(old, dict) or not isinstance(old.get("id"), str) or not isinstance(old.get("locations"), list):
            continue
        if old["id"] in merged:
            for key in ("notes", "user_metadata"):
                if key in old:
                    merged[old["id"]][key] = old[key]
        for location in old["locations"]:
            if not isinstance(location, dict) or not isinstance(location.get("source_id"), str) or not isinstance(location.get("path"), str):
                continue
            key = (location["source_id"], location["path"])
            replaced = key in seen_locations
            if replaced and (old["id"] in merged or not any(k in old for k in ("notes", "user_metadata"))):
                continue
            status = source_status.get(location["source_id"], "untracked")
            previous_location = dict(location)
            previous_location["availability"] = "replaced" if replaced else "offline" if status == "offline" else "unknown" if status == "partial" else "untracked" if status == "untracked" else "missing"
            if isinstance(previous_location.get("files"), list):
                previous_location["files"] = [{**f, "last_known_available": f.get("last_known_available", f.get("available", False)),
                                                "available": False} if isinstance(f, dict) else f
                                               for f in previous_location["files"]]
            item = merged.setdefault(old["id"], {**old, "locations": []})
            item["locations"].append(previous_location)
            seen_locations.add(key)
    for item in merged.values():
        item["locations"].sort(key=lambda loc: (loc.get("priority", 50), loc["path"]))
        available = [loc for loc in item["locations"] if loc.get("availability") in ("available", "present_unverified")]
        item["preferred_path"] = available[0]["path"] if available else None
        item["availability"] = "present" if available else "not_currently_available"
    return sorted(merged.values(), key=lambda item: (item.get("name", "").casefold(), item["id"]))


def scan(specs: list[dict], previous: dict | None = None, manifest_dir: Path | None = None) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    hashes, warnings = known_hashes(manifest_dir) if manifest_dir else ({}, [])
    statuses, records, inventory_sources = {}, [], []
    mounted = {spec["id"]: mount_available(spec) for spec in specs}
    allowed = [str(Path(spec["path"]).resolve(strict=False)) for spec in specs
               if mounted[spec["id"]] and Path(spec["path"]).is_dir()]
    for spec in specs:
        source = {key: spec[key] for key in ("id", "path", "mode", "priority", "ephemeral")}
        source["uuid"] = spec.get("uuid")
        local_warnings = []
        path = Path(spec["path"])
        if not mounted[spec["id"]]:
            status = "offline"
        elif not path.is_dir():
            status = "missing"
        else:
            status = "online"
            try:
                mode = spec["mode"]
                handlers = {"gguf": scan_gguf, "colibri": scan_colibri, "media": scan_media,
                            "hf": scan_hf, "strata": scan_strata}
                records.extend(handlers[mode]({**spec, "allowed_link_roots": allowed}, hashes, now, local_warnings))
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                local_warnings.append(f"Scan failed: {exc}")
            if local_warnings:
                status = "partial"
        statuses[spec["id"]] = status
        source["status"] = status
        if local_warnings:
            source["warnings"] = local_warnings[:100]
            source["warnings_truncated"] = len(local_warnings) > 100
        inventory_sources.append(source)
    for old in (previous or {}).get("sources", []):
        if isinstance(old, dict) and isinstance(old.get("id"), str) and old["id"] not in statuses:
            inventory_sources.append({**old, "status": "untracked"})
            statuses[old["id"]] = "untracked"
    models = merge_records(records, statuses, previous)
    shared = defaultdict(set)
    for model in models:
        for location in model["locations"]:
            for file in location.get("files", []):
                if (isinstance(file, dict) and file.get("available") and not file.get("is_symlink") and
                        file.get("hardlink_count", 0) > 1 and "device_id" in file and "inode" in file):
                    shared[(file["device_id"], file["inode"])].add(file["path"])
    hardlinks = [{"device_id": device, "inode": inode, "paths": sorted(paths),
                  "warning": "These paths share data; an in-place write through one path changes all of them."}
                 for (device, inode), paths in sorted(shared.items()) if len(paths) > 1]
    result = {"schema_version": 1, "managed_by": MANAGED_BY, "library_root": str(NVME_LIBRARY_ROOT),
              "generated_at_utc": now, "description": "Local model and auxiliary-weight inventory; presence is not compatibility or quality qualification.",
              "sources": inventory_sources, "models": models, "hardlinks": hardlinks,
              "summary": {"records": len(models), "currently_present": sum(m["availability"] == "present" for m in models),
                          "not_currently_available": sum(m["availability"] != "present" for m in models)},
              "limitations": ["GGUF copies are grouped by filename and shard sizes, not assumed byte-identical.",
                              "Only file presence, lightweight headers and known previous checksums are recorded; model compatibility and quality are not verified.",
                              "Hugging Face, Colibri and diffusion collection completeness is not independently verified."]}
    if warnings:
        result["warnings"] = warnings
    if previous:
        for key in ("notes", "user_metadata"):
            if key in previous:
                result[key] = previous[key]
    return result


def read_previous(output: Path) -> tuple[dict | None, tuple | None]:
    if output.is_symlink():
        raise ValueError("Inventory destination must not be a symlink.")
    if not output.exists():
        return None, None
    fd = os.open(output, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > MAX_JSON or
                info.st_dev != output.parent.stat().st_dev):
            raise ValueError("Inventory destination is not a bounded user-owned regular file.")
        encoded = stream.read(MAX_JSON + 1)
        after = os.fstat(stream.fileno())
        fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if len(encoded) > MAX_JSON or fingerprint != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("Inventory changed while reading; it was not overwritten.")
    value = json.loads(encoded.decode("utf-8"))
    if (not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("managed_by") != MANAGED_BY or
            value.get("library_root") != str(NVME_LIBRARY_ROOT) or not isinstance(value.get("models"), list)):
        raise ValueError("Refusing to overwrite an unmanaged inventory file.")
    return value, fingerprint


def publish(output: Path, data: dict, previous_fingerprint: tuple | None) -> None:
    encoded = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > MAX_JSON:
        raise ValueError("Inventory exceeds the 16 MiB limit.")
    require_nvme_mount()
    if output.parent.is_symlink() or output.parent.resolve(strict=True) != NVME_LIBRARY_ROOT.resolve(strict=True):
        raise ValueError("Inventory parent is not the selected model library.")
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = ".available-models-" + secrets.token_hex(12) + ".tmp"
    created = False
    try:
        if os.fstat(directory_fd).st_dev != output.parent.stat().st_dev:
            raise ValueError("Inventory parent changed during publication.")
        free = os.fstatvfs(directory_fd)
        if free.f_bavail * free.f_frsize < RESERVE + HEADROOM + len(encoded):
            raise ValueError("Writing the inventory would violate the 16 GiB reserve.")
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
        created = True
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        require_nvme_mount()
        if previous_fingerprint is None:
            os.link(temporary, output.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
            os.unlink(temporary, dir_fd=directory_fd)
            created = False
        else:
            info = os.stat(output.name, dir_fd=directory_fd, follow_symlinks=False)
            current = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            if not stat.S_ISREG(info.st_mode) or current != previous_fingerprint:
                raise ValueError("Inventory changed during scanning; it was not overwritten.")
            os.replace(temporary, output.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            created = False
        os.fsync(directory_fd)
    finally:
        if created:
            try:
                require_nvme_mount()
            except ValueError:
                pass
            else:
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
        os.close(directory_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Scan and print a summary without writing the inventory")
    parser.add_argument("--sources", type=Path, help="Local JSON file listing additional inventory sources")
    args = parser.parse_args()
    try:
        require_nvme_mount()
        output = NVME_LIBRARY_ROOT / "available_local_models.json"
        previous, fingerprint = read_previous(output)
        data = scan(sources(args.sources), previous, NVME_LIBRARY_ROOT / ".vrambridge-manifests")
        if not args.dry_run:
            publish(output, data, fingerprint)
        print(json.dumps({"status": "preview" if args.dry_run else "updated", "path": str(output),
                          **data["summary"], "source_status": {s["id"]: s["status"] for s in data["sources"]}}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"model inventory: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
