#!/usr/bin/env python3
"""Explicit, verified model working copies; never evict archive originals."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vrambridge.app import AppError, model_metadata, readable_gguf

DEFAULT_ROOT = Path("/home/mj420/Desktop/TOR/llm/model-staging")
RESERVE = 16 * 1024 ** 3
COPY_OVERHEAD = 16 * 1024 ** 2
CHUNK = 8 * 1024 ** 2
MARKER = ".vrambridge-staging.json"
ID = re.compile(r"[a-f0-9]{32}")


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 ** 2:
        raise ValueError(f"Expected a bounded regular metadata file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if len(serialized.encode()) > 1024 ** 2:
        raise ValueError("Staging metadata exceeds 1 MiB.")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def initialize(root: Path) -> dict:
    if root.is_symlink():
        raise ValueError("Staging root must not be a symlink.")
    root = root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker = root / MARKER
    if not marker.exists():
        if any(root.iterdir()):
            raise ValueError("Refusing to adopt a nonempty directory without a staging marker.")
        if root.stat().st_uid != os.getuid():
            raise ValueError("Staging root must belong to the current user.")
        root.chmod(0o700)
        write_json(marker, {"schema_version": 1, "kind": "vrambridge_model_staging",
                            "owner_uid": os.getuid(), "root": str(root),
                            "filesystem_device": root.stat().st_dev, "reserve_free_bytes": RESERVE})
    return configuration(root)


def configuration(root: Path) -> dict:
    if root.is_symlink():
        raise ValueError("Staging root must not be a symlink.")
    root = root.resolve(strict=True)
    value = read_json(root / MARKER)
    if (value.get("schema_version") != 1 or value.get("kind") != "vrambridge_model_staging" or
            value.get("root") != str(root) or value.get("owner_uid") != os.getuid() or
            root.stat().st_uid != os.getuid() or value.get("filesystem_device") != root.stat().st_dev or
            value.get("reserve_free_bytes") != RESERVE):
        raise ValueError("Staging ownership, device or 16 GiB reserve policy does not match its marker.")
    return value


@contextmanager
def locked(root: Path):
    configuration(root)
    descriptor = os.open(root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another staging operation is running.") from exc
        yield


def same_filesystem(files: list[Path], root: Path) -> bool:
    return all(path.stat().st_dev == root.stat().st_dev for path in files)


def discover(first: Path) -> list[Path]:
    first = readable_gguf(str(first), "Model")
    metadata = model_metadata(first)
    count = metadata.get("split.count", 1)
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("Invalid or unbounded GGUF split count.")
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", first.name, re.IGNORECASE)
    if match:
        if int(match[2]) != 1 or int(match[3]) != count:
            raise ValueError("Select the first shard; filename and declared split count must agree.")
        files = [first.with_name(f"{match[1]}-{index:05d}-of-{count:05d}.gguf")
                 for index in range(1, count + 1)]
    elif count != 1:
        raise ValueError("Multipart GGUF must retain its standard shard filenames.")
    else:
        files = [first]
    resolved = [readable_gguf(str(path), "Shard") for path in files]
    if any(path.name != original.name for path, original in zip(resolved, files)):
        raise ValueError("Resolved shard filenames differ; preserve standard GGUF names.")
    identities = {(path.stat().st_dev, path.stat().st_ino) for path in resolved}
    if len(identities) != len(resolved):
        raise ValueError("Several shard names refer to the same underlying file.")
    return resolved


def fingerprint(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def plan(root: Path, first: Path) -> dict:
    configuration(root)
    files = discover(first)
    if any(path.is_relative_to(root) for path in files):
        raise ValueError("An existing staged copy cannot be its own archive source.")
    sources = [{"source": str(path), "name": path.name, "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns} for path in files]
    cache_id = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:32]
    reuse = same_filesystem(files, root)
    required = 0 if reuse else sum(item["size_bytes"] for item in sources)
    free = shutil.disk_usage(root).free
    return {"cache_id": cache_id, "files": sources, "reuse_existing_filesystem_copy": reuse,
            "required_copy_bytes": required, "free_bytes": free, "reserve_free_bytes": RESERVE,
            "metadata_headroom_bytes": COPY_OVERHEAD,
            "copy_fits_headroom": free >= required + RESERVE + COPY_OVERHEAD, "model_path": str(files[0])}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(CHUNK):
            digest.update(data)
    return digest.hexdigest()


def load_entry(root: Path, cache_id: str) -> tuple[Path, dict]:
    if not ID.fullmatch(cache_id):
        raise ValueError("Expected a 32-character hexadecimal cache ID.")
    directory = root / cache_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Cache entry must be a real directory inside the staging root.")
    manifest = read_json(directory / "manifest.json")
    files = manifest.get("files")
    if (manifest.get("kind") != "vrambridge_staged_model" or manifest.get("schema_version") != 1 or
            manifest.get("cache_id") != cache_id or manifest.get("root") != str(root) or
            manifest.get("owner_uid") != os.getuid() or not isinstance(files, list) or not 1 <= len(files) <= 100):
        raise ValueError("Invalid staged-model ownership or manifest.")
    names = []
    for item in files:
        name = item.get("name") if isinstance(item, dict) else None
        if (not isinstance(name, str) or Path(name).name != name or not name.lower().endswith(".gguf") or
                not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) or
                type(item.get("size_bytes")) is not int or item["size_bytes"] < 4 or
                not isinstance(item.get("source"), str) or not Path(item["source"]).is_absolute() or
                Path(item["source"]).resolve().is_relative_to(root)):
            raise ValueError("Unsafe staged-model file metadata.")
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError("Staged file is missing, changed, or a symlink.")
        names.append(name)
    if len(set(names)) != len(names) or {path.name for path in directory.iterdir()} != {*names, "manifest.json"}:
        raise ValueError("Unexpected or duplicate files in cache entry; no deletion is allowed.")
    return directory, manifest


def verify(root: Path, cache_id: str) -> dict:
    directory, manifest = load_entry(root, cache_id)
    for item in manifest["files"]:
        if sha256(directory / item["name"]) != item["sha256"]:
            raise ValueError("Staged copy checksum mismatch.")
        original = Path(item["source"])
        if not original.is_file() or sha256(original) != item["sha256"]:
            raise ValueError("Archive original is missing or changed; preserving staged copy.")
    return {"status": "verified", "cache_id": cache_id,
            "model_path": str(directory / manifest["files"][0]["name"]), "originals_preserved": True}


def stage(root: Path, first: Path) -> dict:
    description = plan(root, first)
    if description["reuse_existing_filesystem_copy"]:
        return {**description, "status": "reuse_existing_copy", "copied_bytes": 0}
    destination = root / description["cache_id"]
    if destination.exists() or destination.is_symlink():
        return verify(root, description["cache_id"])
    if not description["copy_fits_headroom"]:
        raise ValueError("Copy would leave less than the required 16 GiB free; nothing copied.")
    partial = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
    owned_files = []
    published = False
    started = time.monotonic()
    remaining = description["required_copy_bytes"]
    source_versions = {}
    try:
        for item in description["files"]:
            original = Path(item["source"])
            before = fingerprint(original)
            source_versions[str(original)] = before
            if before[2:] != (item["size_bytes"], item["mtime_ns"]):
                raise ValueError("Archive changed after planning.")
            target = partial / item["name"]
            print(f"Copying {item['name']} ({item['size_bytes']} bytes)", file=sys.stderr)
            digest = hashlib.sha256()
            with original.open("rb") as source, target.open("xb") as output:
                owned_files.append(target)
                while data := source.read(CHUNK):
                    if shutil.disk_usage(root).free < remaining + RESERVE + COPY_OVERHEAD:
                        raise ValueError("Disk headroom changed during copy; stopping.")
                    output.write(data)
                    digest.update(data)
                    remaining -= len(data)
                output.flush()
                os.fsync(output.fileno())
            if fingerprint(original) != before or target.stat().st_size != item["size_bytes"]:
                raise ValueError("Source changed or staged copy length differs.")
            item["sha256"] = digest.hexdigest()
            if sha256(target) != item["sha256"]:
                raise ValueError("Destination readback checksum mismatch.")
        if any(fingerprint(Path(path)) != version for path, version in source_versions.items()):
            raise ValueError("An earlier archive shard changed before publication.")
        manifest = partial / "manifest.json"
        owned_files.append(manifest)
        write_json(manifest, {"schema_version": 1, "kind": "vrambridge_staged_model",
                             "cache_id": description["cache_id"], "root": str(root),
                             "owner_uid": os.getuid(), "files": description["files"]})
        partial.rename(destination)
        published = True
    finally:
        if not published:
            for path in owned_files:
                path.unlink(missing_ok=True)
            partial.rmdir()
    return {"status": "staged", "cache_id": description["cache_id"],
            "model_path": str(destination / description["files"][0]["name"]),
            "copied_bytes": description["required_copy_bytes"],
            "copy_and_verify_seconds": time.monotonic() - started, "originals_preserved": True}


def evict(root: Path, cache_id: str, confirm_unloaded: bool) -> dict:
    if not confirm_unloaded:
        raise ValueError("Unload the model, then explicitly pass --confirm-unloaded.")
    verify(root, cache_id)
    directory, manifest = load_entry(root, cache_id)
    paths = [directory / item["name"] for item in manifest["files"]]
    result = subprocess.run(["fuser", "-I", *map(str, paths)], capture_output=True, text=True)
    if result.returncode == 0:
        raise RuntimeError("Staged files are in use; eviction refused.")
    if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
        raise RuntimeError("Cannot confirm staged files are unused: " + result.stderr.strip())
    for path in paths:
        path.unlink()
    (directory / "manifest.json").unlink()
    directory.rmdir()
    return {"status": "evicted", "cache_id": cache_id, "originals_preserved": True}


def listing(root: Path) -> dict:
    entries, unmanaged = [], []
    for path in sorted(root.iterdir()):
        if path.name in (MARKER, ".lock"):
            continue
        if ID.fullmatch(path.name):
            _, manifest = load_entry(root, path.name)
            entries.append({"cache_id": path.name, "model": manifest["files"][0]["name"],
                            "size_bytes": sum(item["size_bytes"] for item in manifest["files"])})
        else:
            unmanaged.append(path.name)
    return {"entries": entries, "unmanaged_entries_not_deleted": unmanaged,
            "free_bytes": shutil.disk_usage(root).free, "reserve_free_bytes": RESERVE}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("list")
    for command in ("plan", "stage"):
        commands.add_parser(command).add_argument("model", type=Path)
    commands.add_parser("verify").add_argument("cache_id")
    eviction = commands.add_parser("evict")
    eviction.add_argument("cache_id")
    eviction.add_argument("--confirm-unloaded", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "init":
            result = initialize(args.root)
        else:
            if args.root.is_symlink():
                raise ValueError("Staging root must not be a symlink.")
            root = args.root.resolve(strict=True)
            with locked(root):
                if args.command == "list":
                    result = listing(root)
                elif args.command == "plan":
                    result = plan(root, args.model)
                elif args.command == "stage":
                    result = stage(root, args.model)
                elif args.command == "verify":
                    result = verify(root, args.cache_id)
                else:
                    result = evict(root, args.cache_id, args.confirm_unloaded)
    except (AppError, OSError, ValueError, RuntimeError) as exc:
        print(f"vrambridge-model-staging: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
