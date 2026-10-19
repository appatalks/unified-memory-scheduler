#!/usr/bin/env python3
"""Verify shared GGUF/Strata pack inputs and create owned zero-copy references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_region(first, first_offset, second, second_offset, count):
    with first.open("rb") as left, second.open("rb") as right:
        left.seek(first_offset)
        right.seek(second_offset)
        remaining = count
        while remaining:
            size = min(remaining, 1024 * 1024)
            a, b = left.read(size), right.read(size)
            if len(a) != size or a != b:
                raise ValueError("Pack bytes differ from the original GGUF: " + first.name)
            remaining -= size


def strata_profile_bytes(layers, experts, ranking):
    if not (1 <= layers <= 65535 and 1 <= experts <= 65535 and layers * experts <= 16 * 1024**2):
        raise ValueError("Unsupported profile dimensions.")
    if not ranking or len(set(ranking)) != len(ranking):
        raise ValueError("Profile needs a nonempty, unique ranked expert set.")
    if any(not (0 <= layer < layers and 0 <= expert < experts) for layer, expert in ranking):
        raise ValueError("Profile expert outside its descriptor.")
    table = [-1] * (layers * experts)
    for slot, (layer, expert) in enumerate(ranking):
        table[layer * experts + expert] = slot
    return (b"STRP" + struct.pack("<5I", 1, layers, experts, len(ranking), len(ranking))
            + b"".join(struct.pack("<HH", *pair) for pair in ranking)
            + struct.pack("<" + "i" * len(table), *table))


def verify(args):
    sys.path.insert(0, str(args.gguf_python.resolve(strict=True)))
    import gguf
    model = args.model.resolve(strict=True)
    match = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", model.name)
    if not match:
        raise ValueError("A complete standard first-shard filename is required.")
    count = int(match[2])
    if not 1 <= count <= 100:
        raise ValueError("Unsupported shard count.")
    shards = [model.with_name(f"{match[1]}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)]
    readers = [gguf.GGUFReader(str(path.resolve(strict=True))) for path in shards]
    tensors = {}
    for path, reader in zip(shards, readers):
        for tensor in reader.tensors:
            if tensor.name in tensors:
                raise ValueError("Duplicate tensor across shards: " + tensor.name)
            tensors[tensor.name] = path, tensor
    fields = readers[0].fields
    architecture = fields["general.architecture"].contents()
    geometry = {}
    for key in ("block_count", "embedding_length", "expert_feed_forward_length", "expert_count", "expert_used_count"):
        geometry[key] = fields[architecture + "." + key].contents()
    conversions = json.loads((args.pack / "conversions.json").read_text())
    if conversions.get("tensors") or conversions.get("compat_bf16") is not False:
        raise ValueError("Initial comparison requires a pack without conversions or compatibility rounding.")
    if [(p.name, p.stat().st_size) for p in shards] != [(r["name"], r["size"]) for r in conversions["source_shards"]]:
        raise ValueError("Pack source shard names/sizes do not match.")
    native_rows = []
    for line in (args.pack / "native_experts.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if len(values) not in (8, 9):
            raise ValueError("Unsupported native-expert index version.")
        layer, gu_type, down_type, _, blob, gate, up, down = map(int, values[:8])
        source = model if len(values) == 8 else model.with_name(values[8])
        logical = 0
        for part, offset, expected_type in (("gate", gate, gu_type), ("up", up, gu_type), ("down", down, down_type)):
            path, tensor = tensors[f"blk.{layer}.ffn_{part}_exps.weight"]
            if path != source or tensor.data_offset != offset or int(tensor.tensor_type) != expected_type:
                raise ValueError("Expert native offset/type does not match the GGUF.")
            if int(tensor.shape[-1]) != geometry["expert_count"]:
                raise ValueError("Expert tensor domain differs from model descriptor.")
            logical += int(tensor.n_bytes)
        if logical != blob * geometry["expert_count"]:
            raise ValueError("Expert blob accounting differs from native tensors.")
        native_rows.append({"layer": layer, "gate_up_type": gu_type, "down_type": down_type,
                            "bytes_per_expert": blob})
    if sorted(r["layer"] for r in native_rows) != list(range(geometry["block_count"])):
        raise ValueError("Native expert index does not cover every layer.")
    dense_tensors = dense_bytes = 0
    dense = args.pack / "dense.bin"
    for line in (args.pack / "index.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split()
        name, kind, offset, size = columns[0], columns[2], int(columns[3]), int(columns[4])
        if name not in tensors:
            raise ValueError("Pack index names a missing tensor: " + name)
        path, tensor = tensors[name]
        if kind == "0":
            if columns[9] != "8":
                raise ValueError("Unsupported non-native pack tensor.")
            continue
        expected_type = {"2": 0, "4": 30, "5": 1}.get(kind)
        if expected_type is None or int(tensor.tensor_type) != expected_type or int(tensor.n_bytes) != size:
            raise ValueError("Dense tensor type/length would change target representation: " + name)
        compare_region(path, int(tensor.data_offset), dense, offset, size)
        dense_tensors += 1
        dense_bytes += size
    seed = []
    for line in args.seed.read_text().splitlines():
        layer, expert = map(int, line.split())
        if not (0 <= layer < geometry["block_count"] and 0 <= expert < geometry["expert_count"]):
            raise ValueError("Seed expert is outside the loaded model descriptor.")
        seed.append((layer, expert))
    if len(set(seed)) != len(seed):
        raise ValueError("Duplicate seed expert.")
    sizes = {r["layer"]: r["bytes_per_expert"] for r in native_rows}
    payload = sum(sizes[layer] for layer, _ in seed)
    if payload > 6144 * 1024**2:
        raise ValueError("Common seed exceeds its declared6GiB payload budget.")
    print("Native pack offsets/types and", dense_tensors, "dense tensors match original bytes; hashing complete shards.", flush=True)
    hashes = []
    for path in shards:
        hashes.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
        print("Verified complete shard:", path.name, flush=True)
    args.references.mkdir(mode=0o700, parents=True, exist_ok=False)
    links = {}
    for name, target in {"models": model.parent, "pack": args.pack.resolve(strict=True),
                         "mtp": args.mtp.resolve(strict=True)}.items():
        path = args.references / name
        path.symlink_to(target, target_is_directory=True)
        links[name] = str(target)
    manifest = {"schema_version": 1, "kind": "owned_read_only_model_references", "originals_modified": False,
                "weight_bytes_copied": 0, "links": links}
    (args.references / "references.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.references / "references.json").chmod(0o600)
    profile = args.output.parent / "common-strata-profile.bin"
    with profile.open("xb") as stream:
        stream.write(strata_profile_bytes(geometry["block_count"], geometry["expert_count"], seed))
    report = {"schema_version": 1, "kind": "architecture_shared_asset_verification",
        "model_architecture": architecture, "geometry": geometry, "shards": hashes,
        "native_expert_index_complete": True, "native_expert_rows": native_rows,
        "dense_tensors_byte_identical": dense_tensors, "dense_bytes_compared": dense_bytes,
        "dense_pack_sha256": digest(dense), "conversion_manifest_sha256": digest(args.pack / "conversions.json"),
        "source_seed_sha256": digest(args.seed), "common_seed_experts": len(seed), "common_seed_payload_bytes": payload,
        "strata_profile_sha256": digest(profile), "strata_profile": str(profile),
        "references": str(args.references), "weight_bytes_copied": 0,
        "source_sha256": digest(Path(__file__)), "complete": True}
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "pack", "mtp", "seed", "references", "gguf-python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.references.exists():
            raise ValueError("Refusing to overwrite an asset manifest/reference tree.")
        verify(args)
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"vrambridge-architecture-assets: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
