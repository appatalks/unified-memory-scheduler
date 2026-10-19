#!/usr/bin/env python3
"""Restore a native research MTP derivative from verified existing BF16 tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(args):
    output = args.output
    manifest_output = output.with_suffix(".manifest.json")
    partial = output.with_suffix(".gguf.part")
    allowed = (Path("/home/mj420/Desktop/TOR/llm").resolve(),
               Path("/media/mj420/ExpansionXFS1/ai-models").resolve())
    if not any(output.parent.resolve(strict=True).is_relative_to(root) for root in allowed):
        raise ValueError("Derivative must be stored under an approved model root.")
    if any(p.exists() for p in (output, manifest_output, partial)):
        raise ValueError("Refusing to overwrite a derivative, ownership manifest, or partial.")
    if shutil.disk_usage(output.parent).free < 16 * 1024**3 + 4 * 1024**3:
        raise ValueError("Derivative would violate the16GiB reserve plus bounded working allowance.")
    source = args.source.resolve(strict=True)
    inventory = json.loads((source / "mtp-manifest.json").read_text())
    if len(inventory) != 31 or any(r["dtype"] != "BF16" for r in inventory):
        raise ValueError("Unsupported raw MTP manifest.")
    verified = []
    for row in inventory:
        path = (source / row["file"]).resolve(strict=True)
        if not path.is_relative_to(source) or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Raw MTP tensor failed path/length/SHA verification: " + row["name"])
        verified.append(path)
    sys.path[:0] = [str(args.backend_source / "gguf-py"), str(args.backend_source)]
    import gguf
    import numpy as np
    import torch
    from conversion.qwen4exp import Qwen4ExpTextModel

    torch.set_num_threads(8)
    target = gguf.GGUFReader(str(args.target.resolve(strict=True)))
    fields = target.fields
    if fields["general.architecture"].contents() != "qwen4exp":
        raise ValueError("This specialization requires the native qwen4exp capability.")
    n_trunk = fields["qwen4exp.block_count"].contents()
    converter = object.__new__(Qwen4ExpTextModel)
    converter.hparams = {"indexer_n_heads": fields["qwen4exp.attention.indexer.head_count"].contents(),
                        "indexer_head_dim": fields["qwen4exp.attention.indexer.key_length"].contents()}
    converter.block_count = n_trunk + 1
    converter.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN4EXP, converter.block_count)
    converter._mtp_fc = {}
    converter.fuse_gate_up_exps = False
    converter.fuse_qkv = False
    Qwen4ExpTextModel._original_block_count = n_trunk
    Qwen4ExpTextModel.no_mtp = False
    Qwen4ExpTextModel.mtp_only = True
    writer = gguf.GGUFWriter(str(partial), "qwen4exp")
    try:
        for key, field in fields.items():
            if (key.startswith(("GGUF.", "split.", "qwen4exp.ple."))
                    or key in ("general.architecture", "qwen4exp.embedding_length_per_layer_input")):
                continue
            value = field.contents()
            if key == "qwen4exp.block_count":
                value = converter.block_count
            elif key == "qwen4exp.attention.compress_ratios":
                value = list(value) + [max(value)]
            elif key == "qwen4exp.attention.recurrent_layers":
                value = list(value) + [False]
            elif key == "general.file_type":
                value = int(gguf.LlamaFileType.MOSTLY_Q4_0)
            elif key == "general.name":
                value = str(value) + " research native MTP derivative"
            writer.add_key_value(key, value, field.types[0],
                                 field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
        writer.add_nextn_predict_layers(1)
        names = []
        for row, path in zip(inventory, verified):
            count = int(np.prod(row["shape"]))
            tensor = torch.from_file(str(path), shared=False, size=count, dtype=torch.bfloat16).reshape(row["shape"]).float()
            rewritten = Qwen4ExpTextModel.filter_tensors((row["name"], lambda: tensor))
            if rewritten is None:
                raise ValueError("Pinned converter unexpectedly discarded an MTP tensor.")
            name = rewritten[0]
            for mapped, value in converter.modify_tensors(tensor, name, n_trunk):
                array = value.detach().contiguous().numpy()
                if "_exps.weight" in mapped:
                    array = gguf.quantize(array, gguf.GGMLQuantizationType.Q4_0)
                    writer.add_tensor(mapped, array, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
                else:
                    force_f32 = array.ndim <= 1 or mapped.endswith("_norm.weight") or any(
                        converter.match_model_tensor_name(mapped, key, n_trunk) for key in
                        (gguf.MODEL_TENSOR.FFN_GATE_INP, gguf.MODEL_TENSOR.FFN_GATE_INP_SHEXP))
                    writer.add_tensor(mapped, array.astype(np.float32 if force_f32 else np.float16))
                names.append(mapped)
            del tensor
        for tensor in target.tensors:
            if tensor.name in ("token_embd.weight", "output.weight"):
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
                names.append(tensor.name)
        if len(names) != 34 or len(set(names)) != 34 or converter._mtp_fc:
            raise ValueError("Native MTP transform did not produce the required34 unique tensors.")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    reader = gguf.GGUFReader(str(partial))
    if len(reader.tensors) != 34 or reader.fields["qwen4exp.nextn_predict_layers"].contents() != 1:
        raise ValueError("Written derivative failed its structural gate.")
    heads = {t.name: t for t in target.tensors if t.name in ("token_embd.weight", "output.weight")}
    for tensor in reader.tensors:
        if tensor.name in heads and not np.array_equal(tensor.data, heads[tensor.name].data):
            raise ValueError("Target embedding/output bytes changed in derivative.")
    del reader
    os.link(partial, output)
    partial.unlink()
    report = {"schema_version": 1, "kind": "owned_native_mtp_derivative", "originals_modified": False,
        "derived_file": str(output), "derived_bytes": output.stat().st_size, "derived_sha256": digest(output),
        "raw_source_manifest_sha256": digest(source / "mtp-manifest.json"), "raw_tensors_sha_verified": len(verified),
        "target_embedding_and_head_byte_identical": True, "native_tensors": len(names),
        "native_architecture": "qwen4exp", "nextn_layers": 1, "expert_quantization": "Q4_0",
        "hidden_width": fields["qwen4exp.embedding_length"].contents() * fields["qwen4exp.hyper_connection.count"].contents(),
        "upstream_transforms_reused": True, "source_sha256": digest(Path(__file__)),
        "converter_sha256": digest(args.backend_source / "conversion/qwen4exp.py"),
        "numerical_and_quality_qualified": False, "automatic_eviction_allowed": False}
    with manifest_output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("derived_bytes", "derived_sha256", "native_tensors", "hidden_width")}))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "backend-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        restore(args)
        return 0
    except (OSError, ValueError, KeyError, ImportError) as error:
        print("vrambridge-restore-mtp:", error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
