#!/usr/bin/env python3
"""Create small deterministic, untrained MoE GGUFs for graph correctness, never quality or speed claims."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys


PINNED_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
ARCHITECTURES = ("llama", "qwen2moe", "qwen3moe", "gpt-oss", "llama4", "cohere2moe", "grok")
TYPES = ("F32", "F16", "BF16", "Q4_0", "Q4_K", "Q8_0", "MXFP4")


def generate(args) -> dict:
    commit = subprocess.check_output(["git", "-C", str(args.llama_source), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError("Fixture generation requires the pinned llama.cpp source.")
    sys.path.insert(0, str(args.llama_source / "gguf-py"))
    import gguf
    import numpy as np

    quantizer = ctypes.CDLL(str(args.ggml_base.resolve(strict=True)))
    quantizer.ggml_quantize_init.argtypes = [ctypes.c_int]
    quantizer.ggml_quantize_init.restype = None
    quantizer.ggml_quantize_chunk.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
    ]
    quantizer.ggml_quantize_chunk.restype = ctypes.c_size_t
    dtype = gguf.GGMLQuantizationType[args.expert_type]
    quantizer.ggml_quantize_init(int(dtype))
    rng = np.random.default_rng(args.seed)
    partial = Path(str(args.output) + ".part")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(partial, args.architecture)
    writer.add_name("VRAMBridge SYNTHETIC UNTRAINED MoE correctness fixture")
    writer.add_context_length(2048)
    writer.add_embedding_length(args.embedding)
    writer.add_block_count(args.layers)
    writer.add_feed_forward_length(args.feed_forward)
    writer.add_expert_feed_forward_length(args.feed_forward)
    writer.add_expert_count(args.experts)
    writer.add_expert_used_count(args.active)
    writer.add_head_count(4)
    writer.add_head_count_kv(4)
    writer.add_rope_dimension_count(args.embedding // 4)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_layer_norm_eps(1e-5)
    writer.add_uint32(f"{args.architecture}.vocab_size", 259)
    writer.add_file_type({"F32": 0, "F16": 1, "BF16": 32, "Q4_0": 2,
                          "Q4_K": 15, "Q8_0": 7, "MXFP4": 38}[args.expert_type])
    writer.add_tokenizer_model("llama")
    writer.add_token_list(["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)])
    writer.add_token_types([2, 3, 3] + [6]*256)
    writer.add_token_scores([0.0]*259)
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(False)
    writer.add_chat_template("{% for m in messages %}{{ m['content'] + '\\n' }}{% endfor %}assistant:")
    if args.architecture in ("gpt-oss", "cohere2moe"):
        writer.add_uint32(f"{args.architecture}.attention.sliding_window", 128)
    if args.architecture == "qwen2moe":
        writer.add_expert_shared_feed_forward_length(args.feed_forward)
    if args.architecture == "llama4":
        writer.add_uint32("llama4.interleave_moe_layer_step", 2)
        writer.add_uint32("llama4.attention.sliding_window", 128)
    if args.architecture == "cohere2moe":
        writer.add_uint32("cohere2moe.leading_dense_block_count", 1)
        writer.add_float32("cohere2moe.logit_scale", 1.0)
        writer.add_expert_weights_norm(True)

    def tensor(name, shape, norm=False, zero=False):
        if norm:
            data = np.ones(shape, dtype=np.float32)
        elif zero:
            data = np.zeros(shape, dtype=np.float32)
        else:
            data = rng.normal(0, 0.02, shape).astype(np.float32)
        if "_exps.weight" in name and dtype != gguf.GGMLQuantizationType.F32:
            block, size = gguf.GGML_QUANT_SIZES[dtype]
            packed = np.empty((*shape[:-1], shape[-1] // block * size), dtype=np.uint8)
            written = quantizer.ggml_quantize_chunk(int(dtype), data.ctypes.data, packed.ctypes.data,
                                                    0, data.size // shape[-1], shape[-1], None)
            if written != packed.nbytes:
                raise ValueError("Expert quantization returned the wrong byte count.")
            writer.add_tensor(name, packed, raw_dtype=dtype)
        else:
            writer.add_tensor(name, data)

    width, ff, count = args.embedding, args.feed_forward, args.experts
    tensor("token_embd.weight", (259, width))
    tensor("output_norm.weight", (width,), norm=True)
    tensor("output.weight", (259, width))
    routed = []
    for layer in range(args.layers):
        prefix = f"blk.{layer}."
        tensor(prefix + "attn_norm.weight", (width,), norm=True)
        for projection in ("q", "k", "v"):
            tensor(prefix + f"attn_{projection}.weight", (width, width))
        tensor(prefix + "attn_output.weight", (width, width))
        if args.architecture == "gpt-oss":
            tensor(prefix + "post_attention_norm.weight", (width,), norm=True)
            tensor(prefix + "attn_sinks.weight", (4,), zero=True)
            tensor(prefix + "attn_output.bias", (width,))
        elif args.architecture != "cohere2moe":
            tensor(prefix + "ffn_norm.weight", (width,), norm=True)
        if args.architecture == "qwen3moe":
            tensor(prefix + "attn_q_norm.weight", (width // 4,), norm=True)
            tensor(prefix + "attn_k_norm.weight", (width // 4,), norm=True)
        if args.architecture == "grok":
            tensor(prefix + "attn_output_norm.weight", (width,), norm=True)
            tensor(prefix + "post_ffw_norm.weight", (width,), norm=True)
        is_moe = not ((args.architecture == "llama4" and (layer + 1) % 2) or
                      (args.architecture == "cohere2moe" and layer == 0))
        if is_moe:
            routed.append(layer)
            tensor(prefix + "ffn_gate_inp.weight", (count, width))
            if args.architecture == "cohere2moe":
                tensor(prefix + "ffn_gate_up_exps.weight", (count, ff*2, width))
            else:
                for projection in ("gate", "up"):
                    tensor(prefix + f"ffn_{projection}_exps.weight", (count, ff, width))
            tensor(prefix + "ffn_down_exps.weight", (count, width, ff))
            if args.architecture == "gpt-oss":
                tensor(prefix + "ffn_gate_inp.bias", (count,))
                tensor(prefix + "ffn_gate_exps.bias", (count, ff))
                tensor(prefix + "ffn_up_exps.bias", (count, ff))
                tensor(prefix + "ffn_down_exps.bias", (count, width))
            if args.architecture in ("qwen2moe", "llama4"):
                for projection in ("gate", "up"):
                    tensor(prefix + f"ffn_{projection}_shexp.weight", (ff, width))
                tensor(prefix + "ffn_down_shexp.weight", (width, ff))
                if args.architecture == "qwen2moe":
                    tensor(prefix + "ffn_gate_inp_shexp.weight", (width,))
        else:
            for projection in ("gate", "up"):
                tensor(prefix + f"ffn_{projection}.weight", (ff, width))
            tensor(prefix + "ffn_down.weight", (width, ff))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    os.link(partial, args.output)
    partial.unlink()
    return {"schema_version": 1, "synthetic_untrained": True, "quality_or_speed_claim": False,
            "architecture": args.architecture, "layers": args.layers, "routed_layers": routed,
            "embedding": width, "feed_forward": ff, "experts": count, "active": args.active,
            "expert_type": args.expert_type, "seed": args.seed, "size_bytes": args.output.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--ggml-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--expert-type", choices=TYPES, default="F32")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--feed-forward", type=int, default=128)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--active", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    block = 256 if args.expert_type == "Q4_K" else 32
    if (not 2 <= args.layers <= 4 or not 2 <= args.experts <= 16 or not 1 <= args.active <= args.experts or
            not 32 <= args.embedding <= 512 or not 32 <= args.feed_forward <= 1024 or
            args.embedding % block or args.feed_forward % block or not 0 <= args.seed < 2**32):
        parser.error("Require bounded fixture sizes, active <= experts, and quantization-block-aligned dimensions.")
    if args.output.exists() or Path(str(args.output) + ".part").exists():
        parser.error("Refusing to overwrite an existing fixture or partial.")
    try:
        print(json.dumps(generate(args), sort_keys=True))
        return 0
    except (OSError, ValueError, ImportError, AttributeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"MoE fixture: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
