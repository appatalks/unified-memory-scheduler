# Benchmark Ledger

This ledger is the reviewed index for benchmark evidence committed under `results/`.
It records the conditions required to compare runs, links raw artifacts, and
separates valid measurements from failed-command evidence. New result JSON is
ignored by default and is added here only after review.

## Test Host

- AMD Ryzen 9 9950X: 16 physical cores
- 123.48 GiB host RAM
- NVIDIA GeForce RTX 5070 Ti: 15.45 GiB VRAM
- PCIe 5.0 x8; measured pinned H2D: 28.520 GB/s
- measured sequential RAM read: 42.160 GB/s
- model storage: Samsung 990 PRO NVMe
- llama.cpp: `b21e4de74567f5eef213765c9476a843c2e43f0d`

Unless a row states otherwise, measurements use three repetitions, a 512-token
prompt, a 32-token generation, and a 1,024 MiB auto-fit margin at a 4,096-token
fit context.

## Reviewed Results

| Model | Placement | Prompt tok/s | Generation tok/s | Evidence |
| --- | --- | ---: | ---: | --- |
| Calme 3.2 78B Q4_0 | auto-fit, 26 of 86 dense blocks in VRAM | 185.74 +/- 0.55 | 1.35 +/- 0.02 | [matrix](results/20260822T153837Z-calme-3_2-instruct-78b-Q4_0.json) |
| GPT-OSS 20B MXFP4 | auto-fit, full GPU | 9,289.33 +/- 394.37 | 217.18 +/- 1.52 | [run](results/20260823T143325Z-gpt-oss-20b-MXFP4.json) |
| Qwen3-Coder 30B-A3B Q3_K_L | auto-fit, near/full GPU | 4,048.34 +/- 120.65 | 204.87 +/- 6.82 | [run](results/20260823T143354Z-Qwen3-Coder-30B-A3B-Instruct-Q3_K_L.json) |
| Qwen3.6 35B-A3B Q8_0 | all routed experts on CPU | 313.22 +/- 3.62 | 30.83 +/- 0.17 | [run](results/20260823T143533Z-Qwen3_6-35B-A3B-Q8_0.json) |
| Qwen3.6 35B-A3B Q8_0 | auto-fit GPU/CPU expert placement | 487.95 +/- 7.19 | 41.41 +/- 0.67 | [run](results/20260823T143433Z-Qwen3_6-35B-A3B-Q8_0.json) |
| GPT-OSS 120B MXFP4 | all routed experts on CPU | 232.54 +/- 1.47 | 19.30 +/- 0.12 | [run](results/20260823T151527Z-gpt-oss-120b-MXFP4-00001-of-00002.json) |
| GPT-OSS 120B MXFP4 | auto-fit GPU/CPU expert placement | 305.95 +/- 13.27 | 22.75 +/- 0.31 | [run](results/20260823T151455Z-gpt-oss-120b-MXFP4-00001-of-00002.json) |

The dense 78B result is host-RAM-bandwidth-bound. The large MoE results instead
compute CPU-resident routed experts in place; their generation PCIe traffic is
activation-scale rather than full-expert transfer traffic. See
[frontier-moe-feasibility.md](../docs/research/frontier-moe-feasibility.md) for
the GGUF payload accounting and phase telemetry.

## Context Sweep

Qwen3.6 35B-A3B Q8_0 used auto-fit placement with Q8_0 K/V cache. Each point
uses a prompt half as long as its fit context and generates 32 tokens.

| Fit context | Prompt | Prompt tok/s | Generation tok/s | Evidence |
| ---: | ---: | ---: | ---: | --- |
| 16K | 8,192 | 457.53 +/- 14.27 | 40.96 +/- 0.20 | [run](results/20260823T150402Z-Qwen3_6-35B-A3B-Q8_0.json) |
| 32K | 16,384 | 437.16 +/- 6.69 | 40.55 +/- 0.76 | [run](results/20260823T150701Z-Qwen3_6-35B-A3B-Q8_0.json) |
| 64K | 32,768 | 426.94 +/- 13.07 | 40.55 +/- 0.44 | [run](results/20260823T151238Z-Qwen3_6-35B-A3B-Q8_0.json) |

## Retained Failed Evidence

| Artifact | Reason retained |
| --- | --- |
| [initial dense matrix](results/20260822T153837Z-calme-3_2-instruct-78b-Q4_0.json) | Its nominal CPU run was not strict CPU because CUDA operation/KV offload remained active. Its Unified Memory command used invalid `-ngl all` syntax. Valid manual and auto-fit runs remain comparable. |
| [Qwen3.6 context attempt](results/20260823T150213Z-Qwen3_6-35B-A3B-Q8_0.json) | Pinned `llama-bench` does not accept a general `-c` context override. The corrected context series uses `--fit-ctx` and Q8_0 K/V cache types. |

## Pending Candidates

These are not benchmark results and must not be compared with the rows above.

| Candidate | Status | Next gate |
| --- | --- | --- |
| GLM-5.2 UD-Q4_K_M | Download in progress; 466 GB expected GGUF, 753B total / 40B active | Verify full shard set and header, then run a fit-only probe. The complete expert pool exceeds current host RAM, so an interactive run is not expected on this host. |
| Qwen3.6 27B Q4_K_M or Q6_K | Candidate dense quality comparison | Run fit-only probe first, then the same 512/32/3 matrix if it fits. |

## Tracking Rules

1. Retain raw JSON for every reviewed conclusion, including invalid commands and failed runs that changed the interpretation.
2. Record model file, quantization, llama.cpp commit, fit margin/context, prompt and generation tokens, repetitions, and K/V cache types.
3. Do not compare token rates across different model architectures as quality scores. Compare placement policies only within an identical model and workload.
4. Add a result to this ledger only after focused validation and a review of placement and telemetry.