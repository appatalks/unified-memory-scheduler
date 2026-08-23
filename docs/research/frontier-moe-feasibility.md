# Frontier MoE Feasibility

Date: 2026-08-23

## Answer

A 16 GiB RTX 5070 Ti and 128 GiB host-RAM system can run selected frontier-scale sparse MoE models at interactive generation speed with upstream llama.cpp. This is not a result of treating system RAM as VRAM or of CUDA Unified Memory. llama.cpp keeps the transformer spine and a selected prefix of routed experts in VRAM, while the remaining experts are static CPU-resident tensors.

This machine is not similarly suitable for dense frontier models. The earlier 44.34 GB dense 78B baseline generated at 1.35 tok/s because every token reread approximately 31.50 GB of host-resident weights. CUDA Unified Memory was slower still because it migrated pages over PCIe.

## Host And Method

- CPU: AMD Ryzen 9 9950X, 16 physical cores
- RAM: 123.48 GiB
- GPU: NVIDIA GeForce RTX 5070 Ti, 15.45 GiB VRAM
- PCIe: Gen5 x8, 31.51 GB/s theoretical per direction
- measured RAM read: 42.160 GB/s
- measured pinned H2D: 28.520 GB/s
- model storage: Samsung 990 PRO NVMe at `/media/mj420/E65882B158827FD3/0/AI-MODELS/models`
- engine: official llama.cpp commit `b21e4de74567f5eef213765c9476a843c2e43f0d`, native CUDA 12.0 build
- workload: 512 prompt tokens, 32 generated tokens, three repetitions
- auto-fit policy: 4,096-token fit context with a 1,024 MiB VRAM margin

Generated raw evidence is retained locally under `benchmarks/results/` and ignored by default pending a dedicated review/commit.

## Local Model Fits

| Model | Architecture | File size | Routed experts | Auto-fit at 4K, 1 GiB margin | Feasibility |
| --- | --- | ---: | ---: | --- | --- |
| Gemma 4 E4B Q4_K_M | MoE | 5.34 GB | metadata not exposed in this header | all GPU | full-GPU control |
| gpt-oss-20b MXFP4 | MoE | 12.11 GB | 4 of 32 per layer | all GPU | full GPU |
| gpt-oss-120b MXFP4 | MoE | 63.39 GB, two GGUF shards | 4 of 128 across 36 blocks | GPU spine plus CPU experts | oversized but interactive |
| Qwen3-Coder-30B-A3B Q3_K_L | MoE | 14.58 GB | 8 of 128 across 48 blocks | all GPU for benchmark workload | near-VRAM-bound |
| Qwen3.6-35B-A3B Q8_0 | MoE | 36.90 GB | 8 of 256 across 40 blocks | GPU spine plus CPU experts | oversized but interactive |
| Calme 3.2 78B Q4_0 | dense | 44.34 GB | none | 26 of 86 complete blocks in VRAM | RAM-bandwidth limited |

`llama-fit-params` selected all layers for the first two models. For Qwen3.6 it selected GPU execution plus CPU tensor overrides for the expert tensors in the later blocks. Fit results are capacity estimates; throughput results below are the authoritative evidence.

## Context Sweep

Qwen3.6 was measured at 4K, 16K, 32K, and 64K auto-fit contexts with both K and V caches in Q8_0. Each point uses a prompt half as long as the fit context, then generates 32 tokens. The pinned `llama-bench` interface uses `--fit-ctx` to control the placement context; it has no general `-c` context argument. The rejected `-c 16384` attempt is retained as failed command evidence.

| Fit context | Measured prompt | Prompt tok/s | Generation tok/s | Peak framebuffer |
| ---: | ---: | ---: | ---: | ---: |
| 4K | 512 | 487.95 +/- 7.19 | 41.41 +/- 0.67 | 14.85 GiB |
| 16K | 8,192 | 457.53 +/- 14.27 | 40.96 +/- 0.20 | 14.74 GiB |
| 32K | 16,384 | 437.16 +/- 6.69 | 40.55 +/- 0.76 | 14.80 GiB |
| 64K | 32,768 | 426.94 +/- 13.07 | 40.55 +/- 0.44 | 14.73 GiB |

Generation remains within 2.1% of the 4K result through the 64K fit context. Prompt throughput declines by 12.5% from the 512-token probe to the 32K-token probe, while remaining above 426 tok/s. Q8_0 KV cache preserves the selected expert placement under this load; no CUDA Unified Memory fallback or out-of-memory event occurred.

## Measurements

| Model and placement | Prompt tok/s | Generation tok/s | Peak process RSS | Peak framebuffer |
| --- | ---: | ---: | ---: | ---: |
| gpt-oss-20b MXFP4, auto-fit all GPU | 9,289.33 +/- 394.37 | 217.18 +/- 1.52 | 11.90 GiB | 12.73 GiB |
| gpt-oss-120b MXFP4, all routed experts CPU | 232.54 +/- 1.47 | 19.30 +/- 0.12 | 59.09 GiB | 3.46 GiB |
| gpt-oss-120b MXFP4, auto-fit expert placement | 305.95 +/- 13.27 | 22.75 +/- 0.31 | 59.68 GiB | 14.79 GiB |
| Qwen3-Coder-30B-A3B Q3_K_L, auto-fit | 4,048.34 +/- 120.65 | 204.87 +/- 6.82 | 14.20 GiB | 14.60 GiB |
| Qwen3.6-35B-A3B Q8_0, all routed experts CPU | 313.22 +/- 3.62 | 30.83 +/- 0.17 | 34.52 GiB | 4.09 GiB |
| Qwen3.6-35B-A3B Q8_0, auto-fit expert placement | **487.95 +/- 7.19** | **41.41 +/- 0.67** | 35.04 GiB | 14.52 GiB |
| Calme 3.2 78B Q4_0 dense, auto-fit | 185.74 +/- 0.55 | 1.35 +/- 0.02 | 41.75 GiB | 14.58 GiB |

The Qwen3.6 auto-fit policy improves generation by 34.3% and prompt throughput by 55.8% over computing every routed expert on the CPU.

The gpt-oss-120b auto-fit policy improves generation by 17.8% and prompt throughput by 31.6% over computing every routed expert on the CPU. The smaller percentage reflects that most of this 120B expert pool must remain in RAM even after fitting the GPU prefix.

## GPT-OSS 120B Weight Accounting

The multipart MXFP4 GGUF has 63.374 GB of tensor payload across two files:

- routed-expert tensors: 61.073 GB;
- shared FFN tensors: 0.053 GB;
- all remaining tensors: 2.248 GB;
- 36 MoE blocks, each with 1.696 GB of routed experts;
- four active experts of 128 per block.

At a 1 GiB 4K-context margin, auto-fit leaves about six complete expert blocks in VRAM and places about 50.894 GB of experts in RAM. The host side activates 1.590 GB of expert weight data per token. At the measured 22.75 tok/s, that is 36.18 GB/s, or 85.8% of the 42.16 GB/s measured RAM-read bandwidth.

The all-CPU-expert placement activates 1.909 GB/token. At 19.30 tok/s, it consumes 36.84 GB/s, or 87.4% of the RAM-read benchmark. Generation-phase PCIe traffic remains small in both configurations: auto-fit median RX/TX was 62/21 MiB/s, while the all-CPU baseline was 72/59 MiB/s. The model uses static CPU expert execution, not per-token PCIe expert transfers or managed-memory migration.

## Qwen3.6 Weight Accounting

The Q8_0 GGUF contains 36.892 GB of tensor payload:

- routed-expert tensors: 34.226 GB
- shared FFN tensors: 0.218 GB
- all remaining tensors: 2.449 GB
- expert payload per block: 855.638 MB
- one block's active expert set: 26.739 MB/token, from 8 of 256 experts

The fitted layout retains the expert pool in blocks 0 through 12 in VRAM (11.123 GB) and leaves blocks 13 through 39 in RAM (23.102 GB). The CPU-resident portion therefore activates about 722 MB of expert weights per generated token. At 41.41 tok/s, this is roughly 29.9 GB/s of host expert traffic, below the measured 42.16 GB/s RAM-read ceiling.

The all-CPU-expert placement activates approximately 1.070 GB/token. At 30.83 tok/s, that implies about 33.0 GB/s of host expert traffic. It consumes more CPU (493.5% whole-process average versus 329.6% for auto-fit) and leaves most of VRAM unused.

During auto-fit generation, median PCIe traffic was only 125 MiB/s RX and 77 MiB/s TX. The model does not move full expert weights over PCIe per token. It computes CPU-resident experts in place and transfers only activation-scale values between the CPU and GPU portions.

## Implications

1. Sparse MoE changes the relevant bandwidth equation. Total parameter count is a capacity requirement, but active-expert bytes per token govern decode bandwidth.
2. Static expert placement is already an effective baseline. It is materially different from dense whole-layer offload and avoids CUDA Unified Memory page thrashing.
3. A future VRAMBridge MoE cache should be evaluated against this 41.41 tok/s static baseline, not against the dense 78B result. It must show that dynamically caching experts yields more benefit than the existing static prefix placement.
4. A cache experiment needs routing telemetry: per-block expert IDs, temporal locality, cache hit rate, transfer bytes, CPU expert time, and end-to-end generation latency.
5. The full-GPU gpt-oss-20b result is the current high-throughput local frontier-model path. The 35B-A3B result is the current high-capacity sparse-model path.

## Next Experiments

1. Sweep the automatic fit margin from 1,024 MiB down to the highest safe value at the intended 64K context, then retain the best stable configuration.
2. Measure time-to-first-token with an interactive runner at 16K, 32K, and 64K contexts; `llama-bench` records throughput but not interactive TTFT.
3. Instrument Qwen3.6 and gpt-oss-120b routing to measure actual temporal locality and derive an LRU cache upper bound.
4. Compare a fixed expert-cache policy with static auto-fit placement, holding prompt, context, sampling, and expert quantization constant.
5. Add CPU/PCIe overlap microbenchmarks sized to real active-expert sets: 26.739 MB per Qwen3.6 block and 53.015 MB per gpt-oss-120b block.

Do not use the unverified FreeToken desktop distribution as a production dependency. Its design is useful as a research reference, but this baseline is produced entirely through the pinned, open llama.cpp checkout and local GGUF files.
