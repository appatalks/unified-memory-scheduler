# Calme 3.2 78B Baseline

Date: 2026-08-22

## Answer

For this 44.34 GB dense Qwen2 model on a 15.45 GiB RTX 5070 Ti, upstream llama.cpp automatic static placement is the fastest measured existing method. With a 1 GiB fit margin it selects 26 of 86 transformer blocks for VRAM and leaves the remaining tensors in system RAM.

- Prompt processing: 185.7449 +/- 0.5519 tokens/s.
- Token generation: 1.3499 +/- 0.0221 tokens/s.
- Manual 25-layer offload is within 2.1% for prompt processing and 0.5% for generation.
- CUDA Unified Memory is 49.0% slower for prompt processing and 84.5% slower for generation.

Steady-state generation under static placement is limited by system RAM read bandwidth, not PCIe or GPU compute. Unified Memory is limited by bidirectional page migration and oversubscription thrash.

## Scope

Model:

- `calme-3.2-instruct-78b-Q4_0.gguf`
- architecture: Qwen2, dense
- parameters: 77,965,463,552
- transformer blocks: 86
- quantized tensor payload: 44,335,136,640 bytes (41.29 GiB)
- context capacity: 32,768 tokens

Host:

- AMD Ryzen 9 9950X, 16 physical cores / 32 threads
- 123.48 GiB RAM
- NVIDIA GeForce RTX 5070 Ti, 15.45 GiB VRAM
- PCIe 5.0 x8, 31.51 GB/s theoretical in each direction
- NVIDIA driver 595.58.03 and CUDA 13.1
- model stored on a rotational XFS disk

Software and workload:

- llama.cpp commit `b21e4de74567f5eef213765c9476a843c2e43f0d`
- 512 prompt tokens and 32 generated tokens
- three measured repetitions
- 16 llama.cpp worker threads
- warm page cache for the comparable matrix

Evidence:

- [hardware report](../../benchmarks/results/hardware-20260822.json)
- [manual/auto-fit and retained invalid-run evidence](../../benchmarks/results/20260822T153837Z-calme-3_2-instruct-78b-Q4_0.json)
- [corrected strict-CPU and Unified Memory runs](../../benchmarks/results/20260822T155123Z-calme-3_2-instruct-78b-Q4_0.json)

The first artifact's `cpu` run is not used below because `-ngl 0` still permitted CUDA operation and KV offload. Its Unified Memory run is retained as a command-parser failure. The corrected configurations use `-ngl 0 -dev none` and `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 -ngl -1`.

## Results

| Placement | Effective GPU blocks | Prompt tok/s | Generation tok/s | Peak process RSS | Peak total framebuffer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict CPU | 0 | 22.5819 +/- 0.2680 | 0.9554 +/- 0.0018 | 80.09 GiB | 1,254 MiB |
| Manual static offload | 25 | 181.9481 +/- 1.3599 | 1.3436 +/- 0.0090 | 41.64 GiB | 14,065 MiB |
| Upstream auto-fit | 26 | **185.7449 +/- 0.5519** | **1.3499 +/- 0.0221** | 41.75 GiB | 14,575 MiB |
| CUDA Unified Memory | all requested | 94.8080 +/- 0.9907 | 0.2088 +/- 0.0024 | 41.64 GiB | 15,804 MiB |

Auto-fit is 8.23 times strict CPU prompt throughput and 1.41 times strict CPU generation throughput. It is 1.96 times Unified Memory prompt throughput and 6.47 times Unified Memory generation throughput.

The strict CPU framebuffer number includes the display/driver baseline and is not process-attributed VRAM. The `-dev none` command disables accelerator selection even though the CUDA-linked binary still initializes the backend.

## Static Placement Bottleneck

The measured sequential RAM read bandwidth is 42.160 GB/s. The GGUF tensor table gives the following host-resident payloads when llama.cpp offloads the final blocks:

| Placement | Host-resident tensor payload | Payload rate at measured generation speed | RAM-read benchmark utilization |
| --- | ---: | ---: | ---: |
| Strict CPU | 44.335 GB/token | 42.358 GB/s | 100.5% |
| Manual, 25 blocks | 31.990 GB/token | 42.983 GB/s | 102.0% |
| Auto-fit, 26 blocks | 31.497 GB/token | 42.519 GB/s | 100.9% |

The approximately 2% spread is within the difference expected between a simple sequential memory benchmark and quantized matrix-vector kernels. All three placements land on the same host-memory roofline. During auto-fit generation, median PCIe RX is only 19 MiB/s, median PCIe TX is 3 MiB/s, median GPU SM activity is 5%, and median GPU memory-controller activity is 1%. The GPU waits while the CPU consumes the host-resident weights at RAM speed.

Prompt processing has a different balance because batched matrix-matrix work amortizes weight reads. During auto-fit prompt processing the GPU averages 80.2% SM activity and 12,585 MiB/s PCIe RX. Static placement therefore gives high prompt throughput even though autoregressive generation remains bounded by host RAM.

## Unified Memory Bottleneck

Unified Memory generation takes 4.790 seconds per token. During the measured generation interval:

- GPU SM activity averages 100.0%, but memory-controller activity averages only 8.0%.
- PCIe RX averages 10,016 MiB/s and PCIe TX averages 10,987 MiB/s.
- Framebuffer use averages 15,733 MiB, effectively filling the GPU.
- Estimated bidirectional PCIe traffic is 98.24 GiB per generated token, 2.38 times the complete model payload.

The combination of full reported SM activity, low device-memory activity, full VRAM, and sustained traffic in both PCIe directions identifies managed-page eviction and refaulting as the bottleneck. The SM utilization is not evidence of productive arithmetic in this case; the low power and memory-controller utilization show that kernels spend most of their time stalled around migration.

Prompt processing reaches 94.81 tokens/s because migration is amortized over a 512-token batch, but it still trails static placement by almost two times.

## Storage Effect

The first cold manual-offload probe took 388.44 seconds wall time while its measured prompt and generation work totaled only 3.66 seconds. Roughly 385 seconds were startup, model loading, and warm-up overhead from the rotational model disk. Once the file was in the page cache, comparable run load/warm-up estimates were 1 to 8 seconds.

Storage dominates cold-start latency but does not explain steady-state token throughput. An NVMe copy should be used for future cold-load and time-to-first-token experiments.

## Milestone Decision

Naive dense layer streaming should not be implemented as the next optimization. Auto-fit leaves 31.497 GB of weights in RAM. Even the measured pinned H2D rate of 28.520 GB/s requires at least 1.104 seconds to transfer that payload once per token, an upper bound of 0.906 tokens/s before GPU computation. That is already below static auto-fit's 1.350 tokens/s. Pageable transfer would be slower still.

The next useful experiments are:

1. Treat upstream auto-fit/static offload as the dense-model baseline.
2. Add higher-resolution phase telemetry with CUPTI or Nsight Systems to count managed-memory faults and migrations directly.
3. Test MoE expert placement or caching, where the active weights are sparse and a working set smaller than the full model can exist.
4. Test longer contexts separately to locate the point where KV-cache capacity and transfer behavior replace weight bandwidth as the limit.

For this dense model and hardware, the fastest policy is static: fill VRAM with complete resident layers, compute the remaining layers on the CPU, and avoid moving their weights across PCIe for every generated token.
