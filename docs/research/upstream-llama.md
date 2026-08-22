# Upstream llama.cpp Memory Research

## Pinned Revision

This audit was performed on 2026-08-22 against:

- Repository: `https://github.com/ggml-org/llama.cpp`
- Branch observed: `master`
- Commit: `b21e4de74567f5eef213765c9476a843c2e43f0d`
- Commit date: `2026-08-22T16:33:47+02:00`
- Subject: `mtmd: use ggml_rope_set_offset (#27521)`

All source links below include the pinned commit. Upstream moves quickly, so VRAMBridge result files must record the tested llama.cpp commit and must not silently substitute a newer build.

## Current Mechanisms

### CUDA Unified Memory

`GGML_CUDA_ENABLE_UNIFIED_MEMORY` is tested for presence, not for a truthy value. Even `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0` enables the path. CUDA device allocations made through `ggml_cuda_device_malloc` use `cudaMallocManaged` instead of `cudaMalloc` when it is present.

On Linux, the same setting also makes the CUDA backend's free-memory query use host `MemAvailable`/swap or free hugetlb pages. This allows fitting logic to consider a larger allocation envelope. It does not remove PCIe migration costs or make host RAM equivalent to VRAM. No CUDA prefetch/advice policy for discrete NVIDIA GPUs is applied in this allocation path.

Sources: [allocation](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/ggml/src/ggml-cuda/ggml-cuda.cu#L138-L165), [memory reporting](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/ggml/src/ggml-cuda/ggml-cuda.cu#L4770-L4810), [build documentation](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/docs/build.md).

### Automatic Fitting

Common CLI tools default `--fit` to on and use a 1024 MiB per-device margin. `common_fit_params` creates no-allocation model/context instances and reads a model/context/compute memory breakdown by backend buffer type. It may reduce an automatic context down to `--fit-ctx`, then searches layer placement against per-device targets.

Dense models are filled back-to-front in contiguous device slices. For MoE, fitting first tests dense portions while sparse expert weights overflow to host memory, then may add full or partial layer fractions through generated tensor buffer overrides. The implementation can distinguish attention, up, gate, and sparse-MoE fractions.

Important constraints:

- The fit API assumes system memory is unlimited.
- It only changes model parameters that still equal llama.cpp defaults; context changes only when `n_ctx == 0`.
- It is not thread safe because it temporarily changes global logging.
- Tensor split mode is not supported by the fit implementation.
- Explicit user `n_gpu_layers`, tensor split, or tensor buffer overrides can prevent automatic adjustment.
- `llama-bench` activates fitting with `--fit-target`; unlike the common CLI, its default benchmark path does not automatically fit unless requested.

Sources: [`fit.h`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/common/fit.h), [`fit.cpp`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/common/fit.cpp), [`llama-bench.cpp`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/tools/llama-bench/llama-bench.cpp).

### Layer and Tensor Placement

`--n-gpu-layers` accepts an integer, `auto`, or `all`. At the public API, a negative `llama_model_params::n_gpu_layers` means all layers; common CLI state uses `-1` as the automatic sentinel until fitting resolves it.

`--override-tensor` parses regular-expression patterns into a null-terminated `llama_model_tensor_buft_override` list. During tensor creation, the first matching pattern selects a backend buffer type after compatibility checks. CPU overrides consider CPU extra buffer types. Upstream warns that CPU overrides combined with mmap may perform poorly and recommends `--load-mode none` for that case.

`--cpu-moe` is syntax sugar for a CPU buffer override matching sparse expert feed-forward tensors. `--n-cpu-moe N` creates equivalent patterns for the first `N` layers. These are static model-load placements, not a dynamic expert cache.

Sources: [`arg.cpp`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/common/arg.cpp#L2734-L2778), [`llama.h`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/include/llama.h#L302-L346), [`llama-model-loader.cpp`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/src/llama-model-loader.cpp#L1160-L1235).

### Operation Offload and Backend Scheduler

The ggml backend scheduler owns compute-buffer allocation, operation assignment, graph splitting, and cross-backend copies. Preallocated weight location strongly influences operation placement. With operation offload enabled, an operation whose weight input is in host memory may be assigned to a higher-priority accelerator only when that backend supports the operation and its device-specific `offload_op` policy opts in.

The scheduler exposes reserve, split, explicit tensor-backend assignment, asynchronous graph compute, synchronization, buffer-size queries, and an evaluation callback. It already contains a specialized optimization that can copy only selected experts for certain cross-backend MoE operations. It does not provide a persistent general-purpose cross-token weight cache.

`--no-op-offload` disables the host-operation promotion policy. It does not change the model weight's buffer allocation and is not equivalent to tensor placement.

Sources: [`ggml-backend.h`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/ggml/include/ggml-backend.h#L266-L363), [`ggml-backend.cpp`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/ggml/src/ggml-backend.cpp).

### Model mmap and Loading

Load mode `auto` starts with mmap and falls back when a selected backend lacks mmap support. Linux mappings use `MAP_SHARED`, sequential file advice, and optionally `MAP_POPULATE`. Compatible host-backed tensors can use mapped file regions directly; offloaded tensors are allocated in their backend buffer and populated from the mapped data.

The non-mmap loader already implements bounded asynchronous upload staging: four backend host buffers, normally 1 MiB each or about 64 MiB for aligned direct I/O, plus backend events. This path accelerates model loading only. It is not inference-time transformer-layer streaming and must be treated as an upstream baseline for any VRAMBridge loading work.

Sources: [`llama-mmap.cpp`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/src/llama-mmap.cpp), [`llama-model.cpp`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/src/llama-model.cpp#L1288-L1714), [`llama-model-loader.cpp`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/src/llama-model-loader.cpp#L1431-L1702).

### CUDA Buffers

The CUDA backend exposes device, split, and host buffer types. Regular device buffers allocate through the CUDA allocator (`cudaMalloc` or managed allocation under the environment switch), initialize tensor padding where needed, and use CUDA backend methods for sets, gets, and copies. Buffer allocation is static for model weights and compute reservations; it is not a demand-managed VRAM tensor cache.

Source: [`ggml-cuda.cu`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/ggml/src/ggml-cuda/ggml-cuda.cu).

### KV Cache Placement

The context constructs K/V tensors per model layer and groups allocation by backend buffer type. With KV offload enabled, each layer's KV buffer follows `model.dev_layer(il)`; with `--no-kv-offload`, it uses the CPU buffer type. The entire selected cache allocation is created up front and cleared. Current CLI controls K/V data types and static offload, but there is no supported API for migrating old KV regions between VRAM and RAM while decoding.

Unified KV in current terminology controls whether sequences share one cache stream; it is unrelated to CUDA Unified Memory.

Source: [`llama-kv-cache.cpp`](https://github.com/ggml-org/llama.cpp/blob/b21e4de74567f5eef213765c9476a843c2e43f0d/src/llama-kv-cache.cpp#L64-L300).

## Integration Decision

Milestone 1 uses llama.cpp out of process through `llama-bench` JSON. The next integration step should link the public llama API and use model parameters plus tensor buffer overrides. Telemetry can initially use a very small instrumentation patch around graph execution and backend copies.

Candidate extension order:

1. Public llama API and existing placement controls.
2. ggml backend scheduler evaluation callback for operation/layer timing where sufficient.
3. Thin, versioned instrumentation patch for copy events and layer identity.
4. Custom backend buffer type or scheduler extension for explicit residency and prefetch.

`src/llama-ext.h` is explicitly a WIP staging API that permits breaking changes. VRAMBridge may experiment with it behind a commit-specific adapter, but must not make it the stable project boundary.

## Unknowns Requiring Measurement

- Whether managed allocations on this Blackwell consumer GPU exhibit repeatable migration behavior across prompt processing and token generation.
- Whether the current scheduler's selected-expert copy path overlaps transfers sufficiently to compete with static CPU MoE.
- How much operation offload helps when weight copies approach the active PCIe Gen5 x8 ceiling.
- Whether mmap plus CPU tensor overrides causes page-fault or copy behavior significant enough to prefer non-mmap loading for mixed placement.
- Which minimal hook can expose layer and copy timing without a long-lived fork.