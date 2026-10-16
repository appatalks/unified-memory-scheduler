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
- model storage: Samsung 990 PRO NVMe at `/models` (sanitized publication path)
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

## 120B Browser App Tuning

Screening measurements on 2026-10-16 UTC use the same two-shard
GPT-OSS-120B MXFP4 model on the host above, with official llama.cpp build 11669
(`620d2225c`), a 16,384-token context, and one server slot. The fixed chat prompt
is 98 tokens under that backend's template. Requests use temperature zero, seed
42, low reasoning effort, prompt caching disabled, and exactly 256 output tokens
with EOS ignored. These are bounded decode screening runs, not natural-answer
quality scores or a claim of sustained speed on all prompts.

| Configuration | Runs | Median generation tok/s |
| --- | ---: | ---: |
| Initial automatic placement, mmap, 30 generation/prompt threads | 1 | 11.34 |
| Automatic placement, mmap, 16 generation/prompt threads | 2 | 21.33 |
| Automatic placement, mmap, 8 generation / 16 prompt threads | 2 | 22.20 |
| Automatic placement, mmap, 4 generation / 16 prompt threads | 2 | 22.26 |
| Automatic placement, copy into RAM, 8 generation / 16 prompt threads | 2 | 22.14 |
| All experts on host plus upstream 8 GiB GPU expert cache, 8/16 threads | 2 | 14.59 |
| All experts on host plus upstream 10 GiB GPU expert cache, 8/16 threads | 2 | 17.15 |

All 16/8/4-thread static and explicit-RAM runs produced identical generated-text
SHA-256 values. The cache runs produced a different, internally repeatable text
hash and were slower; they are not enabled as a production optimization.
Raw local reports are not published without provenance and privacy review.

Profiling the initial 30-thread run measured 2,576% mean whole-process CPU usage
(about 26 logical cores), 59.5 GiB peak process residency, and 18.1% mean GPU
utilization. Most resident host pages were file-backed mappings. A monitor that
excludes file-backed pages from application memory can make this look like empty
RAM, but `/proc/PID/status` and CPU-time deltas show real residency and execution.
Using more SMT threads was counterproductive for this workload.

The restored official 4-generation/16-prompt-thread control was additionally
qualified with three exact 512-token generations at 16K context. Wall-clock
decode rates were 22.11, 22.28, and 22.07 tok/s; all three generated-text hashes
matched. Including prompt processing, request throughput was 19.63-19.81 tok/s.
This is a stable control, not a breakthrough beyond the published baseline.

A GUI probe without an explicit sampling temperature returned the incorrect
answer 289 for 17 times 19. Two subsequent tests of that exact prompt with
temperature zero, seed 42, and prompt caching disabled both returned 323.
This is not a general quality pass: sampling must be controlled before assigning
answer differences to a placement change, and correctness remains a separate gate.

These stock-backend results do not reach even the initial 30 tok/s milestone,
let alone the revised 60 tok/s minimum. The older 22.75 tok/s
result above used a different backend, 4K fit context, and shorter generation;
it is not a matched speedup comparison. Subsequent backend/policy candidates must
retain the same weights, actual context, and token-count requirements, with
separate correctness and representative natural-answer checks.

An explicitly opted-in native build of `ik_llama.cpp` at
`8313388ab528503bf503aee0c51e7b30e1cda378` was also tested. It used CUDA SM 120,
native AVX-512/VNNI/BF16 CPU flags, the fork's automatic tensor fitting, and the
same 16K requested context and 256-token chat workload. Median decode rates
were 21.54 tok/s at 8 generation threads, 21.91 at 4 threads, and 21.60 at
16 threads (two runs each, 16 prompt threads throughout). This did not improve
the official-backend control. Backend templates/numerical paths produced a
different text hash, so this is a compatibility/performance screen, not a
bit-identical kernel comparison. The official 4-generation/16-prompt-thread
profile was restored; no fork replaced the default executable.

### Speculative Screening

A target-trained 0.8B DFlash drafter was checksum-verified and converted separately
for the two backends. No target weights were changed. The fork loaded its own
format but measured only 10.16 wall-clock decode tok/s in a 128-token smoke test.
It accepted 74 of 253 proposed tokens; this did not establish a speedup.

The official loader required the optional attention-bias compatibility patch
documented in the app guide. Its shared-target draft memory estimate failed,
so a 1 GiB reserve caused OOM; these successful tests explicitly reserved 4 GiB.
The locally built official server used the CUDA plugin from the identical
upstream commit, not an ABI-mismatched backend.

| Maximum draft tokens | Output tokens/run | Runs | Median wall-clock decode tok/s |
| --- | ---: | ---: | ---: |
| 3 | 256 | 2 | 15.37 |
| 5 | 256 | 1 | 13.56 |
| 9 | 256 | 2 | 10.32 |

Wall-clock decode excludes backend-reported prompt processing from total request
time and divides the remaining interval into accepted output tokens; it agrees
with the backend's reported generation rate. It does not count proposed/rejected
draft tokens as output. These results are slower than the non-speculative control,
are not a representative quality/equivalence evaluation, and do not meet the
60 tok/s minimum or 90 tok/s stretch goal. Speculation is disabled by default.

### Cache Feasibility Gate

The next experiment should collect actual per-layer, per-token expert IDs, then
simulate expert-granular residency before changing execution. Trace collection
must not be treated as a performance measurement because copying routing IDs
can introduce synchronization.

Using the published 4K-layout estimate of 1.59 GB of host expert bytes/token and
the measured 42.16 GB/s read bandwidth gives these optimistic, zero-overhead bounds:

| Decode target | Maximum cold host expert bytes/token | Reduction from 1.59 GB/token |
| --- | ---: | ---: |
| 60 tok/s | 0.703 GB | 55.8% |
| 90 tok/s | 0.468 GB | 70.5% |

These are not raw expert-hit-rate requirements or guarantees of achievable speed.
The actual 16K layout and expert byte sizes must be measured. GPU computation,
activation transfers, cache warm-up, PCIe weight uploads, and synchronization
consume additional time, so real residency/reuse requirements are higher.

Compare static placement, LRU, LFU, and an explicitly labeled future-aware oracle
on held-out coding, reasoning, and prose traces. Account for variable expert bytes,
reuse distance, cold CPU execution, transfer/eviction costs, and per-layer critical
paths. Misses need not trigger migration: cold experts should remain eligible for
CPU execution. A simulator that omits transfers or sums unrealistically perfect
overlap cannot establish viability.

The upstream cache tested above uploads misses into device slots; its poor result
does not disprove a hybrid GPU-hit/CPU-miss policy. Reuse its tensor/slot machinery
only after investigating expert-granular dispatch and correct partial-output
reduction. Prototype that path only if the calibrated oracle bound leaves
headroom above 60 tok/s.

## Real Routing And Costed Replay

The first trace/simulator implementation was validated on the actual 120B model
at 16K context with the official build 11669 libraries, four decode threads, and
16 prompt threads. The public routing corpus contains three training and three
held-out requests covering coding, reasoning, and prose, each limited to 256
generated tokens. All six produced 256 routed generated tokens: 1,536 total,
55,296 layer/token observations, and 221,184 selected-expert references.
Two independent collections had 27,648 identical shared-prefix layer/token
observations. The final trace records its collector source hash and initialization
parameters; the report records its own source hash and the input trace hash.

Collection is decode-only, greedy, and instrumented. Prefill can prune the last
FFN to output positions; enforcing full prefill coverage would be incorrect.
The callback adds synchronization, so collection time is not a throughput result.
Routing does not establish answer quality or token/logit equivalence.

The original loaded tensors give 36 routed layers, 128 experts per layer, four
selected per token, and 13,253,760 storage bytes per complete expert group,
including biases. Actual GPU expert storage is 10,744,381,440 bytes (about
10.006 GiB), enough for 810 complete groups with some space left. The fitted
layout includes mixed host/device components in layer 6, not just whole blocks.
The analyzer preserves that partial placement in the static baseline.

The estimates below use measured RAM/PCIe inputs of 42.16/28.52 GB/s, provisional
calibration to the uninstrumented 22.12 tok/s control, and the same expert byte
budget for every hypothetical cache. Missing GPU-expert and extra synchronization
costs are explicitly zero as optimistic assumptions. The calibrated residual is
about 7.90 ms/token and is assumed unchanged, which is not validated for a new
executor. Host storage is not itself proof of where every operation executes.

| Policy | GPU byte coverage | Weight uploads GB/token | Staged optimistic estimate tok/s | No-staging-copy sensitivity tok/s |
| --- | ---: | ---: | ---: | ---: |
| Actual static placement | 17.59% | 0 | 22.12 (calibration) | 22.12 |
| Training-only fixed hot set | 51.13% | 0 | 33.30 | 33.30 |
| LRU with repeat admission | 65.71% | 0.6501 | 14.35 | 25.74 |
| LFU with repeat admission | 53.44% | 0.0197 | 32.92 | 33.96 |
| Future-aware equal-size hit oracle | 83.44% | 0.2192 | 32.26 | 48.54 |

These are **simulation estimates, not achieved inference speeds**. Staged
uploads include a read+write staging copy plus DMA's RAM read; CPU and DMA share
the RAM bus. The no-staging case assumes already-pinned source data and is not
an implemented memory-management mechanism. Warm-up uploads and cache
initialization are separate fields in the generated reports.

The training-derived plan distributes 810 whole expert groups across all 36
layers, rather than keeping complete blocks. Its per-category estimates are
32.82 tok/s for coding, 29.20 for reasoning, and 39.42 for prose under the model.
A hypothetical future-aware *fractional fixed* hot set reaches 56.78% byte
coverage and a RAM-only upper bound of 51.11 tok/s under the supplied bandwidth
assumption, even before other costs. This rules out 60 for that fixed-residency
relaxation on this small sample; it does not rule out every adaptive algorithm
or establish the physical peak bandwidth of the machine.

The hit oracle's no-staging scenario leaves only about 3.97 ms/token for other
work at a 60 tok/s target, compared with the provisional 7.90 ms residual.
High hit rate alone therefore does not establish viability. Cost-aware admission,
lower staging traffic, and lower dispatch/synchronization overhead need evidence
before a live kernel implementation is justified. Prefill cache effects and larger,
more representative corpora remain untested.

Raw local traces/reports remain ignored, not automatically published.
The [workflow and schemas](../../benchmarks/README.md#routing-collection-and-cache-replay)
document reproduction, validation, privacy, and the limits of each oracle.
The collection and replay do not replace the browser app's non-speculative control.

### Follow-Up: Causal Cost-Aware Admission

A replay-only policy now uses a rolling 64-generated-token history to estimate
future reuse. It admits only when predicted CPU-read savings exceed transfer
and staging cost plus the lost benefit of evicted entries. Reuse predictions
are causal; the policy never consults future test routing. Supplied GPU expert
cost is deducted from expected savings. This is not a new live executor.

On the same completed 1,536-token trace and staged cost inputs, this policy
covered 62.54% of held-out expert bytes while uploading about 0.0635 GB/token,
versus LRU's 0.6501 GB/token. It rejected 27,841 held-out admissions. The
conditional serial/optimistic estimates were 33.22/34.03 tok/s, compared with
LRU's 12.98/14.35 tok/s. In the explicitly optimistic no-staging-copy sensitivity
case, cost-aware admission gives a 37.53 tok/s estimate at 63.58% byte coverage
with 0.0950 GB/token of uploads. Neither scenario reaches the target.
The 64-token window was not tuned against held-out
results. These estimates still depend on the earlier bandwidth measurements,
zero unspecified GPU/synchronization costs, and an assumed unchanged baseline
residual; they are not measured performance or proof of 60 tok/s feasibility.

Before reboot, the subsequent observer-equivalence collection matched generated token sequences
for four requests, then was interrupted by a CUDA launch failure. Its `.part`
file is incomplete and is not promoted to a validated trace. On resumption,
`cuInit(0)` and the official backend reported CUDA error 999 / no usable GPU,
even though NVML/`nvidia-smi` remained responsive. No driver reset or reboot was
performed by the assistant. At that point full six-request token-equivalence
validation and restoring the GPU-loaded app were blocked. Replay of the earlier
complete trace explicitly reported equivalence as unverified.
The collector now checks initialized device enumeration before loading weights,
rather than relying on a compile-time GPU-support flag. The app also explicitly
stops automatic loads when CUDA initialization reports failure; CPU-only mode
remains intentional.

### Post-Reboot Verification And Refreshed Calibration

After the user rebooted on 2026-10-16, `cuInit(0)` returned zero and the official
backend detected the RTX 5070 Ti again. The full six-request collection completed:
all 1,536 generated token IDs and stop conditions matched replay with routing
observation inactive in the same context. The resulting trace contains all
55,296 expected layer/token records. This validates the observer for these
requests, not answer quality, logit tolerances, or a hypothetical hybrid executor.
Interrupted `.part` artifacts were never promoted into completed traces.

Auto-fit placed 11,309,875,200 expert bytes in VRAM after reboot, versus
10,744,381,440 before. Of the shared layer/token positions, 32,442 expert
selections differed across the two complete traces. Placement and numerical
paths changed; do not attribute those differences to an observational failure
or reuse the old routing as the new run. New reports use the newly verified trace.

The earlier 42.16 GB/s probe was single-threaded, not an absolute multi-core
bandwidth ceiling. The new optional parallel read probe uses a fixed 1 GiB
buffer, ten timed passes, and disjoint per-thread chunks. Three runs per thread
count gave these medians:

| Read probe | Median GB/s |
| --- | ---: |
| Single-thread read, across nine runs | 42.47 |
| Parallel read, 4 threads | 45.06 |
| Parallel read, 8 threads | 44.81 |
| Parallel read, 16 threads | 43.37 |

These are sequential-read microbenchmarks with unpinned workers, not sparse
expert-kernel measurements. They show no hidden 2x gain from adding threads in
this probe. No CPU governor or global affinity policy was changed.

The restored official control used unchanged MXFP4 target weights, actual 16K
context, four generation/16 prompt threads, and no speculation. Three separate
uninstrumented requests each generated exactly 512 tokens with matching text
hashes. Wall-clock decode rates were 24.76, 24.99, and 24.41 tok/s; median 24.76.
Whole-request throughput, including prompt work, was 21.60-22.07 tok/s. Because
auto-fit and runtime conditions changed, this is a refreshed control, not a
claimed scheduler speedup from the old 22.12 tok/s run.

The following are **conditional simulation estimates**, recalibrated with that
median, 45.058 GB/s RAM reads, the earlier 28.52 GB/s PCIe input, and the new trace.
The control prompt differs from the trace corpus, so the common residual is
still provisional; new GPU/synchronization costs are not separately measured.

| Policy | GPU byte coverage | Staged optimistic tok/s | No-staging-copy sensitivity tok/s |
| --- | ---: | ---: | ---: |
| Actual static placement | 18.52% | 24.76 (calibration) | 24.76 |
| Training-only fixed hot set | 53.25% | 38.94 | 38.94 |
| LRU with repeat admission | 67.02% | 16.27 | 29.66 |
| LFU with repeat admission | 55.52% | 38.42 | 39.75 |
| Causal cost-aware admission | 64.03% / 64.99% | 39.34 | 43.64 |
| Future-aware equal-size hit oracle | 84.36% | 37.88 | 58.36 |

The future-aware fractional fixed-residency RAM-only bound is 57.24 tok/s under
the refreshed inputs. It is not a proof about every execution algorithm or
physical peak bandwidth. The no-staging scenario is not an implemented
copy-elision mechanism. Neither scenario establishes the 60 tok/s target, and
there is still no live expert-granular CPU/GPU executor. The next implementation
must justify reduced transfer/dispatch cost and correctness, rather than claim
the simulated hot set as a working acceleration.

The app has been restored with the 120B control loaded and responsive. Raw traces
and reports remain local/ignored. Full CPU tests, optional collector tests, report
schemas, and source/trace hashes are checked separately from inference timing.

## Live Execution Screening After Reboot

A reproducible local-app runner now screens explicit host-operation offload,
polling, GPU-fit reserve, and operation-batch threshold settings. It confirms
the backend's reported actual slot context, validates exact output counts and
wall-clock decode intervals, records repeated output hashes, and restores the
original model settings even after an ordinary candidate failure.

These runs use the same official build 11669, MXFP4 weights, actual 16K context,
four generation/16 prompt threads, and fixed 98-token prompt. Each configuration
has two complete 512-token generations, temperature zero, seed 42, low reasoning
effort, no prompt caching, and EOS ignored. These are short screening runs, not
representative sustained quality/performance qualification.

| Profile | Median wall-clock decode tok/s | Whole-request tok/s | First token seconds | Text matches control |
| --- | ---: | ---: | ---: | --- |
| Control: polling 50, host-operation offload on, 1 GiB reserve | 25.06 | 22.08 | 2.94 | yes |
| Host-operation offload off | 25.36 | 24.00 | 1.32 | no |
| Polling zero | 24.97 | 22.02 | 2.92 | yes |
| Host-operation offload off, polling zero | 25.12 | 23.77 | 1.33 | no |
| 512 MiB reserve | 25.06 | 22.14 | 2.87 | no |
| 512 MiB reserve, host-operation offload off | 25.43 | 24.09 | 1.29 | no |

Every profile was internally repeatable on this screen, but execution-path and
fit changes can change rounding and generated text. None established a material
decode improvement or justified reducing the default VRAM safety reserve.
Zero polling did not improve decode, so its lower busy-wait behavior is not
reported as an acceleration.

The upstream CUDA source exposes `GGML_OP_OFFLOAD_MIN_BATCH`, default 32.
Raising it to 128 allows small eligible host-operation batches to stay on CPU
while larger ones can still run on GPU. This is an upstream batch heuristic,
not a new kernel or exact mapping from user prompt length. In a separate matched
screen it reduced first-token time from 2.93 to 1.34 seconds; decode changed only
from 25.15 to 25.26 tok/s, and text differed. Whole-request throughput changed
from 22.15 to 23.89 tok/s on that fixed short-prompt workload.

The candidate was then checked against four deterministic workloads:

| Workload | Control first token seconds | Threshold 128 first token seconds | Result |
| --- | ---: | ---: | --- |
| Coding, AST-only validation | 3.04 | 1.43 | both passed |
| Quantitative reasoning, exact numeric JSON | 3.38 | 1.63 | control passed; candidate failed |
| Prose, limited keyword smoke check | 2.89 | 1.34 | both passed |
| 7,740-token exact retrieval | 26.80 | 26.08 | both passed |

The failed candidate returned `transfer_tps: 28.500` instead of the expected
`28.5 / 31.5 = 0.905`; the other numeric values were correct. A repeat probe
reproduced that answer. This establishes an answer regression for the tested
prompt, not by itself a kernel-arithmetic bug or a general quality score.
No model-generated Python was executed: coding validation compared parsed ASTs.
The prose check is not a factual correctness evaluation.

Because decode did not materially improve and the quantitative check failed,
threshold 128 was rejected as a default. The control remains host-operation
offload on, polling 50, minimum batch 32, 1 GiB reserve, and speculation disabled.
No KV-cache precision or target-weight change was introduced. The reported
60 tok/s goal is still unmet. The next acceleration mechanism must change
expert-granular execution or amortize real work, rather than claim these
latency-only changes as generation acceleration.

## Real-Weight Hybrid Layer Prototype

The optional `vrambridge-hybrid-layer` now implements actual split CPU/GPU expert
execution, not just policy replay. It reads four selected expert groups
(IDs 0, 7, 42, 113) from a real layer's original MXFP4 tensors and F32 biases,
executes CPU/GPU branches independently, transfers CPU outputs through bounded
pinned staging, and reconciles them on a separate GPU backend stream. Router
weights are deliberately asymmetric (0.05, 0.15, 0.30, 0.50) and are never
renormalized separately in the two branches.

**This is an isolated warm-cache microbenchmark with deterministic synthetic
activations**, not live model hidden states, real routing probabilities, or a
full-model token/sec result. The four selected groups total 53,015,040 bytes,
which may benefit from CPU/GPU caches when repeatedly exercised. The all-GPU
case does not imply all experts of all layers fit in VRAM. The working browser
model remained loaded and idle; no driver/global CPU settings were changed.

Correctness checks preserve the gate/up/down biases and OpenAI SwiGLU alpha=1.702,
limit=7 behavior. All warm-up and timed executions compare with an unsplit CPU
graph. The fixed numerical gate is:

```text
max_abs_error <= 0.00001 + 0.0001 * reference_max_abs
relative_l2_error <= 0.0001
```

Both tested reduction strategies passed across layers 0 and 35, all five expert
splits, five synthetic activation vectors, and 50 repetitions of each serialized
and concurrent schedule: 2,000 timed executions plus warm-ups. In the final
runs the worst relative L2 errors were approximately 4.25e-6 for layer 0 and
8.5e-8 for layer 35. A separate F32 self-test also checks an independent scalar
reference, so correctness is not based solely on comparing two copies of the
same graph construction.

Timings below are medians in milliseconds. Serial/concurrent ordering is
interleaved and alternates first within each repetition.

| Layer | GPU / CPU experts | Weighted partials, serial | Weighted partials, concurrent | Full GPU reduction, concurrent |
| --- | --- | ---: | ---: | ---: |
| 0 | 0 / 4 | 0.7950 | 0.7664 | 0.7713 |
| 0 | 1 / 3 | 0.5849 | 0.5548 | 0.5656 |
| 0 | 2 / 2 | 0.3835 | 0.3495 | 0.3483 |
| 0 | 3 / 1 | 0.2407 | 0.1922 | 0.1940 |
| 0 | 4 / 0 | 0.1121 | 0.0974 | 0.0939 |
| 35 | 0 / 4 | 0.7636 | 0.7729 | 0.7510 |
| 35 | 1 / 3 | 0.5876 | 0.5622 | 0.5656 |
| 35 | 2 / 2 | 0.3851 | 0.3508 | 0.3634 |
| 35 | 3 / 1 | 0.2415 | 0.1939 | 0.2067 |
| 35 | 4 / 0 | 0.1120 | 0.0970 | 0.0943 |

These short runs measure schedules, not GPU occupancy timelines or statistically
qualified full-model gains. The all-GPU control also benefits from fewer
intermediate fences; therefore the entire serial/concurrent difference must not
be attributed purely to CPU/GPU compute overlap.

In mixed splits, weighted-partial reconciliation plus dependency waiting
measured about 10-15 microseconds, compared with about 171-533 microseconds for
CPU expert input/computation. The default transfers one 11,520-byte weighted
CPU partial vector, independent of how many CPU experts contributed.
Sending every unreduced CPU expert vector instead requires 11,520-46,080 bytes
and measured roughly 16-20 microseconds of merge/wait in the mixed cases.
The transfer and wait phases are host intervals, not isolated reduction-kernel
times; medians do not necessarily add exactly.

The simpler weighted-partial strategy is the better starting point: GPU-only
weighting/reduction did not consistently improve mixed execution. A separate
merge stream works without reserving a physical GPU slice; reserving compute
does not address the larger CPU expert cost demonstrated here.

Timing includes activation upload, dispatch, CPU execution, partial-output
staging/copy, merge, and completion synchronization. Graph construction and
weight upload are outside timing; output readback for verification is outside
the real-weight timing interval. There is no measured disk tier, no inference
cache/admission loop, and no integration into the application's decode path.
The full-model 60 tok/s target remains unachieved.

See the [probe workflow and schema](../../benchmarks/README.md#isolated-cpugpu-expert-execution).
Generated raw reports remain local/ignored and include the compiled source hash.

## Experimental Whole-Model Expert Dispatch

The isolated probe above is now followed by a **separate modified backend**,
not a change to the verified official executable. The app now defaults to
model-specific selection when a pinned hybrid installation is configured:
matching GPT-OSS-120B MXFP4 selects direct DMA, other architectures select
standard execution. The experimental warning and failed quality gates remain
visible; promotion of a default does not turn these screens into qualification. The
[reproduction guide](../../benchmarks/README.md#experimental-whole-model-expert-dispatch)
describes the pinned-source patch and explicit opt-ins.

The executor distributes resident expert groups across all 36 layers using only
the training half of the routing trace. GPU slot zero contains zero weights and
biases for misses. A packed CPU dispatcher gives resident experts CPU ID `-1`,
which the patched raw/repacked CPU matmul explicitly skips and zeros. Cold
experts still execute directly in RAM. Both branches retain the original four
router selections, probabilities, gate/up/down weights, biases, and activation;
there is no expert pruning or per-device probability renormalization.

These short, single-request screens used the original MXFP4 model, actual 16K
context, four generation threads, 16 prompt threads, and exactly 512 output
tokens on one fixed technical-explanation prompt:

| Profile | Measured wall decode tok/s | Interpretation |
| --- | ---: | --- |
| Frozen 800-expert bank, packed dispatch | 33.07 | Initial screen |
| Causal adaptive 800-expert bank | 36.22 | Initial screen; repeat about 35.5 |
| Adaptive 800-expert bank, current paired control | 35.60 | Raw CPU layout |
| Adaptive 850-expert bank, four threads | 35.38 | Larger bank did not improve this sample |
| Frozen 800-expert bank, CPU-repacked layout | 33.11 | Lossless native-layout GPU export |
| Adaptive 800-expert bank, CPU-repacked layout | 33.80 | Not a performance improvement |
| Adaptive 800-expert bank, synchronous uploads | 35.65 | Paired transfer screen |
| Same policy, asynchronous staged uploads | 34.67 | Not a performance improvement |
| Adaptive raw layout, `--no-host`, synchronous uploads | 35.73 | Paired direct-DMA control |
| Same policy, immutable-source direct DMA | 38.96 | First direct-DMA screen, not qualification |
| Direct DMA, current-source four-thread repeat | 38.81 | Same six output hashes as the paired copy screen |
| Direct DMA, six generation threads | 39.63 | Single exploratory sample |
| Direct DMA, eight generation threads | 39.99 | Single exploratory sample; not a qualified winner |

These are **screening results, not three-repeat cross-workload qualification**.
Arithmetic and output-count checks are not general answer-quality checks.
These results used the original prefill graph and one custom decode position
per call. A separately opt-in small-block path is now being tested for exact
target verification; its batch capacity is not generation throughput. Nothing
here establishes the 60 tok/s goal.

Adaptive admission uses a causal recent-use window and protects every expert
selected in the current layer. The initial implementation stages one component
through a 4,406,400-byte pinned buffer and uploads synchronously. In an
arithmetic-plus-512-token run it admitted 2,347 expert groups and uploaded
31,106,574,720 logical weight bytes. About 66.8% of expert selections were
resident before admission; about 70% executed on GPU including demand uploads.
Logical payload bytes are not measured PCIe counters.

The optional asynchronous path uses a 53,015,040-byte pinned pool, enough for
four complete selected expert groups. Pool reuse waits for previous readers and
DMA; uploads are ordered on the same backend stream before consumer kernels.
Teardown synchronizes while backend handles are still alive. A shutdown failure
in the first test exposed the cache/backend member-lifetime ordering; that
ordering was fixed and the retry exited successfully. Both copies of this test
are retained privately, not relabeled as successful evidence.

The separate CPU-repacked path reverses the eight-row MXFP4 storage interleave
only when uploading an expert. Byte round-trip tests cover multiple row groups,
quantization blocks, and expert slices, with no requantization. Masked matmul
tests cover F32, raw MXFP4, repacked MXFP4, two token positions, all/partial/no
active experts, stale-output clearing, and one/four threads. Repacking requires
`--no-host` to avoid the preferred CUDA host buffer, plus the explicit research
opt-in; merely removing `--no-repack` did not activate repacking. Those
nonactivated attempts were rejected rather than counted as repacked results.

Direct DMA then improved the paired 512-token screen from 35.73 to 38.96 tok/s
(about 9%). It page-aligns/coalesces and registers 61,073,604,608 immutable source
bytes in 73 ranges, capped at 64 GiB with 24 GiB available-RAM headroom.
Unlike the staged path, it avoids CPU staging copies and does not reuse mutable
host storage. Bank overwrites and subsequent consumer kernels are ordered on
the same backend stream; teardown synchronizes before unregistration.
Registration and initial model loading are outside decode timing.
It matched synchronous admission exactly at 96 teacher-forced positions, exited
cleanly, and produced identical full-output hashes on all six screened
requests. The normal app was restored after the private processes closed.
This remains a resource-intensive, opt-in research path, not a default.

### Numerical And Answer-Quality Limits

The teacher-forced checker compares all 201,088 vocabulary logits at 32
positions in each of three held-out requests. CPU-reference repeats matched
exactly. Packed-dispatch refactoring matched the previous hybrid path exactly,
and asynchronous admission matched synchronous admission exactly at all 96
positions, including clean process exit after the lifetime fix.

The separately declared diagnostic gate remains `max_abs <= 0.01` and
`relative_l2 <= 0.001`. Frozen/adaptive placement **fails this strict gate**:
maximum errors are roughly 1.9-3.8 and relative L2 up to about 0.21. Ordinary
unmodified automatic placement also fails it, with sampled maximum error about
5.0 and relative L2 about 0.23. The prototypes were inside that sampled stock
envelope, but an envelope on 96 positions is not a general correctness or
quality guarantee. The gate was not silently relaxed and the root cause of all
execution-path differences has not been isolated.

### Small-Block Verification And DFlash

An explicit batch opt-in now handles up to 32 positions in the hybrid graph.
Each position retains its original expert IDs/probabilities and independent
masked outputs. Residency is frozen throughout a multi-position call, so a
later position cannot evict an expert that an earlier position still needs.
Zero-output prefill graphs remain on the original path; the first live server
test caught and fixed an incorrect attempt to dispatch such a graph.

Enabling a four-position limit while still decoding one position preserved all
96 reference positions exactly. Actual four-position teacher-forced comparison
failed the unchanged strict gate: category maximum errors about
3.48/2.43/2.59, relative L2 about 0.139/0.097/0.096, with top-1 matches
32/32, 32/32, and 31/32. This is a diagnostic result, not a correctness pass
or an accepted-token speed measurement.

The real DFlash screen used the existing target-specific BF16 drafter and the
separate attention-bias fix. A smaller 640-expert plan and 8,192 MiB bank left
VRAM for the drafter. All profiles used actual 16K context, one request, four
generation threads, Jinja formatting, flash attention, and exactly 512 output
tokens. Drafting, rejected proposals, verification, and synchronization are
included in wall decode timing:

| Profile | Accepted generation tok/s |
| --- | ---: |
| Frozen 640-expert control, no drafter | 29.28 |
| DFlash, three proposals, 16 batch threads | 22.81 |
| Same, four batch threads | 22.12 |
| Same, eight batch threads | 24.47 |

Each draft profile proposed 674 tokens and accepted 286 of them; total output,
including target fallback tokens, was 512, not 674. Draft-profile output hashes
matched across thread counts, but differed from the non-speculative control.
Earlier invalid CLI/opt-in attempts are retained separately and were not
counted as inference results. This avenue did not improve performance and is
not enabled in the app. The 60 tok/s goal remains unmet.

Both synchronous and asynchronous adaptive profiles also returned `28.5`
instead of the expected `0.905` in the transfer-throughput field of a quantitative
JSON question. This failed answer check is retained as a deployment blocker.
Passing primitive or copy-equivalence tests does not override it. Additional
unambiguous quantitative questions and paired controls are needed to distinguish
prompt/model reasoning sensitivity from a placement-induced regression.
In the direct-DMA pair, both profiles also passed a separately added question
that explicitly states that **both** RAM and PCIe transfer 31.5 GB per token,
as well as AST-only clamp-code validation and a keyword-only prose smoke check.
The original failed question remains failed; these small additional checks do
not establish general quality or resolve its interpretation.

## Tensor-Derived Cross-Model Dispatcher

The next revision replaces the first prototype's GPT-specific scheduling
assumptions. Counts, active experts, layer/key ranges, cache groups, component
sets, and staging/history sizes derive from tensors. Native and hybrid
branches now share the backend's expert-math builder, preserving activation,
bias, scaling/normalization order, shared experts, and weighting-before-FFN
semantics where supported. Scaled expert tensors, LoRA/Hadamard transforms,
and multiple hybrid GPUs still fail explicitly.

Complete deterministic synthetic graphs cover seven families and seven
storage types. The [compatibility workflow](../../benchmarks/README.md#generic-moe-graph-compatibility)
records all 56 executions and the unchanged strict diagnostic outcome:
28 comparisons passed, 14 failed, with 14 separate reference creations.
Llama/Mixtral-style, Qwen2-MoE-style, Llama4-style, and Cohere-MoE-style
fixtures passed all strict comparisons; GPT-style mixed paths passed.
Quantized Qwen3/Grok and some native-GPU placement comparisons still fail
relative-L2 gates. This is not a claim of trained-model quality or speed.

Cross-layout tests exposed invalid placement of views and duplicate zero
expert IDs on float CUDA kernels. Views now keep their source storage, and
each active GPU miss receives a distinct reserved zero ID. Raw staged
admissions are capped at 64 MiB, chunked when necessary, and synchronized
before pool reuse. Failed startup/fixture trials were retained rather than
reclassified as valid measurements.

The real GPT-OSS-120B frozen path matched the previous hybrid reference
**exactly** for all 201,088 logits at each of 96 teacher-forced positions
(32 coding, 32 reasoning, 32 prose). This is a regression test against the
previous hybrid, not a pass of the earlier CPU-reference gate.

A subsequent matched 8/16-thread, actual-16K, three-by-512-token screen
measured a standard median of **25.84 tok/s** and generic adaptive direct-DMA
median of **37.81 tok/s**, with hybrid samples **39.71, 37.33, 37.81**.
Both small arithmetic and AST-only code checks passed. This retains a roughly
46% paired gain, but is below the earlier 39.46 median and does not establish
no performance regression; history, additional reserved slots, and scheduling
overhead need further isolation. Do not substitute the best sample for the
median. The app was restored after the test. The 60 tok/s target remains unmet.

Version-2 app profiles now bind trained plans to model descriptors rather than
embedding a GPT-only whitelist in dispatch. Other routed layouts can cold-start
with bounded staging; small unsharded weight files prefer native fitting to
avoid unnecessary split overhead. None of the other downloaded trained models
is another larger-than-VRAM MoE, so real cross-family throughput is still an
open validation gate, not an accomplished result.

## Guarded Overlap And Resident Projection Fusion

The [reviewed decode summary](../../benchmarks/results/gpt120-decode-optimization-summary.json)
retains the separate profiling, policy, projection, speculation, natural-workload,
and final-source screens. All use the original GPT-OSS-120B MXFP4 target weights.
During these screens the new switches were research options, not changes to
the installed app default. The subsequent user-requested experimental manual
trial enables the tested bundle with installation `decode_pipeline: true`,
hot880/11,264 MiB residency and 16K context. The UI distinguishes backend
configuration acknowledgement from observed guarded overlap and fused
projection graphs. Failed quality gates remain failed.

The critical-path investigation found that the CPU hidden-input copy waited
behind resident GPU expert work. Preparing the exact 3D input view earlier
lets the scheduler reuse its existing CPU copy. Guarded overlap then permits
only independent, device-only CUDA expert computation and host-only CPU expert
computation to run concurrently. Unsupported operations, new CPU inputs,
multi-copy schedules, and evaluation callbacks retain barriers. Pending
asynchronous host-source reads are fenced before scratch reuse; synchronous
copies need no extra fence. Existing reconciliation dependencies remain.

Initial profiling was invalid: querying CUDA device properties per submission
reduced throughput to about 9.65 tok/s. Caching backend classification removed
that observer effect: corrected control/profiled measurements were 38.62/38.34.
Host GPU-submit intervals are not CUDA execution times. Copy intervals include
dependency waits, not just PCIe transfer time.

Interleaved resident gate/up banks reuse native fused projection math, original
weights and separate biases. Actual logical GPU expert projections decreased
from 108 to 72 per complete 36-layer decode graph. Ranked admission and shorter
history were also screened; CPU CCD affinity was slower. Removing redundant
post-copy fences did not establish an additional throughput gain.

The final-source technical screen, with actual 16K context, 8/16 threads,
11,264 MiB cache and a training-only 880-group plan, generated three exact
512-token samples: **49.03, 44.73, 46.94 tok/s**, median **46.94**.
An earlier two-repeat screen peaked at 50.25; that peak is not a sustained result.

Natural-style prompts were separately screened with `ignore_eos=false` and a
512-token output cap, three repetitions per category:

| Resident cache | Coding median | Reasoning median | Prose median |
| --- | ---: | ---: | ---: |
| 10,240 MiB / hot800 | 42.32 | 43.92 | 57.58 |
| 11,264 MiB / hot880 | 43.18 | 46.10 | 60.54 |

The larger-cache prose samples were **58.60, 60.54, 61.94**. This crosses 60 on
one workload's median, not the overall sustained milestone: the first prose
sample and all coding/reasoning samples remained below 60. These are capped
generation segments, not complete long answers or semantic code validation.
Cache history persists across requests; repeated requests are not fresh seeds.
Both configurations correctly retrieved two keys from an actual 12,852-token
prompt without an allocation failure.

Small arithmetic and AST-only clamp checks passed, but a new quantitative
question still failed exact rounding: expected `37.33` ms and `26.79` tok/s,
with reported rates `26.75`/`26.78`. The first test implementation allowed 0.02
error, too loose for its requested rounded answer. The reviewed summary
recomputes exact numeric equality and retains that original loose-gate result.
Neither configuration is quality-qualified. This does not isolate placement
as the cause; the earlier CPU-reference and quantitative failures remain open.

### Cross-Layout Regression And Rejected Candidates

An attempted immutable QKV alias did not activate on the target model:
its Q8_0 projections have 256-byte inter-tensor padding. On the synthetic
Qwen3 fixture it did activate and introduced a strict regression
(`relativeL2 = 0.0015963`). Isolation showed zero error for overlap alone,
gate/up alone, and their combination, while QKV alone reproduced the failure.
The QKV candidate and its coupled attention change were removed; requesting
its former switch now fails explicitly. Unchanged weight bytes alone do not
guarantee numerically equivalent projection execution.

After removal, all seven layouts passed optimized-versus-prior-hybrid
comparisons. The full screen retained **72 executions: 21 reference creations,
37 passing comparisons and 14 failing comparisons**. The remaining failures
include native controls and the previously documented quantized diagnostics;
they are not waived. Callback-enabled and no-input-staging controls passed
with zero overlap, while optimized controls recorded actual overlap.
The real GPT frozen optimization screens retained zero full-logit error over
96 teacher-forced positions each. Synthetic results are not trained-model
quality or cross-family throughput evidence.

Top-16 F32 stress again passed strict logits while admitting 32 experts and
uploading **192 MiB through a 64 MiB bounded staging pool**. All 16 configured
test targets passed, with one retained oversized-HTTP-body `BrokenPipeError`
flake in the local-app target passing on a single unchanged rerun.

A derived Q8_0 DFlash drafter was also tested without modifying the target:
one/three-token draft limits measured **33.94/29.91 accepted tok/s** against a
matched **43.44** control. Actual proposed/accepted counters were verified;
rejected proposals and draft overhead were not counted as useful output.
This candidate was rejected. No new large model was downloaded.

The final patch is independently checked against the pinned source index and
the separate DFlash-bias patch, with reverse applicability checked against the
built tree. Natural timings precede removal of the unused QKV experiment;
those profiles did not enable it. Final-source performance and logit checks
are recorded separately with the exported patch hash. The app was restored
to its previous 16K default after the private tests.

### Next Optimization Queue: Measure Before Removing Waits

The user reported approximately **52 tok/s** during the updated manual trial.
This is useful operational feedback, not a matched benchmark or a replacement
for the repeated workload results above. At 52 tok/s, mean token latency is
19.23 ms; 60 tok/s requires 16.67 ms, a reduction of approximately **2.56 ms**
or 15.4% more throughput. No individual mechanism below has demonstrated that
additional gain yet.

The archived final technical profiler recorded roughly 12.5 ms of synchronous
CPU submission and 7.7 ms of copy/wait intervals per filtered graph. These
are not timings of the manual 52 tok/s request, isolated kernel times, or
independent bandwidth ceilings.

Proceed in this order:

1. **Separate synchronization from transfers.** Time existing event/backend
   waits separately from synchronous copy calls, asynchronous enqueue calls,
   and copy callbacks. Classify directions using cached backend/buffer data.
   A synchronous copy interval can still contain driver/internal waits; do not
   relabel it pure DMA time. Compare profiling off/on before using results.
2. **Prototype a combined routing packet if the measurements justify it.**
   The current dispatcher receives selected IDs, probabilities, and the exact
   hidden-input view separately. A single GPU-to-CPU packet could reduce
   transfer/dependency overhead. Preserve expert-ID exactness, original
   probabilities, strides, CPU bias IDs, weighting-before-FFN semantics,
   and the hidden-input lifetime. Packing itself adds work and must be timed.
   Any asynchronous return of GPU slot IDs needs bounded storage and explicit
   completion-before-reuse protection, not blanket barrier removal.
3. **Share CPU preparation and eliminate genuinely empty cold work.**
   Gate/up CPU matmuls can prepare the same activation independently. Reusing
   the native conversion once is distinct from the already enabled resident
   GPU fusion. An all-resident selection already skips CPU dot products, but
   may still incur conversion, bookkeeping and branch operations. Any empty
   path must clear the correct output, preserve numerical/nonfinite behavior,
   and leave shared experts and residual work untouched.
4. **Price admissions in time, not frequency alone.** Current causal history
   already includes a bounded payback margin and optional seed/recency ties.
   Compare those controls with measured saved CPU time, upload/eviction cost
   and bytes. Do not assume more migrations or higher hit rate improves the
   critical path. Derive costs/eligibility from tensor descriptors, not names.

Require unchanged target weights and selected experts, existing strict logit
thresholds, cross-layout regression controls, actual optimization counters,
and at least three matched 512-token repetitions. Retain prior diagnostic
failures and report answer checks separately. Restore the installed manual
profile after private experiments, including failed tests.

Do not prioritize disk streaming while the target fits RAM, repeat the rejected
QKV alias, or retry the same slower DFlash setup without a new causal reason.
Avoid new dependencies, global machine tuning, and approximate expert pruning.

## Next Experiments

The subsequent [downloaded-model matrix](../../benchmarks/RESULTS.md#october-16-model-aware-defaults-and-downloaded-model-matrix)
adds three matched 512-token repetitions per profile at 16K and 8/16 threads.
GPT-OSS-120B median wall decode improved from 25.69 to 39.46 tok/s (about 53.6%).
Four other locally available architectures ran successfully on the standard
backend. None is another larger-than-VRAM expert-dispatch test. Adaptive GPT
output hashes varied between repeats; earlier strict diagnostic/answer failures
are not superseded by the two small passing answer checks. The app now selects
the supported hybrid profile by default when installed, with warnings intact.

1. Broaden paired numerical, AST-only coding, prose-smoke, and long-retrieval checks without discarding failed questions.
2. Investigate reducing small cross-backend routing/input transfers and CPU projection overhead, measuring synchronization separately from transfer time; do not infer a win from fewer calls alone.
3. If single-position execution remains below target, investigate exact multi-token verification rather than counting aggregate batching or approximate expert pruning as success.
4. Qualify any candidate across held-out coding, reasoning, and prose with at least three repeated 512-token runs, actual 16K context, and all synchronization/migration overhead included.

Do not use the unverified FreeToken desktop distribution as a production dependency. Its design is useful as a research reference, but this baseline is produced entirely through the pinned, open llama.cpp checkout and local GGUF files.
