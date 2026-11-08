# tandem.cpp: expert offload and execution study

Date: 2026-11-08 UTC (November 7 local time).
Status: pinned builds and local compatibility/speed screens completed.
The initial hybrid/default speed gains are numerically unqualified;
the matched cache-only path passes bounded checks but has no broad
performance win. **No production promotion.**

## Source and claim boundary

- Project: [ewhacc/tandem.cpp](https://github.com/ewhacc/tandem.cpp).
- Inspected revision: `e1acb8877bfcf92b4ae7c9f0e0f1d896dbf6c683`.
- llama.cpp base: `f5222f25c18da062ab2b8a2dbbfbf4feae398566`
  (October 28, 2026), resolved from the project's `BASE_COMMIT`.
- `tandem.patch` SHA-256:
  `8a2e5898d4b750127f9fc474a28b7781615724ea1dbac4ac1c35633df1c1fb5d`.
- MIT license, including the ggml authors' existing copyright notice.
  Retain those notices if any licensed source is later ported.

The full patch applies cleanly against that exact base: 35 changed paths,
31 pre-existing source files and four new files. Source acquisition and
inspection used a separate, detached session checkout. The project's build
script was not executed and the installed backend was not changed.

Separate unpatched-reference and patched-candidate CUDA builds completed
with GNU 14.2.0, Release, `CMAKE_CUDA_ARCHITECTURES=120`, the same CMake
options and four build workers. Both executables run `--version`; the
candidate exposes expert-cache flags in `--help`. The existing
`sequence-decode-probe.cpp` compiles against the candidate and its synthetic
numerical self-test rejects deliberately changed logits. No model was
loaded by these checks, so CUDA compilation is not evidence of working
GPU inference. Frozen executable/library identities are in
[`tandem-source-build-summary.json`](../../benchmarks/results/tandem-source-build-summary.json).
Both executables report the unpatched base commit in `--version`; hashes,
not that version string alone, distinguish the experiment.

The pinned [README](https://github.com/ewhacc/tandem.cpp/blob/e1acb8877bfcf92b4ae7c9f0e0f1d896dbf6c683/README.md)
reports approximately 3.3x Qwen3.8 IQ3_S decode with MTP and 1.73-1.85x
GPT-OSS-120B without a draft on an i5-12400F / RTX 4090 24GB system.
Those are the authors' results, not this machine's measurements.
Their Qwen target is the original ISTA GSQ-RCO conversion, not our modified
SC117 target. The same README retains a slower-than-upstream DeepSeek
SSD-bound case. Neither broad wins nor transfer to 16GB VRAM is assumed.

Its benchmark counts completion tokens over the interval between first and
last streamed text/reasoning pieces, with forced-length generation.
That differs from our HTTP aggregate rate including prefill/queueing and
from `(outputs - 1) / decode wall` estimators. Do not mix these rates.

## What the patch actually changes

The pinned [patch](https://github.com/ewhacc/tandem.cpp/blob/e1acb8877bfcf92b4ae7c9f0e0f1d896dbf6c683/tandem.patch)
is a substantial backend bundle, not one universal speed switch:

1. **Hybrid expert-cache misses.** GPU-resident experts execute on CUDA;
   uncached selected experts are divided between host computation and
   upload. A CPU worker starts before uploads and its output is joined
   with GPU results. The automatic upload share uses measured link
   bandwidth and CPU expert throughput, including their shared RAM
   contention. Source uses `L / (L + k * R)`, default `k = 1.85`.
   A separate A/B controller can disable the CPU path when its fixed
   per-layer cost outweighs the savings. Source: patch lines 3970-4020,
   4190-4270 and `src/llama-moe-cache.cpp`.
2. **Prefill sharing and work layout.** Large prompt batches have their
   own CPU/upload sharing control; CPU gate/up and down work-item layouts
   are tunable. These are separate from concurrent-agent scheduling.
3. **Host/device orchestration.** Packed host-input uploads, activation
   quantization reuse, CUDA graphs keyed by shape, and graph allocation
   pools reduce repeated dispatch/allocation work.
4. **Architecture-specific paths.** GDN state access, combined alpha/beta
   projections, convolution-state handling, hyper-connection fusion and
   per-layer embedding placement depend on supported layouts. MTP and
   EAGLE3 drafting are separate optional mechanisms, not inherent cache
   speedups.

There is direct conceptual overlap with the existing
[`llama-b11669-hybrid-experts.patch`](../../third_party/patches/llama-b11669-hybrid-experts.patch)
and [Classic/Hosted research](sc117-hosted-porting-baseline.md).
That research already records CPU-cold/GPU-hot sharing, LFU admission and
CUDA host-call execution. Therefore compare against both matched
unpatched tandem base and the existing qualified engine; improvements
over plain upstream are not automatically improvements over Synapse.
Do not stack a second expert cache under the existing hybrid engine.

## Accuracy and integration hazards

- The README's eight-token, Qwen3.8 Q2_0 teacher-forced screen reports
  zero mean KLD with defaults disabled, but approximately 0.0091-0.0093
  with defaults and 95.5-96.1% matching top tokens. This is not our
  full-vocabulary raw-logit contract on SC117 IQ3_S or another family.
  Reported GSM8K accuracy within noise does not establish equivalence.
- Source applies a bundle of defaults in `common_init`. Disabling
  release defaults does not remove feature variables already inherited
  from the environment. Every arm must use a clean, explicit environment.
- The project documents timing-dependent CPU/GPU placement and
  non-repeatable greedy output. Its suggested fixed upload share is an
  experimental control, not a guarantee of our required equivalence.
- `LLAMA_MOE_TAIL_SKIP` exists and deliberately omits selected experts.
  It is off by default and **must remain explicitly zero**. No pruning,
  target requantization, lower-precision KV or weight-changing shortcuts.
- Some fusion paths fold scale into runtime weight representations and
  change rounding. Keep those disabled in the initial preserving arm;
  evaluate numerical effects separately rather than declaring them exact.
- `hyb_shape` in `src/llama-graph.cpp` checks activation, bias, scales and
  weight-before-FFN layout. Unsupported CPU paths warn and upload every
  miss. Our harness must classify this as an unsupported mechanism,
  not silently record it as a hybrid-performance success.
- The README targets single-user decode. Concurrent requests are not its
  primary win and can lose with a small cache. This does not solve the
  current Clover eight-agent numerical blocker.
- RAM heuristics use physical RAM rather than cgroup limits, and mmap
  pinning/embedding copies can consume substantial RAM. Disable them
  initially, measure actual available memory, and bound allocations.
- Linux/CUDA is the tested platform. No CPU-only, macOS, Windows or
  multi-GPU speed claim is inferred.

## Local qualification sequence

Keep all new engine code here in research and all build/results directories
private and isolated. Reuse fully hashed local weights, without modifying
them. Do not download draft heads for the initial study.

1. Build unpatched and patched backends at the same exact base with the
   same compiler/CUDA flags. Freeze executable and shared-library hashes
   and verify actual runtime binding. First establish loader compatibility
   separately for Coder Q4_K_M, SC117 Qwen3.8 IQ3_S, GPT-OSS-120B MXFP4
   and Clover Q4_K_M. Retain explicit unsupported/load failures.
2. With drafts absent and defaults disabled, compare scalar controls,
   cache-only, and a fixed-share hybrid path independently. Enable only
   the studied mechanism; pin F16 KV, 16K capacity, sampling, thread
   budgets, cache size and model bytes. Require startup and execution
   evidence of the intended CPU/GPU paths and zero skipped experts.
3. Run the existing raw-logit contract, whole-state restoration and
   occupied-context/chunking probes before accepting speed. Compare
   placements and caching histories explicitly; start with one user
   before testing 2/4/8 streams. Keep the Clover long-stream and
   Coder/GPT failures open; this is another backend, not a retroactive fix.
4. Measure three alternating fresh-load arms on held-out coding,
   reasoning and prose, both short and actual long input. Report
   accepted output tokens, decode, prefill, TTFT, whole-request and
   aggregate throughput, latency, placement, memory and transfers.
   Match available GPU memory with upstream and existing Classic/Hosted.
5. Ablate packed uploads, graph reuse, expert work layout and automatic
   share policy one at a time. Profile CPU expert work, H2D and required
   synchronization. Numerical failures or no end-to-end gain stop an
   ablation; do not optimize a failed path.
6. Only after cross-family qualification, study drafts and architecture
   fusions separately. Any eventual Synapse port must use capabilities
   and tensor metadata, retain activation guards, and enter
   Automatic -> Auto Tune -> remember without new engine toggles.

The initial deliverable was a source/build checkpoint and bounded study
protocol. The local follow-up below does not establish a release candidate.

## Local compatibility and initial speed follow-up

The [reviewed qualification summary](../../benchmarks/results/tandem-qualification-summary.json)
records the local evidence. The
[`vrambridge-tandem-screen.py`](../../tools/benchmark/vrambridge-tandem-screen.py)
runner owns loopback servers, checks GPU contamination, uses temporary
configuration, verifies all model shards against a private manifest, and
requires the requested cache/CPU mechanism and actual execution counters.
It does not load or change the installed Synapse application.

Coder Q4_K_M, the existing SC117 Qwen3.8 IQ3_S two-shard target and
GPT-OSS-120B MXFP4 two-shard target load and finish the tested requests.
Clover Q4_K_M fails in both pinned builds with
`unknown model architecture: 'clover_moe'`. This is a limitation of these
exact builds, not a claim that all upstream revisions reject Clover.

The first screen uses one fresh load per arm, one 16K slot, F16 KV,
8 generation/16 batch threads, 64 excluded warmup tokens, and three
held-out coding/reasoning/prose prompts with 128 forced greedy outputs.
Drafts, prompt reuse, expert skipping and pinned-weight copies are off.
Upstream automatic fit and tandem automatic caching use a 1,024 MiB
free-memory margin but have different expert placement.
The fixed arm disables the default bundle and uploads 20% of misses;
the default arm enables the published bundle and its adaptive sharing.
The table is decode tok/s, **not repeated-load medians or qualified wins**:

| Model / workload | Upstream fit | Tandem fixed | Tandem defaults |
|---|---:|---:|---:|
| Coder / coding | 79.61 | 99.49 | 112.48 |
| Coder / reasoning | 80.27 | 120.47 | 134.61 |
| Coder / prose | 81.18 | 139.71 | 167.66 |
| SC117 Qwen3.8 / coding | 31.13 | 43.19 | 46.38 |
| SC117 Qwen3.8 / reasoning | 31.03 | 43.75 | 41.65 |
| SC117 Qwen3.8 / prose | 31.02 | 42.32 | 33.24 |
| GPT-OSS / coding | 24.51 | 40.04 | 42.63 |
| GPT-OSS / reasoning | 25.39 | 34.54 | 25.32 |
| GPT-OSS / prose | 25.44 | 34.69 | 35.59 |

Whole-request rates also improve in this screen, but not by the same
amount. For example, Qwen3.8 coding is 22.74 -> 38.51 tok/s with defaults;
GPT coding is 15.04 -> 30.75; Coder coding is 68.70 -> 98.51.
Fixed sharing generally has worse TTFT than the default bundle.
The default arm does not improve every workload: GPT reasoning decode
is slightly slower than control. Natural-answer quality is not measured.

All 18 candidate first-token full-vocabulary log-probability comparisons
fail the unchanged diagnostic thresholds. Complete outputs match only
1/3 Coder, 2/3 Qwen3.8 and 0/3 GPT controls per candidate arm.
This blocks claiming equivalent execution, but does not by itself prove
the generated answers are wrong.

## Separating patch effects from placement effects

The extended
[`sequence-decode-probe.cpp`](../../tools/trace/sequence-decode-probe.cpp)
supports `VRAMBRIDGE_PROBE_SERIAL_ONLY=1`: one 16K slot, eight short prompts,
32 teacher-forced steps and full-vocabulary raw logits, giving 264
prefill/decode comparisons per family. Compile it separately against each
build's matching headers and libraries; do not mix struct ABIs. The
optional `VRAMBRIDGE_PROBE_MOE_CACHE_MIB=4096` is bounded and rejects
non-scalar or CPU-only placement.

Against original CUDA-backbone/CPU-expert scalar references, tandem with
cache and feature defaults disabled passes 792/792 checks with zero error.
The 4 GiB fixed-share cache arm fails all 792 checks, and GPU-only caching
also differs from those CPU-expert references.

That comparison changes the experts' execution placement, so those
differences cannot simply be blamed on a tandem cache bug. A second
control enables upstream's existing GPU-only expert cache at the same
4 GiB size and generates new scalar references. Tandem's GPU-only
cache path, with defaults disabled and no CPU-miss computation, then
passes **792/792 raw-logit checks with zero error**. Cache activation and
actual lookups are confirmed. This localizes a preserving control and
separates existing CPU/GPU arithmetic differences from a patch regression.
It does not qualify the faster hybrid/default bundle.

Three fresh alternating runs per arm then measure this preserving
GPU-only control at the same fixed 4 GiB cache, not the larger automatic
caches of the initial screen. Both arms use defaults off, no CPU miss
computation, the same 16K/F16/8/16 settings and 128 forced outputs.
All **27/27 complete output hashes and 27/27 first-token full-vocabulary
probability comparisons match**. Median decode tok/s:

| Model / workload | Upstream cache | Tandem cache | Change |
|---|---:|---:|---:|
| Coder / coding | 54.43 | 52.61 | -3.35% |
| Coder / reasoning | 51.52 | 49.30 | -4.31% |
| Coder / prose | 80.59 | 80.42 | -0.21% |
| SC117 Qwen3.8 / coding | 24.03 | 23.17 | -3.57% |
| SC117 Qwen3.8 / reasoning | 26.40 | 25.07 | -5.01% |
| SC117 Qwen3.8 / prose | 23.21 | 22.86 | -1.53% |
| GPT-OSS / coding | 14.10 | 13.64 | -3.29% |
| GPT-OSS / reasoning | 10.73 | 10.34 | -3.59% |
| GPT-OSS / prose | 13.08 | 13.07 | -0.06% |

Small changes can be timing noise; this does not establish a useful
end-to-end speed win. In particular, a 4 GiB GPU-only cache is not the
fastest measured way to execute these models on this machine. It is a
matched source-isolation control. Do not compare it to the automatic
hybrid screen as if only one feature changed.

The outcome is a compatible research backend on three families, a
bounded preserving control, and real but unqualified hybrid speed leads.
Further work should isolate the CPU/GPU precision/placement contract and
each fusion before measuring a qualified faster bundle. Do not tune or
promote the failed default arm or port this large patch wholesale.

The raw and timing source hashes are separate: the later probe extension
does not rewrite identities in the earlier Clover summaries.
These are short, single-slot screens; arbitrary occupied context,
chunk-boundary equivalence and cache-path rollback remain unqualified.

For reproduction, supply a private JSON manifest with a `label` and
`files` list of `{path, sha256}` for every shard, then use matching pinned
server paths:

```sh
python3 tools/benchmark/vrambridge-tandem-screen.py \
  --reference /path/to/unpatched/llama-server \
  --candidate /path/to/tandem/llama-server \
  --manifest /private/model-manifest.json \
  --output /private/tandem-initial.json --repetitions 1 --tokens 128

python3 tools/benchmark/vrambridge-tandem-screen.py \
  --reference /path/to/unpatched/llama-server \
  --candidate /path/to/tandem/llama-server \
  --manifest /private/model-manifest.json --labels coder qwen38 gpt120 \
  --arms reference-cache tandem-cache --cache-mib 4096 \
  --output /private/tandem-matched-cache.json --repetitions 3 --tokens 128
```

Labels select report assets only; no execution behavior is keyed on model
names. Raw reports/logs stay private. Do not multiply these upstream
comparison gains by previous Synapse Classic/Hosted results: this screen
does not freshly measure those installed engines.

## Preserving component qualification

The later
[`moe-component-qualification-summary.json`](../../benchmarks/results/moe-component-qualification-summary.json)
separates the CPU-miss path from packed input uploads, graph-shape/pool
reuse and Q8 activation-cache reuse. Each preserving component and the
launch-only combination passes 792/792 raw-logit positions on Coder,
SC117 Qwen3.8 and GPT-OSS against the same upstream 4 GiB cache.
The complete preserving combination also passes at 4 GiB.
CPU-miss sharing still fails, including the first prefill: Coder/Qwen
pass 0/264 positions and GPT 4/264 against the GPU-cache reference.
Do not time that failed path as qualified.

The
[`llama-tandem-component-trace-experimental.patch`](../../third_party/patches/llama-tandem-component-trace-experimental.patch)
adds default-off logging at actual packed-copy completion, graph-pool
reuse, graph-shape key selection and Q8 cache hits. It layers on the
pinned tandem patch and changes no tensor operation. The original
candidate binaries are frozen before rebuilding the diagnostic overlay;
activation cannot be inferred merely from environment variables.
The timing harness rejects missing component markers or unexpected CPU
expert computation in a GPU-cache-only arm.

Increasing the cache to 8 GiB changes Qwen's reference: both the disabled
control and the combined candidate initially pass only 132/264 positions
against the old 4 GiB reference, with identical observed maximum error.
This is not evidence of a component regression. Fresh unpatched 8 GiB
references pass their own repeat, and the preserving combination passes
792/792 positions with zero error against those matching references.
Cache configuration belongs in the numerical contract; do not silently
compare different execution configurations.

Three alternating fresh-load pairs per family then use the same 8 GiB
cache, one 16K slot, F16 KV, 8/16 threads, 64 excluded warmup tokens,
256 forced outputs and held-out coding/reasoning/prose requests.
All 27/27 output hashes and first-token probability comparisons match.
Median decode tok/s:

| Model / workload | Upstream cache | Preserving subset | Change |
|---|---:|---:|---:|
| Coder / coding | 118.50 | 121.40 | +2.45% |
| Coder / reasoning | 134.51 | 139.75 | +3.90% |
| Coder / prose | 159.77 | 165.56 | +3.62% |
| SC117 Qwen3.8 / coding | 35.48 | 34.78 | -2.00% |
| SC117 Qwen3.8 / reasoning | 36.05 | 35.02 | -2.88% |
| SC117 Qwen3.8 / prose | 34.90 | 34.88 | -0.07% |
| GPT-OSS / coding | 19.41 | 18.47 | -4.82% |
| GPT-OSS / reasoning | 16.02 | 15.22 | -4.99% |
| GPT-OSS / prose | 19.56 | 19.52 | -0.24% |

These are modest Coder gains, below the application's 5% selection
margin, and regressions elsewhere. There is no broad default-worthy
speed win. The different cache size/output length also prevents treating
this as a matched comparison with the earlier 4 GiB/128-output screen.
No new engine pin, blanket feature bundle or model-name heuristic is
promoted. The separate Clover attention follow-up is a different backend
and cannot be combined with these rates as a multiplier.
