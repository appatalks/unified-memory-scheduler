# tandem.cpp: expert offload and execution study

Date: 2026-11-08 UTC (November 7 local time).
Status: pinned source reviewed, patch application verified, and matched
CUDA builds completed; **no model-level numerical, performance or
production qualification yet**.

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

The current deliverable is a reproducible source/build checkpoint and
bounded study protocol, not a validated inference candidate or release.
