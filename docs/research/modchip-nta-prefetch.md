# llama-modchip non-temporal prefetch screen

Date: 2026-11-13. **Research only; no Synapse promotion.** The sanitized
[aggregate](../../benchmarks/results/modchip-nta-prefetch-summary.json) retains
paired samples, hashes, activation markers, checks and limitations.

## Candidate and scope

We inspected `synchopate/llama-modchip` v2.5.4 at
`d2a63518f920b59fe6e7654963d65dbfb5c8c6eb` (Apache-2.0), particularly
its `deepseek-v4-flash-fractal-256k.patch`. The candidate adapts its x86
non-temporal weight-prefetch idea; the IQ3_S insertion is our analogous
extension, not a claim that upstream supplied that insertion. It does not
import custom quantization, expert pruning, windowed attention, or the rest
of upstream's patch.

The [separate experimental overlay](../../third_party/patches/llama-b11669-modchip-nta-experimental.patch)
adds bounded `_MM_HINT_NTA` hints to native MXFP4/Q8_0, Q4_K/Q8_K, and
IQ3_S/Q8_K x86 dot loops. It is inert unless the copied CPU plugin is compiled
with `-DVRAMBRIDGE_EXPERIMENTAL_NTA`; it does not change arithmetic, weights,
expert selection, or the canonical hybrid patch.

Both plugins were built from the same copied `620d2225cc436ab0b001e3200e590e3b286ef9ab`
source after applying the canonical hybrid patch and this overlay. The only
compiler-flag difference was the experimental macro. Both used the same
AVX2/FMA/F16C/AVX512/VBMI/VNNI/BF16 flags, and the same locally built CUDA
12.0-architecture backend and server binaries. Here 12.0 denotes GPU compute
capability, not the CUDA toolkit version. The host was a Ryzen 9 9950X,
RTX 5070 Ti and approximately 123.48 GiB RAM. Outputs, profiles and binaries
were isolated; existing installations and model files were not changed.

A separate plugin built with the
[trace overlay](../../third_party/patches/llama-b11669-modchip-nta-trace-experimental.patch)
verified actual Q4_K and IQ3_S execution in both the synthetic and Synapse
Hybrid probes. Its atomic markers perturb execution; it was not used for timing.
The preliminary Coder probe while the backend was still compiling is excluded
from every performance result. A first 96K-capacity cohort overlapped separate
build/test commands and is excluded in full; its raw report is retained privately.
MXFP4 passed kernel parity, but its model-level activation and performance
were initially untested because the GPT-OSS-120B archive was unavailable.
The mounted-archive follow-up below supersedes that limitation.

## Accuracy

The optional `vrambridge-nta-prefetch` test compares native and candidate
direct dot outputs bit-for-bit on finite synthetic inputs: 800 MXFP4, 400
Q4_K and 400 IQ3_S comparisons across supported input sizes. This includes
lengths where a prefetch is not issued. It is not a full-model test.

Separately, the existing full-logit checker compared the 16K/4 GiB Hybrid
execution settings for three held-out 32-position continuations per model.
Coder and SC117 Qwen3.8 each passed **96/96 full-vocabulary positions with
zero maximum absolute and relative-L2 error**. Actual CPU and GPU experts,
adaptive admissions, and all 48 routed layers were observed in both arms.
This does not qualify every context, route or model.

All 36 paired complete-output hashes match across the two 128-output screens,
Coder's 512-output confirmation and SC117's 96K-capacity confirmation. Both
arms also pass the bounded arithmetic
and AST-only code checks. Hashes and these small checks are not general quality
qualification.

## Synapse Hybrid results

The existing isolated Synapse runner measured Classic Hybrid, F16 KV,
16,384 allocated context (98,304 for the last three rows), a 4,096 MiB adaptive staged bank, 8/16 generation/
prefill threads and three fresh alternating load pairs (`AB BA AB`). A
64-output warmup was excluded. The requests use greedy sampling, no prompt
reuse, and forced output lengths; rates are `(outputs - 1) / (wall - prompt)`.
Coder prompts contain 100-123 tokens and SC117 prompts 132-155 tokens; allocated
context is not occupied context. Medians are decode tok/s.

| Model / outputs | Workload | Control | NTA | Paired changes |
|---|---|---:|---:|---|
| Coder Q4_K_M / 128 | Coding | 82.64 | 86.22 | +3.05%, +5.39%, +7.38% |
| Coder / 128 | Reasoning | 70.60 | 72.24 | +0.94%, +3.88%, +2.03% |
| Coder / 128 | Prose | 97.58 | 101.97 | +2.80%, +5.10%, +3.41% |
| SC117 IQ3_S / 128 | Coding | 40.01 | 40.41 | -0.09%, +1.81%, +1.18% |
| SC117 / 128 | Reasoning | 35.43 | 35.53 | +2.46%, +1.00%, +0.13% |
| SC117 / 128 | Prose | 31.59 | 31.54 | +0.43%, -0.72%, +1.01% |
| Coder / 512 | Coding | 83.23 | 85.25 | +2.43%, +4.30%, +1.81% |
| Coder / 512 | Reasoning | 90.57 | 93.95 | +3.77%, +3.59%, +4.32% |
| Coder / 512 | Prose | 111.18 | 114.44 | +3.03%, +2.17%, +2.11% |
| SC117 / 256, 96K capacity | Coding | 39.85 | 39.84 | -0.11%, +0.42%, +0.67% |
| SC117 / 256, 96K capacity | Reasoning | 38.03 | 38.17 | +0.37%, +0.18%, +0.60% |
| SC117 / 256, 96K capacity | Prose | 34.77 | 34.57 | +1.38%, -1.22%, +0.22% |

Coder's longer confirmation does not clear the project's approximately 5%
noise/selection margin. SC117 shows no meaningful gain at 16K or 96K capacity;
the 96K test still uses short inputs and is not a filled-context test.
This rejects **promotion of this candidate on these settings**, not every
prefetch distance, quantization or machine.

For separate static CPU-expert placement, `llama-bench` with 128 synthetic
output tokens and three alternating pairs measured Coder 34.76 -> 36.21 tok/s
and SC117 28.10 -> 27.60 tok/s. These are not natural-request results. The
published screen runner adds environment and identity guards after those
measurements; the aggregate records both runner hashes. A separate 16-token
smoke run validated the published runner, not performance.

## Existing CPU preparation reuse: Coder follow-up

llama-modchip also caches quantized activations and expert row mappings.
Our canonical hybrid backend already has a narrower, default-off CPU
preparation-reuse path: it requires the same input/ID tensor identities,
dot-input type and expert count, and invalidates on non-disjoint intervening
operations. It is not the same implementation as llama-modchip's cache.
Earlier GPT-OSS-120B and Qwen3.8 screens found no repeatable win; see the
[existing preparation study](frontier-moe-feasibility.md#exact-cpu-preparation-reuse).

We tested the previously unmeasured Coder case using the **same control
server and libraries** for both arms. A private launcher adds only
`VRAMBRIDGE_HYBRID_CPU_PREPARE=1` after Synapse's environment sanitization.
The probe confirmed the backend's actual reuse marker. Three fresh alternating
16K/4 GiB Hybrid pairs used the same held-out requests and 128-output protocol:

| Workload | Control | Preparation reuse | Paired changes |
|---|---:|---:|---|
| Coding | 80.54 | 82.43 | +4.02%, -0.52%, +3.90% |
| Reasoning | 70.42 | 70.41 | -1.02%, +1.12%, +0.72% |
| Prose | 97.16 | 97.42 | +0.17%, +1.35%, -0.15% |

All nine paired output hashes and the bounded arithmetic/code checks match.
The same 16K/4 GiB full-logit diagnostic passes **96/96 positions with zero
error**, with 4,752 actual reuse events. The timing differences remain within
noise; this does not justify enabling the existing switch in Synapse or rule
out other activation-cache designs. The shipped research wrapper was verified
in a separate marker smoke test; the timing run used an equivalent private
wrapper, and both hashes are recorded.

## Reproduction and follow-up

Use a private working directory and a copy of the pinned source; never patch
an installed backend. Apply the canonical patch, then the experimental overlay.
Build two otherwise identical dynamic CPU plugins, with the macro set only in
the candidate, and use the exact same copied server and CUDA libraries for
both. Apply the trace overlay only in a third copied source and build it with
the experimental macro to verify actual kernel activation; do not time the
instrumented build. The aggregate records plugin,
patch, corpus and partial model fingerprints. Raw logits and logs stay private.

The optional parity test requires explicit matching libraries:

```sh
cmake -S . -B /private/nta-test -DVRAMBRIDGE_ENABLE_CUDA=OFF \
  -DVRAMBRIDGE_LLAMA_SOURCE=/private/pinned-source \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR=/private/control/bin \
  -DVRAMBRIDGE_TEST_NTA_PREFETCH=ON \
  -DVRAMBRIDGE_CPU_REFERENCE_LIBRARY=/private/control/bin/libggml-cpu.so \
  -DVRAMBRIDGE_NTA_CANDIDATE_LIBRARY=/private/candidate/bin/libggml-cpu.so
cmake --build /private/nta-test --target vrambridge-nta-prefetch-test
ctest --test-dir /private/nta-test -R '^vrambridge-nta-prefetch$' --output-on-failure
```

`tools/benchmark/vrambridge-prefetch-screen.py` provides an isolated static
placement screen. `tools/benchmark/vrambridge-synapse-baseline.py` accepts
private control and candidate Hybrid profiles for the actual app path. Record
positive activation, backend identities, model provenance and failed runs;
match the exact cache/placement contract for full logits.

For preparation-reuse research, copy
`tools/benchmark/vrambridge-prepare-wrapper.py` next to the **copied**
`llama-server` and point only a private candidate profile at it. Verify the
`VRAMBridge shared CPU expert preparation active` marker and compare it with
the same server without the wrapper. The wrapper is not a Synapse feature.

The related [tandem study](tandem-cpp-study.md#preserving-component-qualification)
already measured its own Q8 activation-cache/launch preserving combination:
Coder gained 2.45-3.90%, while Qwen and GPT were flat or slower. That is not
the same CPU activation-cache implementation as llama-modchip and does not
prove it ineffective, but it is not evidence to promote another cache port.
Profile the specific CPU activation-conversion cost and invalidation/aliasing
contract before implementing a broader cache. RouteCrack and prompt snapshots remain separate
research questions with their existing rollback and numerical blockers.

## GPT-OSS-120B mounted-archive follow-up

After the archive was mounted on 2026-11-13, both original GPT-OSS-120B MXFP4
shards were verified as accessible on the XFS HDD. The separate
[follow-up aggregate](../../benchmarks/results/modchip-gpt120-prefetch-summary.json)
records four fresh matched cohorts, partial model fingerprints, actual CPU
plugin identities, output hashes, and physical-read accounting. The prior
unavailable-archive result is retained as historical evidence.

The first instrumented probe loaded successfully but had severe overhead and
reported 89.3 GB of process reads over its lifetime. A first 12 GiB instrumented
probe failed the output-count gate; the ordinary control and a traced retry
both subsequently completed. The retry had **zero** physical reads during its
requests but remained very slow, demonstrating that tracing itself perturbs
execution; neither probe is a speed result. We cannot attribute the first
probe's latency solely to disk. The failed probe is retained, not averaged.

The measured arms used uninstrumented matched CPU plugins. Actual MXFP4
prefetch activation was confirmed separately for both configurations:

- 4,096 MiB adaptive staged bank, generic cold-start plan.
- 12,000 MiB adaptive direct-DMA bank, GPT-bound 944-expert training plan.

Each configuration used 16K allocated context, F16 KV, 8/16 generation/prefill
threads, three alternating fresh-load pairs, excluded 64-token warmups and the
same held-out coding/reasoning/prose prompts. Separate cohorts force 256 or
512 output tokens. Every timed request reports **zero physical bytes read by
the owned backend process** across prefill and generation. This does not
measure other processes' I/O or establish a fully occupied context. Medians
below are decode tok/s; paired percentages retain the run-to-run variation.

| Profile / outputs | Workload | Control | NTA | Paired changes |
|---|---|---:|---:|---|
| 4 GiB staged / 256 | Coding | 35.49 | 36.81 | +2.21%, +3.55%, +3.79% |
| 4 GiB staged / 256 | Reasoning | 31.30 | 32.41 | +1.08%, +3.10%, +5.78% |
| 4 GiB staged / 256 | Prose | 32.58 | 34.32 | +0.88%, +5.37%, +6.52% |
| 4 GiB staged / 512 | Coding | 32.36 | 33.13 | +0.95%, +2.35%, +4.88% |
| 4 GiB staged / 512 | Reasoning | 29.07 | 30.27 | +1.53%, +3.04%, +6.41% |
| 4 GiB staged / 512 | Prose | 31.25 | 31.99 | +0.95%, +2.30%, +7.45% |
| 12,000 MiB direct / 256 | Coding | 55.90 | 56.98 | +4.51%, +2.17%, +0.36% |
| 12,000 MiB direct / 256 | Reasoning | 50.38 | 54.23 | +8.91%, +0.45%, +9.48% |
| 12,000 MiB direct / 256 | Prose | 52.93 | 54.00 | +7.30%, +0.52%, +3.99% |
| 12,000 MiB direct / 512 | Coding | 53.37 | 54.55 | +3.62%, +1.66%, +3.42% |
| 12,000 MiB direct / 512 | Reasoning | 49.24 | 50.78 | +6.05%, -0.11%, +3.12% |
| 12,000 MiB direct / 512 | Prose | 55.28 | 57.87 | +4.13%, +5.35%, +2.59% |

The short direct-bank reasoning screen suggested a gain, but the longer
confirmation is mixed and its median is below 5%. The other longer medians
also remain below 5%. The two banks and transfer policies are not a matched
comparison to each other. These results do not justify promoting the candidate
in Synapse; they do not prove all prefetch strategies are ineffective.

Full-vocabulary teacher-forced comparisons passed **96/96 positions with zero
error in each bank configuration**. The trained-bank diagnostic confirmed all
36 routed layers, 944 initial experts, both CPU/GPU branches and actual direct
admissions. All 36 timed pairs have matching complete output hashes, and all
bounded arithmetic/AST checks pass. These are bounded checks, not broad answer
quality or cross-placement equivalence. Prior GPT CPU-preparation-reuse results
remain in the [existing study](frontier-moe-feasibility.md#exact-cpu-preparation-reuse);
the previously losing identical technique was not retested as a new gain.

The staging helper rejected the existing SSD staging marker, and the complete
59 GiB model plus the 16 GiB reserve would exceed the 57 GiB free on the root
filesystem. No marker was altered, model copied, RAM disk used, cache dropped,
or mount modified. The OS page cache became warm naturally. The research runner
now supports `--audit-physical-reads`; its per-request `/proc` delta includes
prompt and decode and rejects missing or decreasing counters. The executed
runner and the later refactored reproduction runner have distinct recorded
hashes. Installed Synapse settings and backend profiles remain untouched.
