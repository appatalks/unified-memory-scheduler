# Benchmark Ledger

This ledger is the reviewed index for benchmark evidence committed under `results/`.
It records the conditions required to compare runs, links raw artifacts, and
separates valid measurements from failed-command evidence. New result JSON is
ignored by default and is added here only after review.

Published JSON uses `benchmark-host` for the hostname and `/models` for
machine-specific model storage roots, including paths embedded in commands and
captured output. Model filenames, measurements, timestamps, and diagnostics are
otherwise unchanged. These placeholders are not executable local paths.

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

## Copy/Wait Breakdown And Event-Protected GPU IDs

The [reviewed handoff summary](results/gpt120-async-id-handoff-summary.json)
records the first follow-up to research checkpoint `068f69e`. On unchanged
GPT-OSS-120B MXFP4 at actual 16K context, 8/16 threads and 11,264 MiB residency,
three matched 512-token technical samples improved the median from
**47.01 to 49.27 tok/s (+4.8%)**. Natural-style coding/reasoning medians improved
**43.84 -> 45.31** and **46.56 -> 48.26**; prose was nearly flat at
**61.31 -> 61.62**. This does not establish the overall 60 tok/s milestone.

The prototype replaces blocking scheduler slot-ID copies with bounded,
event-protected per-layer transfers and explicit graph dependencies.
H2D destination-wait calls decreased by about 1.02 ms per filtered graph, but
CPU submission rose; only the net end-to-end gain is counted. Logical ID
transfers still occur outside the scheduler-copy counters.
All 12 compared generation hashes and all seven prior-hybrid fixture gates
matched controls. Existing native-reference and exact-answer failures remain.
The optional `async_ids` installation setting is enabled in the local
experimental manual profile after validation; Standard rollback is unchanged.

## Decode Overlap And Projection Fusion Screen

The [reviewed summary](results/gpt120-decode-optimization-summary.json) records
GPT-OSS-120B MXFP4 on the 9950X/5070 Ti with unchanged target weights,
actual 16K context, one request, 8 generation/16 prompt threads, and 512 output
tokens per sample. These conditions supersede the older defaults below for
this section only. The final technical three-repeat median is **46.94 tok/s**.

Separate natural-style prompts disabled forced EOS suppression and retained
the exact 512-token cap/count gate. With an 11,264 MiB resident cache:

| Workload | Three samples, tok/s | Median tok/s |
| --- | --- | ---: |
| Coding | 43.18, 43.25, 42.51 | 43.18 |
| Reasoning | 43.74, 46.10, 47.03 | 46.10 |
| Prose | 58.60, 60.54, 61.94 | 60.54 |

The prose median is not the overall 60 tok/s milestone. These capped segments
do not establish full-answer quality. Exact quantitative answers still failed;
two-key retrieval at 12,852 prompt tokens passed. During these screens the new pipeline options were off in the installed
default, and the previous app configuration was restored. The subsequent
experimental manual trial opts into the bundle with installation
`decode_pipeline: true`; this does not remove the failed quality gates.

Q8-drafter speculation was slower. QKV aliasing failed an isolated synthetic
regression and was removed. All seven final fixtures passed comparisons with
the prior hybrid path; 14 broader strict/native-reference failures remain
visible. Read the [investigation](../docs/research/frontier-moe-feasibility.md#guarded-overlap-and-resident-projection-fusion)
before interpreting these measurements as deployment evidence.

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

## October 16: Model-Aware Defaults And Downloaded-Model Matrix

This series uses pinned llama.cpp commit
`620d2225cc436ab0b001e3200e590e3b286ef9ab`, **actual 16,384-token context,
8 generation / 16 prompt threads, and three exact 512-output-token runs**
per model/profile. It is not comparable to the earlier 512-prompt/32-generation
tables without accounting for the changed workload/backend.

The [complete matrix](results/20261016-downloaded-model-matrix.json) contains
all 30 throughput samples, 20 small answer checks, source/patch/plan hashes,
model architecture/type metadata, selected execution profiles, and successful
restoration of the original GPT-OSS-120B app.

| Existing model | Standard median tok/s | Automatic median tok/s | Automatic execution | Arithmetic / AST-only code checks (both profiles) |
| --- | ---: | ---: | --- | --- |
| GPT-OSS-120B MXFP4 | 25.69 | 39.46 | Custom hybrid direct DMA | Pass / pass |
| Gemma 3 12B Q4_K_M | 80.94 | 80.50 | Standard | Exact-integer check failed / pass |
| DeepSeek-R1-0528-Qwen3 8B Q4_K_M | 130.08 | 128.50 | Standard | Arithmetic completion hit 1,024-token limit / pass |
| Qwen3.5 local Q8_0 (`claude-oss-q8_0.gguf`) | 79.48 | 80.26 | Standard | Pass / pass |
| LFM2 1.2B Q8_0 | 485.67 | 486.55 | Standard | Exact-integer check failed / strict AST check failed |

The Qwen3.5 row is identified by actual GGUF architecture `qwen35`; its local
filename is not evidence of an official Anthropic model. The CLIP projector
was excluded because it is not a standalone language model. No models were
downloaded. Only GPT-OSS-120B exceeds this GPU's VRAM among the current five;
this series cannot demonstrate hybrid benefits on a second larger-than-VRAM
architecture.

GPT-OSS-120B's paired median gain is about **53.6%**. Standard mode uses upstream
auto-fit; automatic mode uses the patched resident bank, CPU cold-expert
execution, adaptive admissions, and direct registered-source DMA. Context,
threads, prompt, output count, seed, and sampling settings are matched;
placement/offload are intentionally different and recorded. Loading and source
registration are outside decode timing; end-to-end generation and load
durations are recorded separately. The initial trained bank resets on each
profile load; adaptive residency persists between its repetitions and answer
checks.

All four unsupported architectures selected standard execution as intended.
Their repeated output hashes matched across standard/automatic profiles;
small timing differences are not evidence of an optimization gain. The
standard GPT-OSS repetitions matched one another, but adaptive GPT-OSS hashes
did **not** all match each other or the standard control. Passing the two small
answer checks is not general model/logit equivalence. Strict CPU-reference
diagnostics and the earlier quantitative JSON failure remain unresolved.

Answer checks deliberately require the exact integer and a specific AST with
no extra content. Failures can include formatting or token-budget issues;
answer text is not stored in this matrix, so these booleans must not be
reinterpreted as a broad factual-quality score. The runner retains failed
checks and exits nonzero even when the throughput runs and restoration succeed.
The 512-token throughput prompt is forced past EOS if necessary, not a guarantee
of 512 natural-answer tokens. This is a repeated one-prompt screen, **not**
sustained coding/reasoning/prose qualification. GPT-OSS-120B's 60 tok/s target
is still unmet; the smaller models' higher rates do not satisfy that goal.

### Published Hybrid Research Evidence

| Artifact | Interpretation |
| --- | --- |
| [Direct-source pair](results/20261016-gpt120-direct-source-screen.json) | First paired 35.73 -> 38.96 tok/s result; six output hashes match between copy paths, but the original quantitative JSON check remains failed. |
| [Direct-DMA thread screen](results/20261016-gpt120-direct-thread-screen.json) | Four/six/eight-thread one-prompt samples 38.81/39.63/39.99 tok/s; not three-repeat per-thread qualification. |
| [CPU dual-layout screen](results/20261016-gpt120-dual-layout-screen.json) | Real CPU-repacked storage with lossless GPU exports; adaptive 33.80 tok/s did not beat the raw-layout screen. |
| [Asynchronous staged-copy screen](results/20261016-gpt120-async-copy-screen.json) | Synchronous 35.65 versus async 34.67 tok/s; identical sampled logits, no timing gain, failed quantitative check retained. |
| [Small-block DFlash screen](results/20261016-gpt120-hybrid-block-dflash-screen.json) | 22.81/22.12/24.47 accepted tok/s versus 29.28 control with a smaller bank; includes draft/rejection overhead, not a successful optimization. |
| [Full-vocabulary diagnostic metrics](results/20261016-gpt120-logit-diagnostics.json) | CPU repeat is exact; stock placement and hybrid fail the unchanged strict CPU-reference gates. Direct-DMA versus synchronous hybrid is exact at 96 sampled positions. Block4 remains a failed strict diagnostic. No token IDs/full logits/activations are published. |
| [Training-only initial plan](results/gpt120-trained-hot800.txt) | 800 expert groups from the public routing corpus's training half only; SHA-256 `1198ae0ea67a9b94f1f51f0913f7ade3378bb5ba01e3ac93ace7eccaa77e98e6`. |

Published copies replace private paths with `/models`, `/opt/vrambridge`,
`/workspace/unified-memory-scheduler`, `/session`, and `/home/benchmark-user`.
Raw synthetic answer strings from old screens are replaced with SHA-256 hashes;
measurements, failed checks, diagnostics, and original run status are retained.
Large routing traces, binary references, activation dumps, and raw process logs
remain ignored. Older screen source hashes describe their own prototype state,
not necessarily the final patch. The current matrix identifies the final patch
with SHA-256
`a0f61f4e86aa6468d9b76ed4c9ee801dbee30ff2ec034566e8f86cf25be472a1`.

The app's new default is **automatic model-specific selection**, not a claim of
universal hybrid support. A pinned local installation selects direct DMA only
for the supported GPT-OSS-120B MXFP4 layout; unsupported models select standard
execution with an explicit reason. `--execution-mode standard` bypasses the
installation profile for rollback. Experimental warnings and readiness/resource
checks remain enabled.

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