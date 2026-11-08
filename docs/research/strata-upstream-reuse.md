# Strata: reuse evidence, test integration gaps

Status: bounded direct Strata MTP compatibility, optional production
provider/Automatic integration and installed-component reload pass.
[Initial reviewed evidence](../../benchmarks/results/strata-upstream-compatibility-summary.json);
[provider/cache follow-up](../../benchmarks/results/synapse-strata-cache-integration-summary.json).

## Evidence reused

The [Strata 0.2.12 release](https://github.com/Niko1221/Strata/releases/tag/v0.2.12)
publishes hardware-specific speed cohorts, operation checks and known
regressions. We reuse those, not rerun the release benchmark matrix.
Its Qwen engine is 0.1.58; the official CUDA 13 archive matches the
published SHA-256. Every extracted component file is verified against
the supplied manifest. The manifest signature is present but was not
independently verified. Its declared build source is
`8a42a9457db62568d1663c8c1bcc01cfffae5355`; documentation and the Python
protocol were inspected at `146df3c468739147f520ac3816de806f71232571`.
These are different provenance fields, not interchangeable revisions.

Upstream already includes the enumerated gated proposal test, proposal
packet tests, GPU/CPU proposal parity and native CPU expert arithmetic
parity across widths 1-8. Those proofs are not repeated here. They also
do not by themselves prove every downstream model/window combination.

The [existing local architecture study](model-agnostic-architecture-study.md)
already measured three alternating scalar/MTP pairs on this machine.
Two-row MTP improves every coding/reasoning/prose pair, with identical
512-token hashes. Those are preserved results, not new measurements.
No repeated speed cohort is necessary to establish that Strata can help.

## What was missing

Failed tandem CPU-miss comparisons and failed native verification ports
are not proof that Strata is incorrect. A useful next test is the actual
Strata runtime against its own scalar evaluator with the same placement
and target weights, not another engine's differently rounded evaluator.

The existing pack references original Qwen GGUF expert offsets and its
previously verified dense tensors. Its first shard hash differs from
the modified SC117 target used in newer native-cache screens. Do not
attach this pack to SC117 by filename/family similarity. Strata's
documented Coder is a Qwen3.8 variant, not evidence for our Qwen3-Coder-30B,
GPT-OSS or Clover. Model capability and pack binding must remain structural
and asset-specific rather than name rules.

## New bounded numerical check

The isolated official engine uses the existing native pack, 3,067-expert
profile, one 16K slot, F16 KV, seven CPU workers plus the host and no
lookup/adaptive swaps/cold-miss GPU tier. Tail skipping, routing prior
and resident routing bias are explicitly zero. `STRATA_IQ_MT_MIN=1`
and `STRATA_SPEC_PROB=1` keep the qualified native arithmetic/proposal
contracts. Dynamic CPU prefill sharing is off in both arms.

Three held-out prompts receive fixed 32-position control continuations.
Upstream's `STRATA_FOLLOW` executes real target verification and commits,
not an independent layer oracle. Its FP32 scoring copies expose the
full head through a diagnostic-only `cudaMemcpy` observer. The observer
records existing D2H copies and never changes data or adds a transfer.

All 96 full-vocabulary rows (248,320 logits each) match scalar versus
two-row MTP with **zero error**. Every selected raw head reproduces
upstream's forced-token and argmax log probabilities; every absolute
position and target matches the fixed control. The final request accepts
14 of 17 proposals, so subsequent target rows also cover rejection.

MTP copies 97 raw rows but scores 96: an accepted last window can contain
an unscored suffix beyond the bounded continuation. Alignment excludes
only that suffix, validates all scored positions, and cannot skip a
failed scored row. Synthetic altered/misaligned/non-finite/truncated
captures are rejected. The corrected review reuses the original capture
instead of repeating the hardware run; original operation-failed
accounting evidence is retained and hashed.

Reproduce with `tools/benchmark/vrambridge-strata-compatibility.py --help`.
Build only the observer using CMake with `VRAMBRIDGE_ENABLE_CUDA=OFF`,
`VRAMBRIDGE_TEST_STRATA_CAPTURE=ON`, then target
`vrambridge-strata-logits-capture`; no Strata rebuild is required.
Raw tokens/logits/logs stay private, never in the application or ledger.

## Actual Synapse boundary

`tools/benchmark/vrambridge-strata-server.py` is a research-only adapter.
It accepts the existing server CLI, verifies the reviewed model/pack
binding and engine component files, copies the profile into temporary
storage, and delegates rendering, streaming, HTTP and shutdown to the
pinned upstream implementation in offline mode.

Before publishing an API it requires actual context, F16, cache count,
MTP activation, no lookup and the declared miss placement. Missing
activation fails rather than accepting a drafter fallback. Unsupported
models/placement/contexts are rejected, not relabelled or silently
delegated. The adapter is not a universal fit policy: its expert profile
is the existing measured control.

The real Synapse app passes load verification, non-streamed chat,
complete SSE chat, slot telemetry and unload at 4K. All owned engines
exit. Temporary settings/tuning/XDG stores isolate the application;
the installed engine and default model profiles are not replaced.
Initial integration attempts exposed current upstream `None` heartbeats,
`close()` clearing its process handle and a lifecycle callback API.
Those were adapter defects, not model numerical failures.

Seven focused tests and three CTest groups pass, including existing
architecture and engine-comparison regressions. No engine CUDA rebuild,
dependency installation, new target weights or benchmark matrix is used.

## Production boundary

The follow-up adds the `strata-mtp` execution contract to Synapse's
catalog and existing detect/measure/remember path. An installed profile
binds the actual model/pack/controller/component and cache residency;
load must confirm the provider's own full-routing/F16/MTP facts.
Unknown/stale bindings and unsupported overrides are rejected.
The provider's Python/tokenizer dependencies enter its build identity.
Only the bound model's choice key includes that provider; native choices
for unrelated models are unchanged. A separate provider choice store
keeps old native/hybrid stores compatible with older application builds.

The cache review also finds the old matched partial ranking's intentional
capacity ceiling. The full ranking loads 3,991 slots and 7,786 MiB with
the same upstream byte-budget proxy, not an invented expert-count rule.
Both same-cache arms pass another 96/96 raw-logit positions with zero
error. Native cache candidates are hardware-derived with two bounded
fit/OOM refinements, not a fixed 12 GiB ceiling or a largest-is-best rule.
The fitting 9 GiB candidate measures 39.69 versus native 31.32 tok/s.

The real 1K comparison selects Strata at 57.26 tok/s and verifies
Automatic persistence/reload. Finished native trials are reused after
a transport-only endpoint fix rather than repeated. Strata's unsupported
native `/tokenize`, `/detokenize` and `/completion` operations are
adapted to the same tokenizer and serialized engine, counting accepted
outputs and validated backend times, never proposals.
A separate 16K capacity trial passes 14,333 input tokens and measures
55.85 decode tok/s. Balanced can recommend the faster supported context
with visible client limits; Maximum never silently reduces it.

The signed optional component/profile is installed with user consent;
native backend/profile files are preserved.
Installed-path reload is checked with an identical measured choice in
temporary stores. The updated app build is required to consume the
new provider, and no new engine toggle is added. After separate explicit
deployment approval, the old app is backed up and stopped gracefully,
the validated archive is replaced atomically, and the same native
backend/profile arguments are reused on restart. Listener, frontend,
provider detection and GPU cleanup checks pass.

This clears the supported Qwen/Strata gate. It is not:

- a native llama.cpp MTP/kernel port;
- qualification of arbitrary occupied context, long rollback, stochastic
  sampling distributions, every hardware/quantization or broad answer quality;
- Clover/diffusion or GPT-OSS support.

Automatic still needs its local comparison before choosing a provider;
upstream speed evidence guides candidates rather than hard-coding a winner.
Registration/installation and commits/pushes require consent. The optional
provider is separate; the native backend pin is unchanged.
