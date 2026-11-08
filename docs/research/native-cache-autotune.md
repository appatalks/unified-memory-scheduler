# Native expert-cache Auto Tune adaptation

Status: bounded application adaptation validated; not a new backend or
a blanket quality/speed qualification. Evidence:
[`synapse-native-cache-autotune-summary.json`](../../benchmarks/results/synapse-native-cache-autotune-summary.json).

## Plan and gates

1. Identify the native installed backend and its cache CLI contract,
   then measure owned, isolated cache/fitting configurations on Coder,
   SC117 Qwen3.8 and GPT-OSS. Keep model weights, F16 KV and full routing.
2. Integrate capability- and metadata-gated native-cache candidates into
   existing Auto Tune. Verify actual allocation; reject disabled/missing
   cache paths. Remember only passing winners and retain the 5% margin.
3. Validate application lifecycle, unsupported platforms, CPU-only paths,
   memory failure, persistence, cancellation and real Automatic reload.
4. Separately qualify cold-expert CPU/GPU overlap before changing its
   scheduling/precision implementation. Current numerical failures block
   default promotion.
5. Separately qualify model-compatible drafts, multi-position target
   verification and rollback before MTP/diffusion integration. Clover's
   prototype support does not qualify the installed backend.

Steps 1-3 are the completed native-cache application feature. Steps 4-5
remain gated research, not silently enabled components.
The [direct Strata follow-up](strata-upstream-reuse.md) now establishes
bounded same-engine Qwen scalar/MTP parity and a working research
Synapse API adapter. It does not establish a native port or production
Automatic-provider integration.

## What was implemented

No engine patch or installer pin is changed. Synapse's existing native
`--cpu-moe --moe-cache-mib` operation becomes an Auto Tune candidate,
using a guarded `standard-native-cache` profile. Matching is by MoE
metadata, declared backend CLI support and qualified platform/hardware,
not model names.

The first candidate scope is Linux/NVIDIA with F16 KV. CPU-only, dense,
non-qualified platforms and unsupported backends retain normal trials.
Budgets up to 12 GiB are tested only within the card's capacity minus a
reserve; actual backbone/KV/compute fit is checked by loading. Allocation
failure tries smaller budgets, but missing activation is not an
allocation retry. Existing Standard-only opt-out does not rewrite the
previous Automatic winner or enable experimental Hybrid.

Native cache loads enable reliable backend logging and must confirm a
non-CPU cache allocation and the requested KV precision before ready.
They suppress inherited experimental sharing/routing/default bundles.
Measured cache budgets survive applying/reloading through Automatic.
Profiles carry definition fingerprints; native shared-library identities
also enter the machine/build key. Saved files are not migrated or
rewritten just by inspecting them.

The capacity check also sends native `reasoning_effort=low`. GPT-OSS does
not use the Qwen-style `/no_think` instruction alone; without the effort
request the short probe failed before cache trials. This generic native
API control fixes that failure without changing ordinary chat preferences
or adding a model-name heuristic.

## Local pinned-backend screens

The native executable is b11669,
`620d2225cc436ab0b001e3200e590e3b286ef9ab`, inspected and run read-only.
Temporary runtime/settings/tuning stores isolate all experiments.
The first allocation checks lacked sufficient retained activation
logging and are excluded from cache-performance conclusions; the logged
follow-up confirms the requested operation.

One fresh load per configuration at 16K/F16, one slot, 8/16 threads,
64 excluded warmup tokens and three 128-output held-out workloads show:

| Target | Native fit decode range | Native 8 GiB cache decode range |
|---|---:|---:|
| Coder Q4_K_M | 68.53-70.38 | 111.77-140.30 |
| SC117 Qwen3.8 IQ3_S | 27.48-27.78 | 30.84-33.32 |
| GPT-OSS-120B MXFP4 | 23.27-23.70 | 15.65-20.25 |

The 4 GiB cache does not offer a broad win across these workloads and
families. Coder's strict coding predicate
fails on fitting and both cache sizes; arithmetic passes. Qwen/GPT pass
both narrow predicates. Do not interpret these results as broad coding
quality or independent numerical equivalence across CPU/GPU placement.

Actual Auto Tune then runs the load/probe/measure/remember/apply/reload
flow with an isolated store. A bounded 1K maximum selects a 12 GiB cache
for Coder (201.96 vs 85.75 decode-probe tok/s), an 8 GiB cache for Qwen
(37.30 vs 29.95), and keeps uncached GPT fitting (25.14 tok/s).
Qwen's oversized 12 GiB trial fails allocation and falls back correctly.
These figures are short raw Auto Tune decode probes, not normal-chat
rates or full-context speed guarantees.

Coder is additionally tuned through 8K and 16K, with 14,330 actual input
tokens in the largest successful capacity check. The 12 GiB trial fails
allocation; the passing 8 GiB winner measures 141.57 vs 75.27 raw decode
tok/s and reloads verified through Automatic. Actual 32K/maximum-context
and other hardware configurations remain local qualification work.

## Validation and production boundary

Focused candidate/activation/profile tests and the complete
lifecycle/API suite pass (142 cases including inherited-environment
suppression), as do 10 tuning/profile tests, 61 browser tests
and the actual zipapp smoke checks. Isolated wheel imports verify the
app, tuner and five-profile catalog without installing it. Release packaging includes the
catalog; no model data or private reports enter the application.

This feature chooses an existing native implementation by local
measurement rather than declaring a global best preset. It does not
enable expert skipping, routing bias, new activation/KV precision,
unqualified CPU-miss sharing, drafting or concurrent application chats.
Neither installed backend/profile nor the user's saved state is changed.
Research commits/pushes and any future backend pin/deployment still
require separate consent.
