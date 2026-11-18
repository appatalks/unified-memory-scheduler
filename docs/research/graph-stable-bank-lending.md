# Graph-stable bank lending: a feasibility screen

**Research direction, not a measured inference speedup or a claim of prior-art
novelty.** The idea is to reserve stable GPU virtual addresses for Hybrid's
shared expert-bank chunks, then temporarily map their physical pages as a
prefill workspace. After prefill, the same pages return to their original bank
addresses and the overwritten expert bytes are refilled. CUDA graphs that
refer to the stable bank address would only run while the bank is restored.
The [reviewed aggregate](../../benchmarks/results/vmm-bank-lending-research-summary.json)
records the measurements and failed accuracy gate.

This differs from a direct [FLE lending port](fle-prefill-audit.md), which
frees and reallocates whole per-layer cache buffers. Our cache groups layers
into shared banks; safe lending therefore needs explicit chunk ownership and
must prevent access to unmapped slots. CUDA VMM is an established primitive,
not an invention here: the experiment asks whether this composition is useful
for our allocator and scheduling constraints.

## Isolated CUDA experiment

The [probe](../../tools/trace/vmm-bank-lending-probe.cpp) owns its CUDA
context, physical allocation and two virtual addresses. It synchronizes,
unmaps the bank, maps the *same* physical backing as scratch, writes scratch,
then returns that backing to the bank address. It verifies all bytes before
and after aliasing, refills from a pageable host buffer, and replays the
**same instantiated device-to-device read graph** using the original bank
address. It checks every output byte. The sink is a separate allocation;
this is not evidence that an unmodified allocator will reuse lent memory.

RTX 5070 Ti, 2 MiB VMM granularity; three independent processes per size,
one warmup and 12 measured cycles per process. Table values are medians of
the three per-process medians. Mapping timings include unmap, map and access
setup; refill is a synchronous pageable-host-to-device copy. Graph replay
is the test's separate device-to-device copy and synchronization.

| Backing | Lend mapping | Return mapping | Pageable refill | Graph read replay |
| --- | ---: | ---: | ---: | ---: |
| 64 MiB | 0.128 ms | 0.105 ms | 4.427 ms | 0.209 ms |
| 512 MiB | 0.160 ms | 0.142 ms | 36.332 ms | 1.468 ms |
| 1,024 MiB | 0.176 ms | 0.149 ms | 72.642 ms | 3.088 ms |

All 108 measured cycles passed byte and graph-read checks. The cost of the
refill dominated remapping in this isolated test; the table excludes actual
expert dispatch, prefill, cache hit-rate changes, graph rebuilds, memory
pressure and end-to-end inference. It is not a projection of model speed.
The initial failed attempt omitted synchronization
after a scratch write and produced `CUDA_ERROR_ILLEGAL_ADDRESS`; the passing
probe synchronizes before every remap and after refill.

## Larger-ubatch accuracy gate

Lending is only useful for larger prefill compute buffers if that execution
is accurate. On the existing SC117 16K/F16, frozen 4 GiB bank contract, a
2,048-token prompt with matched full-length warmup was compared against the
physical-ubatch-512 full-vocabulary reference. The 512 reference had already
matched all five positions in six fresh control/candidate runs. Each larger
ubatch below was screened once; it is not a throughput benchmark.

| Physical ubatch | CUDA compute buffer | Passing raw-logit positions | Worst absolute / relative error |
| --- | ---: | ---: | ---: |
| 512 | 548.57 MiB | 5/5 in each prior 2K control | 0 / 0 |
| 1,024 | 1,096.67 MiB | 0/5 | 5.712663 / 0.272303 |
| 2,048 | 2,192.86 MiB | 0/5 | 4.261751 / 0.253608 |

All argmax tokens happened to match, which does not satisfy the accuracy gate.
The maxima may be at different positions. These screens changed the physical
batch only; they did not use VMM lending. Increasing ubatch is therefore not
a qualified use of this technique, regardless of remap cost. This does not
prove which underlying numerical path caused the differences.

A shorter threshold check used separate matched-warmup references for 512
and 1,024 input tokens. At 512 tokens, both the 512 and 1,024 configured
ubatches matched 5/5; the actual batch fits inside 512 in both cases. At
1,024 tokens, the 512 control matched 5/5, while the 1,024 ubatch failed
0/5 (maximum absolute 3.225783, relative L2 0.189434). These are different
synthetic prompts, so this narrows the conditions to investigate but does
not identify a particular kernel or prove causation.

## Required design and go/no-go gates

1. Build grouped banks from independently mapped chunks from the outset;
   the CUDA driver requires unmapping complete mapped ranges. Ensure expert
   slices, zero slots, metadata and group LRU ownership cannot straddle a
   lent chunk without explicit handling.
2. Before lending, quiesce in-flight GPU work and uploads, prevent cache
   lookup or graph execution against unmapped chunks, and allocate the
   scratch workspace from the same backing. Returning pages to the bank does
   **not** restore their overwritten contents; refill all previously live
   slots and atomically republish mappings only after synchronization.
3. Keep baseline routing, weight bytes and selected experts unchanged.
   Measure actual resident bytes needing refill, phase transition time,
   prefill gain, post-prefill decode and memory headroom. Compare with the
   existing allocator and with no lending in alternating, warmup-excluded
   trials. The microbenchmark alone cannot justify a production variant.
4. Require the established full raw-logit and cross-family gates, including
   Coder, SC117 Qwen and GPT-OSS, before any opt-in engine promotion.

The probe builds against the CUDA driver headers and links `libcuda`; pass
the size in MiB (16-1024) or `--self-test`. Do not run beside an unrelated
GPU compute process. Source SHA-256 for the measured version:
`724619a1fb7f76231514c566ef17833184b52042ddcbc037dea59f440a5db649`.
