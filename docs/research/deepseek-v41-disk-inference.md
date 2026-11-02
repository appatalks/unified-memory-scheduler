# DeepSeek V4.1: explicit disk-tier feasibility

Date: 2026-11-01. Status: initial architecture and storage screens complete;
replacement downloaded, payload checksums skipped at user request;
native 4K text smoke/baseline complete; broad quality, numerical parity and
explicit disk-tier optimization gates pending. Two live RAM expert-cache
candidates completed matched comparisons and were rejected; see section 11.
The user reports repairing the
WD volume in Windows; bounded read tests resumed successfully. The earlier
controller outage remains an unresolved reliability finding; see sections 7-9.

## Research question and current answer

Can exact scheduling make a model substantially larger than physical memory
interactive on consumer hardware? **Possibly for the right sparse workload;
not established by this study.** Capacity is not an absolute prohibition,
but neither mmap nor asynchronous I/O removes compulsory reads.

The first promising distinction is architectural. V4.1 has a large,
deterministically addressed lookup table, not just a large MoE expert pool.
Those two kinds of weight need different disk policies. Engram looks
latency-feasible on the measured SSD paths. Sustained expert misses look
bandwidth-limited unless actual demand is highly concentrated in a cache.

Research stays here. No installed Synapse backend/profile/engine choices were
changed, and the previous V4 Flash 284B Q4 files remain intact. The b11669
patch is not a disk-streaming implementation. Do not transplant experimental
disk code into Synapse before its measurements and correctness gates pass.

Machine-readable [initial evidence](../../benchmarks/results/deepseek-v41-disk-feasibility.json)
contains the pinned file manifest, tensor offsets/types, 21 storage screens,
and computed bounds. It is a dated snapshot, not live acquisition status.

The user explicitly stopped payload SHA-256 verification after download.
All six local sizes, GGUF shard indices and tensor descriptors were checked,
and Engram was renamed to the required sixth-shard filename. The previously
verified metadata hash remains valid; full weight integrity must be labeled
unverified. This permits exploratory runtime tests, not a checksum-verified
replacement or automatic retirement of the old model.

## 1. Replacement assets and compatibility

Selected with user approval:
[ISTA-DASLab GSQ-RCO IQ2_XXS](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/tree/65d53bc8be65123e255ff02f3ded958b6dbf54fd),
revision `65d53bc8be65123e255ff02f3ded958b6dbf54fd`.

| Component | Files | Actual manifest size |
|---|---:|---:|
| IQ2_XXS transformer backbone | metadata shard plus four payload shards | 138.239 GB |
| Shared IQ4_NL Engram | one payload file, loaded as shard 6 | 110.595 GB |
| Text-only total | six shards | **248.834 GB** |

The publisher recommends IQ2_XXS and reports parity with the base model on
three reasoning/coding benchmarks. This is **publisher-reported narrow
evaluation**, not proof of general quality, lossless quantization, or
interactive performance here. Research correctness means preserving all
bytes of the selected quantized checkpoint and every router-selected expert;
it does not mean that IQ2 is identical to unquantized weights.

The initial acquisition excludes the optional BF16 vision projector and
DSpark draft. Text inference, target-only decoding and routing are qualified
before adding vision or speculation.

Header verification fetched only the 5,780,704-byte metadata shard and five
1 MiB HTTP range prefixes. The metadata shard's SHA-256 matches the pinned
manifest. Across the prefixes, all 1,043 tensor descriptors are present,
shard indices/counts agree, names are unique, quantization blocks are integral,
and aligned tensor ranges neither overlap nor exceed expected file lengths.
This verifies **descriptors**, not unread payload hashes or loadability.

The GGUF is `deepseek41`, not `deepseek4`: 40 blocks, 384 routed experts per
block, six active experts, width 5,120 and expert intermediate width 2,304.
The downloaded target tensor descriptors contain approximately 748.495B
scalar elements; the advertised family count is 763.2B. Do not substitute
the advertised count for actual stored bytes, or silently assume identical
counting boundaries for target, vision and draft assets.

### File naming and storage

The normal first filename is:

```text
DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf
```

Shards 2-5 follow the same prefix. Remote Engram is named
`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`; llama.cpp needs its bytes under:

```text
DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf
```

The user approved a dedicated directory on the WD_BLACK SN850XE NVMe/exFAT
volume. It has about 1.2 TiB free before acquisition. **exFAT has no symlinks
or hard links:** after SHA-256 verification, rename the Engram file to the
required sixth-shard name without making another 110.6 GB copy. The acquisition
journal retains its remote name/hash. Do not use the publisher's `ln -s`
recipe on this filesystem.

The root SATA SSD has about 112 GiB free. Neither the entire build nor Engram
alone fits there while preserving the existing 16 GiB staging reserve.
The archive HDD has about 3.6 TiB free, but is a poor random-row working tier.
The Samsung NTFS drive and existing staging policy remain untouched.

### Isolated backend and rollback

The model card requires llama.cpp b11870 or later. Pin the first supported
release for the initial baseline:

- upstream commit `74bca3950054eed1795f525cb9cf06bfe796492d`;
- official `llama-b11870-bin-ubuntu-cuda-12.8-x64.tar.gz`;
- archive SHA-256
  `9a60e70bd94487b124a9ffdd96438e48726f65d68a9c81ca1631c869aeff4f87`.

The archive hash, `--version`, `--list-devices` and `--lazy-mode` help were
verified in a separate session build directory. The new GPU plugin resolves
the existing official NVIDIA CUDA libraries and detects the RTX 5070 Ti.
That establishes binary startup, **not** V4.1 model execution.

The user's b11669 executable, libraries and profiles remain the rollback.
Use matching server/plugins from the isolated b11870 directory and a temporary
configuration directory; never mix its CPU/CUDA libraries with b11669.
Do not apply the old hybrid patch blindly: its registration of full immutable
expert source ranges (64 GiB cap) conflicts with truly out-of-core experts.
Rebase and qualify a bounded staging design separately, retaining the old
canonical patch until a replacement passes the existing model-family gates.

## 2. Tensor-aware bandwidth accounting

Parsed IQ2_XXS tensor descriptors, not nominal bit-width estimates:

| Weight class | Payload | Access per generated token |
|---|---:|---|
| Routed experts | 132.073 GB | **2.064 GB** if all selected experts are read |
| Other target weights | 6.160 GB | heterogeneous dense/shared/attention operations |
| Engram | 110.595 GB | 48 selected rows, not a sweep of the entire table |

Expert gate/up/down slices together are 7.741-10.230 MB per expert, depending
on layer-specific quantization. There are 240 selected expert evaluations per
token. Cache metrics must be **bytes weighted**; expert-count hit rates are
not equivalent when formats differ across layers.

Engram is one `[256, 768022850]` IQ4_NL tensor. A row is 144 bytes. Three
n-gram lengths, eight heads and two Engram sites require **48 rows/token**:
6,912 logical bytes. At 4 KiB page granularity, unrelated cold rows imply
roughly 196,608-393,216 physical bytes, depending on alignment, before
filesystem read-ahead. Coalescing and cache hits can reduce this; readahead
can increase it. Measure actual pages rather than treating logical bytes as
physical I/O.

Upstream b11870 already computes Engram hash IDs at graph-input setup and
gathers its rows before the transformer blocks. Its lazy-read mode
memory-maps large lookup tensors rather than preloading the entire table.
The row addresses therefore do not require guessing a future router:
all current-token Engram requests can be submitted early. This is a better
first explicit-I/O prototype than replacing expert routing.

## 3. Measured storage baseline

fio 3.39, read-only files, `direct=1`, `libaio`, one worker, eight seconds per
case. SSD/HDD windows use existing model files; the WD volume uses an owned
2 GiB fixture. No device writes, cache drops or remounts. Key random cases
were repeated; the table reports their two-screen averages.

| Current storage path | Random 8 MiB, QD1 | Random 4 KiB, QD8 | 4 KiB QD1 mean latency |
|---|---:|---:|---:|
| Samsung 870 EVO SATA/ext4 | **558.6 MB/s** | 62,875 IOPS | 0.098 ms |
| Seagate 24 TB SATA/XFS HDD | **163.7 MB/s** | 385 IOPS | 5.892 ms |
| WD_BLACK SN850XE NVMe/exFAT | **312.7 MB/s** | 37,135 IOPS | 0.054 ms |

The WD controller reports PCIe Gen4 x4, but **this mounted path did not
deliver multi-GB/s large reads in these screens**. Its sequential QD8 result
was approximately 325 MB/s. The cause is not established. Do not assume an
advertised NVMe bandwidth in the feasibility model; investigating filesystem,
host path, concurrent I/O and sustained behavior is a performance work item,
not justification for changing mounts or buying hardware now.

O_DIRECT bypasses the OS page cache, not controller caches. Small, eight-second
fixtures do not establish steady thermal behavior or model-sized random-I/O
performance. Independently measured large-read bandwidth and random IOPS
cannot both be fully consumed by one workload.

### Optimistic bounds

Define `A` as selected expert bytes/token, `h` as the combined RAM+VRAM
**byte** hit rate, `R` as read amplification, `p` as useful prefetch fraction,
and `B` as measured disk bandwidth:

```text
physical expert bytes/token = A * (1 - h) * R / p
I/O floor >= physical bytes/token / B
fully overlapped token time >= max(I/O time, resident compute time)
```

This is a lower bound on latency, not a speed prediction. Current-token
layer routing dependencies, queueing tails, Engram misses, PCIe transfers,
decompression and CPU/GPU math can make actual latency larger.

With ideal amplification/prefetch and zero compute cost, **10 tok/s** needs:

| Path | Minimum combined expert byte hit rate | Remaining logical misses/token |
|---|---:|---:|
| SATA SSD | 97.30% | 55.7 MB |
| Archive HDD | 99.22% | 16.2 MB |
| WD NVMe, current measured path | 98.49% | 31.1 MB |

At only 90% hit rate, the WD path's optimistic I/O ceiling is approximately
**1.5 tok/s**. A hypothetical measured 5 GB/s NVMe path would allow about
24 tok/s for the same cold expert bytes before computation/dependencies;
that is a sensitivity calculation, not this machine's measured result.

For 48 cold Engram pages/token alone, the HDD QD8 throughput bound is about
8 tok/s, even before expert reads, and serial faults are slower. Early batched
SSD row reads are plausible; HDD placement is not.

Use the [tested roofline calculator](../../tools/benchmark/vrambridge-disk-roofline.py):

```sh
python3 tools/benchmark/vrambridge-disk-roofline.py \
  --active-bytes 2063646720 --bandwidth-mbps 312.67 --iops 37135 \
  --rows 48 --row-bytes 144 --hit-rate 0.90 --target-tps 10
```

The calculator debits wasted prefetch/read amplification, separates full-overlap
and serial references, and checks random-row/compute necessary conditions.
It does not infer a hit rate from model size or invent generation speed.

## 4. Relevant primary-source designs

| Source | Useful mechanism | Qualification boundary |
|---|---|---|
| [DeepSeek Engram](https://github.com/deepseek-ai/Engram) | Deterministic n-gram addressing, separate conditional-memory tier | Demo/research code; host-memory overhead claims do not prove disk latency for the full table |
| [SSD-LLaMA](https://arxiv.org/html/2609.18110v1) | Contiguous per-expert gate/up/down pack, bounded io_uring pipeline, lossless per-expert compression | Exact relative to the chosen quantized checkpoint; reports 8.89 to 5.93 GB/token and a 1+ tok/s objective on NVMe, not 10+ tok/s on this path |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | Request-local expert tracing, cache admission and prefetch | RAM-resident backing pool in reported offloading setting; not proof of SSD residency |
| [Fiddler](https://arxiv.org/abs/2402.07033) | Decide CPU execution versus GPU weight movement; move activations where cheaper | CPU compute/DRAM assumptions and full backing memory differ from SSD cold misses |
| [FlexGen](https://arxiv.org/abs/2303.06865) | Joint weight/KV/activation scheduling across three tiers | Large-batch throughput is not single-user interactive decode; optional quantization separate |
| [PowerInfer-2](https://arxiv.org/abs/2406.06282) | Fine-grained neuron-cluster I/O pipeline and hardware-aware planner | Smartphone UFS/TurboSparse targets; not unchanged V4.1 expert execution |
| [Edge0](https://arxiv.org/html/2609.18063v1) | Overlap storage with cross-token prediction | Its prediction is consumed as routing and recovery adapters change the model; **not an eligible exact path here** |

No headline speed from another hardware/model/workload is treated as a
qualification result. Lossless transport must be byte-checked, including
scales and non-compressible fallback formats; GSQ IQ2 compression ratios
cannot be assumed from another model's packed 4-bit histogram.

## 5. Incremental experiment queue and gates

1. **Replacement qualification and native baseline.** Finish acquisition and
   SHA-256 all six files. At 4K, text only, target only, isolated b11870 and
   lazy Engram enabled, verify startup, chat encoding, short arithmetic,
   structured output and executable coding checks. Compare cold load,
   first request, repeated prompt and multiple distinct held-out prompts.
   Preserve the old V4 files until these gates pass; retire only the resolved
   five old shard paths, never a broad model root.

2. **Explicit Engram row tier.** First use smaller synthetic and real
   row-gather fixtures with a forced tiny page cache. Compute exact current
   row IDs, merge/deduplicate pages, issue bounded asynchronous reads before
   graph consumption, and gather only requested rows. Compare bytes and
   graph output with native lazy mmap. Pin only a bounded staging pool, not
   the 110.6 GB table. Require lower p95 row wait/TTFT without extra
   approximation, unsafe buffers or unaccounted RAM.

3. **Measured expert-trace replay.** Collect `(request, token, layer, expert,
   tensor offset/type/bytes)` plus ready/deadline timestamps for each family.
   Split calibration from held-out prompts. Compare byte-budget LRU, aging
   LFU, request-local and cost-aware policies at realistic RAM/VRAM budgets.
   Resolve both caches before disk I/O. Sweep next-token/cross-layer
   prediction but always use actual router choices with mandatory miss
   fallback. Reject predictive policies that spend more disk bytes or
   increase p95 demand wait despite better nominal hit rates.

4. **Exact disk-expert delivery.** On Coder/SC117/gpt-oss, artificially limit
   resident memory to force true out-of-core access without downloading
   another huge model. Start from a read-only GGUF slice manifest; compare
   three component reads with aligned per-expert packing. Then bounded
   io_uring reads -> staging readiness -> CPU compute or H2D -> CUDA event
   -> visible cache slot. Never reuse a buffer before readers/DMA complete,
   evict an in-flight selected expert, drop a contribution, or preload the
   entire source pool through host registration.

5. **Compression and adaptive scheduler.** Add lossless transport only if
   representative expert measurements save end-to-end cold-miss time
   including decompression. Choose CPU/GPU placement from observed service
   costs, bandwidth contention and deadlines, not model names. Add online
   request-local adaptation only after fixed-policy causal baselines pass.

Every stage records generated/accepted token counts, first token/text and
backend decode time separately; `/proc/<pid>/io` physical reads, major/minor
faults, RSS/anonymous/file pages/swap; NVML VRAM and PCIe traffic; useful and
wasted prefetch bytes; RAM/GPU/disk hits; read latency distributions; graph
and CPU service intervals. RSS and page cache overlap, so do not double-count
them. Attributable physical traffic is required before calling a window
disk-backed. Report read amplification as physical bytes / useful tensor
bytes, not syscall `rchar`.

Use **5, 10 and 20 tok/s as research targets**, not promises. Promotion
requires numerical consistency with the selected quantized reference
(existing max-absolute 0.01 / relative-L2 0.001 gates where applicable),
unchanged-weight checks, held-out answer checks, repeated alternating runs,
tail latency and explicit resource safety. Warm repeated prompts alone are
not sufficient.

An experiment is stopped for unsafe available-memory/swap pressure, failed
buffer/DMA ownership, missing observations, changed routing/math or exhausted
time budget. Memory-availability snapshots include reclaimable cache and do
not guarantee an allocation will succeed. Failure stays in the record.
Only the benchmark's owned process is stopped.

## 6. Native 4K baseline after acquisition

The guarded isolated b11870 run completed in 559.9 seconds. The backend log
confirmed 4K context and lazy Engram. Loading took **452.9 seconds** and
attributed **113.91 GB** of physical reads to the owned process.
No cache drops were performed; this is not a controlled cold-storage load.

| Request, 128 generated tokens | Generation tok/s | First text | Request physical reads | Generation-window major faults |
|---|---:|---:|---:|---:|
| First hash-table prompt | 2.95 | 12.72 s | 12.329 GB | 49,376 |
| Same prompt repeated | 14.29 | 0.70 s | 0.000328 GB | 0 |
| Distinct AVL-tree prompt | 5.40 | 2.96 s | 4.209 GB | 21,415 |

The generation-window physical reads were respectively 9.017 GB,
0.000254 GB and 3.588 GB. These windows start at the first received text,
not an exact graph execution boundary. They nevertheless establish a large
physical-I/O difference between the warm repeated prompt and new requests.
The warm **14.29 tok/s is not evidence of interactive SSD streaming**:
almost all of that window's weight access was already cached.

Peak observed process RSS was 112.71 GiB and peak device-wide VRAM 14.73 GiB.
File-backed RSS overlaps reclaimable page cache; neither is a fully locked
residency budget. The available-memory counter therefore must not be added
to these figures or interpreted as a guarantee of further allocatable memory.

Natural-completion arithmetic returned `323`; the structured-output request
returned `{"product": 323, "word": "cedar"}`. Both checks passed, but there is
no coding execution, broad quality evaluation or full-vocabulary numerical
comparison yet. The native backend was stopped after the run; Synapse's
installed backend, profile and saved choices remain unchanged. SC117 was
unloaded through the app API with explicit approval, leaving the app running.

The detailed local report is
`benchmarks/results/deepseek-v41-native-20261101T220815Z.json`.
The sanitized summary in the initial evidence links its SHA-256 and keeps
the checksum-skipped and qualification limits explicit.

This demonstrates that the replacement **can execute on this machine despite
its total disk size exceeding physical memory**. It also provides a real
baseline bottleneck: different prompt/routing patterns trigger compulsory
physical reads and many major faults. The next prototype must reduce those
miss stalls; it cannot claim the repeated-prompt warm rate as its disk result.

## 7. Exact row-stream prototype and storage blocker

The [bounded row-stream prototype](../../tools/benchmark/vrambridge-row-stream.py)
implements a read-only positional page source, explicit futures/readiness,
byte-identical row reconstruction across page boundaries, page deduplication,
a bounded LRU page cache and bounded worker/staging storage. It supports
buffered `pread` and aligned `O_DIRECT`/`preadv` reads. Consumers receive all
requested rows in their original order, including duplicates. Short/failed
reads and out-of-range inputs raise errors rather than returning zero rows.
There is only one outstanding request per store; closing drains readers
before releasing file descriptors or direct-read buffers.

This is an **experimental primitive, not an integrated inference engine**.
The declared payload bound excludes Python object/container overhead. The
tests cover bytes/order, page boundaries, request/cache limits, cache-hit I/O
avoidance, failed/short reads, backing-file changes, direct-read alignment
and closing while work remains pending.

Source review corrected an important baseline assumption: b11870 already
hashes both Engram sites in graph-input setup, merges their page ranges and
prefetches them. Linux issues `madvise(MADV_WILLNEED)` per merged range;
Windows uses a batched virtual-memory prefetch API. The new comparison must
therefore include **existing upstream prefetch**, not compare only against
plain serial page faults and call that an improvement over llama.cpp.

The first real-file microbench used synthetic numeric-token histories hashed
according to the metadata/upstream rule, not live tokenizer or routing traces.
Every completed result matched the same local GGUF row bytes. Hash parity
against a running native graph is not yet qualified. No compute or GPU work
was simulated, no global caches were dropped, and both completed novel-ID
phases had zero measured initial page residency.

| Completed method, 48 rows/request | Novel-row median | Novel-row p95 | Immediate-reuse median |
|---|---:|---:|---:|
| Plain mmap | 4.72 ms | 5.19 ms | 0.0018 ms |
| Upstream-style merged madvise + mmap | 1.66 ms | 1.71 ms | 0.0145 ms |

Only 32 timed novel requests plus immediate repeats completed per method in
one trial. This confirms the value of the **already present** upstream
prefetch path; it does not establish a new optimization gain or infer model
tok/s. Per-token row latency on these screens is small relative to the
disk-heavy native generation times, so expert misses remain the likely
larger optimization target, subject to actual per-class attribution.

The concurrent buffered-read phase failed with `errno 5` (I/O error).
Read-only follow-up found the WD NVMe controller `nvme1` in kernel state
**`dead`**, with the block device reporting **0B**. An independent read of the
small metadata shard also failed. The direct-read comparison did not run.
Kernel logs were unavailable to the current user, so no root cause, physical
damage or data-loss claim is justified.

The [partial measurement/blocker record](../../benchmarks/results/deepseek-v41-row-stream-blocked.json)
preserves both positive partial results and the failure. No filesystem repair,
controller reset, remount or model deletion was attempted. Storage recovery
must precede further accesses or inference; the previous V4 model remains on
the separate archive HDD. After recovery, validate the files again before
continuing the interrupted method comparison.

## 8. Bounded system-SSD fixture and research pause

After reboot, the WD controller remained visible on PCIe but had no bound
NVMe driver or usable block device. With user approval, a separate
[row-stream study runner](../../tools/benchmark/vrambridge-row-stream-study.py)
was added for the system SATA SSD. It creates at most a 256 MiB private
fixture, preserves a 16 GiB free-space reserve, refuses tmpfs/rotational
backing, and removes its owned fixture on completion or ordinary failure.
It never accesses the WD research volume.

The fixture has deterministic, non-repeating page bytes. An independent
generated reference checks every gathered row without reading the fixture
and warming its page cache. Matched synthetic requests, explicit warm-up,
rotating method order, per-request physical-read counters and separate
immediate-reuse controls support comparison of mmap, upstream-style
prefetch, serial/direct concurrent reads and the bounded page cache.
The runner does not generate model tokens or emulate expert routing.

The first 128 MiB screen is retained as a
[failed qualification result](../../benchmarks/results/system-ssd-row-stream-20261101.json).
Only the two warm mmap methods completed; both observed zero physical reads.
The direct-read control also observed no physical I/O in that run, so the
runner explicitly rejected the result instead of claiming disk throughput.
A separate 4 MiB diagnostic observed the O_DIRECT flag and nonzero physical
reads from both main-thread and worker-thread reads; this does not explain
or repair the failed control. Further attribution remains pending.

All 16 row-stream/study tests, six roofline tests and ten existing I/O-study
tests pass. No SSD fixture remains, and no new end-to-end inference
optimization has been qualified. The user then explicitly paused research
to address the drive; do not restart workloads automatically.

## 9. Recovered-volume checks and completed bounded screens

Later on 2026-11-01, the user reported repairing the drive in Windows.
Linux again enumerated the WD controller as `live`, with the correct capacity
and a read/write exFAT mount. All six model files retained their expected
lengths; small header, middle and end reads succeeded. These checks do not
establish full weight integrity or a root cause for the earlier outage.
Payload SHA-256 remains skipped at user request, and the previous V4 model
has not been retired. Privileged kernel/health diagnostics remain unavailable.

### Corrected system-SSD control

The small [recovery control](../../benchmarks/results/system-ssd-row-recovery-control-20261101.json)
reproduced the earlier missing-physical-I/O rejection. The cause was then
identified in the fixture **runner**, not the row primitive: its direct cases
did not forward `direct=True` into `RowStore`. Thus those named direct
controls were buffered reads. The physical-I/O guard correctly rejected
them. Both failed records remain intact; no disk performance is inferred
from them.

The flag wiring is corrected and a regression test verifies all direct
methods request direct I/O. Unit-test physical counters are mocked because
temporary directories can be tmpfs; actual direct-I/O traffic is separately
attested in the block-backed SSD run.

The [corrected screen](../../benchmarks/results/system-ssd-row-stream-fixed-20261101.json)
used a 32 MiB owned SATA-SSD fixture, 48 rows/request, 16 timed novel sets
with immediate repeats, three rotating-order trials and two workers. The
fixture was removed. Means of the three phase medians:

| Method | Novel-set gather | Immediate-repeat gather |
|---|---:|---:|
| Direct serial | 3.91 ms | 3.85 ms |
| Direct two-worker | 2.12 ms | 2.10 ms |
| Direct two-worker + 1 MiB page cache | 2.10 ms | 0.10 ms |

Serial/concurrent cases each observed 24.31 MB of physical reads across all
three cases including warm-up. The cache control observed 11.83 MB; repeated
sets had no submitted page reads. This demonstrates exact concurrent I/O
and bounded caching on that artificial workload, **not** model-routing
locality, inference tok/s, or a benefit over native warm mmap. The mmap
methods had zero physical reads from the recently written fixture and are
only warm-cache references.

### Real Engram screen after recovery

The [completed Engram screen](../../benchmarks/results/deepseek-v41-row-stream-recovered.json)
used two workers, two rotating-order trials and eight timed novel synthetic
token histories per method, each immediately repeated. All novel phases had
zero measured initial page residency. No cache drops, filesystem changes,
controller resets or inference processes were involved.

| Method | Novel-set mean of phase medians | Immediate-repeat mean of phase medians |
|---|---:|---:|
| Plain mmap | 9.43 ms | 0.0019 ms |
| Existing upstream-style madvise + mmap | **5.30 ms** | **0.0149 ms** |
| Explicit two-worker buffered pread + cache | 6.25 ms | 0.111 ms |
| Explicit two-worker direct preadv + cache | 5.97 ms | 0.124 ms |

All quantized rows matched the same local GGUF bytes. The controller still
reported `live` after the screen. The two explicit implementations have a
declared payload bound of 1,588,256 bytes, excluding Python/container and OS
cache overhead. The direct screen's whole phase interval also includes
reference mmap reads, so its total physical traffic must not be attributed
solely to the prototype; it is not a read-amplification measurement.

**No new Engram speedup is qualified.** Explicit two-worker row reads were
slower than the already-present upstream path on this limited comparison,
including immediate reuse where warm mmap is much cheaper than Python
future/cache bookkeeping. Request distributions are synthetic and different
between methods, phase medians differ across trials, and host conditions
changed since the pre-outage run. This is a screening rejection for this
prototype/configuration, not proof that every native io_uring implementation
or queue depth would lose.

The completed screen supplies no justification for patching production
Synapse with this row path. Keep native Engram prefetch as the baseline.
Prioritize actual routed-expert bytes/cache misses and physically measured
per-layer waits before adding a larger reader pool or replacing backend I/O.
The 17 row-stream/study tests, six roofline tests and ten existing I/O tests
pass; no installed backend/profile/saved state was changed.

## 10. Optional expert-I/O observer and causal cache replay

The existing [routing collector](../../tools/trace/routing_trace.cpp) now
accepts optional Linux `--trace-io`. Its ordinary capture path remains the
default. The observer records:

- actual top-k expert IDs and metadata-derived original expert byte sizes;
- process physical `read_bytes` and major/minor-fault deltas from the routed
  top-k boundary to that layer's `ffn_moe_out` completion;
- Engram gather boundary intervals, when present;
- whole-token decode intervals for distinguishing covered from unattributed
  physical reads.

Those intervals include scheduler fences and callback overhead. Engram
prefetch begins before its gather, so gather intervals alone do not measure
all Engram-related I/O. A read occurring in an expert interval is an observed
boundary attribution, not proof that every byte came from that expert
tensor. The observer changes synchronization, so its wall time is **not a
generation-speed benchmark**. Each request retains the existing callback-
inactive replay with the same prompt and greedy sampling; the generated
tokens and stop condition must match before the trace is published.

The default CMake target and commit remain b11669. The same collector was
separately compiled against the exact b11870 source and matching official
libraries using an explicit `VRAMBRIDGE_TRACE_LLAMA_COMMIT` definition.
There is no change to the canonical backend patch or installed application.
Do not use the optional pin to mix headers and libraries from different
versions. For an isolated b11870 source/library pair:

```sh
source=/path/to/llama-b11870-source
libraries=/path/to/llama-b11870-official-libraries
collector_sha=$(sha256sum tools/trace/routing_trace.cpp | cut -d' ' -f1)
g++ -std=c++17 -O2 -pthread \
  -I"$source/include" -I"$source/common" -I"$source/src" \
  -I"$source/ggml/include" -I"$source/vendor" \
  -DVRAMBRIDGE_TRACE_LLAMA_COMMIT='"74bca3950054eed1795f525cb9cf06bfe796492d"' \
  -DVRAMBRIDGE_LLAMA_LIBRARY_DIR="\"$libraries\"" \
  -DVRAMBRIDGE_TRACE_SOURCE_SHA256="\"$collector_sha\"" \
  tools/trace/routing_trace.cpp \
  -L"$libraries" -Wl,-rpath,"$libraries" \
  -lllama -lllama-common -lggml -lggml-base -o /path/to/vrambridge-trace-b11870
/path/to/vrambridge-trace-b11870 --io-self-test
```

The deferred expert-output and Engram boundaries, ID preservation and
missing-boundary rejection passed bounded CPU fixture self-tests on both
pinned builds. Collector input/self-test suites also passed for both builds.
This establishes the observer plumbing, not actual full-model coverage.

The [disk-trace analyzer](../../tools/benchmark/vrambridge-disk-trace.py)
reuses the existing routing validator, byte-budget cache and training hot-set
helpers. It requires a complete trace and successful token-sequence checks.
It retains unattributed reads and rejects missing/out-of-order intervals or
read sums exceeding whole-token observations.

The analyzer separately replays a training-only fixed cache, LRU and the
existing basic unaged LFU policy over held-out routes. Misses are measured
before admissions; it does not change routing or consult future events.
Byte hit fractions account for layer-dependent expert sizes. Combined
RAM+VRAM budgets cover only retained expert bytes: staging, KV/dense weights
and OS memory need independent reserves. These policies are neither the
production LFU bundle nor an implemented disk cache.

```sh
python3 tools/benchmark/vrambridge-disk-trace.py /path/to/complete-routing.jsonl \
  --capacity-gib 32 64 96 108 --bandwidth-mbps 312.67 \
  --output /path/to/new-disk-cache-report.json
```

Bandwidth-only miss-service floors are sensitivity calculations, not predicted
tok/s. They omit seeks, amplified reads, wasted prefetch, Engram, transfer and
computation. Calibration is short and native reference replays warm OS pages
between requests; neither is hidden or presented as sustained held-out speed.

The guarded V4.1 observer protocol uses one training and two held-out
prompts, up to 32 generated tokens each, 4K context, target-only, lazy Engram,
8/16 threads and 128-position batches. A 30-minute limit, available-RAM/swap
reserves and a live-controller check stop only the owned collector. Full
payload hashes remain skipped at the user's request.

### Completed V4.1 attribution run

The first attempt stopped before model loading because `--reasoning` is a
server-only option, not a COMMON collector option. The corrected collector
sets `enable_thinking=false` in the model's inspected Jinja template; the
unrecognized CLI flag was removed. The failed launch remains recorded and
is not counted as a model-compatibility failure.

The corrected run completed in 649 seconds including loading and reference
replays. All three requests produced 32 fully routed tokens across all 40
layers. Every token had two completed Engram gather boundaries. All three
callback-inactive greedy token-sequence comparisons passed, and the owned
collector stopped normally. The
[sanitized observer summary](../../benchmarks/results/deepseek-v41-expert-attribution-summary.json)
and [full cache-analysis output](../../benchmarks/results/deepseek-v41-disk-trace-20261102.json)
retain source/trace identities and qualification limits.

| Request, 32 observed tokens | Whole decode physical reads | Expert-interval reads | Engram-gather reads | Expert-interval major faults |
|---|---:|---:|---:|---:|
| Training: hash tables | 2.299 GB | 2.287 GB | 4.01 MB | 12,258 |
| Held-out: AVL trees | 1.048 GB | 1.040 GB | 2.62 MB | 5,797 |
| Held-out: logic | 1.103 GB | 1.093 GB | 3.19 MB | 6,046 |

Approximately **99.2-99.5% of observed decode reads occurred inside routed
expert intervals**. Earlier Engram prefetch is not fully covered by gather
timing, but the small explicit gathers and retained unattributed read
remainder do not support making it the primary bandwidth target. Expert
interval reads averaged 32.5-34.2 MB/token on the two held-out requests.
These are observed physical reads, not proof of a particular full-expert
cache-hit percentage.

Peak process RSS was 114.86 GiB, device-wide VRAM 14.65 GiB and process swap
165.9 MiB. The native allocation included approximately 6.936 GB of expert
weights on GPU. RSS includes reclaimable mapped pages; it is not a pinned
cache budget. No controller outage occurred during this run.

### Replay finding and its important limitation

The causal byte-cache screen counts misses before admission, starts with
only the short training route, and continues adapting on held-out routes:

| Simulated combined expert-cache budget | Training-only fixed hit rate | LRU hit rate | Basic LFU hit rate |
|---|---:|---:|---:|
| 32 GiB | 57.19% | 84.92% | 84.77% |
| 64 / 96 / 108 GiB | 57.19% | 85.24% | 85.24% |

The larger adaptive budgets all ended with only **37.57 GiB** resident,
because this short trace never touched enough distinct experts to fill them.
Training-only residency was 19.42 GiB. Therefore these numbers neither show
that 64 GiB suffices for general V4.1 workloads nor identify an optimal cache
size. The fixed training policy lacks information about new-request experts;
it is not an optimized full-model preload.

Replay logical misses at the larger budgets were approximately 304.5
MB/token, substantially more than native observed physical reads. This is
expected: native startup and prefill already populated many file-backed
weight pages, while the simulated cache started only from training decode.
Do not substitute replay misses into the native run, call the resulting
bandwidth floor a measured throughput, or mistake repeated low-capacity
results for a validated memory savings.

The evidence supports **request-aware adaptive residency over a short fixed
training seed**, but not a live cache speedup or an LRU/LFU winner. The next
experiment needs longer, varied calibration and held-out workloads, prefill
access accounting and a matched initial-residency policy. Only then should
a bounded disk-expert delivery prototype be compared against native page
caching. All existing and new observer/replay tests pass; the standard
collector's default backend pin remains unchanged.

## 11. Live exact source-page expert-cache prototype

The next experiment is an actual runtime path, not another cache simulation:
[expert_disk_cache.cpp](../../tools/trace/expert_disk_cache.cpp) intercepts
native CPU `ggml_graph_compute` in an isolated b11870 process. For supported
single-position `MUL_MAT_ID` graphs it reads the actual external expert IDs,
resolves their original file mappings and gathers the required source pages
into a bounded anonymous RAM cache before native expert computation.

Each original weight tensor has a sparse virtual address window with the
same data layout. Only pages owned by resident selected expert slices are
committed. Expert slices can share boundary pages; reference counts keep a
shared page alive until all owners are evicted. All current graph selections
are protected from eviction. Page-aligned adjacent misses are merged into
positional O_DIRECT reads; at most two read workers are used in the pilot.
Workers complete before visibility to compute. LRU eviction releases
unowned pages, and selected raw tensor data pointers are restored after
native computation or failure.

The path preserves tensor types, dimensions, strides, selected IDs, native
kernels and reduction order. It does not implement GPU-expert admission or
replace routing. Source changes, invalid IDs, unsupported/non-file-backed
layouts, short reads and an active set larger than the RAM budget are
explicit failures. Multi-position expert graphs remain native; this is not
a prefill optimization. Cache activation and nonzero intercepted graphs
must be confirmed by the benchmark.

This is a research-only, single-load process experiment. It requires raw
`--no-repack`, CPU-MoE expert placement and no operation offload, with the
matching pinned CPU library explicitly supplied. It is not an installation
plugin, cross-version ABI, model-reload integration or production capability.
The 64 GiB bound covers committed source-page payload; descriptor metadata,
activations, KV, native mapped pages and OS/interpreter memory are separate
and monitored. The native source mapping is retained, so the prototype can
duplicate already-cached file data. Do not call it a total-process 64 GiB cap
or memory saving without measuring those other allocations.

The byte-cache self-test covers hits avoiding direct reads, overlapping
pages, eviction, readmission and rejection of oversized active sets. The
[native-kernel fixture](../../tests/test_expert_disk_cache.cpp) compares
native and cached output bit arrays over four expert selections for F32,
Q4_0, IQ2_XXS, IQ3_S and IQ1_M, and checks restoration of source pointers.
All five format comparisons passed. This proves fixture behavior, not
full-vocabulary/model-quality equivalence.

### Build and qualify in isolation

```sh
source=/path/to/llama-b11870-source
libraries=/path/to/matching-b11870-libraries
output=/path/to/private-cache-build
g++ -std=c++17 -O2 -pthread -fPIC -shared \
  -I"$source/ggml/include" -I"$source/vendor" \
  tools/trace/expert_disk_cache.cpp \
  -L"$libraries" -Wl,-rpath,"$libraries" -lggml-base -ldl \
  -o "$output/libexpert-disk-cache.so"
g++ -std=c++17 -O2 -pthread -DVRAMBRIDGE_EXPERT_DISK_SELF_TEST \
  -I"$source/ggml/include" -I"$source/vendor" \
  tools/trace/expert_disk_cache.cpp \
  -L"$libraries" -Wl,-rpath,"$libraries" -lggml-base \
  -o "$output/cache-self-test"
"$output/cache-self-test"
```

Run only an owned server with this library preloaded. Explicit controls:
`VRAMBRIDGE_EXPERT_DISK_LIBRARY` names its matching CPU plugin,
`VRAMBRIDGE_EXPERT_DISK_OUT` a new private statistics file,
`VRAMBRIDGE_EXPERT_DISK_MIB=65536` sets the managed-page budget and
`VRAMBRIDGE_EXPERT_DISK_WORKERS=2` bounds parallel reads. Existing backend,
profile and saved state remain untouched.

### V4.1 pilot and matched-comparison protocol

The 32-token V4.1 pilot completed with the cache marker and 1,280 intercepted
expert graphs. It served 66.04 GB of logical expert demand, of which 40.97 GB
hit the new cache; requested direct reads and peak managed payload were both
25.09 GB. Preparing weights took approximately 95.8 seconds, versus 2.0
seconds inside native expert compute. The first response generated
**0.34 tok/s** after a 495-second model load.

That cold-cache cost is a negative finding, not a performance lead. The
native file/page cache can already contain source pages that this prototype
then copies again through direct I/O. A warm repeated prompt alone cannot
justify this duplication or its startup cost.

The bounded comparison used two fresh-process rounds in
baseline/cache then cache/baseline order, 64 output tokens per prompt,
one retained warm-up, four distinct measured tasks and a delayed repeated
prompt. Both modes use b11870, 4K context, 8/16 threads, raw no-repack
weights, all transformer layers on GPU with routed experts on CPU, no
operation offload, no speculation, and native multi-position prefill.
This placement differs from earlier auto-fit results; compare only the
matched controls, not those old headline speeds.

Every request records backend timing, first text, generated-text hash,
physical reads and faults, with sampled RSS/VRAM/swap and cache counters.
Greedy text must match across modes; arithmetic must return `323`.
Each owned process has a 30-minute deadline plus RAM/swap/controller guards.
No global cache eviction or filesystem changes occur. Both modes share
the same physical hardware limit, but their cache policies are not equal
configured RSS caps. Report load/cold cost, varied prompts and reuse
separately before stating a pass/fail for this prototype.

### First completed live comparison: direct-fill cache rejected

The [first live-cohort summary](../../benchmarks/results/deepseek-v41-live-cache-v1-summary.json)
contains four completed runs. All native/cache greedy text hashes matched
for all six tasks across both rounds; arithmetic passed in every run.
The source snapshot for this version has SHA-256
`34ff33cb32e47acd8c7ffe912ccd90093fe4c3babbd40c8e0a032adeda99724b`.
The original library and source snapshot were retained separately before
the refinement below.

Across eight distinct-task samples per method (four tasks, two rounds):

| Measurement | Matched native paging | Direct-fill 64 GiB cache |
|---|---:|---:|
| Median generation | **7.15 tok/s** | **1.87 tok/s** |
| Aggregate backend generation | 6.38 tok/s | 1.64 tok/s |
| Median first text | **1.40 s** | **11.24 s** |
| Total distinct-request physical reads | **12.45 GB** | **91.26 GB** |

This is a clear **negative performance result** for the direct-fill cache,
not a production optimization. In the first round, peak RSS was 116.71
GiB native versus 117.98 GiB cached: a 64 GiB managed budget did not imply
a reduction in total process residency.

The implemented cache is exact, but its misses are **managed-cache
misses**, not necessarily physical-memory misses. It rereads entire source
pages for uncached expert slices even when native file pages are already
resident. Native startup/prefill and the OS cache already serve much of
that data. Creating a second anonymous copy can increase I/O and memory
pressure rather than solve them. The first-round cache warm-up requested
39.3 GB of physical reads, versus 8.86 GB for the native request.

The distinct-task cache byte hits increased from 88.9% to 96.3% within the
first round, but preparing weights still dominated native expert compute.
The delayed repeated prompt reached 97.1% managed byte hits yet generated
only 2.84 tok/s, versus native 14.78 tok/s. More nominal cache hits are
therefore not, by themselves, evidence of an efficient delivery path.

The execution service stopped waiting after 30 minutes, but the owned
comparison continued under its per-run resource guards and finished all
four runs. No duplicate workload was launched. The final result was
validated from the completed report, not inferred from partial logs.

### Source-residency-aware refinement

The next version targets the measured over-reading rather than enlarging
the cache blindly. Before filling a missing source-page range, it checks
the original mapping's page residency with `mincore`. Pages reported
resident are copied from that mapping; only absent page ranges receive
direct reads. Destination protection changes are merged per range rather
than issued once per page. The source file's device/inode identity must
match its original mapping, and its size/mtime are checked before and after
delivery. Actual selected weights, tensor layout and kernels are unchanged.

The internal research control is
`VRAMBRIDGE_EXPERT_DISK_SOURCE_AWARE=1`; it is not a Synapse UI setting.
The new library is built under a separate filename, leaving the running
first-version binary intact. All five native-kernel format comparisons
remain bitwise equal, and six fixture checks pass, including source-aware
RAM copying, explicit-config failures, output-file preservation and
rejection of invalid worker counts.

`mincore` is an observation, not a page lock: a source page can be evicted
between observation and copy, causing a fault/read. The RAM-copy byte counter
does not guarantee zero physical I/O; the actual process I/O counters remain
the authority. The anonymous cache can still duplicate native file pages,
so this refinement also needs measured total residency and is not assumed
to win.

A second reversed-order, matched cohort was executed as
four separate bounded commands so each completes below the execution
service's 30-minute limit. Each run has a 25-minute guard, and the harness
refuses any overlapping llama-server. Individual reports do not claim
cross-mode equality: the four outputs must first be combined and validated
before a final cohort result is published.

### Final result of this prototype phase

Both four-run cohorts completed. All eight runs stopped their owned
backends, passed the arithmetic check and preserved matched greedy text
within their respective cohorts. The drive remained accessible. The
[consolidated final record](../../benchmarks/results/deepseek-v41-live-cache-final-summary.json)
retains the metrics, per-task reads/timings, source identities, residency
observations and qualification limits.
The [exact six-prompt corpus](../../benchmarks/disk-cache-corpus.jsonl)
retains the warm-up, distinct tasks and delayed identical prompt; the
duplicate is intentional. It is a performance corpus, not input for the
routing collector's distinct-prompt calibration validator.

For four distinct measured tasks over two reversed-order rounds:

| Cohort and method | Median generation | Aggregate generation | Median first text | Total request physical reads |
|---|---:|---:|---:|---:|
| First cohort: native paging | **7.15 tok/s** | 6.38 tok/s | **1.40 s** | **12.45 GB** |
| First cohort: direct-fill cache | 1.87 tok/s | 1.64 tok/s | 11.24 s | 91.26 GB |
| Second cohort: native paging | **6.94 tok/s** | 6.27 tok/s | **0.84 s** | **11.70 GB** |
| Second cohort: source-aware cache | 3.06 tok/s | 2.79 tok/s | 10.07 s | 56.56 GB |

The source-aware design reduced the excessive I/O relative to the first
cache version, but **both candidates lost to their matched native controls**.
Do not compare versions across cohorts as a precise speedup estimate:
file-cache history, time and backend conditions changed.

The second cohort's native per-task generation ranged from approximately
3.96 to 8.37 tok/s on distinct prompts. Its delayed identical hash-table
prompt generated 14.23 and 15.00 tok/s with virtually no physical reads.
That repeated-prompt result remains a warm positive control, not demonstrated
interactive throughput for changing requests.

There was no meaningful total-RSS saving. Median per-run peak RSS in the
second cohort was **117.11 GiB native versus 117.30 GiB source-aware**.
The managed cache stayed within its 64 GiB source-page budget, but also
retained the original mapping and native prefill path. It therefore did
not replace the full source/page-cache residency system.

Per source-aware run, approximately 46.6-47.2 GB of fills were copied from
pages reported resident and 26.0-26.6 GB were requested through direct I/O
over the complete workload. This demonstrates that the residency-aware
branch ran; it does not establish its profitability. Preparing weights
still took roughly 109-112 seconds, versus 23-36 seconds inside native CPU
expert graphs across all requests.

The observed high first-text latency and increased reads are consistent
with two remaining design costs: duplicating native cached data into
anonymous pages, and leaving multi-position prefill to access original
source mappings that compete with those copies for memory. This is a
code-and-counter-supported explanation, not a full causal ablation of
every fault, NUMA effect or CPU/cache interaction.

**Disposition: reject both live-cache versions for production and retain
native paging/Engram prefetch as the measured baseline.** The provisional
10 tok/s changing-prompt goal was not reached in this cohort. No GPU expert
cache, routing predictor, cross-layer compute/I/O overlap or native-buffer
replacement was implemented here, and the research question is not
declared impossible or solved by these results.

A future prototype must own the actual expert backing/residency lifecycle
rather than layer a second cache over eager native mappings, account for
prefill as well as decode, and preserve accurate demand-time fallback.
This is a different architectural experiment, not permission to ship either
rejected candidate or silently relax weight/routing fidelity.

The six cache/ownership/kernel fixture checks passed, including exact
F32/Q4_0/IQ2_XXS/IQ3_S/IQ1_M output bits, source-aware fills and explicit
failure paths. These checks plus matched full-model text/arithmetic tests
are still narrower than full-vocabulary numerical and broad quality gates.
The model payload checksums remain skipped, installed Synapse is unchanged,
and the prior V4 model is preserved.

The isolated native-kernel fixture and Python contract suite can be run
without downloading or loading a model:

```sh
g++ -std=c++17 -O2 -pthread \
  -I"$source/ggml/include" -I"$source/vendor" \
  tests/test_expert_disk_cache.cpp \
  -L"$libraries" -Wl,-rpath,"$libraries" \
  -l:libggml-cpu-zen4.so -lggml-base -o "$output/kernel-self-test"
python3 tests/test_expert_disk_cache.py \
  "$output/cache-self-test" "$output/kernel-self-test" \
  "$output/libexpert-disk-cache.so" "$libraries/libggml-cpu-zen4.so"
```

The selected CPU plugin is explicit for this host's tested build; do not
substitute incompatible plugins or assume the same ISA is available on
another system. This contract checks primitives and errors, not model
throughput.

## Next decision, not a production rollout

Keep checking storage reliability. Retain the existing native Engram and
expert paths as the measured controls; do not deploy the rejected caches.
Any subsequent live disk-expert design must replace duplicated backing
state and include prefill/residency accounting before another end-to-end
comparison.
Use held-out traces to decide whether achievable expert byte hit rates meet
the measured I/O budget. If they do not, test narrower exact techniques or
state the interactive-speed limit honestly rather than substituting routing
approximation. None of these stages requires treating existing llama.cpp
as the optimal design, or treating disk capacity as a performance solution.
