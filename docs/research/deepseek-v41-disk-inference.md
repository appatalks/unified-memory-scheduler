# DeepSeek V4.1: explicit disk-tier feasibility

Date: 2026-11-01. Status: initial architecture and storage screens complete;
replacement downloaded, payload checksums skipped at user request;
native 4K text smoke/baseline complete; broad quality, numerical parity and
explicit disk-tier optimization gates pending. Further experiments are blocked
by the WD NVMe controller becoming unavailable; see section 7.

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

## Next decision, not a production rollout

Restore reliable access to the WD research volume first. Then extend the
native smoke test to multiple held-out workloads, explain the WD
path's measured large-read limitation, and prototype deterministic Engram gathering.
Use held-out traces to decide whether achievable expert byte hit rates meet
the measured I/O budget. If they do not, test narrower exact techniques or
state the interactive-speed limit honestly rather than substituting routing
approximation. None of these stages requires treating existing llama.cpp
as the optimal design, or treating disk capacity as a performance solution.
