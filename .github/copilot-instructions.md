# Project Guidelines

## Sister projects: research vs. production

- **This repository (VRAMBridge) is the research area**: backend patches, experiments, benchmarks, raw results and
  negative findings. The canonical llama.cpp patch is
  [third_party/patches/llama-b11669-hybrid-experts.patch](../third_party/patches/llama-b11669-hybrid-experts.patch)
  against commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`.
- **[appatalks/Synapse](https://github.com/appatalks/Synapse) is the production application.** It receives only the
  learned, validated subset of this research: defaults, guards, opt-in profile flags, docs and tests.
- Flow: investigate and measure here → record evidence and the patch here → apply the validated result in Synapse.
  When the user asks for end-to-end work, both repositories may be edited in the same task; keep experimental
  engine code here and production behavior there.
- Synapse also changes independently. Read its current code before relying on it; don't assume prior sessions.
- Synapse's installer pins a research commit and the patch SHA-256 (`scripts/setup_acceleration.py`). Commit and
  push here first, verify the raw URL's checksum, then bump the pins in Synapse. Ask before committing or pushing.
- Explain which evidence supports every recommendation; report negative and mixed results honestly.
- Synapse users never pick engine internals. A validated technique ships in Synapse as an engine variant that
  Synapse detects, Auto Tune measures, and Automatic reuses per model and machine; mixed results are fine as long
  as the technique can be measured and is never the untested default.

## Research principles

- **Model-agnostic.** Validate speed changes on several MoE families (at minimum Qwen3-Coder-30B-A3B, a
  many-expert Qwen3.8 model, gpt-oss-120b). A gain on one model that regresses another stays opt-in.
- **Accurate only.** No lossy shortcuts; verify answers (and text identity where the math is unchanged).
- Benchmark with warm-up excluded and medians reported; noise is about ±5%.
- Develop in a copied backend tree; never replace the user's installed backend or profile without consent.
- Stop only processes you started; never `pkill` every `llama-server`.

## Local model paths and storage

- Use these model roots for discovery, testing, downloads, persistent storage, and inference inputs:
  - `/home/mj420/Desktop/TOR/llm`
  - `/media/mj420/ExpansionXFS1/ai-models`
- Check both roots for existing weights before downloading another copy. Verify actual files, symlink targets, mount availability, and free space before use; keep model weights out of this repository.
- Use `/home/mj420/Desktop/TOR/llm/model-staging` on the existing root-drive SSD for explicit temporary working copies. The user selected a **16 GiB minimum free-space reserve**. Reuse weights already on that filesystem instead of duplicating them; see the [verified staging workflow](../benchmarks/README.md#explicit-ssd-model-staging).
- Preserve archive originals and evict only manifest-owned staged copies after unloading them. Do not automatically evict models or use the Samsung NTFS volume as the working tier; the user chose the existing root drive instead.
- `/mnt/ramdisk` is an optional temporary staging location for experiments that copy model files into RAM and read them from there instead of disk. Verify that it is mounted and RAM-backed, with sufficient capacity, before copying.
- Charge RAM-disk allocations against host RAM alongside model working memory, KV caches, expert caches, and pinned buffers. Preserve adequate headroom; RAM-disk storage is not additional RAM or VRAM.
- Retain persistent originals and treat RAM-disk contents as disposable across reboots. Do not move or delete originals, or create, resize, or remount the RAM disk, without explicit approval.
- Measure whether RAM-disk staging helps the actual workload. Account for copy/setup time and the OS page cache; warm inference may already read weights from RAM without a RAM disk. Do not assume a generation-speed benefit.
