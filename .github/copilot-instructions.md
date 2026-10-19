# Project Guidelines

## Research and implementation workflow

- Treat [appatalks/unified-memory-scheduler](https://github.com/appatalks/unified-memory-scheduler) as the research workspace: inspect its code, documentation, tests, and results to investigate ideas and develop findings.
- Treat the Synapse repository, [appatalks/Synapse](https://github.com/appatalks/Synapse), as a downstream consumer of this research, not as a workspace for us to edit or push to.
- Synapse may be updated independently, including by the user working there directly; do not assume its current code or decisions are reflected in this repository or in prior conversation.
- When a task depends on Synapse's current state, inspect the latest Synapse code or user-provided context read-only if available. If its state cannot be verified, say so and ask for the relevant context rather than making assumptions.
- When research produces useful findings, present them clearly so Synapse can apply them through its own agent or coding sessions. Do not make code changes or push commits to Synapse directly.
- Keep work in this repository focused on research and artifacts that support it; do not implement findings in `unified-memory-scheduler` as a substitute for Synapse's own implementation work.
- Keep research changes here limited to artifacts that directly support the investigation, and explain which evidence or results support any recommendations.

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
