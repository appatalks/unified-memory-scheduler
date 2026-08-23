# Benchmarks

Schemas are versioned under `schema/`. Generated hardware and llama.cpp results go under `results/` and are not committed by default.

Tests never download models. Benchmark commands require paths to local GGUF files.

[RESULTS.md](RESULTS.md) is the reviewed benchmark ledger. It links each committed evidence artifact, distinguishes valid measurements from retained failed commands, and records the fixed comparison conditions. New generated JSON remains ignored until it has been reviewed and added to that ledger.