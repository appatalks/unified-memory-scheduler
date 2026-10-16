# Benchmarks

Schemas are versioned under `schema/`. Generated hardware and llama.cpp results go under `results/` and are not committed by default.

Tests never download models. Benchmark commands require paths to local GGUF files.

[RESULTS.md](RESULTS.md) is the reviewed benchmark ledger. It links each committed evidence artifact, distinguishes valid measurements from retained failed commands, and records the fixed comparison conditions. New generated JSON remains ignored until it has been reviewed and added to that ledger.

Raw files under `results/`, including nested output and logs, are ignored unless explicitly allowlisted in [`.gitignore`](../.gitignore). Before allowlisting evidence, review hostnames, model paths, command arguments, stdout, and stderr for credentials and private machine or user identifiers. Use neutral placeholders for identifying details without changing measurements. Only sanitized environment examples should be committed; local environment files, model weights, and user-specific CMake presets are ignored.

Ignore rules do not remove already tracked content or historical copies. Before publishing the repository, review both the current files and Git history, including commit author and committer identities.