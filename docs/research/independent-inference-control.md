# Independent inference-control prototype

Status: research only, November 14, 2026. The control loop and isolated expert executor are not a standalone inference engine or a production replacement. The running Synapse installation was not changed.

## Scope and provenance

`vrambridge-engine` owns prompt tokenization, bounded prefill chunks, sequence positions, greedy token selection, and explicit backend selection. It supports a single raw text prompt, optional token IDs or full-logit JSONL diagnostics, and CPU or explicitly requested GPU placement. Its model loader, tensor graphs, kernels, and tokenization implementation remain in the linked llama.cpp/GGML libraries. It has no server, chat templating, model-quality qualification, or Synapse integration.

A separate experimental expert dispatcher partitions the selected experts between CPU and GPU, runs both branches concurrently, and reduces in original routing order. It includes a bounded GGUF expert reader, native CPU quantized projections, and CUDA down-projection kernels for IQ4_NL/Q8_0, Q6_K/Q8_K, and MXFP4/Q8_0. The CUDA arithmetic emulates the tested x86 CPU plugin's reduction order; it is not assumed compatible with other CPU implementations. Each live intervention shadows GPU expert outputs against CPU bits and fails closed on a mismatch. The executor is used only by the diagnostic probe, not by `vrambridge-engine` or Synapse.

No Strata or Edge0 source was incorporated. The llama.cpp and GGML libraries are upstream MIT-licensed components; their source is [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`, with local research modifications. The CUDA reduction follows that source's x86 quantized-dot contract; the MXFP4 kernel also builds on this repository's earlier `tools/trace/compatible_down.cu` experiment. The CPU plugin used for validation was an existing research build, not a newly installed backend. Source commit alone does not identify this modified build. The tested shared-library SHA-256 values were:

| Library | SHA-256 |
| --- | --- |
| libllama.so | `30c0ddb27eeb81108bed123e6f10ca561d572777373c87df4bd4243e4244be4c` |
| libggml.so | `55e734dcfa156e7eae5b65cf8ad3dede85ed399ee5939c70` |
| libggml-base.so | `7162196429d539d0f5f0d4874b1a0b948934bf48b762f8e71f3d36aed29c2545` |
| libggml-cpu-zen4.so | `155a87103fdc442b309b656862bb2ea9b9f77be88f3434f9c044b71a572279e4` |

## Correctness evidence

The focused test builds deterministic, untrained Llama, Qwen3-MoE, and GPT-OSS fixtures. For each, the prototype and a separate reference control loop use the same library, CPU backend, prompt, context, physical batch setting, and prefill partition. Tests compare all 259 logits at up to six sampled positions for chunk sizes 1, 8, and 512, and require identical greedy token IDs and maximum absolute error at most `1e-5`. These fixtures establish control-loop behavior only; they do not establish trained-model quality or throughput.

A separate CPU smoke test of the existing LFM2-1.2B Q8_0 model compared all 65,536 logits at three positions with chunk size 512. The serialized float values and greedy tokens matched exactly. This is one small-model check, not validation of SC117 or the GPU path.

The baseline itself is sensitive to prefill partitioning. Comparing the reference loop with chunk sizes 1 and 512 across six positions gave the following maximum absolute and relative-L2 differences; greedy tokens matched in these samples. These measurements do not identify a defect or establish text identity across real models.

| Untrained fixture | Max absolute | Max relative L2 |
| --- | ---: | ---: |
| Llama | `6.0e-8` | `1.37e-7` |
| Qwen3-MoE | `7.07e-5` | `1.50e-4` |
| GPT-OSS | `1.23e-5` | `2.21e-5` |

Consequently, a future scheduling change must report its actual batching and compare complete logits under both matched and changed execution shapes; merely retaining the same weights does not establish bitwise equality. The earlier SC117 fresh-context instability and multi-sequence failures remain unresolved.

## Build and exercise

Configure with `VRAMBRIDGE_EXPERIMENTAL_ENGINE=ON`, `VRAMBRIDGE_LLAMA_SOURCE` set to the pinned source tree, and `VRAMBRIDGE_LLAMA_LIBRARY_DIR` set to its matching built libraries. Build targets `vrambridge-engine` and `vrambridge-engine-reference`; run CTest `vrambridge-experimental-engine`. The fixture test also needs the source tree's GGUF Python package and NumPy.

The executable's `--help` describes its deliberately limited interface. A CPU backend plugin and model are explicit inputs; a GPU plugin and layer count are required for GPU placement. `--trace-logits` writes the complete vocabulary per output position and is intended for bounded diagnostics, not normal chat. Never use private prompts in command-line arguments on a shared machine.

## Exact expert dispatch: decode-only controls

The [reviewed numeric summary](../../benchmarks/results/independent-exact-expert-summary.json) records the source and backend hashes, all per-process timing samples, and the final control results. Three synthetic prompt categories and three layers per model were run in separate processes at 512 context and four GPU layers. Each run checked the full vocabulary at prompt completion and four teacher-forced continuation positions. Only one CPU-resident expert layer at a time was replaced, and only during decode; original prefill remained in place.

| Model / GPU down format | Layers | Qualified runs | Decode interventions | Largest serialized-logit difference |
| --- | --- | ---: | ---: | ---: |
| SC117 Qwen3.8 / IQ4_NL | 0, 17, 35 | 9/9 | 36 | 0 |
| Qwen3-Coder 30B / Q6_K | 0, 17, 35 | 9/9 | 36 | 0 |
| GPT-OSS 120B / MXFP4 | 0, 17, 31 | 9/9 | 36 | 0 |

The native graph still computed the original experts before the diagnostic replaced their output. These are observer-qualified, one-layer interventions, **not an integrated engine or throughput test**. The first prompt and continuations are synthetic; longer contexts, multiple layers replaced together, sampling, concurrency, and answer quality remain unqualified. GPT-OSS layer 32 was explicitly rejected in three runs because its native output was not CPU-resident; it was not silently redirected.

Precision controls matter. An earlier FP32 GPU down-projection differed from native by only `2.98e-8` at one SC117 layer yet changed full-logit max absolute error by `1.54`; original-output and CPU-native placebo interventions stayed exact. The native-compatible GPU kernel was checked bit-for-bit against the selected CPU expert outputs before use. The naive Q6 intervention over prefill passed only 4/9 Coder cases under the default reference; explicitly disabling native CPU tiled matmul gave 9/9, but also changes the reference execution and is **not a recommended default**. Decode-only intervention preserved native prefill and passed the nine controls above. For GPT-OSS with four GPU layers, observing prefill itself failed the full-logit gate in 6/9 earlier cases; a fusion-disabled diagnostic passed its layer-0 control, while decode-only observation passed at the three CPU-resident layers above. These controls do not establish the underlying numerical cause.

The bit-compatible CUDA dot kernels passed biased and unbiased synthetic fixtures, including zeros and packed signed-byte edge cases. CUDA Compute Sanitizer reported zero memcheck errors and zero racecheck hazards for each of the three formats. This is bounded operator coverage, not a full-engine sanitizer qualification.

### Warm single-layer timing

At layer 17, with one synthetic reasoning request's actual decode activations, four timed repetitions per process, one excluded warmup, and three fresh processes, the medians of process medians were:

| Model | Two CPU workers | CPU + exact GPU down | Paired process changes |
| --- | ---: | ---: | --- |
| SC117 | 1.499 ms | 1.334 ms | +5.65%, +6.44%, +11.91% |
| Qwen3-Coder | 0.393 ms | 0.338 ms | +15.74%, +27.32%, +2.68% |
| GPT-OSS | 1.484 ms | 1.318 ms | +8.59%, +7.63%, +11.53% |

Positive percentages mean lower latency. The candidate runs CPU gate/up and native activation for every selected expert, then GPU down for a fixed subset. The CPU control uses two workers; device weights are already warm, and the timed region includes input packing, per-call allocations, transfer, synchronization and reduction. The IQ4/Q6/MXFP4 timed outputs were compared bitwise with the CPU control after timing. No model load, real dynamic cache admission, native optimized graph timing, full request, or tok/s measurement is included. The Coder variation includes a result within the roughly 5% noise band. These numbers justify further investigation, not production selection.

## Qualification path

Next, integrate the executor into an isolated backend graph without first executing the original branch, preserving native prefill and explicitly rejecting unsupported devices, quantizations and multi-token shapes. Test many layers together, longer prompts and sequences, full-vocabulary controls and held-out answers on all three MoE families. Compare end-to-end performance against the existing optimized hybrid engine, with warmup excluded and repeated medians. A candidate remains a research opt-in unless Synapse's existing detect, Auto Tune, and per-machine selection gates can verify it.
