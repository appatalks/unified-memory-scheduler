# VRAMBridge Local App

A Linux-first browser preview over ordinary llama.cpp inference. The service and
UI use only Python's standard library and browser APIs, with no frontend build or
network-loaded assets. Python 3.10+ is required.

## Start

Install or build `llama-server` using the [official llama.cpp instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
Select a CPU or GPU build appropriate to your machine; VRAMBridge does not install
drivers, download executables, or run shell commands on your behalf.

From the repository root:

```bash
python3 tools/vrambridge/app.py --llama-server /path/to/llama-server
```

Alternatively put `llama-server` on `PATH` or set `LLAMA_SERVER`. The browser opens
automatically; `--no-browser` suppresses it. The default port is 8080; `--port 0`
selects a free port and prints its address.

The CMake install includes the app and assets under
`<prefix>/share/vrambridge`; run `python3 <prefix>/share/vrambridge/app.py`.
It does not install Python or llama.cpp.

1. Enter the absolute path to an existing GGUF file. For sharded models, select the
   first shard and keep the remaining shards beside it.
2. Leave the 4,096-token context and automatic placement selected initially.
3. Click **Load model**. State changes from loading to ready only after the backend
   accepts an authenticated model-list request.
4. Chat, or configure an external client using the displayed base URL and
   `local-model`. New chat clears the page's conversation.
5. **Stop** cancels a chat; **Unload** also cancels model loading and releases the
   backend. Ctrl+C or SIGTERM shuts down the service and its backend.

### Reasoning, Output Limits, And Speed

The chat output budget defaults to **8,192 tokens** and is adjustable per request
with **Max output tokens** (1-65,536). This counts both reasoning and the final
answer. Reasoning models can exhaust small budgets before producing any answer;
the UI shows their separate reasoning stream in an expandable section and an
explicit thinking status rather than leaving the response blank.

If generation stops at a token limit, the UI warns that the answer may be
incomplete. A reasoning-only or empty completion is also reported explicitly.
Increase the output budget or use a more focused request as appropriate.

The output budget is separate from the loaded context. With a 4,096-token
context, a long prompt/conversation plus a long answer may require context
shifting or hit backend limits. The UI flags an output budget larger than the
loaded context. To retain more history, unload and reload with a larger context
in **Run settings**, subject to the model's supported window and available memory.

During generation the UI displays the backend's **tokens/second**, output-token
count (including reasoning), and total elapsed time. It requests
`timings_per_token` and streaming usage, reads `timings.predicted_per_second`
and `predicted_n`, and uses the final `usage.completion_tokens` when available.
SSE chunks are not counted as tokens. Generation speed excludes prompt
processing; total elapsed time includes it. Backends without timing data show
speed as pending/unavailable rather than estimating a misleading rate.

The UI renders responses as plain text, including code and HTML, without executing
model output. Chat and settings are not saved. Backend output is kept in a bounded
in-memory log; it may contain local paths. Do not publish logs without review.

## Placement And Resource Limits

Automatic placement passes `--n-gpu-layers auto --fit on --fit-target 1024` to
llama.cpp. These flags were checked against the project's documented upstream
commit `b21e4de74567f5eef213765c9476a843c2e43f0d`; older builds may reject them.
Such failures appear in backend output rather than silently changing policy.
The 1 GiB margin applies to GPU fitting, not total system RAM.
If the backend reports a CUDA initialization failure during automatic placement,
the app stops that load with an explicit error instead of presenting CPU-only
fallback as working GPU inference. Explicit CPU-only mode does not require CUDA.
`nvidia-smi` alone is not a compute-readiness check: the driver can report GPU
statistics even while CUDA initialization fails.

CPU-only mode passes `--n-gpu-layers 0 --device none`. The service uses one backend
slot. By default generation and prompt threads count physical cores within the
process's CPU affinity, not SMT/hyperthreads. Fewer threads can be substantially
faster for memory-bound MoE decode; defaults are not an optimal-model claim. This is not a
guarantee of desktop responsiveness or protection against system OOM: start with
a model that fits available RAM. No fit estimator, GPU utilization monitor, disk
tiering, or model catalog is implemented in
this preview. Model and architecture support are inherited from
the chosen llama.cpp build; extra multimodal projectors are not supported here.

### Advanced Performance Controls

Under **Run settings > Advanced performance**, generation and prompt thread
counts can be set separately. The API accepts `cpu_threads` and `batch_threads`,
each bounded by the number of logical CPUs in the process affinity.

On the official backend, `host_op_offload` (default `true`) controls whether
host-tensor operations may be offloaded to the GPU. `false` passes
`--no-op-offload`; it does not move attention or all model tensors to the CPU.
It can change numerical execution paths and prompt-processing behavior.
`cpu_poll` (default 50, range 0-100) passes upstream `--poll`; zero reduces
busy waiting but may increase synchronization latency. Both controls are visible
under advanced settings and in status. The experimental fork adapter rejects
nondefault values rather than assuming matching flag semantics.

`op_offload_min_batch` exposes the official CUDA backend's
`GGML_OP_OFFLOAD_MIN_BATCH` (default 32, range 1-8192). This is an upstream
operation-batch heuristic, not a promise that it exactly equals user prompt length.
The application sets it explicitly instead of inheriting an unseen shell override.
Increasing it can keep smaller host-tensor batches on CPU while allowing larger
batches to run on GPU. The tested value 128 improved short-prompt latency but
failed a deterministic quantitative-answer check, so it is not recommended or
enabled automatically. Unchanged default behavior remains the control.

`load_mode` is `auto` (upstream mmap default) or `none` (copy weights into RAM).
Copying weights can increase non-reclaimable memory pressure; it is not equivalent
to a faster CPU kernel, and it did not improve the measured 120B workload.

`expert_cache_mib` defaults to zero. On a supporting official backend, a positive
value opts into upstream `--cpu-moe --moe-cache-mib N` with the nonexpert layers
on the GPU. It consumes VRAM and changes the execution path; it can be slower and
can change numerical rounding. It is not a VRAMBridge-invented scheduler and is
never enabled silently. Unsupported flags and insufficient VRAM are explicit
backend errors. See the [120B tuning measurements](../../docs/research/frontier-moe-feasibility.md#120b-browser-app-tuning).

The status panel reports the backend process's sampled CPU usage in equivalent
logical cores and as a percentage of CPUs available to the process. RAM includes
resident file-backed model pages, with anonymous and file-backed residency shown
separately. Linux can count file-backed pages as reclaimable cache in
`MemAvailable`; that does not mean those weights are unused. Telemetry failure
is displayed explicitly rather than reported as zero usage.

### Experimental ik_llama.cpp Adapter

The official llama.cpp backend remains the default. An explicitly selected
side-by-side experimental backend can be launched with:

```bash
python3 tools/vrambridge/app.py \
  --backend-kind ik_llama.cpp \
  --llama-server /path/to/ik_llama.cpp/build/bin/llama-server
```

The adapter uses the fork's `--fit --fit-margin 1024 --n-gpu-layers 999`
syntax, and maps RAM loading to `--no-mmap`. It currently supports automatic
placement without an explicit expert-cache allocation; unverified modes are
rejected. Backend kind is visible in status and is not inferred from a filename.
The application never downloads or replaces a backend automatically.

The fork has narrower architecture/backend compatibility than upstream and is
not a universal performance upgrade. Building for this machine used native CPU
instructions, the AVX-512/VNNI/BF16 build options documented by the fork, and CUDA
SM 120. That binary is machine-specific, not a portable release artifact.

Loading has a ten-minute deadline and is cancellable. Readiness is not a warm-up
benchmark. One chat request is admitted at a time; concurrent attempts return 409.
Backend requests have a 120-second socket inactivity timeout. A disconnected
client closes the upstream connection to cancel generation; backend resource
release still depends on llama.cpp handling cancellation.

## Local API

Application endpoints use `/api/v1`; chat compatibility endpoints use `/v1`:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/v1/status` | Backend availability, state, model, settings, busy flag, and bounded logs |
| POST | `/api/v1/model/load` | Load `{"model":"/models/model.gguf","context":4096,"placement":"auto"}` |
| POST | `/api/v1/model/unload` | Cancel loading or unload; send `{}` |
| GET | `/v1/models` | Forward the ready backend's model list |
| POST | `/v1/chat/completions` | Forward chat, including SSE streaming, using the loaded model |

Load requests can additionally set `cpu_threads`, `batch_threads`, `load_mode`,
`expert_cache_mib`, `fit_target_mib`, `host_op_offload`, `cpu_poll`, `op_offload_min_batch`, `draft_model`,
and `draft_tokens`. Status reports the actual requested execution settings,
explicit backend kind, and sampled process resources.

### Opt-In Speculative Research

DFlash is disabled by default. Launch with `--experimental-speculation` to
allow a local, target-trained draft GGUF. The UI and API expose the draft path,
number of draft tokens, and explicit VRAM fitting reserve. CPU-only placement
and simultaneous explicit expert caching are rejected for this research path.
The target model must verify proposals; draft tokens are not output tokens.

The tested 0.8B GPT-OSS-120B DFlash draft came from
`z-lab/gpt-oss-120b-DFlash` revision
`1278df34f0a7bd2c8588a27f49048aaa05c7db00`; its safetensors SHA-256 is
`4546a44a2b4f3da784003964085f979ef1e59fb22955dafbac11535b7f413827`.
Conversion used the matching backend's converter and pinned target tokenizer
metadata. Fork and official GGUF draft formats are not interchangeable.

Official build 11669's DFlash loader does not consume this draft's attention
bias tensors. The optional
[compatibility patch](../../third_party/patches/llama-b11669-dflash-attention-bias.patch)
against `620d2225cc436ab0b001e3200e590e3b286ef9ab` loads those optional tensors
and applies biases before attention normalization and after output projection,
including K/V injection. Its upstream MIT notice is preserved in
[LLAMA-LICENSE](../../third_party/patches/LLAMA-LICENSE). It is a loader/graph
compatibility fix, not a claimed throughput optimization or a generally validated
backend release.

```bash
git -C /path/to/llama.cpp apply --check /path/to/llama-b11669-dflash-attention-bias.patch
git -C /path/to/llama.cpp apply /path/to/llama-b11669-dflash-attention-bias.patch
```

Build that checkout using its supported backend configuration. The local test
build used `GGML_BACKEND_DL=ON`, `GGML_NATIVE=OFF`, and
`GGML_CPU_ALL_VARIANTS=ON`; its matching official CUDA plugin was loaded using
`GGML_BACKEND_PATH=/path/to/libggml-cuda.so` (a library file, not a directory).
Do not mix plugin versions. Neither the default executable nor installed GPU
drivers were replaced.

The shared-target draft could not be measured by upstream automatic memory
fitting, which fell back to target-only fitting. A 1 GiB reserve then caused
OOM; the tests explicitly reserved 4 GiB. This is not a universal fit guarantee.
At a fixed 16K context, tested 3/5/9-token blocks were slower than the
non-speculative control. Therefore speculation remains experimental and
disabled, with no claim that the 60 tok/s target or output-equivalence criterion
has been reached. See the [measurement record](../../docs/research/frontier-moe-feasibility.md#120b-browser-app-tuning).

The [whole-model hybrid experiment](../../benchmarks/README.md#experimental-whole-model-expert-dispatch)
screens near 39-40 tok/s with direct expert DMA. It can now be deliberately
enabled in this app for manual testing, using the launch profile below.
Its strict model diagnostics and an answer-quality screen remain failed;
it is not quality-qualified, and the 60 tok/s target is still unmet.
Automatic model-specific selection is now the default. With a local hybrid
installation profile, matching GPT-OSS-120B MXFP4 models use direct DMA.
Other architectures use the standard backend, explicitly labeled as such.
Without an installation profile, the app remains standard-only; no backend is
silently downloaded or substituted.

### Installing The Default Hybrid Profile

Build the [pinned backend patch](../../benchmarks/README.md#experimental-whole-model-expert-dispatch)
first. Copy [hybrid-profile.example.json](hybrid-profile.example.json) to
`~/.local/share/vrambridge/hybrid-profile.json` and replace all placeholder paths
with absolute paths to the patched executable, reviewed training-only plan,
and matching CUDA plugin. The installation file is machine-local, not a file
to commit. `--hybrid-profile /path/to/profile.json` can select another installation.
The profile requires the pinned commit identity, readable files, and a valid
resident plan. An invalid installed profile stops startup with an error.

Then the normal command is sufficient:

```bash
python3 tools/vrambridge/app.py
```

**Automatic model-specific defaults** chooses from bounded GGUF metadata, never
from a model filename. The supported hybrid layout is `gpt-oss`, 36 blocks,
width 2,880, FFN width 2,880, 128 experts with four selected, and MXFP4 file
type 38. Select the first shard. This does not establish support for other
MoE families or generalize the measured gains to them. A malformed GGUF,
unsupported metadata type, or missing first-shard architecture is an error,
not a silently successful standard fallback.

In Automatic mode, the browser sends only the selected model and policy so
settings from a previous model cannot leak into the next model's configuration.
Choose **Standard backend (manual settings)** for standard tuning or CPU-only
execution. **Require hybrid direct DMA** rejects unsupported layouts instead
of selecting standard execution. A selected hybrid that fails readiness or
resource checks remains an error; the app does not mask it by running standard.

API clients can set `"execution_mode": "auto"`, `"standard"`, or `"hybrid"` on
model-load requests. Omitted policy uses the launch default. The status API
reports requested `execution_mode`, actual `execution_profile`,
`selection_reason`, `hybrid_verified`, and the plan hash. Automatic callers may
provide explicit performance settings for reproducible tests; conflicting
hybrid settings are rejected rather than silently overridden.

### Manual Hybrid Direct-DMA Trial

Only use this with the pinned, patched backend and its matching CUDA plugin.
The same browser chat, streaming metrics, Stop, Unload, and API remain available:

```bash
python3 tools/vrambridge/app.py \
  --llama-server /path/to/patched/llama-server \
  --experimental-hybrid-plan /path/to/training-only-hot800.txt \
  --hybrid-cuda-plugin /path/to/matching/libggml-cuda.so
```

The launch profile persists for this app process, including unload/reload.
Select the original GPT-OSS-120B MXFP4 first shard; other layouts are rejected
before launch in required-hybrid mode. Launch paths and plan entries are validated; no weights,
executables, or plugins are downloaded. The normal executable/symlink is not
replaced. Restart using the same arguments to resume experimental mode.

This profile fixes context at **16,384**, residency at **10,240 MiB**, raw
default host loading, no repacking, no host-op offload, and **fitting off**.
The reserve field is disabled and does not promise spare VRAM. Generation and
prompt threads default to **8/16** (bounded by CPU affinity) and can be changed
after unloading. Polling remains adjustable. Drafting/speculation and CPU-only
placement are rejected in this profile. Conflicting API settings are errors,
not silently overridden.

The backend executes cold experts on CPU and admits reusable experts into the
GPU bank. It temporarily registers about **57 GiB of read-only model pages**
for direct DMA, capped at 64 GiB with 24 GiB available-RAM headroom checks.
It does not make the model smaller. Unload terminates the backend and releases
registration/VRAM. New chat clears conversation but does not reset the adaptive
expert bank; unload/reload resets both. Output can differ from ordinary
placement; independently verify important answers.

The page displays an experimental warning, actual profile, and activation
verification. Ready requires an authenticated backend response **and** log
confirmation of adaptive hybrid dispatch, direct registered-source DMA,
registration limits, and the actual 16K slot. An unpatched backend cannot be
silently labeled as a working hybrid. Research environment overrides are
removed from child processes; only this explicit profile injects its known
plan/plugin and direct-DMA switches.

For a useful manual comparison, start a new chat with a detailed technical,
coding, or reasoning prompt and let it generate at least 512 tokens. Read the
final backend generation rate, not the first few streaming updates. The
displayed rate includes reasoning tokens but excludes prompt processing; short
answers and longer conversations need not match the 39-40 tok/s screen.
Report the prompt, output-token count, rate, elapsed time, and any wrong answer
or backend error. No claim of sustained 60 tok/s follows from this trial.

To roll back, unload, stop the app (Ctrl+C or SIGTERM), then relaunch without
the hybrid flags using the original official executable:

```bash
python3 tools/vrambridge/app.py --execution-mode standard --llama-server /path/to/original/llama-server
```

Load the model again with the desired standard settings. The API/base URL is
unchanged when using the same port.

For example, while a model is ready:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"stream":true}'
```

Clients requiring an API key may use `local` as a placeholder; the application API
does not authenticate local clients. A private, per-launch backend key is used
internally and redacted from captured backend output. Placement-related
`LLAMA_ARG_*` settings and the CUDA managed-memory override are not inherited.
The proxy always uses the loaded model, regardless of the client's model string.
This is a chat subset, not a complete OpenAI API implementation: embeddings,
Responses API, file uploads, and cross-origin browser integrations are not exposed.

## Local-Only Boundary

Both servers bind to `127.0.0.1`. Host and browser-origin checks reject nonlocal
hosts, cross-site requests, and foreign origins. Static assets are allowlisted;
model paths are passed as subprocess arguments, never through a shell.

Trusted local processes can use the API, including loading readable GGUF files and
unloading the model. This is a single-user convenience tool, not a security
boundary against other users or processes on the same machine. Do not expose it
through a reverse proxy, port forward, tunnel, or public bind address. The browser
must run on the same machine as the service.

## Validation

```bash
python3 tests/test_local_app.py tools/vrambridge/app.py
node --test tests/test_chat_stream.mjs  # developer tests; Node.js 18+
node --input-type=module --check < tools/vrambridge/web/app.js
```

The service regression tests also run in CTest/CI. They use a deterministic fake
backend and do not download models or require a GPU. They verify process lifecycle,
timeouts, failed loads, API errors, streaming, cancellation, input validation,
environment isolation, and local-origin restrictions; they do not establish real
model compatibility, output correctness, or inference performance.
Stream unit tests also run in CTest when Node.js is available. They cover
reasoning, fragmented UTF-8/SSE, backend timing and usage counters, token-limit
warnings, empty answers, and interrupted streams. Node.js is not required to run
the application.