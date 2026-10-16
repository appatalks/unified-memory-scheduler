"use strict";

import { ChatStream } from "./chat-stream.mjs";

const el = (id) => document.getElementById(id);
let status = null;
let controller = null;
let actionPending = false;
let history = [];
let settingsInitialized = false;

function notice(message = "") {
  el("notice").textContent = message;
  el("notice").hidden = !message;
}

async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || `Request failed (${response.status}).`);
  return data;
}

function controls() {
  const active = status && ["loading", "ready", "stopping"].includes(status.state);
  el("load").disabled = !status?.backend_available || active || actionPending || !!controller;
  el("unload").disabled = !active || status?.state === "stopping" || actionPending;
  el("send").disabled = status?.state !== "ready" || !!controller || actionPending || status?.chat_busy;
  el("stop").disabled = !controller;
  el("clear").disabled = !!controller;
  el("max-output").disabled = !!controller;
  for (const id of ["model", "context", "placement", "cpu-threads", "batch-threads", "cpu-poll", "host-op-offload", "op-offload-min-batch", "load-mode", "expert-cache", "fit-target", "draft-model", "draft-tokens"]) {
    el(id).disabled = active || actionPending;
  }
  el("execution-mode").disabled = active || actionPending;
  const selectedMode = el("execution-mode").value;
  if (selectedMode === "auto") {
    for (const id of ["context", "placement", "cpu-threads", "batch-threads", "cpu-poll", "host-op-offload", "op-offload-min-batch", "load-mode", "expert-cache", "fit-target", "draft-model", "draft-tokens"]) {
      el(id).disabled = true;
    }
  }
  for (const id of ["draft-model", "draft-tokens"]) {
    el(id).disabled = active || actionPending || selectedMode === "auto" || !status?.speculation_enabled;
  }
  if (status?.backend_kind !== "llama.cpp") {
    el("cpu-poll").disabled = true;
    el("host-op-offload").disabled = true;
    el("op-offload-min-batch").disabled = true;
  }
  if (selectedMode === "hybrid") {
    for (const id of ["placement", "context", "load-mode", "expert-cache", "fit-target", "host-op-offload", "op-offload-min-batch"]) {
      el(id).disabled = true;
    }
  }
  el("placement").querySelector('option[value="cpu"]').disabled =
    status?.backend_kind === "ik_llama.cpp" || selectedMode !== "standard";
}

function renderStatus() {
  const labels = { stopped: "Stopped", loading: "Loading model", ready: "Ready", stopping: "Unloading", error: "Backend error" };
  const experimental = status.backend_kind === "ik_llama.cpp";
  const hybrid = status.execution_profile === "experimental-hybrid-direct";
  el("execution-mode").querySelector('option[value="hybrid"]').disabled = !status.hybrid_available;
  el("selection-reason").textContent = status.selection_reason || "";
  el("profile-note").textContent = status.hybrid_available
    ? "Automatic uses direct DMA for matching GPT-OSS-120B MXFP4 models and standard execution for other architectures. Automatic uses profile defaults; select Standard for manual settings. No cross-model hybrid speedup is implied."
    : "No hybrid installation profile is configured. Automatic uses the standard backend. Select Standard for manual settings.";
  el("execution-warning").hidden = !status.execution_warning;
  el("execution-warning").textContent = status.execution_warning || "";
  el("execution-footer").textContent = hybrid
    ? "Local experiment - Hybrid CPU/GPU expert dispatch - Adaptive direct DMA - Not quality-qualified"
    : `Local preview - ${experimental ? "Experimental ik_llama.cpp" : "Standard llama.cpp"} execution - Hybrid scheduling is not enabled`;
  el("placement-note").textContent = hybrid
    ? "Experimental GPT-OSS-120B profile: fixed 16K context and 10,240 MiB resident expert bank. Automatic fitting, repacking, and host-op offload are disabled. Unload releases pinned RAM."
    : "Automatic uses llama.cpp fitting with the selected VRAM margin. It is not a RAM-fit guarantee. Larger contexts need more memory.";
  el("cache-note").textContent = hybrid
    ? "Resident experts are spread across all 36 layers. Cold experts execute on CPU; causal admissions upload directly from registered RAM. Model weights and router probabilities are unchanged. New chat clears conversation, not learned residency; unload/reload resets it."
    : "Fewer threads can be faster. RAM loading can increase memory pressure. The upstream experimental expert cache uses VRAM and may be slower. Zero disables it.";
  el("speculation-note").textContent = status.speculation_enabled
    ? "Experimental speculation is enabled. A compatible target-trained draft and backend are required; speed is not guaranteed."
    : "Speculation is disabled by default. Research trials require --experimental-speculation at launch.";
  el("cpu-threads").max = status.logical_cpus;
  el("batch-threads").max = status.logical_cpus;
  el("placement").querySelector('option[value="cpu"]').disabled = experimental || hybrid;
  el("placement").querySelector('option[value="auto"]').textContent = hybrid
    ? "Experimental hybrid CPU / GPU (direct DMA)" : "Automatic CPU / GPU placement";
  el("expert-cache").max = experimental ? "0" : "65536";
  el("state").textContent = labels[status.state] || status.state;
  el("model-status").textContent = status.model || "No model loaded.";
  el("run-status").textContent = status.model
    ? `${status.backend_kind}${experimental || hybrid ? " (experimental)" : ""} · ${hybrid ? `Hybrid direct DMA${status.hybrid_verified ? " (activation verified)" : " (verifying activation)"}` : status.placement === "cpu" ? "CPU only" : "Automatic placement"} · ${status.context.toLocaleString()} context · ${status.cpu_threads} generation / ${status.batch_threads} prompt threads · ${status.load_mode} loading · ${status.expert_cache_mib} MiB ${hybrid ? "resident expert bank · fitting off" : `expert cache · ${status.fit_target_mib} MiB reserve`} · polling ${status.cpu_poll} · host-op offload ${status.host_op_offload ? "on" : "off"} (min batch ${status.op_offload_min_batch})${status.draft_model ? ` · DFlash (${status.draft_tokens} draft tokens)` : ""}`
    : "Load an existing GGUF to begin.";
  el("logs").textContent = status.logs.join("\n") || "No output yet.";
  const resources = status.resources;
  if (status.resource_error) {
    el("resources").textContent = status.resource_error;
  } else if (resources) {
    const gib = (bytes) => (bytes / 1024 ** 3).toFixed(1);
    const cpu = resources.cpu_logical_cores === null ? "sampling"
      : `${resources.cpu_logical_cores.toFixed(1)} logical cores (${(100 * resources.cpu_logical_cores / status.logical_cpus).toFixed(0)}% of machine)`;
    el("resources").textContent = `CPU: ${cpu}. Resident RAM: ${gib(resources.rss_bytes)} GiB `
      + `(${gib(resources.file_backed_bytes)} GiB file-backed, ${gib(resources.anonymous_bytes)} GiB private). `
      + `System available: ${gib(resources.system_available_ram_bytes)} GiB, including reclaimable cache.`;
  } else {
    el("resources").textContent = status.model ? "Collecting process resource usage..." : "";
  }
  if (status.model && ["loading", "ready", "stopping"].includes(status.state)) {
    el("model").value = status.model;
    el("context").value = status.context;
    el("placement").value = status.placement;
  }
  if (!settingsInitialized || (status.model && ["loading", "ready", "stopping"].includes(status.state))) {
    el("execution-mode").value = status.execution_mode || "standard";
    if (hybrid) {
      el("context").value = status.context;
      el("placement").value = status.placement;
    }
    el("cpu-threads").value = status.cpu_threads;
    el("batch-threads").value = status.batch_threads;
    el("cpu-poll").value = status.cpu_poll;
    el("host-op-offload").value = String(status.host_op_offload);
    el("op-offload-min-batch").value = status.op_offload_min_batch;
    el("load-mode").value = status.load_mode;
    el("expert-cache").value = status.expert_cache_mib;
    el("fit-target").value = status.fit_target_mib;
    el("draft-model").value = status.draft_model || "";
    el("draft-tokens").value = status.draft_tokens;
    settingsInitialized = true;
  }
  el("backend-help").hidden = status.backend_available;
  el("backend-help").textContent = "Backend missing. Install llama.cpp, then restart with: python3 tools/vrambridge/app.py --llama-server /path/to/llama-server";
  if (status.error) notice(status.error);
  contextNote();
  controls();
}

function contextNote() {
  el("context-note").textContent = status?.state === "ready" && Number(el("max-output").value) > status.context
    ? `Loaded context: ${status.context.toLocaleString()} tokens. This may stop generation before the output budget is reached. Unload and reload with a larger context in Run settings.`
    : "";
}

el("max-output").addEventListener("input", contextNote);
el("execution-mode").addEventListener("change", () => {
  if (el("execution-mode").value !== "auto") {
    const hybrid = el("execution-mode").value === "hybrid";
    el("context").value = hybrid ? "16384" : "4096";
    el("placement").value = "auto";
    el("cpu-threads").value = Math.min(8, status.logical_cpus);
    el("batch-threads").value = Math.min(16, status.logical_cpus);
    el("host-op-offload").value = String(!hybrid);
    el("expert-cache").value = hybrid ? "10240" : "0";
    el("fit-target").value = "1024";
    el("op-offload-min-batch").value = "32";
    el("load-mode").value = "auto";
    el("draft-model").value = "";
  }
  controls();
});

async function poll() {
  try {
    const next = await request("/api/v1/status");
    const recovered = !status;
    status = next;
    if (recovered) notice();
    renderStatus();
  } catch (error) {
    status = null;
    el("state").textContent = "Disconnected";
    notice(`Cannot reach the local service: ${error.message}`);
    controls();
  } finally {
    window.setTimeout(poll, 1500);
  }
}

el("endpoint").value = `${window.location.origin}/v1`;
el("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(el("endpoint").value);
    el("copy").textContent = "Copied";
    window.setTimeout(() => { el("copy").textContent = "Copy"; }, 1500);
  } catch (error) {
    notice(`Clipboard unavailable. Select and copy the Base URL manually. ${error.message}`);
  }
});

el("load-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  actionPending = true;
  notice();
  controls();
  try {
    const settings = {
      model: el("model").value,
      execution_mode: el("execution-mode").value,
      context: Number(el("context").value),
      placement: el("placement").value,
      cpu_threads: Number(el("cpu-threads").value),
      batch_threads: Number(el("batch-threads").value),
      cpu_poll: Number(el("cpu-poll").value),
      host_op_offload: el("host-op-offload").value === "true",
      op_offload_min_batch: Number(el("op-offload-min-batch").value),
      load_mode: el("load-mode").value,
      expert_cache_mib: Number(el("expert-cache").value),
      fit_target_mib: Number(el("fit-target").value),
      draft_model: status?.speculation_enabled ? el("draft-model").value.trim() || null : null,
      draft_tokens: Number(el("draft-tokens").value),
    };
    status = await request("/api/v1/model/load", settings.execution_mode === "auto"
      ? { model: settings.model, execution_mode: "auto" } : settings);
    clearChat();
    renderStatus();
  } catch (error) {
    notice(error.message);
  } finally {
    actionPending = false;
    controls();
  }
});

el("unload").addEventListener("click", async () => {
  controller?.abort();
  actionPending = true;
  controls();
  notice();
  try {
    status = await request("/api/v1/model/unload", {});
    renderStatus();
  } catch (error) {
    notice(error.message);
  } finally {
    actionPending = false;
    controls();
  }
});

function clearChat() {
  history = [];
  el("messages").replaceChildren();
  const empty = document.createElement("p");
  empty.id = "empty";
  empty.textContent = "Start a new conversation. Your chat is not saved.";
  el("messages").append(empty);
  el("timing").textContent = "";
}

el("clear").addEventListener("click", () => { clearChat(); notice(); });
el("stop").addEventListener("click", () => controller?.abort());

function message(role, text) {
  el("empty")?.remove();
  const item = document.createElement("div");
  item.className = "message";
  item.dataset.role = role;
  const heading = document.createElement("strong");
  heading.textContent = role === "user" ? "You" : "Assistant";
  const content = document.createElement("div");
  content.className = "text";
  content.textContent = text;
  item.append(heading, content);
  el("messages").append(item);
  item.scrollIntoView({ block: "nearest" });
  return content;
}

function assistantMessage() {
  const output = message("assistant", "Waiting for model...");
  const details = document.createElement("details");
  details.className = "reasoning";
  details.hidden = true;
  const summary = document.createElement("summary");
  summary.textContent = "Reasoning";
  const reasoning = document.createElement("div");
  reasoning.className = "text";
  details.append(summary, reasoning);
  output.before(details);
  return { output, details, reasoning, summary };
}

el("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = el("prompt").value.trim();
  if (!prompt || controller || status?.state !== "ready") return;
  const maxTokens = Number(el("max-output").value);
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 65536) {
    notice("Max output tokens must be an integer between 1 and 65,536.");
    return;
  }
  notice();
  controller = new AbortController();
  controls();
  el("prompt").value = "";
  message("user", prompt);
  const view = assistantMessage();
  const started = performance.now();
  const previous = [...history, { role: "user", content: prompt }];
  const stream = new ChatStream();
  let phase = "Waiting for model";
  const updateMetrics = () => {
    el("timing").textContent = stream.metrics((performance.now() - started) / 1000, phase);
  };
  const timer = window.setInterval(updateMetrics, 500);
  updateMetrics();
  let reader;
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "local-model", messages: previous, stream: true, max_tokens: maxTokens,
        stream_options: { include_usage: true }, timings_per_token: true,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error?.message || `Chat failed (${response.status}).`);
    }
    reader = response.body.getReader();
    while (!stream.done) {
      const chunk = await reader.read();
      stream.push(chunk.value, chunk.done);
      phase = stream.content ? "Answering" : stream.reasoning ? "Thinking" : "Waiting for model";
      view.details.hidden = !stream.reasoning;
      view.reasoning.textContent = stream.reasoning;
      view.summary.textContent = stream.content ? "Reasoning" : "Reasoning (in progress)";
      view.output.textContent = stream.content || (stream.reasoning
        ? "Thinking... expand Reasoning to see progress." : "Waiting for model...");
      updateMetrics();
      el("messages").scrollTop = el("messages").scrollHeight;
    }
    phase = stream.finishReason === "length" ? "Token limit reached" : "Complete";
    view.summary.textContent = "Reasoning";
    const warning = stream.warning(maxTokens);
    if (warning) notice(warning);
    if (stream.content.trim()) {
      history = [...previous, { role: "assistant", content: stream.content }];
    } else {
      view.output.textContent = warning;
      el("prompt").value = prompt;
    }
  } catch (error) {
    const interrupted = error.name === "AbortError";
    phase = interrupted ? "Stopped" : "Failed";
    view.summary.textContent = interrupted ? "Reasoning (stopped)" : "Reasoning (interrupted)";
    if (stream.content.trim()) {
      history = [...previous, { role: "assistant", content: stream.content }];
      view.output.textContent = stream.content;
    }
    if (!stream.content.trim()) view.output.textContent = interrupted ? "Generation stopped." : "No final answer.";
    notice(interrupted ? "Generation stopped. Any partial answer is retained." : error.message);
    if (!stream.content.trim()) el("prompt").value = prompt;
  } finally {
    window.clearInterval(timer);
    updateMetrics();
    if (reader) {
      if (!stream.done && !controller.signal.aborted) {
        try { await reader.cancel(); } catch (error) { notice(`Could not close the chat stream: ${error.message}`); }
      }
      reader.releaseLock();
    }
    controller = null;
    controls();
  }
});

poll();
