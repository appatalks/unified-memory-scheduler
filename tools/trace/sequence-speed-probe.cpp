#define main sequence_decode_contract_main
#include "sequence-decode-probe.cpp"
#undef main

#include <chrono>
#include <cstdlib>

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "Usage: %s MODEL CONCURRENCY TOKENS [gpu-cpu-experts]\n", argv[0]);
        return 2;
    }
    try {
#if !defined(_WIN32)
        umask(0077);
#endif
        const int concurrency = std::atoi(argv[2]);
        const int tokens = std::atoi(argv[3]);
        const bool gpu = argc == 5 && std::strcmp(argv[4], "gpu-cpu-experts") == 0;
        require(argc == 4 || gpu, "invalid placement");
        require((concurrency == 1 || concurrency == 2 || concurrency == 4 || concurrency == 8) &&
                tokens >= 2 && tokens <= 512, "invalid concurrency or output length");
        const bool hybrid = bounded_env("VRAMBRIDGE_SPEED_HYBRID", 0, 0, 1);
        const bool serial_tokens = bounded_env("VRAMBRIDGE_SPEED_SERIAL_TOKENS", 0, 0, 1);
        const int slots = bounded_env("VRAMBRIDGE_PROBE_SLOTS", 8, 1, 8);
        const int sequence_context = bounded_env("VRAMBRIDGE_PROBE_CONTEXT", 16384, 512, 16384);
        require(concurrency <= slots, "concurrency exceeds reserved slots");
        if (hybrid) {
            const char * cpu_plugin = std::getenv("VRAMBRIDGE_PROBE_CPU_PLUGIN");
            const char * cuda_plugin = std::getenv("VRAMBRIDGE_PROBE_CUDA_PLUGIN");
            require(gpu && serial_tokens && cpu_plugin && cuda_plugin &&
                    ggml_backend_load(cpu_plugin) && ggml_backend_load(cuda_plugin),
                    "Hybrid speed screen requires serialized tokens and explicit GPU/CPU plugins");
        }
        llama_backend_init();
        ggml_backend_dev_t devices[] = {nullptr};
        ggml_backend_dev_t gpu_devices[] = {nullptr, nullptr};
        llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr}
        };
        auto mp = llama_model_default_params();
        if (hybrid) {
            mp.no_host = true;
            mp.use_extra_bufts = false;
        }
        if (gpu) {
            gpu_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            require(gpu_devices[0] != nullptr && llama_supports_gpu_offload() &&
                    std::strncmp(ggml_backend_dev_name(gpu_devices[0]), "CUDA", 4) == 0, "CUDA unavailable");
            mp.devices = gpu_devices;
            mp.n_gpu_layers = -1;
            mp.tensor_buft_overrides = overrides;
        } else {
            mp.devices = devices;
            mp.n_gpu_layers = 0;
        }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = sequence_context * slots;
        cp.n_seq_max = slots;
        cp.n_batch = 2048;
        cp.n_ubatch = 512;
        cp.n_threads = 8;
        cp.n_threads_batch = 16;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = gpu;
        cp.op_offload = gpu && !hybrid;
        cp.kv_unified = false;
        if (hybrid) {
            cp.moe_cache_size = static_cast<size_t>(bounded_env("VRAMBRIDGE_PROBE_MOE_CACHE_MIB", 4096, 1, 8192)) * 1024 * 1024;
        }
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), cp), llama_free);
        require(context != nullptr && llama_n_ctx_seq(context.get()) == static_cast<uint32_t>(sequence_context),
                "context mismatch");
        const auto * vocab = llama_model_get_vocab(model.get());
        const int vocabulary = llama_vocab_n_tokens(vocab);
        const std::vector<std::string> prompts = {
            "Agent: Review a bounded work queue.", "Agent: Find a cancellation race.",
            "Agent: Explain how a slot keeps state.", "Agent: Propose tests for worker errors.",
            "Agent: Describe GPU memory pressure.", "Agent: Compare two queue designs.",
            "Agent: Explain why prompts should not leak.", "Agent: Summarize a safe shutdown strategy."
        };
        std::vector<std::vector<llama_token>> inputs;
        for (const auto & prompt : prompts) {
            const int count = -llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()), nullptr, 0, true, true);
            require(count >= 3 && count <= 128, "unexpected prompt length");
            inputs.emplace_back(count);
            require(llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()),
                                  inputs.back().data(), count, true, true) == count, "tokenization failed");
        }
        if (hybrid) {
            std::vector<llama_token> warm_next(concurrency);
            for (int seq = 0; seq < concurrency; ++seq) {
                const auto logits = prefill(context.get(), inputs[seq], seq, vocabulary);
                warm_next[seq] = static_cast<llama_token>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            }
            Batch warm_batch(1);
            for (int step = 0; step < 32; ++step) {
                for (int seq = 0; seq < concurrency; ++seq) {
                    warm_batch.value.n_tokens = 0;
                    add(warm_batch.value, warm_next[seq], static_cast<int>(inputs[seq].size()) + step, seq, true);
                    require(llama_decode(context.get(), warm_batch.value) == 0, "warmup decode failed");
                    const auto * logits = llama_get_logits_ith(context.get(), 0);
                    require(logits != nullptr, "missing warmup logits");
                    warm_next[seq] = static_cast<llama_token>(std::max_element(logits, logits + vocabulary) - logits);
                }
            }
        } else {
            prefill(context.get(), inputs[0], 0, vocabulary);
        }
        llama_memory_clear(llama_get_memory(context.get()), true);
        double prompt_seconds = 0, decode_seconds = 0;
#if defined(VRAMBRIDGE_SPEED_GRAPH_STATS)
        int64_t decode_reuses = 0, decode_reallocs_same = 0, decode_reallocs_diff = 0;
#endif
        std::vector<std::vector<llama_token>> outputs(8);
        std::vector<double> first_token_seconds(8), completion_seconds(8);
        const auto cohort_start = std::chrono::steady_clock::now();
        for (int wave = 0; wave < 8; wave += concurrency) {
            llama_memory_clear(llama_get_memory(context.get()), true);
            std::vector<llama_token> next(concurrency);
            auto start = std::chrono::steady_clock::now();
            for (int seq = 0; seq < concurrency; ++seq) {
                const auto logits = prefill(context.get(), inputs[wave + seq], seq, vocabulary);
                next[seq] = static_cast<llama_token>(std::max_element(logits.begin(), logits.end()) - logits.begin());
                first_token_seconds[wave + seq] = seconds_since(cohort_start);
            }
            prompt_seconds += seconds_since(start);
            Batch batch(concurrency);
#if defined(VRAMBRIDGE_SPEED_GRAPH_STATS)
            const auto graph_before = llama_perf_context(context.get());
#endif
            start = std::chrono::steady_clock::now();
            for (int step = 0; step < tokens - 1; ++step) {
                if (serial_tokens) {
                    for (int seq = 0; seq < concurrency; ++seq) {
                        batch.value.n_tokens = 0;
                        outputs[wave + seq].push_back(next[seq]);
                        add(batch.value, next[seq], static_cast<int>(inputs[wave + seq].size()) + step, seq, true);
                        require(llama_decode(context.get(), batch.value) == 0, "serial decode failed");
                        const auto * logits = llama_get_logits_ith(context.get(), 0);
                        require(logits != nullptr, "missing serial logits");
                        next[seq] = static_cast<llama_token>(std::max_element(logits, logits + vocabulary) - logits);
                        if (step == tokens - 2) completion_seconds[wave + seq] = seconds_since(cohort_start);
                    }
                } else {
                    batch.value.n_tokens = 0;
                    for (int seq = 0; seq < concurrency; ++seq) {
                        outputs[wave + seq].push_back(next[seq]);
                        add(batch.value, next[seq], static_cast<int>(inputs[wave + seq].size()) + step, seq, true);
                    }
                    require(llama_decode(context.get(), batch.value) == 0, "decode failed");
                    for (int seq = 0; seq < concurrency; ++seq) {
                        const auto * logits = llama_get_logits_ith(context.get(), seq);
                        require(logits != nullptr, "missing logits");
                        next[seq] = static_cast<llama_token>(std::max_element(logits, logits + vocabulary) - logits);
                        if (step == tokens - 2) completion_seconds[wave + seq] = seconds_since(cohort_start);
                    }
                }
            }
            decode_seconds += seconds_since(start);
#if defined(VRAMBRIDGE_SPEED_GRAPH_STATS)
            const auto graph_after = llama_perf_context(context.get());
            decode_reuses += graph_after.n_reused - graph_before.n_reused;
            decode_reallocs_same += graph_after.n_reallocs_same - graph_before.n_reallocs_same;
            decode_reallocs_diff += graph_after.n_reallocs_diff - graph_before.n_reallocs_diff;
#endif
            for (int seq = 0; seq < concurrency; ++seq) outputs[wave + seq].push_back(next[seq]);
        }
        const double wall = seconds_since(cohort_start);
        require(prompt_seconds > 0 && decode_seconds > 0, "invalid timing");
        std::printf("{\"concurrency\":%d,\"requests\":8,\"output_tokens_per_request\":%d,\"placement\":\"%s\","
                    "\"context_per_sequence\":%d,\"warmup_decode_tokens\":%d,\"serial_tokens\":%s,\"serialized_prefill_seconds\":%.9g,"
                    "\"decode_seconds\":%.9g,\"cohort_seconds\":%.9g,"
                    "\"aggregate_decode_tok_s\":%.9g,\"aggregate_end_to_end_tok_s\":%.9g,\"tokens\":[",
                    concurrency, tokens, gpu ? "gpu-cpu-experts" : "cpu", sequence_context, hybrid ? 32 * concurrency : 0,
                    serial_tokens ? "true" : "false", prompt_seconds, decode_seconds, wall,
                    8.0 * (tokens - 1) / decode_seconds, 8.0 * tokens / wall);
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (i) std::printf(",");
            std::printf("[");
            for (size_t j = 0; j < outputs[i].size(); ++j) {
                if (j) std::printf(",");
                std::printf("%d", outputs[i][j]);
            }
            std::printf("]");
        }
        std::printf("]");
#if defined(VRAMBRIDGE_SPEED_GRAPH_STATS)
        std::printf(",\"decode_graph_reuses\":%lld,\"decode_graph_reallocs_same\":%lld,\"decode_graph_reallocs_diff\":%lld",
                    static_cast<long long>(decode_reuses), static_cast<long long>(decode_reallocs_same),
                    static_cast<long long>(decode_reallocs_diff));
#endif
        std::printf(",\"first_token_seconds\":[");
        for (size_t i = 0; i < first_token_seconds.size(); ++i) {
            if (i) std::printf(",");
            std::printf("%.9g", first_token_seconds[i]);
        }
        std::printf("],\"completion_seconds\":[");
        for (size_t i = 0; i < completion_seconds.size(); ++i) {
            if (i) std::printf(",");
            std::printf("%.9g", completion_seconds[i]);
        }
        std::printf("]}\n");
        std::fflush(stdout);
        context.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "probe failed: %s\n", exc.what());
        return 1;
    }
}
