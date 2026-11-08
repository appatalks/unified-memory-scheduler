#define main sequence_decode_contract_main
#include "sequence-decode-probe.cpp"
#undef main
static_assert(kSteps >= 3, "Rollback probe needs three reference positions");

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "Usage: %s MODEL PRIVATE_REFERENCE [gpu-cpu-experts]\n", argv[0]);
        return 2;
    }
    try {
#if !defined(_WIN32)
        umask(0077);
#endif
        const bool gpu = argc == 4 && std::strcmp(argv[3], "gpu-cpu-experts") == 0;
        require(argc == 3 || gpu, "invalid placement");
        int rollback_step = 0;
        if (const char * step_env = std::getenv("VRAMBRIDGE_PROBE_ROLLBACK_STEP")) {
            char * end = nullptr;
            const auto step = std::strtoul(step_env, &end, 10);
            require(end != step_env && *end == '\0' && step < static_cast<unsigned long>(kSteps) &&
                    static_cast<unsigned long>(kSteps) - step >= 3,
                    "Rollback step must leave three bounded reference positions");
            rollback_step = static_cast<int>(step);
        }
        llama_backend_init();
        ggml_backend_dev_t devices[] = {nullptr, nullptr};
        auto mp = llama_model_default_params();
        llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}
        };
        if (gpu) {
            devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            require(devices[0] && std::strncmp(ggml_backend_dev_name(devices[0]), "CUDA", 4) == 0, "CUDA unavailable");
            mp.n_gpu_layers = -1;
            mp.tensor_buft_overrides = overrides;
        } else {
            mp.n_gpu_layers = 0;
        }
        mp.devices = devices;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 131072;
        cp.n_seq_max = 8;
        cp.n_batch = 2048;
        cp.n_ubatch = 512;
        cp.n_threads = 8;
        cp.n_threads_batch = 16;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = gpu;
        cp.op_offload = gpu;
        cp.kv_unified = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(ctx && llama_n_ctx_seq(ctx.get()) == 16384, "context mismatch");
        const auto * vocab = llama_model_get_vocab(model.get());
        const int count = llama_vocab_n_tokens(vocab);
        const std::vector<std::string> prompts = {
            "Agent: Review a bounded work queue.", "Agent: Find a cancellation race.",
            "Agent: Explain how a slot keeps state.", "Agent: Propose tests for worker errors.",
            "Agent: Describe GPU memory pressure.", "Agent: Compare two queue designs.",
            "Agent: Explain why prompts should not leak.", "Agent: Summarize a safe shutdown strategy."
        };
        std::vector<Reference> refs;
        for (const auto & prompt : prompts) {
            const int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
            require(n > 0 && n < 128, "invalid token count");
            Reference ref;
            ref.prompt.resize(n);
            require(llama_tokenize(vocab, prompt.c_str(), prompt.size(), ref.prompt.data(), n, true, true) == n,
                    "tokenization failed");
            refs.push_back(std::move(ref));
        }
        load_reference(argv[2], refs, count, gpu);
        bool passed = true;
        for (int sequences : {2, 4, 8}) {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            for (int seq = 0; seq < sequences; ++seq) prefill(ctx.get(), refs[seq].prompt, seq, count);
            for (int step = 0; step < rollback_step; ++step) decode(ctx.get(), refs, sequences, step, count);
            std::vector<std::vector<uint8_t>> snapshots;
            for (int seq = 0; seq < sequences; ++seq) {
                const size_t size = llama_state_seq_get_size_ext(ctx.get(), seq, LLAMA_STATE_SEQ_FLAGS_NONE);
                require(size > 0 && size <= 128u * 1024u * 1024u, "snapshot exceeds bounded storage");
                snapshots.emplace_back(size);
                require(llama_state_seq_get_data_ext(ctx.get(), snapshots.back().data(), size, seq,
                        LLAMA_STATE_SEQ_FLAGS_NONE) == size, "state capture failed");
            }
            const auto before = decode(ctx.get(), refs, sequences, rollback_step, count);
            decode(ctx.get(), refs, sequences, rollback_step + 1, count);
            decode(ctx.get(), refs, sequences, rollback_step + 2, count);
            for (int seq = 0; seq < sequences; ++seq) {
                const auto & state = snapshots[seq];
                require(llama_state_seq_set_data_ext(ctx.get(), state.data(), state.size(), seq,
                        LLAMA_STATE_SEQ_FLAGS_NONE) == state.size(), "state restore failed");
            }
            const auto after = decode(ctx.get(), refs, sequences, rollback_step, count);
            for (int seq = 0; seq < sequences; ++seq) {
                passed = compare("before_reference", sequences, seq, rollback_step,
                                 refs[seq].decode[rollback_step], before[seq]) && passed;
                passed = compare("restored_reference", sequences, seq, rollback_step,
                                 refs[seq].decode[rollback_step], after[seq]) && passed;
                passed = compare("restore_repeatability", sequences, seq, rollback_step, before[seq], after[seq]) && passed;
            }
        }
        ctx.reset();
        model.reset();
        llama_backend_free();
        return passed ? 0 : 2;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "rollback probe failed: %s\n", exc.what());
        return 1;
    }
}
