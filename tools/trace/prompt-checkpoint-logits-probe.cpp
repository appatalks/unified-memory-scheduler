#define main sequence_decode_contract_main
#include "sequence-decode-probe.cpp"
#undef main
#include "qsa-indexer-observer.h"
#include "qsa-mask-observer.h"
#include <chrono>

namespace {

constexpr int kPrompt = 8192;
constexpr int kBatch = 2048;
constexpr int kCheckpoint = 4096;
constexpr size_t kSnapshotLimit = size_t(2) * 1024 * 1024 * 1024;
bool numerical_pass = true;
bool rollback_supported = true;

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    const int count = -llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), nullptr, 0, special, true);
    require(count > 0, "token count invalid");
    std::vector<llama_token> tokens(count);
    require(llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), tokens.data(), count, special, true) == count,
            "tokenization failed");
    return tokens;
}

void process(llama_context * ctx, const std::vector<llama_token> & tokens, int begin, int end, bool output) {
    require(begin >= 0 && begin < end && end <= static_cast<int>(tokens.size()), "invalid input range");
    for (int first = begin; first < end; first += kBatch) {
        const int last = std::min(first + kBatch, end);
        Batch batch(last - first);
        for (int pos = first; pos < last; ++pos) add(batch.value, tokens[pos], pos, 0, output && pos + 1 == end);
        require(llama_decode(ctx, batch.value) == 0, "prompt decode failed");
    }
    require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == end - 1, "unexpected sequence position");
}

std::vector<uint8_t> save(llama_context * ctx, llama_state_seq_flags flags) {
    const size_t size = llama_state_seq_get_size_ext(ctx, 0, flags);
    require(size > 0 && size <= kSnapshotLimit, "snapshot size exceeds bound");
    std::vector<uint8_t> data(size);
    require(llama_state_seq_get_data_ext(ctx, data.data(), data.size(), 0, flags) == size, "snapshot failed");
    return data;
}

bool restore(llama_context * ctx, const std::vector<uint8_t> & data, llama_state_seq_flags flags, int prefix) {
    if (flags == LLAMA_STATE_SEQ_FLAGS_NONE) llama_memory_clear(llama_get_memory(ctx), true);
    require(llama_state_seq_set_data_ext(ctx, data.data(), data.size(), 0, flags) == data.size(), "restore failed");
    const bool removed = llama_memory_seq_rm(llama_get_memory(ctx), 0, prefix, -1);
    const int pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    std::printf("{\"event\":\"restore\",\"flags\":%d,\"prefix\":%d,\"bytes\":%zu,\"tail_removal\":%s,\"position\":%d}\n",
                static_cast<int>(flags), prefix, data.size(), removed ? "true" : "false", pos);
    std::fflush(stdout);
    return removed && pos == prefix - 1;
}

struct Outputs {
    std::vector<llama_token> forced;
    std::vector<std::vector<float>> values;
};

Outputs collect(llama_context * ctx, int position, int vocab, const Outputs * reference = nullptr) {
    Outputs result;
    const float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "missing prompt logits");
    result.values.emplace_back(logits, logits + vocab);
    for (int i = 0; i < kSteps; ++i) {
        const auto token = reference ? reference->forced.at(i) : static_cast<llama_token>(
            std::max_element(result.values.back().begin(), result.values.back().end()) - result.values.back().begin());
        result.forced.push_back(token);
        Batch batch(1);
        add(batch.value, token, position + i, 0, true);
        require(llama_decode(ctx, batch.value) == 0, "teacher-forced decode failed");
        logits = llama_get_logits_ith(ctx, 0);
        require(logits != nullptr, "missing decode logits");
        result.values.emplace_back(logits, logits + vocab);
    }
    return result;
}

void check(const char * name, const Outputs & reference, const Outputs & candidate) {
    require(reference.values.size() == candidate.values.size(), "output count mismatch");
    for (size_t i = 0; i < reference.values.size(); ++i) {
        numerical_pass = compare(name, 1, 0, static_cast<int>(i) - 1, reference.values[i], candidate.values[i]) && numerical_pass;
    }
}

void report_graph_reuse(llama_context * ctx, const char * context_name) {
    const auto stats = llama_perf_context(ctx);
    std::printf("{\"event\":\"graph_reuse\",\"context\":\"%s\",\"count\":%d}\n", context_name, stats.n_reused);
    std::fflush(stdout);
}

void evaluate(llama_context * ctx, const char * case_name, const std::vector<llama_token> & source,
              const std::vector<llama_token> & target, int prefix, int vocab) {
    auto memory = llama_get_memory(ctx);
    llama_memory_clear(memory, true);
    process(ctx, target, 0, target.size(), true);
    const Outputs reference = collect(ctx, target.size(), vocab);

    llama_memory_clear(memory, true);
    process(ctx, target, 0, target.size(), true);
    check((std::string(case_name) + "_cold_repeat").c_str(), reference, collect(ctx, target.size(), vocab, &reference));

    llama_memory_clear(memory, true);
    process(ctx, target, 0, prefix, false);
    process(ctx, target, prefix, target.size(), true);
    check((std::string(case_name) + "_split").c_str(), reference, collect(ctx, target.size(), vocab, &reference));

    const int ragged = prefix == kCheckpoint ? prefix + 257 : prefix - 4;
    llama_memory_clear(memory, true);
    process(ctx, target, 0, ragged, false);
    process(ctx, target, ragged, target.size(), true);
    check((std::string(case_name) + "_ragged_split").c_str(), reference, collect(ctx, target.size(), vocab, &reference));

    for (const auto flags : {LLAMA_STATE_SEQ_FLAGS_NONE, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY}) {
        llama_memory_clear(memory, true);
        process(ctx, source, 0, prefix, true);
        const auto state = save(ctx, flags);
        if (prefix < static_cast<int>(source.size())) {
            process(ctx, source, prefix, source.size(), true);
        } else {
            Batch batch(1);
            add(batch.value, source.back(), prefix, 0, true);
            require(llama_decode(ctx, batch.value) == 0, "source continuation failed");
        }
        const char * kind = flags == LLAMA_STATE_SEQ_FLAGS_NONE ? "full_restore" : "partial_restore";
        if (!restore(ctx, state, flags, prefix)) {
            rollback_supported = false;
            std::printf("{\"case\":\"%s_%s\",\"status\":\"unsupported_tail_removal\"}\n", case_name, kind);
            std::fflush(stdout);
            continue;
        }
        process(ctx, target, prefix, target.size(), true);
        check((std::string(case_name) + "_" + kind).c_str(), reference, collect(ctx, target.size(), vocab, &reference));
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        if (sequence_decode_contract_main(argc, argv) != 0) return 1;
        Outputs reference, changed;
        reference.values = {{1.0f, -2.0f}};
        changed.values = {{1.02f, -2.0f}};
        check("self_gate_equal", reference, reference);
        const bool equal_passed = numerical_pass;
        check("self_gate_changed", reference, changed);
        return equal_passed && !numerical_pass && qsa_observer::self_test() && qsa_mask_observer::self_test() ? 0 : 1;
    }
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s MODEL\n", argv[0]);
        return 2;
    }
    try {
#if !defined(_WIN32)
        umask(0077);
#endif
        const char * cpu = std::getenv("VRAMBRIDGE_PROBE_CPU_PLUGIN");
        const char * cuda = std::getenv("VRAMBRIDGE_PROBE_CUDA_PLUGIN");
        require(cpu && cuda && ggml_backend_load(cpu) && ggml_backend_load(cuda), "explicit plugins required");
        llama_backend_init();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU), nullptr};
        require(devices[0] && std::strncmp(ggml_backend_dev_name(devices[0]), "CUDA", 4) == 0, "CUDA unavailable");
        llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = -1;
        mp.tensor_buft_overrides = overrides;
        mp.no_host = true;
        mp.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = bounded_env("VRAMBRIDGE_PROBE_CONTEXT", 16384, 16384, 98304);
        cp.n_seq_max = 1;
        cp.n_batch = kBatch;
        cp.n_ubatch = bounded_env("VRAMBRIDGE_PROBE_UBATCH", 512, 1, kBatch);
        cp.n_threads = 8;
        cp.n_threads_batch = 16;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = true;
        cp.op_offload = false;
        cp.kv_unified = false;
        cp.moe_cache_size = size_t(bounded_env("VRAMBRIDGE_PROBE_MOE_CACHE_MIB", 4096, 0, 8192)) * 1024 * 1024;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(ctx && llama_n_ctx_seq(ctx.get()) == cp.n_ctx, "context mismatch");
        const auto * vocab = llama_model_get_vocab(model.get());
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const int prompt_tokens = bounded_env("VRAMBRIDGE_PROBE_PROMPT_TOKENS", kPrompt, 512, kPrompt);
        std::string body;
        char row[128];
        for (int i = 0; i < kPrompt / 10 + 1024; ++i) {
            std::snprintf(row, sizeof(row), "Row %06d: amber birch silver; item %07d.\n", i, i * 7919 % 1000003);
            body += row;
        }
        auto source = tokenize(vocab, body, true);
        require(source.size() >= size_t(prompt_tokens), "insufficient prompt tokens");
        source.resize(prompt_tokens);
        auto edited = source;
        const int edit_position = prompt_tokens / 2 + std::min(257, prompt_tokens / 4);
        edited[edit_position] = tokenize(vocab, "quartz", false).front();
        require(edited[edit_position] != source[edit_position], "edit did not change token");
        auto append = source;
        const auto suffix = tokenize(vocab, "\nContinue explaining the engineering tradeoffs in detail.", false);
        append.insert(append.end(), suffix.begin(), suffix.end());
        std::printf("{\"event\":\"configuration\",\"context\":%u,\"ubatch\":%u,\"prompt_tokens\":%d,"
                    "\"suffix_tokens\":%zu,\"vocabulary\":%d,\"steps\":%d}\n",
                    cp.n_ctx, cp.n_ubatch, prompt_tokens, suffix.size(), n_vocab, kSteps);
        std::fflush(stdout);
        process(ctx.get(), source, 0, 128, true);
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        const char * reference_mode = std::getenv("VRAMBRIDGE_PROBE_CROSS_REFERENCE");
        if (reference_mode) {
            require(std::strcmp(reference_mode, "write") == 0 || std::strcmp(reference_mode, "read") == 0,
                    "cross-reference mode must be write or read");
            const char * path = std::getenv("VRAMBRIDGE_PROBE_REFERENCE_FILE");
            require(path && path[0] == '/', "cross-reference path must be absolute");
            std::vector<Reference> references(1);
            references[0].prompt = edited;
            if (std::strcmp(reference_mode, "read") == 0) load_reference(path, references, n_vocab, true);
            if (bounded_env("VRAMBRIDGE_PROBE_TIMING_WARMUP", 0, 0, 1)) {
                process(ctx.get(), edited, 0, edited.size(), true);
                llama_synchronize(ctx.get());
                llama_memory_clear(llama_get_memory(ctx.get()), true);
            }
            const auto start = std::chrono::steady_clock::now();
            process(ctx.get(), edited, 0, edited.size(), true);
            if (bounded_env("VRAMBRIDGE_PROBE_TIMING", 0, 0, 1)) {
                llama_synchronize(ctx.get());
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                require(seconds > 0 && std::isfinite(seconds), "invalid prompt timing");
                std::printf("{\"event\":\"prompt_timing\",\"tokens\":%zu,\"seconds\":%.9f}\n", edited.size(), seconds);
                std::fflush(stdout);
            }
            if (std::strcmp(reference_mode, "write") == 0) {
                const auto result = collect(ctx.get(), edited.size(), n_vocab);
                references[0].next = result.forced;
                references[0].prefill = result.values.front();
                references[0].decode.assign(result.values.begin() + 1, result.values.end());
                save_reference(path, references, n_vocab, true);
                std::printf("{\"event\":\"reference_written\",\"steps\":%d}\n", kSteps);
            } else {
                Outputs expected;
                expected.forced = references[0].next;
                expected.values.push_back(references[0].prefill);
                expected.values.insert(expected.values.end(), references[0].decode.begin(), references[0].decode.end());
                check("cross_reference", expected, collect(ctx.get(), edited.size(), n_vocab, &expected));
            }
            report_graph_reuse(ctx.get(), "cross_reference");
        } else if (bounded_env("VRAMBRIDGE_PROBE_QSA_MASK_SCREEN", 0, 0, 1)) {
            constexpr int early = 2048;
            process(ctx.get(), edited, 0, early, true);
            const auto reference = collect(ctx.get(), early, n_vocab);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            process(ctx.get(), edited, 0, early, true);
            check("mask_unobserved_repeat", reference, collect(ctx.get(), early, n_vocab, &reference));
            report_graph_reuse(ctx.get(), "unobserved");
            ctx.reset();
            qsa_mask_observer::Observer observer;
            cp.cb_eval = qsa_mask_observer::callback;
            cp.cb_eval_user_data = &observer;
            ctx.reset(llama_init_from_model(model.get(), cp));
            require(ctx && llama_n_ctx_seq(ctx.get()) == cp.n_ctx, "mask observer context mismatch");
            process(ctx.get(), source, 0, 128, true);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            observer.active = true;
            process(ctx.get(), edited, 0, early, true);
            observer.active = false;
            require(observer.error.empty(), observer.error.c_str());
            int layers = 0, frames = 0, rows = 0;
            for (int layer = 3; layer < 48; layer += 4) {
                require(observer.frames[layer] == 4 && observer.rows[layer] == early, "incomplete QSA mask coverage");
                ++layers; frames += observer.frames[layer]; rows += observer.rows[layer];
            }
            check("mask_observer_vs_unobserved", reference, collect(ctx.get(), early, n_vocab, &reference));
            std::printf("{\"event\":\"qsa_mask_screen\",\"layers\":%d,\"frames\":%d,\"rows\":%d,"
                        "\"elements\":%llu,\"bit_differences\":%llu,\"value_differences\":%llu,\"different_rows\":%llu,"
                        "\"same_storage\":%llu,\"causal_differences\":%llu}\n",
                        layers, frames, rows, (unsigned long long) observer.total.elements,
                        (unsigned long long) observer.total.bit_differences, (unsigned long long) observer.total.value_differences,
                        (unsigned long long) observer.total.different_rows, (unsigned long long) observer.same_storage,
                        (unsigned long long) observer.causal_differences);
            std::fflush(stdout);
            numerical_pass = numerical_pass && observer.total.bit_differences == 0 && observer.total.value_differences == 0 &&
                             observer.causal_differences == 0 && observer.next_layer == 3;
            report_graph_reuse(ctx.get(), "mask_observed");
        } else if (bounded_env("VRAMBRIDGE_PROBE_QSA_OBSERVER", 0, 0, 1)) {
            require(prompt_tokens == kPrompt, "QSA indexer observer requires 8192 prompt tokens");
            process(ctx.get(), edited, 0, edited.size(), true);
            const auto reference = collect(ctx.get(), edited.size(), n_vocab);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            process(ctx.get(), edited, 0, edited.size(), true);
            check("unobserved_cold_repeat", reference, collect(ctx.get(), edited.size(), n_vocab, &reference));
            report_graph_reuse(ctx.get(), "unobserved");
            ctx.reset();
            const int layer = bounded_env("VRAMBRIDGE_PROBE_QSA_LAYER", 3, 3, 47);
            require(layer % 4 == 3, "QSA layer must be a full attention layer");
            qsa_observer::Observer observer(layer);
            cp.cb_eval = qsa_observer::callback;
            cp.cb_eval_user_data = &observer;
            ctx.reset(llama_init_from_model(model.get(), cp));
            require(ctx && llama_n_ctx_seq(ctx.get()) == cp.n_ctx, "observed context mismatch");
            process(ctx.get(), source, 0, 128, true);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            observer.start();
            process(ctx.get(), edited, 0, edited.size(), true);
            auto first = observer.finish();
            const auto observed = collect(ctx.get(), edited.size(), n_vocab, &reference);
            check("observer_vs_unobserved", reference, observed);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            observer.start();
            process(ctx.get(), edited, 0, edited.size(), true);
            auto second = observer.finish();
            const auto repeated = collect(ctx.get(), edited.size(), n_vocab, &reference);
            check("observed_cold_repeat", observed, repeated);
            const auto result = qsa_observer::compare(first, second);
            require(result.invalid == 0, "QSA selections invalid");
            int min_cols = first.front().cols, max_cols = first.front().cols;
            bool device_only = true;
            for (const auto & frame : first) {
                min_cols = std::min(min_cols, frame.cols);
                max_cols = std::max(max_cols, frame.cols);
                device_only = device_only && frame.score_device && frame.ids_device;
            }
            for (const auto & frame : second) device_only = device_only && frame.score_device && frame.ids_device;
            std::printf("{\"event\":\"qsa_comparison\",\"layer\":%d,\"frames\":%zu,\"min_cols\":%d,\"max_cols\":%d,"
                        "\"device_only\":%s,\"rows\":%llu,\"same_score_rows\":%llu,\"changed_sets\":%llu,"
                        "\"changed_visible_sets\":%llu,\"order_only\":%llu,\"same_score_changed_sets\":%llu,"
                        "\"same_score_tie_changed_sets\":%llu,\"first_finite_cutoff_ties\":%llu,"
                        "\"second_finite_cutoff_ties\":%llu}\n", layer, first.size(),
                        min_cols, max_cols, device_only ? "true" : "false", (unsigned long long) result.rows,
                        (unsigned long long) result.same_scores, (unsigned long long) result.changed_sets,
                        (unsigned long long) result.changed_visible_sets, (unsigned long long) result.order_only,
                        (unsigned long long) result.same_score_changed_sets,
                        (unsigned long long) result.same_score_tie_changed_sets,
                        (unsigned long long) result.first_finite_cutoff_ties,
                        (unsigned long long) result.second_finite_cutoff_ties);
            std::fflush(stdout);
            report_graph_reuse(ctx.get(), "observed");
        } else if (bounded_env("VRAMBRIDGE_PROBE_FRESH_CONTEXT", 0, 0, 1)) {
            require(prompt_tokens == kPrompt, "fresh-context mode requires 8192 prompt tokens");
            process(ctx.get(), edited, 0, edited.size(), true);
            const auto reference = collect(ctx.get(), edited.size(), n_vocab);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            process(ctx.get(), edited, 0, edited.size(), true);
            check("middle_edit_cold_repeat", reference, collect(ctx.get(), edited.size(), n_vocab, &reference));
            const int repeats = bounded_env("VRAMBRIDGE_PROBE_EXTRA_REPEATS", 0, 0, 8);
            for (int i = 0; i < repeats; ++i) {
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                process(ctx.get(), edited, 0, edited.size(), true);
                const auto name = std::string("middle_edit_extra_repeat_") + std::to_string(i + 1);
                check(name.c_str(), reference, collect(ctx.get(), edited.size(), n_vocab, &reference));
            }
            report_graph_reuse(ctx.get(), "initial");
            ctx.reset();
            ctx.reset(llama_init_from_model(model.get(), cp));
            require(ctx && llama_n_ctx_seq(ctx.get()) == cp.n_ctx, "fresh context mismatch");
            process(ctx.get(), source, 0, 128, true);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            process(ctx.get(), edited, 0, edited.size(), true);
            check("middle_edit_fresh_context", reference, collect(ctx.get(), edited.size(), n_vocab, &reference));
            report_graph_reuse(ctx.get(), "fresh");
        } else {
            require(prompt_tokens == kPrompt, "checkpoint mode requires 8192 prompt tokens");
            evaluate(ctx.get(), "middle_edit", source, edited, kCheckpoint, n_vocab);
            evaluate(ctx.get(), "append", source, append, kPrompt, n_vocab);
            report_graph_reuse(ctx.get(), "initial");
        }
        ctx.reset();
        model.reset();
        llama_backend_free();
        const char * status = !rollback_supported ? "unsupported" : numerical_pass ? "passed" : "numerical_failure";
        std::printf("{\"event\":\"result\",\"status\":\"%s\"}\n", status);
        return !rollback_supported ? 3 : numerical_pass ? 0 : 2;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "checkpoint probe failed: %s\n", exc.what());
        return 1;
    }
}
