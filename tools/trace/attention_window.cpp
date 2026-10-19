#include "attention_window.h"
#include "ggml-backend.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-moe-cache.h"
#include "llama-batch.h"
#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>

namespace {

using Build = ggml_tensor * (*)(const llm_graph_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
    ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, int64_t, float, int);
using Linear = ggml_tensor * (*)(const llm_graph_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *);
using Expert = ggml_tensor * (*)(const llm_graph_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
    ggml_tensor *, ggml_tensor *);
using Moe = ggml_tensor * (*)(const llm_graph_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
    ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
    int64_t, int64_t, llm_ffn_op_type, bool, float, llama_expert_gating_func_type, int,
    ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *);
using BatchLimit = int64_t (*)(const llama_moe_cache *);
using SlotMap = ggml_tensor * (*)(const llama_moe_cache *, int32_t, int64_t, int64_t);
using Process = int32_t (*)(llama_context *, llama_process_type, llama_batch_ext *);

thread_local int active_moe_window = -1;

struct MoeScope {
    int previous;
    explicit MoeScope(bool eligible) : previous(active_moe_window) {
        active_moe_window = eligible ? 1 : 0;
    }
    ~MoeScope() { active_moe_window = previous; }
};

constexpr const char * symbol =
    "_ZNK17llm_graph_context14build_attn_mhaEP11ggml_tensorS1_S1_S1_S1_S1_S1_lfi";

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "Attention window: %s\n", message);
    std::abort();
}

struct State {
    Build original = nullptr;
    Linear linear = nullptr;
    Expert expert = nullptr;
    Moe moe = nullptr;
    BatchLimit batch_limit = nullptr;
    SlotMap slot_map = nullptr;
    Process process = nullptr;
    bool enabled = false;
    bool products = false;
    bool expert_columns = true;
    bool decode_threads = false;
    FILE * output = nullptr;
    std::atomic<uint64_t> calls{0}, split_calls{0}, split_queries{0}, unsupported_calls{0};
    std::atomic<uint64_t> linear_splits{0}, expert_splits{0};
    std::atomic<uint64_t> prefill_bank_limit_restorations{0};
    std::atomic<uint64_t> prefill_slot_map_bypasses{0};
    std::mutex thread_mutex;
    std::map<std::pair<int32_t, int32_t>, uint64_t> thread_changes;

    State() {
        const char * mode = std::getenv("VRAMBRIDGE_ATTN_WINDOW");
        if (mode && std::strcmp(mode, "off") && std::strcmp(mode, "split")) {
            throw std::runtime_error("mode must be off or split");
        }
        enabled = mode && !std::strcmp(mode, "split");
        const char * library = std::getenv("VRAMBRIDGE_ATTN_WINDOW_LIBRARY");
        if (!library) throw std::runtime_error("explicit matching native llama library required");
        void * handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
        if (!handle) throw std::runtime_error(dlerror());
        original = reinterpret_cast<Build>(dlsym(handle, symbol));
        linear = reinterpret_cast<Linear>(dlsym(handle, "_ZNK17llm_graph_context13build_lora_mmEP11ggml_tensorS1_S1_"));
        expert = reinterpret_cast<Expert>(dlsym(handle, "_ZNK17llm_graph_context16build_lora_mm_idEP11ggml_tensorS1_S1_S1_S1_"));
        moe = reinterpret_cast<Moe>(dlsym(handle,
            "_ZNK17llm_graph_context13build_moe_ffnEP11ggml_tensorS1_S1_S1_S1_S1_S1_S1_S1_S1_ll15llm_ffn_op_typebf29llama_expert_gating_func_typeiS1_S1_S1_S1_S1_S1_S1_"));
        batch_limit = reinterpret_cast<BatchLimit>(dlsym(handle, "_ZNK15llama_moe_cache18hybrid_batch_limitEv"));
        slot_map = reinterpret_cast<SlotMap>(dlsym(handle, "_ZNK15llama_moe_cache12get_slot_mapEill"));
        process = reinterpret_cast<Process>(dlsym(handle, "llama_process"));
        if (!original || !linear || !expert || !moe || !batch_limit || !slot_map || !process) {
            throw std::runtime_error("matching native builders are not exported");
        }
        const char * product_mode = std::getenv("VRAMBRIDGE_ATTN_WINDOW_PRODUCTS");
        if (product_mode && std::strcmp(product_mode, "off") && std::strcmp(product_mode, "columns")) {
            throw std::runtime_error("product mode must be off or columns");
        }
        products = product_mode && !std::strcmp(product_mode, "columns");
        const char * expert_mode = std::getenv("VRAMBRIDGE_ATTN_WINDOW_EXPERT_PRODUCTS");
        if (expert_mode && std::strcmp(expert_mode, "columns") && std::strcmp(expert_mode, "native")) {
            throw std::runtime_error("expert product mode must be columns or native");
        }
        expert_columns = !expert_mode || !std::strcmp(expert_mode, "columns");
        const char * threads = std::getenv("VRAMBRIDGE_ATTN_WINDOW_THREADS");
        if (threads && std::strcmp(threads, "off") && std::strcmp(threads, "decode")) {
            throw std::runtime_error("verification thread policy must be off or decode");
        }
        decode_threads = threads && !std::strcmp(threads, "decode");
        if (!enabled) return;
        const char * scalar = std::getenv("VRAMBRIDGE_HYBRID_SCALAR_ATTN");
        if (scalar && std::strcmp(scalar, "0")) {
            throw std::runtime_error("split Flash Attention cannot be combined with non-flash scalar attention");
        }
        if (products) {
            for (const char * flag : {"VRAMBRIDGE_HYBRID_SCALAR_LINEAR", "VRAMBRIDGE_HYBRID_SCALAR_EXPERT",
                                     "VRAMBRIDGE_HYBRID_SCALAR_ROUTER"}) {
                const char * value = std::getenv(flag);
                if (value && std::strcmp(value, "0")) {
                    throw std::runtime_error("window-scoped products require global scalar-product flags off");
                }
            }
        }
        const char * path = std::getenv("VRAMBRIDGE_ATTN_WINDOW_OUT");
        if (!path) throw std::runtime_error("explicit private output required");
        struct stat info{};
        const auto parent = std::filesystem::path(path).parent_path();
        if (parent.empty() || stat(parent.c_str(), &info) || !S_ISDIR(info.st_mode) ||
            info.st_uid != getuid() || (info.st_mode & 077)) {
            throw std::runtime_error("report parent must be owned and private");
        }
        output = std::fopen(path, "wx");
        if (!output || fchmod(fileno(output), 0600)) throw std::runtime_error("cannot create exclusive private output");
        std::fprintf(stderr, "Attention window active: bounded one-stream queries use the original single-query Flash Attention builder\n");
    }

    ~State() {
        if (!enabled) return;
        nlohmann::json changes = nlohmann::json::array();
        for (const auto & [counts, calls] : thread_changes) {
            changes.push_back({{"from_batch_threads", counts.first}, {"to_threads", counts.second}, {"calls", calls}});
        }
        const nlohmann::json report = {{"schema_version", 1}, {"kind", "attention_window"},
            {"complete", true}, {"pid", getpid()}, {"source_sha256", VRAMBRIDGE_ATTN_WINDOW_SOURCE_SHA256},
            {"header_sha256", VRAMBRIDGE_ATTN_WINDOW_HEADER_SHA256},
            {"calls", calls.load()}, {"split_calls", split_calls.load()},
            {"split_queries", split_queries.load()}, {"unsupported_calls", unsupported_calls.load()},
            {"window_scoped_products", products}, {"linear_split_calls", linear_splits.load()},
            {"expert_column_split_enabled", products && expert_columns},
            {"expert_split_calls", expert_splits.load()}, {"eligibility", "bounded multi-token all-output graphs"},
            {"prefill_bank_limit_restorations", prefill_bank_limit_restorations.load()},
            {"prefill_slot_map_bypasses", prefill_slot_map_bypasses.load()},
            {"verification_decode_threads", decode_threads}, {"thread_changes", changes},
            {"target_weights_changed", false}, {"kv_precision_changed", false},
            {"attention_algorithm_replaced", false}, {"full_model_acceleration_claim", false}};
        const auto text = report.dump(2) + "\n";
        if (std::fwrite(text.data(), 1, text.size(), output) != text.size() || std::fclose(output)) {
            fail("cannot finish report");
        }
    }
};

State & shared_state() {
    static State state;
    return state;
}

} // namespace

ggml_tensor * llm_graph_context::build_attn_mha(
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * kq_b,
        ggml_tensor * mask, ggml_tensor * sinks, ggml_tensor * v_mla,
        int64_t n_kv_max, float scale, int layer) const {
    try {
        auto & state = shared_state();
        ++state.calls;
        if (!state.enabled || !attention_window_eligible(n_tokens, n_outputs) ||
            q->ne[2] <= 1 || q->ne[2] > 32) {
            return state.original(this, q, k, v, kq_b, mask, sinks, v_mla, n_kv_max, scale, layer);
        }
        if (!cparams.flash_attn || kq_b || v_mla || k->ne[3] != 1 || q->ne[3] != 1 ||
            q->ne[2] != n_tokens || !mask || mask->type != GGML_TYPE_F16 ||
            mask->ne[2] != 1 || mask->ne[3] != 1) {
            ++state.unsupported_calls;
            return state.original(this, q, k, v, kq_b, mask, sinks, v_mla, n_kv_max, scale, layer);
        }
        ggml_tensor * result = nullptr;
        for (int64_t position = 0; position < q->ne[2]; ++position) {
            const auto [query, column_mask] = attention_window_column(ctx0, q, mask, position);
            auto * column = state.original(this, query, k, v, nullptr, column_mask, sinks, nullptr,
                                          n_kv_max, scale, layer);
            if (column->ne[1] != 1) throw std::runtime_error("original attention output is not one query column");
            result = result ? ggml_concat(ctx0, result, column, 1) : column;
        }
        ggml_build_forward_expand(gf, result);
        ++state.split_calls;
        state.split_queries += q->ne[2];
        return result;
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

ggml_tensor * llm_graph_context::build_lora_mm(
        ggml_tensor * weights, ggml_tensor * input, ggml_tensor * scales) const {
    try {
        auto & state = shared_state();
        const bool floating = weights->type == GGML_TYPE_F32 || weights->type == GGML_TYPE_F16 ||
                              weights->type == GGML_TYPE_BF16;
        if (!state.enabled || !state.products || !attention_window_eligible(n_tokens, n_outputs) || !floating ||
            input->ne[1] <= 1 || input->ne[1] > 32 || input->ne[2] != 1 || input->ne[3] != 1 ||
            weights->ne[2] != 1 || weights->ne[3] != 1) {
            return state.linear(this, weights, input, scales);
        }
        ggml_tensor * result = nullptr;
        for (int64_t position = 0; position < input->ne[1]; ++position) {
            auto * column = attention_window_linear_column(ctx0, input, position);
            auto * value = state.linear(this, weights, column, scales);
            result = result ? ggml_concat(ctx0, result, value, 1) : value;
        }
        ++state.linear_splits;
        return result;
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
        ggml_tensor * weights, ggml_tensor * input, ggml_tensor * ids,
        ggml_tensor * scales, ggml_tensor * slots) const {
    try {
        auto & state = shared_state();
        if (!state.enabled || !state.products || !state.expert_columns || !attention_window_eligible(n_tokens, n_outputs) ||
            !weights->buffer || ggml_backend_buffer_is_host(weights->buffer) || slots ||
            input->ne[2] <= 1 || input->ne[2] > 32 || input->ne[3] != 1) {
            return state.expert(this, weights, input, ids, scales, slots);
        }
        ggml_tensor * result = nullptr;
        for (int64_t position = 0; position < input->ne[2]; ++position) {
            const auto [column, indices] = attention_window_expert_column(ctx0, input, ids, position);
            auto * value = state.expert(this, weights, column, indices, scales, nullptr);
            result = result ? ggml_concat(ctx0, result, value, 2) : value;
        }
        ++state.expert_splits;
        return result;
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

ggml_tensor * llm_graph_context::build_moe_ffn(
        ggml_tensor * cur, ggml_tensor * gate_inp, ggml_tensor * gate_inp_b,
        ggml_tensor * up_exps, ggml_tensor * up_exps_b, ggml_tensor * gate_exps, ggml_tensor * gate_exps_b,
        ggml_tensor * down_exps, ggml_tensor * down_exps_b, ggml_tensor * exp_probs_b,
        int64_t experts, int64_t used, llm_ffn_op_type operation, bool normalize, float weight_scale,
        llama_expert_gating_func_type gating, int layer, ggml_tensor * probs,
        ggml_tensor * gate_up, ggml_tensor * gate_up_bias, ggml_tensor * up_scale,
        ggml_tensor * gate_scale, ggml_tensor * down_scale, ggml_tensor * selected) const {
    try {
        auto & state = shared_state();
        const MoeScope scope(attention_window_eligible(n_tokens, n_outputs));
        return state.moe(this, cur, gate_inp, gate_inp_b, up_exps, up_exps_b, gate_exps, gate_exps_b,
                         down_exps, down_exps_b, exp_probs_b, experts, used, operation, normalize,
                         weight_scale, gating, layer, probs, gate_up, gate_up_bias, up_scale,
                         gate_scale, down_scale, selected);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

int64_t llama_moe_cache::hybrid_batch_limit() const {
    try {
        auto & state = shared_state();
        const int64_t original = state.batch_limit(this);
        if (state.enabled && state.products && active_moe_window == 0 && original > 1) {
            ++state.prefill_bank_limit_restorations;
            return 1;
        }
        return original;
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t layer, int64_t tokens, int64_t used) const {
    try {
        auto & state = shared_state();
        auto * original = state.slot_map(this, layer, tokens, used);
        if (state.enabled && state.products && active_moe_window == 0 && tokens > 1 && original) {
            ++state.prefill_slot_map_bypasses;
            return nullptr;
        }
        return original;
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

extern "C" int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    try {
        auto & state = shared_state();
        if (!state.enabled || !state.decode_threads || !ctx || !batch || type != LLAMA_PROCESS_TYPE_DECODE ||
            batch->tokens.size() <= 1 || batch->tokens.size() > 32 ||
            !std::all_of(batch->tokens.begin(), batch->tokens.end(), [](const auto & token) { return token.output; })) {
            return state.process(ctx, type, batch);
        }
        const int32_t generation = llama_n_threads(ctx), batched = llama_n_threads_batch(ctx);
        if (generation <= 0 || batched <= 0) throw std::runtime_error("native thread limits are invalid");
        struct Restore {
            llama_context * ctx;
            int32_t generation, batched;
            ~Restore() { llama_set_n_threads(ctx, generation, batched); }
        } restore{ctx, generation, batched};
        llama_set_n_threads(ctx, generation, generation);
        if (generation != batched) {
            std::lock_guard lock(state.thread_mutex);
            ++state.thread_changes[{batched, generation}];
        }
        return state.process(ctx, type, batch);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}
