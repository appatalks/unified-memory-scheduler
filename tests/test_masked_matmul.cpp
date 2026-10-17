#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void check(ggml_backend_t backend, ggml_type type, size_t experts, size_t active, size_t tokens,
                  ggml_backend_buffer_type_t repack = nullptr, bool strided_ids = false) {
    ggml_init_params params{1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    require(ctx != nullptr, "metadata allocation failed");
    const size_t columns = 256, rows = 16;
    auto * weights = ggml_new_tensor_3d(ctx.get(), type, columns, rows, experts);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> packed_buffer(
        repack ? ggml_backend_buft_alloc_buffer(repack, ggml_nbytes(weights) + 256) : nullptr,
        ggml_backend_buffer_free);
    if (repack) {
        require(packed_buffer != nullptr, "CPU repack allocation failed");
        require(ggml_backend_tensor_alloc(packed_buffer.get(), weights,
                                         ggml_backend_buffer_get_base(packed_buffer.get())) == GGML_STATUS_SUCCESS,
                "CPU repack tensor allocation failed");
    }
    auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, columns, 1, tokens);
    auto * id_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, active + (strided_ids ? 1 : 0), tokens);
    auto * ids = strided_ids ? ggml_view_2d(ctx.get(), id_storage, active, tokens, id_storage->nb[1], 0) : id_storage;
    auto * output = ggml_mul_mat_id(ctx.get(), weights, input, ids);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "tensor allocation failed");
    if (strided_ids) {
        std::vector<int32_t> padding((active + 1)*tokens, 0);
        ggml_backend_tensor_set(id_storage, padding.data(), 0, padding.size()*sizeof(int32_t));
    }
    std::vector<float> source(columns * rows * experts), activation(columns * tokens);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = std::ldexp(0.2f * std::sin(float(i + 1)), int((i / 32) % 13) - 6);
    }
    for (size_t i = 0; i < activation.size(); ++i) activation[i] = 0.1f * std::cos(float(i + 1));
    if (type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(weights, source.data(), 0, source.size() * sizeof(float));
    } else {
        std::vector<uint8_t> quantized(ggml_nbytes(weights));
        const auto bytes = ggml_quantize_chunk(type, source.data(), quantized.data(), 0, rows * experts, columns, nullptr);
        require(bytes == quantized.size(), "quantized byte count mismatch");
        ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
        if (repack) {
            using export_fn = bool (*)(const ggml_tensor *, int64_t, void *, size_t);
            const auto export_raw = reinterpret_cast<export_fn>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "vrambridge_export_repacked_expert"));
            require(export_raw != nullptr, "CPU repack export hook missing");
            std::vector<uint8_t> raw(weights->nb[2]);
            for (size_t expert = 0; expert < experts; ++expert) {
                require(export_raw(weights, expert, raw.data(), raw.size()), "CPU repack export failed");
                require(std::memcmp(raw.data(), quantized.data() + expert*raw.size(), raw.size()) == 0,
                        "CPU repack round-trip changed original MXFP4 bytes");
            }
            require(!export_raw(weights, -1, raw.data(), raw.size()), "negative expert accepted");
            require(!export_raw(weights, experts, raw.data(), raw.size()), "out-of-range expert accepted");
            require(!export_raw(weights, 0, raw.data(), raw.size() - 1), "invalid expert size accepted");
        }
    }
    ggml_backend_tensor_set(input, activation.data(), 0, activation.size() * sizeof(float));
    std::vector<int32_t> all(active*tokens);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t rank = 0; rank < active; ++rank) all[token*active + rank] = (token + rank*3) % experts;
    }
    const auto set_ids = [&](const std::vector<int32_t> & values) {
        for (size_t token = 0; token < tokens; ++token) {
            ggml_backend_tensor_set(ids, values.data() + token*active, token*ids->nb[1], active*sizeof(int32_t));
        }
    };
    using empty_count_fn = uint32_t (*)(void);
    const auto empty_count = reinterpret_cast<empty_count_fn>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "vrambridge_cpu_empty_count"));
    require(empty_count != nullptr, "CPU empty-path counter hook missing");
    const char * flag = std::getenv("VRAMBRIDGE_HYBRID_CPU_EMPTY");
    const bool enabled = flag && std::strcmp(flag, "1") == 0;
    set_ids(all);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "reference matmul failed");
    std::vector<float> reference(rows * active * tokens), actual(reference.size());
    ggml_backend_tensor_get(output, reference.data(), 0, reference.size() * sizeof(float));
    auto partial = all;
    for (size_t i = 0; i < partial.size(); i += 2) partial[i] = -1;
    for (const std::vector<int32_t> mask : {
             partial,
             std::vector<int32_t>(active*tokens, -1),
             all}) {
        set_ids(mask);
        const auto before = empty_count();
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "masked matmul failed");
        bool all_empty = true;
        for (int32_t id : mask) all_empty = all_empty && id == -1;
        if (enabled && all_empty && !repack) require(empty_count() == before + 1, "empty fast path was not exercised");
        if (!enabled || !all_empty) require(empty_count() == before, "empty fast path ran on active experts");
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        for (size_t token = 0; token < tokens; ++token) {
            for (size_t expert = 0; expert < active; ++expert) {
                for (size_t row = 0; row < rows; ++row) {
                    const size_t index = token * active * rows + expert * rows + row;
                    const float expected = mask[token * active + expert] < 0 ? 0 : reference[index];
                    require(std::isfinite(actual[index]) && std::abs(actual[index] - expected) <= 1e-6f,
                            "masked output is stale, nonzero, or changed an active expert");
                }
            }
        }
    }
}

static void check_shared(ggml_backend_t backend, ggml_type type, bool clobber_input,
                         ggml_backend_buffer_type_t repack = nullptr, int64_t tokens = 2) {
    ggml_init_params params{1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    require(ctx != nullptr, "shared preparation metadata allocation failed");
    constexpr int64_t columns = 256, rows = 16, experts = 7, active = 3;
    auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, columns, 1, tokens);
    auto * storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, active + 1, tokens);
    auto * ids = ggml_view_2d(ctx.get(), storage, active, tokens, storage->nb[1], 0);
    auto * bias_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, active, tokens);
    auto * bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rows, experts);
    auto * up_weights = ggml_new_tensor_3d(ctx.get(), type, columns, rows, experts);
    auto * gate_weights = ggml_new_tensor_3d(ctx.get(), type, columns, rows, experts);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> packed_buffer(
        repack ? ggml_backend_buft_alloc_buffer(repack, ggml_nbytes(gate_weights) + 256) : nullptr,
        ggml_backend_buffer_free);
    if (repack) {
        require(packed_buffer != nullptr, "shared extra-buffer allocation failed");
        require(ggml_backend_tensor_alloc(packed_buffer.get(), gate_weights,
                                         ggml_backend_buffer_get_base(packed_buffer.get())) == GGML_STATUS_SUCCESS,
                "shared extra-buffer tensor allocation failed");
    }
    auto * up = ggml_add_id(ctx.get(), ggml_mul_mat_id(ctx.get(), up_weights, input, ids), bias, bias_ids);
    auto * gate = ggml_mul_mat_id(ctx.get(), gate_weights, input, ids);
    auto * up_graph = ggml_new_graph_custom(ctx.get(), 32, false);
    auto * gate_graph = ggml_new_graph_custom(ctx.get(), 32, false);
    auto * combined = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(up_graph, up);
    ggml_build_forward_expand(gate_graph, gate);
    ggml_build_forward_expand(combined, up);
    if (clobber_input) ggml_build_forward_expand(combined, ggml_scale_inplace(ctx.get(), input, 2.0f));
    ggml_build_forward_expand(combined, gate);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "shared preparation tensor allocation failed");
    std::vector<float> weights(columns*rows*experts);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.2f*std::sin(float(i + 1));
    for (auto * tensor : {up_weights, gate_weights}) {
        if (type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(tensor, weights.data(), 0, weights.size()*sizeof(float));
        } else {
            std::vector<uint8_t> packed(ggml_nbytes(tensor));
            require(ggml_quantize_chunk(type, weights.data(), packed.data(), 0, rows*experts, columns, nullptr) == packed.size(),
                    "shared preparation quantization size mismatch");
            ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size());
        }
        for (auto & value : weights) value *= -0.7f;
    }
    std::vector<float> biases(rows*experts, 0.03f);
    ggml_backend_tensor_set(bias, biases.data(), 0, biases.size()*sizeof(float));
    std::vector<int32_t> selected(active*tokens);
    for (size_t i = 0; i < selected.size(); ++i) selected[i] = i % experts;
    ggml_backend_tensor_set(bias_ids, selected.data(), 0, selected.size()*sizeof(int32_t));
    using count_fn = uint32_t (*)(void);
    const auto count = reinterpret_cast<count_fn>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "vrambridge_cpu_prepare_count"));
    require(count != nullptr, "shared preparation counter missing");
    const char * flag = std::getenv("VRAMBRIDGE_HYBRID_CPU_PREPARE");
    const bool enabled = flag && std::strcmp(flag, "1") == 0;
    std::vector<float> values(columns*tokens), reference_up(rows*active*tokens), reference_gate(reference_up.size()), actual(reference_up.size());
    for (int repetition = 0; repetition < 3; ++repetition) {
        for (size_t i = 0; i < values.size(); ++i) values[i] = 0.13f*std::cos(float(i + 17*repetition));
        auto mask = selected;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (repetition == 1 && i % 2 == 0) mask[i] = -1;
            if (repetition == 2) mask[i] = -1;
        }
        for (int64_t token = 0; token < tokens; ++token) {
            ggml_backend_tensor_set(ids, mask.data() + token*active, token*ids->nb[1], active*sizeof(int32_t));
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
        const auto before_references = count();
        require(ggml_backend_graph_compute(backend, up_graph) == GGML_STATUS_SUCCESS, "shared up reference failed");
        ggml_backend_tensor_get(up, reference_up.data(), 0, reference_up.size()*sizeof(float));
        if (clobber_input) {
            auto scaled = values;
            for (auto & value : scaled) value *= 2;
            ggml_backend_tensor_set(input, scaled.data(), 0, scaled.size()*sizeof(float));
        }
        require(ggml_backend_graph_compute(backend, gate_graph) == GGML_STATUS_SUCCESS, "shared gate reference failed");
        require(count() == before_references, "preparation leaked between separate graph invocations");
        ggml_backend_tensor_get(gate, reference_gate.data(), 0, reference_gate.size()*sizeof(float));
        ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
        const auto before = count();
        require(ggml_backend_graph_compute(backend, combined) == GGML_STATUS_SUCCESS, "shared combined graph failed");
        if (!enabled || clobber_input || repack) require(count() == before, "preparation reused across a disabled, opaque or mutating path");
        if (enabled && !clobber_input && !repack && (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_Q4_K)) {
            require(count() == before + 1, "raw shared preparation was not exercised");
        }
        for (auto pair : {std::make_pair(up, &reference_up), std::make_pair(gate, &reference_gate)}) {
            ggml_backend_tensor_get(pair.first, actual.data(), 0, actual.size()*sizeof(float));
            for (size_t i = 0; i < actual.size(); ++i) {
                require(std::isfinite(actual[i]) && std::abs(actual[i] - (*pair.second)[i]) <= 1e-6f,
                        "shared preparation changed expert math or reused stale activations/IDs");
            }
        }
    }
}

int main() {
    try {
        ggml_backend_load_all_from_path(VRAMBRIDGE_LLAMA_LIBRARY_DIR);
        ggml_backend_dev_t device = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                device = ggml_backend_dev_get(i);
                break;
            }
        }
        require(device != nullptr, "CPU backend missing");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_dev_init(device, nullptr), ggml_backend_free);
        require(backend != nullptr, "CPU backend initialization failed");
        auto threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device), "ggml_backend_set_n_threads"));
        require(threads != nullptr, "CPU thread control missing");
        auto extra_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device), "ggml_backend_dev_get_extra_bufts"));
        require(extra_bufts != nullptr, "CPU extra buffer discovery missing");
        ggml_backend_buffer_type_t repack = nullptr;
        for (auto * buft = extra_bufts(device); buft && *buft; ++buft) {
            if (std::strcmp(ggml_backend_buft_name(*buft), "CPU_REPACK") == 0) repack = *buft;
        }
        require(repack != nullptr, "CPU repack buffer missing (test requires an AVX2 CPU)");
        for (int count : {1, 4}) {
            threads(backend.get(), count);
            for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0,
                              GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, GGML_TYPE_MXFP4}) {
                check(backend.get(), type, 4, 4, 2);
                check(backend.get(), type, 8, 2, 1);
                check(backend.get(), type, 16, 8, 3);
                check(backend.get(), type, 7, 1, 2);
                check(backend.get(), type, 7, 3, 2, nullptr, true);
                const char * empty = std::getenv("VRAMBRIDGE_HYBRID_CPU_EMPTY");
                if (!empty || std::strcmp(empty, "1") != 0) {
                    check_shared(backend.get(), type, false);
                    check_shared(backend.get(), type, true);
                }
            }
            check(backend.get(), GGML_TYPE_MXFP4, 8, 2, 3, repack);
            check_shared(backend.get(), GGML_TYPE_MXFP4, false, repack);
            check_shared(backend.get(), GGML_TYPE_Q4_K, false, nullptr, 64);
        }
        std::cout << "PASS: seven raw types, variable expert/top-k/token counts, masked outputs, lossless MXFP4 export, 1/4 threads\n";
        return 0;
    } catch (const std::exception & exc) {
        std::cerr << "masked matmul test: " << exc.what() << '\n';
        return 1;
    }
}
