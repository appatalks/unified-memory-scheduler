#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static Context context() {
    Context result(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    require(bool(result), "Cannot allocate fixture metadata");
    return result;
}

static void dispatch(ggml_tensor * dst, int ith, int, void * fast) {
    if (ith != 0) return;
    static_cast<int32_t *>(dst->src[1]->data)[2] = 7;
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::microseconds(fast ? 0 : 500);
    while (std::chrono::steady_clock::now() < deadline) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    *static_cast<float *>(dst->data) = 1;
}

int main(int argc, char ** argv) {
    try {
        const bool no_dispatch = argc == 2 && std::string(argv[1]) == "--no-dispatch";
        const bool unload_only = argc == 2 && std::string(argv[1]) == "--unload-only";
        require(argc == 1 || no_dispatch || unload_only, "Unknown fixture argument");
        ggml_backend_load_all_from_path(VRAMBRIDGE_LLAMA_LIBRARY_DIR);
        const auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(device != nullptr, "No CPU fixture backend");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_dev_init(device, nullptr), ggml_backend_free);
        require(bool(cpu), "Cannot initialize fixture backend");
        std::vector<float> outputs;
        for (int model = 0; model < 3; ++model) {
            auto weights = context(), work = context();
            auto * input = ggml_new_tensor_3d(work.get(), GGML_TYPE_F32, 64, 1, 1);
            auto * ids = ggml_new_tensor_2d(work.get(), GGML_TYPE_I32, 2, 1);
            auto * cpu_ids = ggml_new_tensor_2d(work.get(), GGML_TYPE_I32, 2, 1);
            auto * map = ggml_new_tensor_2d(work.get(), GGML_TYPE_I32, 1, 4);
            ggml_set_name(map, "moe_cache.slot_map-0");
            ggml_tensor * sources[]{ids, map};
            auto * marker = ggml_custom_4d(work.get(), GGML_TYPE_F32, 1, 1, 1, 1,
                                          sources, 2, dispatch, 1, unload_only ? ids : nullptr);
            auto * expert_graph = ggml_new_graph_custom(work.get(), 32, false);
            auto * dispatch_graph = ggml_new_graph_custom(work.get(), 16, false);
            ggml_build_forward_expand(dispatch_graph, marker);
            const char * names[]{"gate", "up", "down"};
            ggml_tensor * tensors[3], * projections[3];
            for (int part = 0; part < 3; ++part) {
                tensors[part] = ggml_new_tensor_3d(weights.get(), GGML_TYPE_MXFP4, 64, 1024, 4);
                ggml_format_name(tensors[part], "blk.0.ffn_%s_exps.weight", names[part]);
                projections[part] = ggml_mul_mat_id(work.get(), tensors[part], input, cpu_ids);
                ggml_set_output(projections[part]);
                ggml_build_forward_expand(expert_graph, projections[part]);
            }
            Buffer weight_buffer(ggml_backend_alloc_ctx_tensors(weights.get(), cpu.get()), ggml_backend_buffer_free);
            Buffer work_buffer(ggml_backend_alloc_ctx_tensors(work.get(), cpu.get()), ggml_backend_buffer_free);
            require(bool(weight_buffer) && bool(work_buffer), "Cannot allocate fixture buffers");
            ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (int part = 0; part < 3; ++part) {
                std::vector<uint8_t> packed(ggml_nbytes(tensors[part]));
                for (size_t block = 0; block < packed.size() / 17; ++block) {
                    packed[block * 17] = 123 + model;
                    for (int i = 1; i < 17; ++i) packed[block * 17 + i] = uint8_t(block + part + i);
                }
                ggml_backend_tensor_set(tensors[part], packed.data(), 0, packed.size());
            }
            std::vector<float> activation(64);
            for (int i = 0; i < 64; ++i) activation[i] = 0.2f * std::sin(float(i + model));
            const int32_t chosen[]{0, 2};
            const int32_t cold[]{0, -1};
            ggml_backend_tensor_set(input, activation.data(), 0, ggml_nbytes(input));
            ggml_backend_tensor_set(ids, chosen, 0, sizeof(chosen));
            ggml_backend_tensor_set(cpu_ids, cold, 0, sizeof(cold));
            require(ggml_backend_graph_compute(cpu.get(), expert_graph) == GGML_STATUS_SUCCESS, "Fixture baseline failed");
            std::vector<float> expected(3 * 2048), actual(expected.size());
            for (int part = 0; part < 3; ++part) {
                ggml_backend_tensor_get(projections[part], expected.data() + part * 2048, 0, 2048 * sizeof(float));
            }
            require(std::all_of(expected.begin(), expected.end(), [](float v) { return std::isfinite(v); }),
                    "Fixture baseline is nonfinite");
            for (int iteration = 0; iteration < 20; ++iteration) {
                const int32_t slots[]{-1, 3, -1, 9};
                ggml_backend_tensor_set(map, slots, 0, sizeof(slots));
                if (!no_dispatch) {
                    require(ggml_backend_graph_compute(cpu.get(), dispatch_graph) == GGML_STATUS_SUCCESS,
                            "Fixture dispatcher failed");
                }
                if (unload_only) break;
                require(ggml_backend_graph_compute(cpu.get(), expert_graph) == GGML_STATUS_SUCCESS, "Fixture expert failed");
                for (int part = 0; part < 3; ++part) {
                    ggml_backend_tensor_get(projections[part], actual.data() + part * 2048, 0, 2048 * sizeof(float));
                }
                require(std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) == 0,
                        "Warmer changed fixture projection bits");
            }
            outputs.insert(outputs.end(), expected.begin(), expected.end());
        }
        uint64_t hash = 14695981039346656037ULL;
        for (float value : outputs) {
            uint8_t bytes[sizeof(value)];
            std::memcpy(bytes, &value, sizeof(value));
            for (uint8_t byte : bytes) hash = (hash ^ byte) * 1099511628211ULL;
        }
        std::puts(nlohmann::json({{"all_checks_passed", true}, {"model_reload_rounds", 3},
            {"output_fingerprint", std::to_string(hash)}}).dump().c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Live warmer fixture: %s\n", error.what());
        return 1;
    }
}
