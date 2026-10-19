#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <memory>
#include <stdexcept>

static void dispatch(ggml_tensor * dst, int ith, int, void *) {
    if (ith != 0) return;
    static_cast<int32_t *>(dst->src[1]->data)[2] = 7;
    *static_cast<float *>(dst->data) = 1;
}

int main() {
    try {
        ggml_backend_load_all_from_path(VRAMBRIDGE_LLAMA_LIBRARY_DIR);
        const auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!device) throw std::runtime_error("No CPU backend for the dispatcher fixture");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_dev_init(device, nullptr), ggml_backend_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        if (!cpu || !context) throw std::runtime_error("Cannot allocate dispatcher fixture");
        auto * ids = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, 2, 1);
        auto * map = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, 1, 4);
        ggml_set_name(map, "moe_cache.slot_map-0");
        ggml_tensor * sources[]{ids, map};
        auto * output = ggml_custom_4d(context.get(), GGML_TYPE_F32, 1, 1, 1, 1,
                                      sources, 2, dispatch, 1, nullptr);
        auto * graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(context.get(), cpu.get()), ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("Cannot allocate dispatcher fixture tensors");
        const int32_t selected[]{0, 2}, slots[]{3, -1, -1, 9};
        ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
        ggml_backend_tensor_set(map, slots, 0, sizeof(slots));
        if (ggml_backend_graph_compute(cpu.get(), graph) != GGML_STATUS_SUCCESS ||
            static_cast<int32_t *>(map->data)[2] != 7) {
            throw std::runtime_error("Dispatcher fixture did not execute");
        }
        std::puts("pre-admission dispatcher fixture passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "CPU hint fixture: %s\n", error.what());
        return 1;
    }
}
