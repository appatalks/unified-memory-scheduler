#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void control(ggml_tensor * dst, int ith, int, void *) {
    if (ith) return;
    const auto * ids = static_cast<const int32_t *>(dst->src[0]->data);
    const auto * values = static_cast<const int32_t *>(dst->src[1]->data);
    *static_cast<float *>(dst->data) = float(values[ids[0]] + values[ids[1]]);
}

int main(int argc, char ** argv) {
    try {
        const bool large_copy = argc == 2 && std::strcmp(argv[1], "--large-copy") == 0;
        const bool batch_two = argc == 2 && std::strcmp(argv[1], "--batch-two") == 0;
        require(argc == 1 || large_copy || batch_two, "Unknown fixture option");
        ggml_backend_load_all_from_path(VRAMBRIDGE_LLAMA_LIBRARY_DIR);
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(device != nullptr, "No CPU backend");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_dev_init(device, nullptr), ggml_backend_free);
        require(bool(cpu), "CPU initialization failed");
        const auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_set_n_threads"));
        require(set_threads != nullptr, "CPU thread control unavailable");
        set_threads(cpu.get(), 8);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        require(bool(ctx), "Fixture allocation failed");
        auto * raw = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, large_copy ? 20000 : 2, batch_two ? 2 : 1);
        auto * ids = ggml_cast(ctx.get(), raw, GGML_TYPE_I32);
        auto * map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 4);
        ggml_set_name(map, "moe_cache.slot_map-0");
        ggml_tensor * inputs[]{ids, map};
        auto * output = ggml_custom_4d(ctx.get(), GGML_TYPE_F32, 1, 1, 1, 1, inputs, 2, control, 1, nullptr);
        auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, output);
        auto * numeric = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1024);
        auto * doubled = ggml_add(ctx.get(), numeric, numeric);
        auto * other = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(other, doubled);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()), ggml_backend_buffer_free);
        require(bool(buffer), "Cannot allocate tensors");
        const int32_t values[]{3, 7, 11, 19};
        std::vector<float> numbers(1024);
        for (int i = 0; i < 1024; ++i) numbers[i] = 0.1f * std::sin(float(i));
        ggml_backend_tensor_set(map, values, 0, sizeof(values));
        ggml_backend_tensor_set(numeric, numbers.data(), 0, ggml_nbytes(numeric));
        const std::vector<float> zeros(ggml_nelements(raw), 0);
        ggml_backend_tensor_set(raw, zeros.data(), 0, ggml_nbytes(raw));
        for (int iteration = 0; iteration < 12; ++iteration) {
            const float selected[]{float(iteration % 4), float((iteration + 1) % 4)};
            ggml_backend_tensor_set(raw, selected, 0, sizeof(selected));
            require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS, "Control graph failed");
            float result;
            ggml_backend_tensor_get(output, &result, 0, sizeof(result));
            require(result == values[int(selected[0])] + values[int(selected[1])], "Control result changed");
            require(ggml_backend_graph_compute(cpu.get(), other) == GGML_STATUS_SUCCESS, "Other graph failed");
            std::vector<float> actual(1024);
            ggml_backend_tensor_get(doubled, actual.data(), 0, ggml_nbytes(doubled));
            for (int i = 0; i < 1024; ++i) require(actual[i] == numbers[i] + numbers[i], "Other graph changed");
        }
        std::puts("{\"all_checks_passed\":true}");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Control-thread fixture: %s\n", error.what());
        return 1;
    }
}
