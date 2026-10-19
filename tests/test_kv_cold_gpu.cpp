#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cuda_runtime_api.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "explicit CUDA plugin required");
        auto * registration = ggml_backend_load(argv[1]);
        require(registration && ggml_backend_reg_dev_count(registration) > 0, "CUDA backend unavailable");
        auto * device = ggml_backend_reg_dev_get(registration, 0);
        auto * type = ggml_backend_dev_buffer_type(device);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> kv(
            ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        auto * key = ggml_new_tensor_3d(kv.get(), GGML_TYPE_F16, 128, 512, 1);
        auto * value = ggml_new_tensor_3d(kv.get(), GGML_TYPE_F16, 128, 512, 1);
        ggml_set_name(key, "cache_k_l0");
        ggml_set_name(value, "cache_v_l0");
        ggml_set_name(ggml_view_1d(kv.get(), key, 128, 0), "key_view");
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(kv.get(), type);
        require(buffer != nullptr, "KV allocation failed");
        cudaPointerAttributes attributes{};
        require(cudaPointerGetAttributes(&attributes, key->data) == cudaSuccess &&
                attributes.type == cudaMemoryTypeManaged, "KV allocation is not managed");
        ggml_backend_buffer_clear(buffer, 0);
        int last_prefetch = 0;
        require(cudaMemRangeGetAttribute(&last_prefetch, sizeof(last_prefetch),
                cudaMemRangeAttributeLastPrefetchLocation, key->data, ggml_nbytes(key)) == cudaSuccess &&
                last_prefetch == cudaCpuDeviceId, "KV clear did not complete CPU prefetch");
        std::vector<uint16_t> input(128 * 512), output(input.size());
        for (size_t i = 0; i < input.size(); ++i) input[i] = uint16_t(i % 1024);
        ggml_backend_tensor_set(key, input.data(), 0, input.size() * sizeof(uint16_t));
        ggml_backend_tensor_get(key, output.data(), 0, output.size() * sizeof(uint16_t));
        require(input == output, "KV bytes changed during CPU staging/GPU access");
        const size_t bytes = ggml_backend_buffer_get_size(buffer);
        ggml_backend_buffer_free(buffer);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(
            ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        auto * weight = ggml_new_tensor_2d(weights.get(), GGML_TYPE_F16, 128, 128);
        ggml_set_name(weight, "blk.0.attn_k.weight");
        auto * mixed_key = ggml_new_tensor_3d(weights.get(), GGML_TYPE_F16, 128, 512, 1);
        ggml_set_name(mixed_key, "cache_k_l0");
        ggml_set_name(ggml_new_tensor_3d(weights.get(), GGML_TYPE_F16, 128, 512, 1), "cache_v_l0");
        auto * regular = ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), type);
        require(regular != nullptr, "weight allocation failed");
        require(cudaPointerGetAttributes(&attributes, weight->data) == cudaSuccess &&
                attributes.type == cudaMemoryTypeDevice, "weight allocation was incorrectly made managed");
        require(cudaPointerGetAttributes(&attributes, mixed_key->data) == cudaSuccess &&
                attributes.type == cudaMemoryTypeDevice, "mixed context was incorrectly made managed");
        ggml_backend_buffer_clear(regular, 0);
        ggml_backend_buffer_free(regular);
        std::printf("{\"roundtrip\":true,\"kv_buffer_bytes\":%zu,\"weights_remained_device\":true}\n", bytes);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "cold KV GPU test: %s\n", error.what());
        return 1;
    }
}
