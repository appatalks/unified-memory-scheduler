#include "common.cuh"
#include "ggml.h"
#include <math_constants.h>

#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

using topk_fn = void (*)(ggml_backend_cuda_context &, ggml_tensor *);

topk_fn original_topk() {
    static const auto fn = [] {
        const char * path = std::getenv("VRAMBRIDGE_PROBE_CUDA_PLUGIN");
        if (!path) {
            std::fprintf(stderr, "qsa top-k diagnostic requires VRAMBRIDGE_PROBE_CUDA_PLUGIN\n");
            std::abort();
        }
        void * handle = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
        if (!handle) {
            std::fprintf(stderr, "qsa top-k diagnostic could not find the loaded CUDA plugin\n");
            std::abort();
        }
        auto result = reinterpret_cast<topk_fn>(dlsym(handle, "_Z18ggml_cuda_op_top_kR25ggml_backend_cuda_contextP11ggml_tensor"));
        if (!result) {
            std::fprintf(stderr, "qsa top-k diagnostic could not resolve the original operator\n");
            std::abort();
        }
        return result;
    }();
    return fn;
}

__device__ bool better(float a, int ia, float b, int ib) {
    return a > b || (a == b && ia < ib);
}

__global__ void stable_topk(const float * source, int32_t * output, int cols, int k, int padded) {
    extern __shared__ unsigned char data[];
    float * values = reinterpret_cast<float *>(data);
    int * ids = reinterpret_cast<int *>(values + padded);
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        values[i] = i < cols ? source[size_t(row) * cols + i] : -CUDART_INF_F;
        ids[i] = i < cols ? i : INT_MAX;
    }
    __syncthreads();
    for (int width = 2; width <= padded; width *= 2) {
        for (int step = width / 2; step > 0; step /= 2) {
            for (int i = threadIdx.x; i < padded; i += blockDim.x) {
                const int other = i ^ step;
                if (other > i) {
                    const bool ascending = (i & width) == 0;
                    const bool swap = ascending ? better(values[other], ids[other], values[i], ids[i])
                                                : better(values[i], ids[i], values[other], ids[other]);
                    if (swap) {
                        const float value = values[i];
                        const int id = ids[i];
                        values[i] = values[other]; ids[i] = ids[other];
                        values[other] = value; ids[other] = id;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x) output[size_t(row) * k + i] = ids[i];
}

} // namespace

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const bool diagnostic_all = std::getenv("VRAMBRIDGE_TOPK_DIAG_ALL") != nullptr;
    const bool qsa = std::strncmp(dst->name, "indexer_top_k-", 14) == 0;
    const int64_t cols = src->ne[0], rows = ggml_nrows(src), k = dst->ne[0];
    if ((!qsa && !diagnostic_all) || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(src) || cols <= 1024 || cols > 4096 || rows <= 1 || rows > 8192 || k <= 0 || k > cols) {
        original_topk()(ctx, dst);
        return;
    }
    int padded = 1;
    while (padded < cols) padded *= 2;
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        std::fprintf(stderr, "[qsa-stable-topk] diagnostic interception active\n");
    }
    if (ctx.device < 0 || ctx.device >= GGML_CUDA_MAX_DEVICES || ctx.curr_stream_no < 0 ||
        ctx.curr_stream_no >= GGML_CUDA_MAX_STREAMS) std::abort();
    auto & stream = ctx.streams[ctx.device][ctx.curr_stream_no];
    if (!stream && (cudaSetDevice(ctx.device) != cudaSuccess ||
                    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)) std::abort();
    stable_topk<<<int(rows), 256, padded * (sizeof(float) + sizeof(int)), stream>>>(
        static_cast<const float *>(src->data), static_cast<int32_t *>(dst->data), int(cols), int(k), padded);
    if (cudaGetLastError() != cudaSuccess) std::abort();
}
