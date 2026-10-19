#include "compatible_down.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct FreeDevice {
    void operator()(void * pointer) const noexcept {
        const auto status = cudaFree(pointer);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "compatible-down cudaFree: %s\n", cudaGetErrorString(status));
        }
    }
};

std::unique_ptr<void, FreeDevice> allocate(size_t bytes) {
    void * pointer = nullptr;
    check(cudaMalloc(&pointer, bytes), "compatible-down cudaMalloc");
    return std::unique_ptr<void, FreeDevice>(pointer);
}

__device__ float scale(const uint8_t * x, const uint8_t * y) {
    const uint16_t bits = static_cast<uint16_t>(y[0]) | (static_cast<uint16_t>(y[1]) << 8);
    const unsigned e = x[0];
    const float exponent = __uint_as_float(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
    return __fmul_rn(__half2float(__ushort_as_half(bits)), exponent);
}

__device__ int quad(const uint8_t * x, const uint8_t * y, int lane) {
    constexpr int table[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int sum = 0;
    for (int j = 0; j < 4; ++j) {
        const int index = 4 * lane + j;
        const int nibble = index < 16 ? x[1 + index] & 15 : x[1 + index - 16] >> 4;
        const int a = table[nibble];
        const int b = static_cast<int>(reinterpret_cast<const int8_t *>(y + 2)[index]);
        // Match VPSIGNB, including its signed-byte -128 behavior.
        const int signed_b = a == 0 ? 0 : a > 0 ? b : b == -128 ? -128 : -b;
        sum += (a < 0 ? -a : a) * signed_b;
    }
    return sum;
}

__device__ int packed_quad(const uint8_t * x, const uint8_t * y, int lane) {
    const auto * codes = x + 1 + 4 * (lane % 4);
    uint32_t nibbles = uint32_t(codes[0]) | (uint32_t(codes[1]) << 8) |
        (uint32_t(codes[2]) << 16) | (uint32_t(codes[3]) << 24);
    if (lane >= 4) nibbles >>= 4;
    nibbles &= 0x0f0f0f0f;
    const uint32_t selectors = (nibbles & 7) | ((nibbles >> 4) & 0x70) |
        ((nibbles >> 8) & 0x700) | ((nibbles >> 12) & 0x7000);
    const uint32_t magnitudes = __byte_perm(0x03020100, 0x0c080604, selectors);
    const auto * input = reinterpret_cast<const uint16_t *>(y + 2 + 4 * lane);
    const uint32_t q = uint32_t(input[0]) | (uint32_t(input[1]) << 16);
    const uint32_t negative = __vcmpeq4(nibbles & 0x08080808, 0x08080808);
    const uint32_t signed_q = (negative & __vneg4(q)) | (~negative & q);
    return __dp4a(static_cast<int>(magnitudes), static_cast<int>(signed_q), 0);
}

template<bool PackedDot>
__global__ void down_kernel(const uint8_t * weights, const uint8_t * input, const int32_t * ids,
                            float * output, int blocks, int outputs, int routes) {
    const int index = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
    const int lane = threadIdx.x % 8;
    if (index >= routes * outputs) return;
    const unsigned mask = __activemask();
    const int route = index / outputs;
    const int row = index % outputs;
    const auto * x = weights + (size_t(ids[route]) * outputs + row) * blocks * 17;
    const auto * y = input + size_t(route) * blocks * 34;
    float even = 0, odd = 0;
    for (int block = 0; block < blocks; block += 2) {
        const auto * x0 = x + block * 17;
        const auto * y0 = y + block * 34;
        const int p0 = PackedDot ? packed_quad(x0, y0, lane) : quad(x0, y0, lane);
        const int p1 = PackedDot ? packed_quad(x0 + 17, y0 + 34, lane) : quad(x0 + 17, y0 + 34, lane);
        even = __fmaf_rn(scale(x0, y0), static_cast<float>(p0), even);
        odd = __fmaf_rn(scale(x0 + 17, y0 + 34),
                        static_cast<float>(p1), odd);
    }
    // Native AVX2 adds paired accumulators, then (4+0,5+1,6+2,7+3), then 2, then 1.
    float sum = __fadd_rn(even, odd);
    sum = __fadd_rn(__shfl_xor_sync(mask, sum, 4, 8), sum);
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 2, 8));
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 1, 8));
    if (lane == 0) output[index] = sum;
}

} // namespace

CompatibleDownResult compatible_down(
    const std::vector<uint8_t> & weights, const std::vector<int32_t> & ids,
    int width, int outputs, int experts, int repetitions, bool upload_weights, bool packed_dot,
    const std::function<void(std::vector<uint8_t> &)> & quantize) {
    if (width <= 0 || width > 2880 || width % 64 != 0 || outputs <= 0 || outputs > 2880 ||
        experts < 1 || experts > 128 || ids.empty() || ids.size() > 1024 ||
        repetitions < 1 || repetitions > 1000 ||
        weights.size() != size_t(experts) * outputs * (width / 32) * 17) {
        throw std::runtime_error("Unsupported compatible-down shape or packed weight size.");
    }
    for (int32_t id : ids) {
        if (id < 0 || id >= experts) throw std::runtime_error("Compatible-down expert ID out of range.");
    }
    check(cudaSetDevice(0), "compatible-down cudaSetDevice");
    const size_t input_bytes = ids.size() * (width / 32) * 34;
    const size_t output_bytes = ids.size() * outputs * sizeof(float);
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total), "compatible-down cudaMemGetInfo");
    if (free < weights.size() + input_bytes + output_bytes + ids.size() * sizeof(int32_t) + 256 * 1024 * 1024) {
        throw std::runtime_error("Insufficient GPU headroom for compatible-down probe.");
    }
    auto device_weights = allocate(weights.size()), device_input = allocate(input_bytes);
    auto device_ids = allocate(ids.size() * sizeof(int32_t)), device_output = allocate(output_bytes);
    check(cudaMemcpy(device_weights.get(), weights.data(), weights.size(), cudaMemcpyHostToDevice), "initial weights");
    CompatibleDownResult result;
    result.output.resize(ids.size() * outputs);
    std::vector<float> first(result.output.size());
    std::vector<uint8_t> packed(input_bytes);
    using Clock = std::chrono::steady_clock;
    for (int iteration = -2; iteration < repetitions; ++iteration) {
        const auto start = Clock::now();
        quantize(packed);
        const auto prepared = Clock::now();
        if (packed.size() != input_bytes) throw std::runtime_error("Compatible-down packed input size mismatch.");
        if (upload_weights) {
            check(cudaMemcpy(device_weights.get(), weights.data(), weights.size(), cudaMemcpyHostToDevice), "weights");
        }
        check(cudaMemcpy(device_input.get(), packed.data(), input_bytes, cudaMemcpyHostToDevice), "packed input");
        check(cudaMemcpy(device_ids.get(), ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "IDs");
        const unsigned threads = 128;
        const unsigned blocks = static_cast<unsigned>((ids.size() * outputs * 8 + threads - 1) / threads);
        if (packed_dot) {
            down_kernel<true><<<blocks, threads>>>(
                static_cast<const uint8_t *>(device_weights.get()), static_cast<const uint8_t *>(device_input.get()),
                static_cast<const int32_t *>(device_ids.get()), static_cast<float *>(device_output.get()),
                width / 32, outputs, static_cast<int>(ids.size()));
        } else {
            down_kernel<false><<<blocks, threads>>>(
                static_cast<const uint8_t *>(device_weights.get()), static_cast<const uint8_t *>(device_input.get()),
                static_cast<const int32_t *>(device_ids.get()), static_cast<float *>(device_output.get()),
                width / 32, outputs, static_cast<int>(ids.size()));
        }
        check(cudaGetLastError(), "compatible-down launch");
        check(cudaMemcpy(result.output.data(), device_output.get(), output_bytes, cudaMemcpyDeviceToHost), "raw output");
        check(cudaDeviceSynchronize(), "compatible-down synchronize");
        const auto end = Clock::now();
        if (iteration == -2) first = result.output;
        else result.all_iterations_identical &= std::memcmp(first.data(), result.output.data(), output_bytes) == 0;
        if (iteration >= 0) {
            result.samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            result.quantize_ms.push_back(std::chrono::duration<double, std::milli>(prepared - start).count());
        }
    }
    return result;
}
