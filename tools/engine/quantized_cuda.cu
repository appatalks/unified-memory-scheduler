#include "quantized_cuda.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace vrambridge::experimental {
namespace {

constexpr std::size_t reserve_bytes = 1024ULL * 1024 * 1024;
constexpr std::size_t max_floats = 32 * 1024 * 1024;

void checked(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void require_space(std::size_t bytes) {
    std::size_t free = 0, total = 0;
    checked(cudaMemGetInfo(&free, &total), "GPU memory query failed");
    if (free < reserve_bytes || bytes > free - reserve_bytes)
        throw std::runtime_error("IQ4 expert allocation would violate the 1 GiB GPU reserve");
}

template<class T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t count) { allocate(count); }
    ~DeviceArray() { if (data_) cudaFree(data_); }
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray & operator=(const DeviceArray &) = delete;
    void allocate(std::size_t count) {
        if (data_) throw std::logic_error("GPU buffer already allocated");
        if (count) checked(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)), "GPU allocation failed");
    }
    T * get() const noexcept { return data_; }
private:
    T * data_ = nullptr;
};

template<class T>
void upload(DeviceArray<T> & destination, const std::vector<T> & source) {
    destination.allocate(source.size());
    if (!source.empty()) checked(cudaMemcpy(destination.get(), source.data(), source.size() * sizeof(T),
                                            cudaMemcpyHostToDevice), "GPU expert upload failed");
}

__device__ float half_value(const std::uint8_t * bytes) {
    const std::uint16_t bits = static_cast<std::uint16_t>(bytes[0]) | (static_cast<std::uint16_t>(bytes[1]) << 8);
    return __half2float(__ushort_as_half(bits));
}

__device__ int iq4_quad(const std::uint8_t * x, const std::uint8_t * y, int lane) {
    constexpr int values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    int total = 0;
    for (int j = 0; j < 4; ++j) {
        const int index = 4 * lane + j;
        const int packed = x[2 + index % 16];
        const int weight = values[index < 16 ? packed & 15 : packed >> 4];
        const int activation = static_cast<int>(reinterpret_cast<const std::int8_t *>(y + 2)[index]);
        const int signed_activation = weight >= 0 ? activation : activation == -128 ? -128 : -activation;
        total += (weight < 0 ? -weight : weight) * signed_activation;
    }
    return total;
}

__global__ void iq4_down_kernel(const std::uint8_t * weights, const std::uint8_t * input,
                                const std::uint32_t * ids, const float * bias, float * output,
                                int blocks, int hidden, int jobs, bool has_bias) {
    const int index = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
    const int lane = threadIdx.x % 8;
    if (index >= jobs * hidden) return;
    const auto mask = __activemask();
    const int job = index / hidden, row = index % hidden;
    const auto * matrix = weights + (static_cast<std::size_t>(ids[job]) * hidden + row) * blocks * 18;
    const auto * activation = input + static_cast<std::size_t>(job) * blocks * 34;
    float even = 0, odd = 0;
    for (int block = 0; block < blocks; block += 2) {
        const auto * x0 = matrix + block * 18;
        const auto * y0 = activation + block * 34;
        const float scale0 = __fmul_rn(half_value(y0), half_value(x0));
        const float scale1 = __fmul_rn(half_value(y0 + 34), half_value(x0 + 18));
        even = __fmaf_rn(scale0, static_cast<float>(iq4_quad(x0, y0, lane)), even);
        odd = __fmaf_rn(scale1, static_cast<float>(iq4_quad(x0 + 18, y0 + 34, lane)), odd);
    }
    float sum = __fadd_rn(even, odd);
    sum = __fadd_rn(__shfl_xor_sync(mask, sum, 4, 8), sum);
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 2, 8));
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 1, 8));
    if (lane == 0) output[index] = has_bias ? __fadd_rn(sum, bias[static_cast<std::size_t>(ids[job]) * hidden + row]) : sum;
}

__global__ void q6_down_kernel(const std::uint8_t * weights, const std::uint8_t * input,
                               const std::uint32_t * ids, const float * bias, float * output,
                               int blocks, int hidden, int jobs, bool has_bias) {
    const int index = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
    const int lane = threadIdx.x % 8;
    if (index >= jobs * hidden) return;
    const auto mask = __activemask();
    const int job = index / hidden, row = index % hidden;
    const auto * matrix = weights + (static_cast<std::size_t>(ids[job]) * hidden + row) * blocks * 210;
    const auto * activation = input + static_cast<std::size_t>(job) * blocks * 292;
    float accumulated = 0;
    for (int block = 0; block < blocks; ++block) {
        const auto * x = matrix + block * 210;
        const auto * y = activation + block * 292;
        const auto * scales = reinterpret_cast<const std::int8_t *>(x + 192);
        const auto * codes = reinterpret_cast<const std::int8_t *>(y + 4);
        int integer = 0;
        for (int half = 0; half < 2; ++half) {
            for (int chunk = 0; chunk < 4; ++chunk) {
                const int scale = scales[half * 8 + chunk * 2 + lane / 4];
                for (int j = 0; j < 4; ++j) {
                    const int offset = lane * 4 + j;
                    const auto low = x[half * 64 + (chunk % 2) * 32 + offset];
                    const auto high = x[128 + half * 32 + offset];
                    const int q6 = (chunk < 2 ? low & 15 : low >> 4) | (((high >> (chunk * 2)) & 3) << 4);
                    integer += scale * q6 * static_cast<int>(codes[half * 128 + chunk * 32 + offset]);
                }
            }
        }
        const auto * sums = y + 260 + lane * 4;
        const int sum0 = static_cast<std::int16_t>(static_cast<std::uint16_t>(sums[0]) | (static_cast<std::uint16_t>(sums[1]) << 8));
        const int sum1 = static_cast<std::int16_t>(static_cast<std::uint16_t>(sums[2]) | (static_cast<std::uint16_t>(sums[3]) << 8));
        integer -= 32 * (sum0 * scales[lane * 2] + sum1 * scales[lane * 2 + 1]);
        const std::uint32_t scale_bits = static_cast<std::uint32_t>(y[0]) | (static_cast<std::uint32_t>(y[1]) << 8) |
            (static_cast<std::uint32_t>(y[2]) << 16) | (static_cast<std::uint32_t>(y[3]) << 24);
        const float scale = __fmul_rn(__uint_as_float(scale_bits), half_value(x + 208));
        accumulated = __fmaf_rn(scale, static_cast<float>(integer), accumulated);
    }
    float sum = __fadd_rn(__shfl_xor_sync(mask, accumulated, 4, 8), accumulated);
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 2, 8));
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 1, 8));
    if (lane == 0) output[index] = has_bias ? __fadd_rn(sum, bias[static_cast<std::size_t>(ids[job]) * hidden + row]) : sum;
}

__device__ int mxfp4_quad(const std::uint8_t * x, const std::uint8_t * y, int lane) {
    constexpr int values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int total = 0;
    for (int j = 0; j < 4; ++j) {
        const int index = 4 * lane + j;
        const int packed = x[1 + index % 16];
        const int weight = values[index < 16 ? packed & 15 : packed >> 4];
        const int activation = static_cast<int>(reinterpret_cast<const std::int8_t *>(y + 2)[index]);
        const int signed_activation = weight == 0 ? 0 : weight > 0 ? activation : activation == -128 ? -128 : -activation;
        total += (weight < 0 ? -weight : weight) * signed_activation;
    }
    return total;
}

__device__ float mxfp4_scale(const std::uint8_t * x, const std::uint8_t * y) {
    const unsigned exponent = x[0];
    const float value = __uint_as_float(exponent < 2 ? (0x00200000u << exponent) : ((exponent - 1) << 23));
    return __fmul_rn(half_value(y), value);
}

__global__ void mxfp4_down_kernel(const std::uint8_t * weights, const std::uint8_t * input,
                                    const std::uint32_t * ids, const float * bias, float * output,
                                    int blocks, int hidden, int jobs, bool has_bias) {
    const int index = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
    const int lane = threadIdx.x % 8;
    if (index >= jobs * hidden) return;
    const auto mask = __activemask();
    const int job = index / hidden, row = index % hidden;
    const auto * matrix = weights + (static_cast<std::size_t>(ids[job]) * hidden + row) * blocks * 17;
    const auto * activation = input + static_cast<std::size_t>(job) * blocks * 34;
    float even = 0, odd = 0;
    for (int block = 0; block < blocks; block += 2) {
        const auto * x = matrix + block * 17;
        const auto * y = activation + block * 34;
        even = __fmaf_rn(mxfp4_scale(x, y), static_cast<float>(mxfp4_quad(x, y, lane)), even);
        odd = __fmaf_rn(mxfp4_scale(x + 17, y + 34), static_cast<float>(mxfp4_quad(x + 17, y + 34, lane)), odd);
    }
    float sum = __fadd_rn(even, odd);
    sum = __fadd_rn(__shfl_xor_sync(mask, sum, 4, 8), sum);
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 2, 8));
    sum = __fadd_rn(sum, __shfl_down_sync(mask, sum, 1, 8));
    if (lane == 0) output[index] = has_bias ? __fadd_rn(sum, bias[static_cast<std::size_t>(ids[job]) * hidden + row]) : sum;
}

} // namespace

struct Iq4NativeCudaDown::Impl {
    int device;
    std::size_t hidden, intermediate;
    bool has_bias;
    std::vector<int> slots;
    DeviceArray<std::uint8_t> down;
    DeviceArray<float> bias;
    cudaStream_t stream = nullptr;
    std::mutex mutex;

    Impl(const LoadedExperts & loaded, const std::vector<std::uint8_t> & resident, int device_index)
        : device(device_index), hidden(loaded.weights.hidden), intermediate(loaded.weights.intermediate),
          has_bias(!loaded.weights.down_bias.empty()), slots(loaded.weights.experts, -1) {
        loaded.weights.validate();
        if (device < 0 || resident.size() != loaded.weights.experts || loaded.quant_types[2] != GGML_TYPE_IQ4_NL ||
            intermediate % 64 != 0) throw std::invalid_argument("unsupported IQ4 native GPU configuration");
        const auto stride = hidden * (intermediate / 32) * 18;
        if (loaded.packed_weights[2].size() != stride * loaded.weights.experts)
            throw std::invalid_argument("invalid IQ4 native packed weights");
        checked(cudaSetDevice(device), "GPU device selection failed");
        std::vector<std::uint8_t> selected;
        std::vector<float> selected_bias;
        for (std::size_t expert = 0; expert < resident.size(); ++expert) {
            if (resident[expert] > 1) throw std::invalid_argument("invalid GPU residency value");
            if (!resident[expert]) continue;
            slots[expert] = static_cast<int>(selected_bias.size() / hidden);
            selected.insert(selected.end(), loaded.packed_weights[2].begin() + expert * stride,
                            loaded.packed_weights[2].begin() + (expert + 1) * stride);
            for (std::size_t i = 0; i < hidden; ++i)
                selected_bias.push_back(has_bias ? loaded.weights.down_bias[expert * hidden + i] : 0.0f);
        }
        if (selected.empty()) throw std::invalid_argument("no GPU-resident experts");
        require_space(selected.size() + selected_bias.size() * sizeof(float));
        upload(down, selected); upload(bias, selected_bias);
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "GPU stream creation failed");
    }
    ~Impl() {
        cudaSetDevice(device);
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }
};

Iq4NativeCudaDown::Iq4NativeCudaDown(const LoadedExperts & loaded, const std::vector<std::uint8_t> & resident, int device)
    : impl_(std::make_unique<Impl>(loaded, resident, device)) {}
Iq4NativeCudaDown::~Iq4NativeCudaDown() = default;

std::vector<float> Iq4NativeCudaDown::execute(const std::vector<std::uint8_t> & packed_q8,
                                               const std::vector<ExpertJob> & jobs) {
    std::lock_guard lock(impl_->mutex);
    checked(cudaSetDevice(impl_->device), "GPU device selection failed");
    const auto hidden = impl_->hidden, blocks = impl_->intermediate / 32;
    const auto stride = blocks * 34;
    if (jobs.size() > max_floats / hidden || jobs.size() > max_floats / impl_->intermediate ||
        packed_q8.size() != jobs.size() * stride) throw std::invalid_argument("invalid packed Q8 input shape");
    if (jobs.empty()) return {};
    std::vector<std::uint32_t> ids;
    for (const auto & job : jobs) {
        if (job.expert >= impl_->slots.size() || impl_->slots[job.expert] < 0)
            throw std::invalid_argument("IQ4 GPU expert is out of range or not resident");
        ids.push_back(static_cast<std::uint32_t>(impl_->slots[job.expert]));
    }
    require_space(packed_q8.size() + ids.size() * sizeof(std::uint32_t) + jobs.size() * hidden * sizeof(float));
    DeviceArray<std::uint8_t> input(packed_q8.size());
    DeviceArray<std::uint32_t> device_ids(ids.size());
    DeviceArray<float> result(jobs.size() * hidden);
    std::vector<float> output(jobs.size() * hidden);
    try {
        checked(cudaMemcpyAsync(input.get(), packed_q8.data(), packed_q8.size(), cudaMemcpyHostToDevice, impl_->stream),
                "packed Q8 upload failed");
        checked(cudaMemcpyAsync(device_ids.get(), ids.data(), ids.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                                impl_->stream), "IQ4 route upload failed");
        constexpr int threads = 128;
        const auto blocks_grid = static_cast<unsigned>((jobs.size() * hidden * 8 + threads - 1) / threads);
        iq4_down_kernel<<<blocks_grid, threads, 0, impl_->stream>>>(impl_->down.get(), input.get(), device_ids.get(),
            impl_->bias.get(), result.get(), static_cast<int>(blocks), static_cast<int>(hidden), static_cast<int>(jobs.size()), impl_->has_bias);
        checked(cudaGetLastError(), "IQ4 down launch failed");
        checked(cudaMemcpyAsync(output.data(), result.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream),
                "IQ4 output download failed");
        checked(cudaStreamSynchronize(impl_->stream), "IQ4 down execution failed");
    } catch (...) {
        cudaStreamSynchronize(impl_->stream);
        throw;
    }
    if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("IQ4 GPU produced non-finite output");
    return output;
}

struct Q6NativeCudaDown::Impl {
    int device;
    std::size_t hidden, intermediate;
    bool has_bias;
    std::vector<int> slots;
    DeviceArray<std::uint8_t> down;
    DeviceArray<float> bias;
    cudaStream_t stream = nullptr;
    std::mutex mutex;

    Impl(const LoadedExperts & loaded, const std::vector<std::uint8_t> & resident, int device_index)
        : device(device_index), hidden(loaded.weights.hidden), intermediate(loaded.weights.intermediate),
          has_bias(!loaded.weights.down_bias.empty()), slots(loaded.weights.experts, -1) {
        loaded.weights.validate();
        if (device < 0 || resident.size() != loaded.weights.experts || loaded.quant_types[2] != GGML_TYPE_Q6_K ||
            intermediate % 256 != 0) throw std::invalid_argument("unsupported Q6 native GPU configuration");
        const auto stride = hidden * (intermediate / 256) * 210;
        if (loaded.packed_weights[2].size() != stride * loaded.weights.experts)
            throw std::invalid_argument("invalid Q6 native packed weights");
        checked(cudaSetDevice(device), "GPU device selection failed");
        std::vector<std::uint8_t> selected;
        std::vector<float> selected_bias;
        for (std::size_t expert = 0; expert < resident.size(); ++expert) {
            if (resident[expert] > 1) throw std::invalid_argument("invalid GPU residency value");
            if (!resident[expert]) continue;
            slots[expert] = static_cast<int>(selected_bias.size() / hidden);
            selected.insert(selected.end(), loaded.packed_weights[2].begin() + expert * stride,
                            loaded.packed_weights[2].begin() + (expert + 1) * stride);
            for (std::size_t i = 0; i < hidden; ++i)
                selected_bias.push_back(has_bias ? loaded.weights.down_bias[expert * hidden + i] : 0.0f);
        }
        if (selected.empty()) throw std::invalid_argument("no GPU-resident experts");
        require_space(selected.size() + selected_bias.size() * sizeof(float));
        upload(down, selected); upload(bias, selected_bias);
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "GPU stream creation failed");
    }
    ~Impl() {
        cudaSetDevice(device);
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }
};

Q6NativeCudaDown::Q6NativeCudaDown(const LoadedExperts & loaded, const std::vector<std::uint8_t> & resident, int device)
    : impl_(std::make_unique<Impl>(loaded, resident, device)) {}
Q6NativeCudaDown::~Q6NativeCudaDown() = default;

std::vector<float> Q6NativeCudaDown::execute(const std::vector<std::uint8_t> & packed_q8,
                                              const std::vector<ExpertJob> & jobs) {
    std::lock_guard lock(impl_->mutex);
    checked(cudaSetDevice(impl_->device), "GPU device selection failed");
    const auto hidden = impl_->hidden, blocks = impl_->intermediate / 256;
    const auto stride = blocks * 292;
    if (jobs.size() > max_floats / hidden || jobs.size() > max_floats / impl_->intermediate ||
        packed_q8.size() != jobs.size() * stride) throw std::invalid_argument("invalid packed Q8_K input shape");
    if (jobs.empty()) return {};
    std::vector<std::uint32_t> ids;
    for (const auto & job : jobs) {
        if (job.expert >= impl_->slots.size() || impl_->slots[job.expert] < 0)
            throw std::invalid_argument("Q6 GPU expert is out of range or not resident");
        ids.push_back(static_cast<std::uint32_t>(impl_->slots[job.expert]));
    }
    require_space(packed_q8.size() + ids.size() * sizeof(std::uint32_t) + jobs.size() * hidden * sizeof(float));
    DeviceArray<std::uint8_t> input(packed_q8.size());
    DeviceArray<std::uint32_t> device_ids(ids.size());
    DeviceArray<float> result(jobs.size() * hidden);
    std::vector<float> output(jobs.size() * hidden);
    try {
        checked(cudaMemcpyAsync(input.get(), packed_q8.data(), packed_q8.size(), cudaMemcpyHostToDevice, impl_->stream),
                "packed Q8_K upload failed");
        checked(cudaMemcpyAsync(device_ids.get(), ids.data(), ids.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                                impl_->stream), "Q6 route upload failed");
        constexpr int threads = 128;
        const auto blocks_grid = static_cast<unsigned>((jobs.size() * hidden * 8 + threads - 1) / threads);
        q6_down_kernel<<<blocks_grid, threads, 0, impl_->stream>>>(impl_->down.get(), input.get(), device_ids.get(),
            impl_->bias.get(), result.get(), static_cast<int>(blocks), static_cast<int>(hidden), static_cast<int>(jobs.size()), impl_->has_bias);
        checked(cudaGetLastError(), "Q6 down launch failed");
        checked(cudaMemcpyAsync(output.data(), result.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream),
                "Q6 output download failed");
        checked(cudaStreamSynchronize(impl_->stream), "Q6 down execution failed");
    } catch (...) {
        cudaStreamSynchronize(impl_->stream);
        throw;
    }
    if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Q6 GPU produced non-finite output");
    return output;
}

struct Mxfp4NativeCudaDown::Impl {
    int device;
    std::size_t hidden, intermediate;
    bool has_bias;
    std::vector<int> slots;
    DeviceArray<std::uint8_t> down;
    DeviceArray<float> bias;
    cudaStream_t stream = nullptr;
    std::mutex mutex;

    Impl(const LoadedExperts & loaded, const std::vector<std::uint8_t> & resident, int device_index)
        : device(device_index), hidden(loaded.weights.hidden), intermediate(loaded.weights.intermediate),
          has_bias(!loaded.weights.down_bias.empty()), slots(loaded.weights.experts, -1) {
        loaded.weights.validate();
        if (device < 0 || resident.size() != loaded.weights.experts || loaded.quant_types[2] != GGML_TYPE_MXFP4 ||
            intermediate % 64 != 0) throw std::invalid_argument("unsupported MXFP4 native GPU configuration");
        const auto stride = hidden * (intermediate / 32) * 17;
        if (loaded.packed_weights[2].size() != stride * loaded.weights.experts)
            throw std::invalid_argument("invalid MXFP4 native packed weights");
        checked(cudaSetDevice(device), "GPU device selection failed");
        std::vector<std::uint8_t> selected;
        std::vector<float> selected_bias;
        for (std::size_t expert = 0; expert < resident.size(); ++expert) {
            if (resident[expert] > 1) throw std::invalid_argument("invalid GPU residency value");
            if (!resident[expert]) continue;
            slots[expert] = static_cast<int>(selected_bias.size() / hidden);
            selected.insert(selected.end(), loaded.packed_weights[2].begin() + expert * stride,
                            loaded.packed_weights[2].begin() + (expert + 1) * stride);
            for (std::size_t i = 0; i < hidden; ++i)
                selected_bias.push_back(has_bias ? loaded.weights.down_bias[expert * hidden + i] : 0.0f);
        }
        if (selected.empty()) throw std::invalid_argument("no GPU-resident experts");
        require_space(selected.size() + selected_bias.size() * sizeof(float));
        upload(down, selected); upload(bias, selected_bias);
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "GPU stream creation failed");
    }
    ~Impl() {
        cudaSetDevice(device);
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }
};

Mxfp4NativeCudaDown::Mxfp4NativeCudaDown(const LoadedExperts & loaded, const std::vector<std::uint8_t> & resident, int device)
    : impl_(std::make_unique<Impl>(loaded, resident, device)) {}
Mxfp4NativeCudaDown::~Mxfp4NativeCudaDown() = default;

std::vector<float> Mxfp4NativeCudaDown::execute(const std::vector<std::uint8_t> & packed_q8,
                                                 const std::vector<ExpertJob> & jobs) {
    std::lock_guard lock(impl_->mutex);
    checked(cudaSetDevice(impl_->device), "GPU device selection failed");
    const auto hidden = impl_->hidden, blocks = impl_->intermediate / 32;
    const auto stride = blocks * 34;
    if (jobs.size() > max_floats / hidden || jobs.size() > max_floats / impl_->intermediate ||
        packed_q8.size() != jobs.size() * stride) throw std::invalid_argument("invalid packed Q8 input shape");
    if (jobs.empty()) return {};
    std::vector<std::uint32_t> ids;
    for (const auto & job : jobs) {
        if (job.expert >= impl_->slots.size() || impl_->slots[job.expert] < 0)
            throw std::invalid_argument("MXFP4 GPU expert is out of range or not resident");
        ids.push_back(static_cast<std::uint32_t>(impl_->slots[job.expert]));
    }
    require_space(packed_q8.size() + ids.size() * sizeof(std::uint32_t) + jobs.size() * hidden * sizeof(float));
    DeviceArray<std::uint8_t> input(packed_q8.size());
    DeviceArray<std::uint32_t> device_ids(ids.size());
    DeviceArray<float> result(jobs.size() * hidden);
    std::vector<float> output(jobs.size() * hidden);
    try {
        checked(cudaMemcpyAsync(input.get(), packed_q8.data(), packed_q8.size(), cudaMemcpyHostToDevice, impl_->stream),
                "packed Q8 upload failed");
        checked(cudaMemcpyAsync(device_ids.get(), ids.data(), ids.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                                impl_->stream), "MXFP4 route upload failed");
        constexpr int threads = 128;
        const auto blocks_grid = static_cast<unsigned>((jobs.size() * hidden * 8 + threads - 1) / threads);
        mxfp4_down_kernel<<<blocks_grid, threads, 0, impl_->stream>>>(impl_->down.get(), input.get(), device_ids.get(),
            impl_->bias.get(), result.get(), static_cast<int>(blocks), static_cast<int>(hidden), static_cast<int>(jobs.size()), impl_->has_bias);
        checked(cudaGetLastError(), "MXFP4 down launch failed");
        checked(cudaMemcpyAsync(output.data(), result.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream),
                "MXFP4 output download failed");
        checked(cudaStreamSynchronize(impl_->stream), "MXFP4 down execution failed");
    } catch (...) {
        cudaStreamSynchronize(impl_->stream);
        throw;
    }
    if (!std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("MXFP4 GPU produced non-finite output");
    return output;
}

} // namespace vrambridge::experimental
