#include "f32_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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
        throw std::runtime_error("F32 expert allocation would violate the 1 GiB GPU reserve");
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
                                            cudaMemcpyHostToDevice), "GPU weight upload failed");
}

__global__ void gate_up_kernel(const float * gate, const float * up, const float * gate_bias,
                               const float * up_bias, const float * input, const std::uint32_t * ids,
                               float * intermediate, int hidden, int width, int jobs, bool openai) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= jobs * width) return;
    const int job = index / width, col = index % width;
    const std::size_t matrix = static_cast<std::size_t>(ids[job]) * hidden * width + col * hidden;
    const std::size_t bias = static_cast<std::size_t>(ids[job]) * width + col;
    float g = gate_bias[bias], u = up_bias[bias];
    for (int k = 0; k < hidden; ++k) {
        const float x = input[static_cast<std::size_t>(job) * hidden + k];
        g += gate[matrix + k] * x;
        u += up[matrix + k] * x;
    }
    if (openai) {
        g = fminf(g, 7.0f);
        u = fminf(fmaxf(u, -7.0f), 7.0f);
        intermediate[index] = g / (1.0f + expf(-1.702f * g)) * (u + 1.0f);
    } else {
        intermediate[index] = g / (1.0f + expf(-g)) * u;
    }
}

__global__ void down_kernel(const float * down, const float * bias, const float * intermediate,
                            const std::uint32_t * ids, float * output, int hidden, int width, int jobs) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= jobs * hidden) return;
    const int job = index / hidden, col = index % hidden;
    const std::size_t matrix = static_cast<std::size_t>(ids[job]) * hidden * width + col * width;
    float sum = bias[static_cast<std::size_t>(ids[job]) * hidden + col];
    for (int k = 0; k < width; ++k) sum += down[matrix + k] * intermediate[static_cast<std::size_t>(job) * width + k];
    output[index] = sum;
}

} // namespace

struct F32CudaExperts::Impl {
    int device;
    std::size_t hidden, intermediate;
    bool openai;
    std::vector<int> slots;
    DeviceArray<float> gate, up, down, gate_bias, up_bias, down_bias;
    cudaStream_t stream = nullptr;
    std::mutex mutex;

    Impl(const F32ExpertWeights & weights, const std::vector<std::uint8_t> & resident, int device_index)
        : device(device_index), hidden(weights.hidden), intermediate(weights.intermediate),
          openai(weights.activation == F32Activation::openai_swiglu), slots(weights.experts, -1) {
        weights.validate();
        if (resident.size() != weights.experts || device < 0) throw std::invalid_argument("invalid GPU residency");
        checked(cudaSetDevice(device), "GPU device selection failed");
        std::vector<float> g, u, d, gb, ub, db;
        const auto matrix = hidden * intermediate;
        for (std::size_t expert = 0; expert < weights.experts; ++expert) {
            if (resident[expert] > 1) throw std::invalid_argument("invalid GPU residency value");
            if (!resident[expert]) continue;
            slots[expert] = static_cast<int>(gb.size() / intermediate);
            g.insert(g.end(), weights.gate.begin() + expert * matrix, weights.gate.begin() + (expert + 1) * matrix);
            u.insert(u.end(), weights.up.begin() + expert * matrix, weights.up.begin() + (expert + 1) * matrix);
            d.insert(d.end(), weights.down.begin() + expert * matrix, weights.down.begin() + (expert + 1) * matrix);
            for (std::size_t i = 0; i < intermediate; ++i) {
                gb.push_back(weights.gate_bias.empty() ? 0.0f : weights.gate_bias[expert * intermediate + i]);
                ub.push_back(weights.up_bias.empty() ? 0.0f : weights.up_bias[expert * intermediate + i]);
            }
            for (std::size_t i = 0; i < hidden; ++i)
                db.push_back(weights.down_bias.empty() ? 0.0f : weights.down_bias[expert * hidden + i]);
        }
        if (g.empty()) throw std::invalid_argument("no GPU-resident experts");
        const std::size_t total = (g.size() + u.size() + d.size() + gb.size() + ub.size() + db.size()) * sizeof(float);
        require_space(total);
        upload(gate, g); upload(up, u); upload(down, d);
        upload(gate_bias, gb); upload(up_bias, ub); upload(down_bias, db);
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "GPU stream creation failed");
    }

    ~Impl() {
        cudaSetDevice(device);
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }
};

F32CudaExperts::F32CudaExperts(const F32ExpertWeights & weights, const std::vector<std::uint8_t> & resident, int device)
    : impl_(std::make_unique<Impl>(weights, resident, device)) {}

F32CudaExperts::~F32CudaExperts() = default;

std::vector<float> F32CudaExperts::execute(const std::vector<float> & input, const std::vector<ExpertJob> & jobs,
                                           const F32Transforms & transforms) {
    std::lock_guard lock(impl_->mutex);
    checked(cudaSetDevice(impl_->device), "GPU device selection failed");
    const auto hidden = impl_->hidden, width = impl_->intermediate;
    if (input.empty() || input.size() % hidden || jobs.size() > max_floats / hidden ||
        jobs.size() > max_floats / width ||
        !std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("invalid GPU expert input");
    if (jobs.empty()) return {};
    std::vector<float> packed_input;
    std::vector<std::uint32_t> ids;
    packed_input.reserve(jobs.size() * hidden);
    ids.reserve(jobs.size());
    for (const auto & job : jobs) {
        if (job.row >= input.size() / hidden || job.expert >= impl_->slots.size() || impl_->slots[job.expert] < 0)
            throw std::invalid_argument("GPU job is out of range or not resident");
        ids.push_back(static_cast<std::uint32_t>(impl_->slots[job.expert]));
        std::vector<float> row(input.begin() + job.row * hidden, input.begin() + (job.row + 1) * hidden);
        if (transforms.gate_up_input) transforms.gate_up_input(row);
        if (row.size() != hidden || !std::all_of(row.begin(), row.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("invalid transformed GPU gate/up input");
        packed_input.insert(packed_input.end(), row.begin(), row.end());
    }
    const auto floats = 2 * jobs.size() * hidden + jobs.size() * width;
    require_space(floats * sizeof(float) + ids.size() * sizeof(std::uint32_t));
    DeviceArray<float> device_input(packed_input.size()), activation(jobs.size() * width), result(jobs.size() * hidden);
    DeviceArray<std::uint32_t> device_ids(ids.size());
    std::vector<float> output(jobs.size() * hidden);
    try {
        checked(cudaMemcpyAsync(device_input.get(), packed_input.data(), packed_input.size() * sizeof(float),
                                cudaMemcpyHostToDevice, impl_->stream), "GPU input upload failed");
        checked(cudaMemcpyAsync(device_ids.get(), ids.data(), ids.size() * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice, impl_->stream), "GPU route upload failed");
        const int threads = 128;
        gate_up_kernel<<<static_cast<unsigned>((jobs.size() * width + threads - 1) / threads), threads, 0, impl_->stream>>>(
            impl_->gate.get(), impl_->up.get(), impl_->gate_bias.get(), impl_->up_bias.get(),
            device_input.get(), device_ids.get(), activation.get(), static_cast<int>(hidden), static_cast<int>(width),
            static_cast<int>(jobs.size()), impl_->openai);
        checked(cudaGetLastError(), "GPU gate/up launch failed");
        if (transforms.down_input) {
            std::vector<float> host_activation(jobs.size() * width);
            checked(cudaMemcpyAsync(host_activation.data(), activation.get(), host_activation.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost, impl_->stream), "GPU activation download failed");
            checked(cudaStreamSynchronize(impl_->stream), "GPU activation execution failed");
            for (std::size_t job = 0; job < jobs.size(); ++job) {
                std::vector<float> row(host_activation.begin() + job * width, host_activation.begin() + (job + 1) * width);
                transforms.down_input(row);
                if (row.size() != width || !std::all_of(row.begin(), row.end(), [](float x) { return std::isfinite(x); }))
                    throw std::runtime_error("invalid transformed GPU down input");
                std::copy(row.begin(), row.end(), host_activation.begin() + job * width);
            }
            checked(cudaMemcpyAsync(activation.get(), host_activation.data(), host_activation.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, impl_->stream), "GPU transformed activation upload failed");
            checked(cudaStreamSynchronize(impl_->stream), "GPU transformed activation synchronization failed");
        }
        down_kernel<<<static_cast<unsigned>((jobs.size() * hidden + threads - 1) / threads), threads, 0, impl_->stream>>>(
            impl_->down.get(), impl_->down_bias.get(), activation.get(), device_ids.get(), result.get(),
            static_cast<int>(hidden), static_cast<int>(width), static_cast<int>(jobs.size()));
        checked(cudaGetLastError(), "GPU down launch failed");
        checked(cudaMemcpyAsync(output.data(), result.get(), output.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, impl_->stream), "GPU output download failed");
        checked(cudaStreamSynchronize(impl_->stream), "GPU expert execution failed");
    } catch (...) {
        cudaStreamSynchronize(impl_->stream);
        throw;
    }
    if (!std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("GPU expert produced non-finite output");
    return output;
}

std::vector<float> F32CudaExperts::execute_down(const std::vector<float> & activations,
                                                const std::vector<ExpertJob> & jobs,
                                                const F32Transforms & transforms) {
    std::lock_guard lock(impl_->mutex);
    checked(cudaSetDevice(impl_->device), "GPU device selection failed");
    const auto hidden = impl_->hidden, width = impl_->intermediate;
    if (jobs.size() > max_floats / hidden || jobs.size() > max_floats / width ||
        activations.size() != jobs.size() * width ||
        !std::all_of(activations.begin(), activations.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("invalid GPU down input");
    if (jobs.empty()) return {};
    std::vector<float> packed_input;
    std::vector<std::uint32_t> ids;
    packed_input.reserve(activations.size());
    ids.reserve(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        const auto & job = jobs[i];
        if (job.expert >= impl_->slots.size() || impl_->slots[job.expert] < 0)
            throw std::invalid_argument("GPU down job is out of range or not resident");
        ids.push_back(static_cast<std::uint32_t>(impl_->slots[job.expert]));
        std::vector<float> row(activations.begin() + i * width, activations.begin() + (i + 1) * width);
        if (transforms.down_input) transforms.down_input(row);
        if (row.size() != width || !std::all_of(row.begin(), row.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("invalid transformed GPU down input");
        packed_input.insert(packed_input.end(), row.begin(), row.end());
    }
    require_space((packed_input.size() + jobs.size() * hidden) * sizeof(float) + ids.size() * sizeof(std::uint32_t));
    DeviceArray<float> device_input(packed_input.size()), result(jobs.size() * hidden);
    DeviceArray<std::uint32_t> device_ids(ids.size());
    std::vector<float> output(jobs.size() * hidden);
    try {
        checked(cudaMemcpyAsync(device_input.get(), packed_input.data(), packed_input.size() * sizeof(float),
                                cudaMemcpyHostToDevice, impl_->stream), "GPU down input upload failed");
        checked(cudaMemcpyAsync(device_ids.get(), ids.data(), ids.size() * sizeof(std::uint32_t),
                                cudaMemcpyHostToDevice, impl_->stream), "GPU down route upload failed");
        const int threads = 128;
        down_kernel<<<static_cast<unsigned>((jobs.size() * hidden + threads - 1) / threads), threads, 0, impl_->stream>>>(
            impl_->down.get(), impl_->down_bias.get(), device_input.get(), device_ids.get(), result.get(),
            static_cast<int>(hidden), static_cast<int>(width), static_cast<int>(jobs.size()));
        checked(cudaGetLastError(), "GPU down launch failed");
        checked(cudaMemcpyAsync(output.data(), result.get(), output.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, impl_->stream), "GPU down output download failed");
        checked(cudaStreamSynchronize(impl_->stream), "GPU down execution failed");
    } catch (...) {
        cudaStreamSynchronize(impl_->stream);
        throw;
    }
    if (!std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("GPU down produced non-finite output");
    return output;
}

} // namespace vrambridge::experimental
