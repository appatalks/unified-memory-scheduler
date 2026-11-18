#include "native_cpu.hpp"

#include "ggml-cpu.h"
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vrambridge::experimental {
namespace {
constexpr std::size_t max_floats = 32 * 1024 * 1024;
}

struct NativeCpuExperts::Impl {
    using Traits = const ggml_type_traits_cpu * (*)(ggml_type);
    using Swiglu = void (*)(int, float *, const float *, const float *);
    const LoadedExperts & loaded;
    void * handle = nullptr;
    Traits traits = nullptr;
    Swiglu swiglu = nullptr;

    Impl(const LoadedExperts & weights, const std::string & plugin) : loaded(weights) {
        loaded.weights.validate();
        handle = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error("cannot load native CPU expert plugin");
        traits = reinterpret_cast<Traits>(dlsym(handle, "ggml_get_type_traits_cpu"));
        swiglu = reinterpret_cast<Swiglu>(dlsym(handle, "ggml_vec_swiglu_f32"));
        const auto initialize = reinterpret_cast<void (*)()>(dlsym(handle, "ggml_cpu_init"));
        if (!traits || !swiglu || !initialize) {
            dlclose(handle);
            handle = nullptr;
            throw std::runtime_error("native CPU expert symbols unavailable");
        }
        initialize();
        try {
            for (int i = 0; i < 3; ++i) {
                const auto type_id = loaded.quant_types[i];
                const auto * type = traits(type_id);
                const auto * dot = type ? traits(type->vec_dot_type) : nullptr;
                const auto in_width = i == 2 ? loaded.weights.intermediate : loaded.weights.hidden;
                const auto out_width = i == 2 ? loaded.weights.hidden : loaded.weights.intermediate;
                const auto block = ggml_blck_size(type_id);
                if (!type || !type->vec_dot || !dot || !dot->from_float || block <= 0 ||
                    in_width % static_cast<std::size_t>(block))
                    throw std::runtime_error("unsupported native expert dot format");
                const auto row_bytes = ggml_row_size(type_id, static_cast<std::int64_t>(in_width));
                if (!row_bytes || row_bytes > max_floats * sizeof(float) / out_width ||
                    row_bytes * out_width > max_floats * sizeof(float) / loaded.weights.experts ||
                    loaded.packed_weights[i].size() != row_bytes * out_width * loaded.weights.experts)
                    throw std::runtime_error("invalid native packed expert shape");
            }
        } catch (...) {
            dlclose(handle);
            handle = nullptr;
            throw;
        }
    }
    ~Impl() { if (handle) dlclose(handle); }

    std::vector<std::uint8_t> quantize(const float * values, std::size_t count, ggml_type weight_type) const {
        const auto dot_type = traits(weight_type)->vec_dot_type;
        const auto * type = traits(dot_type);
        const auto block = ggml_blck_size(dot_type);
        if (!type || !type->from_float || block <= 0 || count % static_cast<std::size_t>(block))
            throw std::runtime_error("unsupported native expert input format");
        std::vector<std::uint8_t> result(ggml_row_size(dot_type, static_cast<std::int64_t>(count)));
        type->from_float(values, result.data(), static_cast<std::int64_t>(count));
        return result;
    }

    void validate_jobs(const std::vector<float> & input, const std::vector<ExpertJob> & jobs) const {
        const auto & w = loaded.weights;
        if (input.empty() || input.size() % w.hidden || jobs.size() > max_floats / w.hidden ||
            jobs.size() > max_floats / w.intermediate ||
            !std::all_of(input.begin(), input.end(), [](float value) { return std::isfinite(value); }))
            throw std::invalid_argument("invalid native expert input");
        for (const auto & job : jobs)
            if (job.row >= input.size() / w.hidden || job.expert >= w.experts)
                throw std::invalid_argument("invalid native expert route");
    }

    void validate_activations(const std::vector<float> & values, std::size_t count) const {
        const auto width = loaded.weights.intermediate;
        if (count > max_floats / width || values.size() != count * width ||
            !std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
            throw std::invalid_argument("invalid native down activations");
    }

    std::vector<float> project(int part, std::size_t expert, const std::vector<std::uint8_t> & input,
                               std::size_t in_width, std::size_t out_width) const {
        const auto type = loaded.quant_types[part];
        const auto row_bytes = ggml_row_size(type, static_cast<std::int64_t>(in_width));
        const auto & packed = loaded.packed_weights[part];
        if (expert >= loaded.weights.experts || row_bytes > packed.size() / out_width ||
            expert >= packed.size() / (row_bytes * out_width)) throw std::runtime_error("invalid native expert slice");
        const auto * matrix = packed.data() + expert * row_bytes * out_width;
        std::vector<float> output(out_width);
        const auto dot = traits(type)->vec_dot;
        for (std::size_t row = 0; row < out_width; ++row)
            dot(static_cast<int>(in_width), &output[row], 0, matrix + row * row_bytes, 0, input.data(), 0, 1);
        return output;
    }

    std::vector<float> activate(const std::vector<float> & input, const ExpertJob & job,
                                const F32Transforms & observations) const {
        const auto & w = loaded.weights;
        if (input.empty() || input.size() % w.hidden || job.row >= input.size() / w.hidden || job.expert >= w.experts)
            throw std::invalid_argument("invalid native expert input or route");
        const auto * row = input.data() + job.row * w.hidden;
        auto gate = project(0, job.expert, quantize(row, w.hidden, loaded.quant_types[0]), w.hidden, w.intermediate);
        auto up = project(1, job.expert, quantize(row, w.hidden, loaded.quant_types[1]), w.hidden, w.intermediate);
        for (std::size_t i = 0; i < w.intermediate; ++i) {
            if (!w.gate_bias.empty()) gate[i] += w.gate_bias[job.expert * w.intermediate + i];
            if (!w.up_bias.empty()) up[i] += w.up_bias[job.expert * w.intermediate + i];
        }
        if (observations.observe) { observations.observe(job, F32Stage::gate, gate); observations.observe(job, F32Stage::up, up); }
        std::vector<float> activation(w.intermediate);
        if (w.activation == F32Activation::silu) {
            swiglu(static_cast<int>(w.intermediate), activation.data(), gate.data(), up.data());
        } else {
            for (std::size_t i = 0; i < w.intermediate; ++i) {
                const float g = std::min(gate[i], 7.0f);
                const float u = std::clamp(up[i], -7.0f, 7.0f);
                activation[i] = (g / (1.0f + std::exp(1.702f * (-g)))) * (u + 1.0f);
            }
        }
        if (observations.observe) observations.observe(job, F32Stage::activation, activation);
        return activation;
    }
};

NativeCpuExperts::NativeCpuExperts(const LoadedExperts & weights, const std::string & cpu_plugin)
    : impl_(std::make_unique<Impl>(weights, cpu_plugin)) {}
NativeCpuExperts::~NativeCpuExperts() = default;

std::vector<float> NativeCpuExperts::activations(const std::vector<float> & input, const std::vector<ExpertJob> & jobs,
                                                 const F32Transforms & observations) const {
    impl_->validate_jobs(input, jobs);
    std::vector<float> output;
    output.reserve(jobs.size() * impl_->loaded.weights.intermediate);
    for (const auto & job : jobs) {
        const auto values = impl_->activate(input, job, observations);
        output.insert(output.end(), values.begin(), values.end());
    }
    return output;
}

std::vector<float> NativeCpuExperts::execute(const std::vector<float> & input, const std::vector<ExpertJob> & jobs,
                                             const F32Transforms & observations) const {
    impl_->validate_jobs(input, jobs);
    const auto & w = impl_->loaded.weights;
    std::vector<float> output;
    output.reserve(jobs.size() * w.hidden);
    for (const auto & job : jobs) {
        const auto activation = impl_->activate(input, job, observations);
        auto down = impl_->project(2, job.expert, impl_->quantize(activation.data(), w.intermediate, impl_->loaded.quant_types[2]),
                                   w.intermediate, w.hidden);
        if (!w.down_bias.empty())
            for (std::size_t i = 0; i < w.hidden; ++i) down[i] += w.down_bias[job.expert * w.hidden + i];
        if (observations.observe) observations.observe(job, F32Stage::down, down);
        output.insert(output.end(), down.begin(), down.end());
    }
    return output;
}

std::vector<std::uint8_t> NativeCpuExperts::pack_down(const std::vector<float> & activations) const {
    const auto width = impl_->loaded.weights.intermediate;
    if (activations.size() % width) throw std::invalid_argument("invalid native down activation shape");
    const auto count = activations.size() / width;
    impl_->validate_activations(activations, count);
    std::vector<std::uint8_t> output;
    for (std::size_t i = 0; i < count; ++i) {
        const auto packed = impl_->quantize(activations.data() + i * width, width, impl_->loaded.quant_types[2]);
        output.insert(output.end(), packed.begin(), packed.end());
    }
    return output;
}

std::vector<float> NativeCpuExperts::execute_down(const std::vector<float> & activations,
                                                   const std::vector<ExpertJob> & jobs) const {
    const auto & w = impl_->loaded.weights;
    impl_->validate_activations(activations, jobs.size());
    if (jobs.size() > max_floats / w.hidden) throw std::invalid_argument("native down output exceeds bound");
    std::vector<float> output;
    output.reserve(jobs.size() * w.hidden);
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        const auto & job = jobs[i];
        if (job.expert >= w.experts) throw std::invalid_argument("invalid native down expert");
        auto down = impl_->project(2, job.expert, impl_->quantize(activations.data() + i * w.intermediate,
                                   w.intermediate, impl_->loaded.quant_types[2]), w.intermediate, w.hidden);
        if (!w.down_bias.empty())
            for (std::size_t col = 0; col < w.hidden; ++col) down[col] += w.down_bias[job.expert * w.hidden + col];
        output.insert(output.end(), down.begin(), down.end());
    }
    return output;
}

std::vector<float> NativeCpuExperts::execute_down_packed(const std::vector<std::uint8_t> & packed,
                                                          const std::vector<ExpertJob> & jobs) const {
    const auto & w = impl_->loaded.weights;
    const auto dot_type = impl_->traits(impl_->loaded.quant_types[2])->vec_dot_type;
    const auto stride = ggml_row_size(dot_type, static_cast<std::int64_t>(w.intermediate));
    if (!stride || jobs.size() > max_floats / w.hidden || jobs.size() > max_floats / w.intermediate ||
        jobs.size() > (max_floats * sizeof(float)) / stride || packed.size() != jobs.size() * stride)
        throw std::invalid_argument("invalid packed native down inputs");
    std::vector<float> output;
    output.reserve(jobs.size() * w.hidden);
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        const auto & job = jobs[i];
        if (job.expert >= w.experts) throw std::invalid_argument("invalid native packed down expert");
        const std::vector<std::uint8_t> row(packed.begin() + i * stride, packed.begin() + (i + 1) * stride);
        auto down = impl_->project(2, job.expert, row, w.intermediate, w.hidden);
        if (!w.down_bias.empty())
            for (std::size_t col = 0; col < w.hidden; ++col) down[col] += w.down_bias[job.expert * w.hidden + col];
        if (!std::all_of(down.begin(), down.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("native packed down produced non-finite output");
        output.insert(output.end(), down.begin(), down.end());
    }
    return output;
}

} // namespace vrambridge::experimental
