#include "f32_experts.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vrambridge::experimental {
namespace {

constexpr std::size_t max_floats = 32 * 1024 * 1024;

bool finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

} // namespace

void F32ExpertWeights::validate() const {
    if (!hidden || !intermediate || !experts || hidden > 65536 || intermediate > 65536 || experts > 65536 ||
        hidden > max_floats / intermediate || experts > max_floats / (hidden * intermediate))
        throw std::invalid_argument("invalid or unbounded F32 expert dimensions");
    if (activation != F32Activation::silu && activation != F32Activation::openai_swiglu)
        throw std::invalid_argument("unsupported F32 expert activation");
    const auto matrix = experts * hidden * intermediate;
    if (gate.size() != matrix || up.size() != matrix || down.size() != matrix ||
        (!gate_bias.empty() && gate_bias.size() != experts * intermediate) ||
        (!up_bias.empty() && up_bias.size() != experts * intermediate) ||
        (!down_bias.empty() && down_bias.size() != experts * hidden))
        throw std::invalid_argument("invalid F32 expert tensor shape");
    if (!finite(gate) || !finite(up) || !finite(down) || !finite(gate_bias) || !finite(up_bias) || !finite(down_bias))
        throw std::invalid_argument("non-finite F32 expert weights");
}

std::vector<float> execute_f32_cpu(const F32ExpertWeights & weights,
                                   const std::vector<float> & input,
                                   const std::vector<ExpertJob> & jobs,
                                   const F32Transforms & transforms) {
    weights.validate();
    if (input.empty() || input.size() % weights.hidden || !finite(input) || jobs.size() > max_floats / weights.hidden)
        throw std::invalid_argument("invalid F32 expert input");
    const auto rows = input.size() / weights.hidden;
    std::vector<float> result(jobs.size() * weights.hidden);
    std::vector<float> intermediate(weights.intermediate);
    for (std::size_t job_index = 0; job_index < jobs.size(); ++job_index) {
        const auto & job = jobs[job_index];
        if (job.row >= rows || job.expert >= weights.experts) throw std::invalid_argument("invalid F32 expert job");
        const auto input_offset = job.row * weights.hidden;
        const auto weight_offset = static_cast<std::size_t>(job.expert) * weights.hidden * weights.intermediate;
        std::vector<float> transformed;
        const float * row = input.data() + input_offset;
        if (transforms.gate_up_input) {
            transformed.assign(row, row + weights.hidden);
            transforms.gate_up_input(transformed);
            if (transformed.size() != weights.hidden || !finite(transformed))
                throw std::runtime_error("invalid transformed gate/up input");
            row = transformed.data();
        }
        std::vector<float> gates, ups;
        if (transforms.observe) { gates.resize(weights.intermediate); ups.resize(weights.intermediate); }
        for (std::size_t i = 0; i < weights.intermediate; ++i) {
            float gate = weights.gate_bias.empty() ? 0.0f : weights.gate_bias[job.expert * weights.intermediate + i];
            float up = weights.up_bias.empty() ? 0.0f : weights.up_bias[job.expert * weights.intermediate + i];
            for (std::size_t k = 0; k < weights.hidden; ++k) {
                gate += weights.gate[weight_offset + i * weights.hidden + k] * row[k];
                up += weights.up[weight_offset + i * weights.hidden + k] * row[k];
            }
            if (transforms.observe) { gates[i] = gate; ups[i] = up; }
            if (weights.activation == F32Activation::openai_swiglu) {
                gate = std::min(gate, 7.0f);
                up = std::clamp(up, -7.0f, 7.0f);
                intermediate[i] = gate / (1.0f + std::exp(-1.702f * gate)) * (up + 1.0f);
            } else {
                intermediate[i] = gate / (1.0f + std::exp(-gate)) * up;
            }
        }
        if (transforms.observe) {
            transforms.observe(job, F32Stage::gate, gates);
            transforms.observe(job, F32Stage::up, ups);
            transforms.observe(job, F32Stage::activation, intermediate);
        }
        if (transforms.down_input) {
            transforms.down_input(intermediate);
            if (intermediate.size() != weights.intermediate || !finite(intermediate))
                throw std::runtime_error("invalid transformed down input");
        }
        for (std::size_t i = 0; i < weights.hidden; ++i) {
            float sum = weights.down_bias.empty() ? 0.0f : weights.down_bias[job.expert * weights.hidden + i];
            for (std::size_t k = 0; k < weights.intermediate; ++k)
                sum += weights.down[weight_offset + i * weights.intermediate + k] * intermediate[k];
            if (!std::isfinite(sum)) throw std::runtime_error("F32 expert produced a non-finite output");
            result[job_index * weights.hidden + i] = sum;
        }
        if (transforms.observe) {
            transforms.observe(job, F32Stage::down,
                std::vector<float>(result.begin() + job_index * weights.hidden, result.begin() + (job_index + 1) * weights.hidden));
        }
    }
    return result;
}

} // namespace vrambridge::experimental
