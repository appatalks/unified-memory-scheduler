#include "expert_dispatch.hpp"
#include "f32_cuda.hpp"
#include "f32_experts.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace vrambridge::experimental;

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void check(F32Activation activation, bool bias) {
    F32ExpertWeights weights{6, 5, 4, activation, {}, {}, {}, {}, {}, {}};
    const auto count = weights.hidden * weights.intermediate * weights.experts;
    for (std::size_t i = 0; i < count; ++i) {
        weights.gate.push_back(std::sin(static_cast<float>(i) * 0.3f) * 0.5f);
        weights.up.push_back(std::cos(static_cast<float>(i) * 0.2f) * 0.7f);
        weights.down.push_back(std::sin(static_cast<float>(i) * 0.1f) * 0.4f);
    }
    if (bias) {
        weights.gate_bias.assign(weights.experts * weights.intermediate, 0.2f);
        weights.up_bias.assign(weights.experts * weights.intermediate, -0.3f);
        weights.down_bias.assign(weights.experts * weights.hidden, 0.1f);
        if (activation == F32Activation::openai_swiglu) {
            weights.gate_bias[0] = 9.0f;
            weights.up_bias[0] = -9.0f;
        }
    }
    const std::vector<float> input{1.0f, -0.5f, 0.2f, 2.0f, 0.1f, 0.3f,
                                   -1.0f, 0.1f, 0.7f, -0.4f, 1.1f, -0.8f};
    const std::vector<std::vector<Route>> routes{{{2, 0.2f}, {0, 0.3f}, {1, 0.5f}},
                                                  {{1, 0.2f}, {3, 0.4f}, {2, 0.4f}}};
    const std::vector<std::uint8_t> resident{1, 0, 1, 0};
    const auto cpu = [&](const std::vector<ExpertJob> & jobs) { return execute_f32_cpu(weights, input, jobs); };
    const auto expected = ExpertPlan(weights.hidden, routes, {0, 0, 0, 0}).execute(cpu, {});
    F32CudaExperts cuda(weights, resident);
    const auto gpu = [&](const std::vector<ExpertJob> & jobs) { return cuda.execute(input, jobs); };
    const ExpertPlan mixed(weights.hidden, routes, resident);
    const auto actual = mixed.execute(cpu, gpu);
    require(expected.size() == actual.size(), "CPU/GPU output size differs");
    for (std::size_t i = 0; i < actual.size(); ++i)
        require(std::abs(actual[i] - expected[i]) <= 1e-5f + 1e-4f * std::abs(expected[i]),
                "CPU/GPU expert output differs");
    F32Transforms transforms;
    transforms.gate_up_input = [](std::vector<float> & row) { for (auto & x : row) x = std::round(x * 64.0f) / 64.0f; };
    transforms.down_input = transforms.gate_up_input;
    const auto transformed_cpu = [&](const std::vector<ExpertJob> & jobs) {
        return execute_f32_cpu(weights, input, jobs, transforms);
    };
    const auto transformed_gpu = [&](const std::vector<ExpertJob> & jobs) { return cuda.execute(input, jobs, transforms); };
    const auto transformed_expected = ExpertPlan(weights.hidden, routes, {0, 0, 0, 0}).execute(transformed_cpu, {});
    const auto transformed_actual = mixed.execute(transformed_cpu, transformed_gpu);
    for (std::size_t i = 0; i < transformed_actual.size(); ++i)
        require(std::abs(transformed_actual[i] - transformed_expected[i]) <= 1e-5f + 1e-4f * std::abs(transformed_expected[i]),
                "CPU/GPU transformed expert output differs");
    const auto staged_gpu = [&](const std::vector<ExpertJob> & jobs) {
        std::vector<float> activations;
        auto observed = transforms;
        observed.observe = [&](const ExpertJob &, F32Stage stage, const std::vector<float> & values) {
            if (stage == F32Stage::activation) activations.insert(activations.end(), values.begin(), values.end());
        };
        execute_f32_cpu(weights, input, jobs, observed);
        return cuda.execute_down(activations, jobs, transforms);
    };
    const auto staged = mixed.execute(transformed_cpu, staged_gpu);
    for (std::size_t i = 0; i < staged.size(); ++i)
        require(std::abs(staged[i] - transformed_expected[i]) <= 1e-5f + 1e-4f * std::abs(transformed_expected[i]),
                "CPU activation / GPU down output differs");
    bool rejected = false;
    try { cuda.execute(input, {{0, 0, 1}}); } catch (const std::exception &) { rejected = true; }
    require(rejected, "nonresident GPU expert was accepted");
}

} // namespace

int main() {
    int count = 0;
    std::size_t free = 0, total = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1 || cudaSetDevice(0) != cudaSuccess ||
        cudaMemGetInfo(&free, &total) != cudaSuccess || free < 1100ULL * 1024 * 1024) {
        std::puts("CUDA expert test skipped: GPU unavailable or insufficient headroom");
        return 77;
    }
    try {
        check(F32Activation::silu, false);
        check(F32Activation::silu, true);
        check(F32Activation::openai_swiglu, true);
        std::puts("real CUDA/CPU expert execution matches the CPU reference");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "CUDA expert test: %s\n", error.what());
        return 1;
    }
}
