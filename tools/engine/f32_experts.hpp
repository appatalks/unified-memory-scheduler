#pragma once

#include "expert_dispatch.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace vrambridge::experimental {

enum class F32Activation { silu, openai_swiglu };
enum class F32Stage { gate, up, activation, down };

struct F32ExpertWeights {
    std::size_t hidden;
    std::size_t intermediate;
    std::size_t experts;
    F32Activation activation;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;
    std::vector<float> gate_bias;
    std::vector<float> up_bias;
    std::vector<float> down_bias;

    void validate() const;
};

struct F32Transforms {
    std::function<void(std::vector<float> &)> gate_up_input;
    std::function<void(std::vector<float> &)> down_input;
    std::function<void(const ExpertJob &, F32Stage, const std::vector<float> &)> observe;
};

std::vector<float> execute_f32_cpu(const F32ExpertWeights & weights,
                                   const std::vector<float> & input,
                                   const std::vector<ExpertJob> & jobs,
                                   const F32Transforms & transforms = {});

} // namespace vrambridge::experimental
