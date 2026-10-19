#pragma once

#include <cstdint>
#include <functional>
#include <vector>

struct CompatibleDownResult {
    std::vector<float> output;
    std::vector<double> samples_ms;
    std::vector<double> quantize_ms;
    bool all_iterations_identical = true;
};

CompatibleDownResult compatible_down(
    const std::vector<uint8_t> & weights, const std::vector<int32_t> & ids,
    int width, int outputs, int experts, int repetitions, bool upload_weights, bool packed_dot,
    const std::function<void(std::vector<uint8_t> &)> & quantize);
