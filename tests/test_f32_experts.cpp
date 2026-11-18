#include "expert_dispatch.hpp"
#include "f32_experts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace vrambridge::experimental;

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void rejects(Function function, const char * message) {
    bool failed = false;
    try { function(); } catch (const std::exception &) { failed = true; }
    require(failed, message);
}

F32ExpertWeights fixture(F32Activation activation) {
    F32ExpertWeights weights{4, 3, 3, activation, {}, {}, {}, {}, {}, {}};
    const auto size = weights.hidden * weights.intermediate * weights.experts;
    for (std::size_t i = 0; i < size; ++i) {
        weights.gate.push_back(std::sin(static_cast<float>(i) * 0.3f) * 0.5f);
        weights.up.push_back(std::cos(static_cast<float>(i) * 0.2f) * 0.7f);
        weights.down.push_back(std::sin(static_cast<float>(i) * 0.1f) * 0.4f);
    }
    weights.gate_bias.assign(weights.experts * weights.intermediate, 0.2f);
    weights.up_bias.assign(weights.experts * weights.intermediate, -0.3f);
    weights.down_bias.assign(weights.experts * weights.hidden, 0.1f);
    if (activation == F32Activation::openai_swiglu) {
        weights.gate_bias[0] = 9.0f;
        weights.up_bias[0] = -9.0f;
    }
    return weights;
}

std::vector<double> reference(const F32ExpertWeights & w, const std::vector<float> & input,
                              std::size_t row, std::size_t expert) {
    std::vector<double> activation(w.intermediate), output(w.hidden);
    const std::size_t offset = expert * w.hidden * w.intermediate;
    for (std::size_t j = 0; j < w.intermediate; ++j) {
        double gate = w.gate_bias.empty() ? 0 : w.gate_bias[expert * w.intermediate + j];
        double up = w.up_bias.empty() ? 0 : w.up_bias[expert * w.intermediate + j];
        for (std::size_t k = 0; k < w.hidden; ++k) {
            gate += static_cast<double>(w.gate[offset + j * w.hidden + k]) * input[row * w.hidden + k];
            up += static_cast<double>(w.up[offset + j * w.hidden + k]) * input[row * w.hidden + k];
        }
        if (w.activation == F32Activation::openai_swiglu) {
            gate = std::min(gate, 7.0);
            up = std::clamp(up, -7.0, 7.0);
            activation[j] = gate / (1 + std::exp(-1.702 * gate)) * (up + 1);
        } else {
            activation[j] = gate / (1 + std::exp(-gate)) * up;
        }
    }
    for (std::size_t col = 0; col < w.hidden; ++col) {
        double sum = w.down_bias.empty() ? 0 : w.down_bias[expert * w.hidden + col];
        for (std::size_t j = 0; j < w.intermediate; ++j)
            sum += static_cast<double>(w.down[offset + col * w.intermediate + j]) * activation[j];
        output[col] = sum;
    }
    return output;
}

void check(F32Activation activation, bool bias) {
    auto weights = fixture(activation);
    if (!bias) { weights.gate_bias.clear(); weights.up_bias.clear(); weights.down_bias.clear(); }
    const std::vector<float> input{1.0f, -0.5f, 0.2f, 2.0f, -1.0f, 0.1f, 0.7f, -0.4f};
    const std::vector<std::vector<Route>> routes{{{2, 0.3f}, {0, 0.7f}}, {{1, 0.6f}, {2, 0.4f}}};
    for (const std::vector<std::uint8_t> residency : {std::vector<std::uint8_t>{0, 0, 0},
                                                       std::vector<std::uint8_t>{1, 0, 1},
                                                       std::vector<std::uint8_t>{1, 1, 1}}) {
        const ExpertPlan plan(weights.hidden, routes, residency);
        const auto runner = [&](const std::vector<ExpertJob> & jobs) { return execute_f32_cpu(weights, input, jobs); };
        const auto actual = plan.execute(runner, runner);
        for (std::size_t row = 0; row < routes.size(); ++row) {
            std::vector<double> expected(weights.hidden, 0);
            for (const auto & route : routes[row]) {
                const auto part = reference(weights, input, row, route.expert);
                for (std::size_t col = 0; col < weights.hidden; ++col) expected[col] += route.weight * part[col];
            }
            for (std::size_t col = 0; col < weights.hidden; ++col)
                require(std::abs(actual[row * weights.hidden + col] - expected[col]) < 1e-5,
                        "CPU expert output does not match scalar reference");
        }
    }
}

} // namespace

int main() {
    try {
        check(F32Activation::silu, false);
        check(F32Activation::silu, true);
        check(F32Activation::openai_swiglu, true);
        auto weights = fixture(F32Activation::silu);
        F32Transforms transforms;
        transforms.gate_up_input = [](std::vector<float> & values) { std::fill(values.begin(), values.end(), 0.0f); };
        transforms.down_input = [](std::vector<float> & values) { std::fill(values.begin(), values.end(), 0.0f); };
        const auto transformed = execute_f32_cpu(weights, {1, 2, 3, 4}, {{0, 0, 0}}, transforms);
        for (const auto value : transformed) require(value == 0.1f, "per-projection transforms were not applied");
        rejects([&] { execute_f32_cpu(weights, {1, 2, 3}, {{0, 0, 0}}); }, "bad input shape accepted");
        rejects([&] { execute_f32_cpu(weights, {1, 2, 3, 4}, {{0, 0, 3}}); }, "bad expert accepted");
        weights.gate[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { execute_f32_cpu(weights, {1, 2, 3, 4}, {{0, 0, 0}}); }, "non-finite weight accepted");
        std::puts("F32 expert math and routed reduction match scalar reference");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "F32 expert test: %s\n", error.what());
        return 1;
    }
}
