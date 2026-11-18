#include "native_cpu.hpp"

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

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    try {
        const auto loaded = load_gguf_experts(argv[1], 0, {2, 0, 1}, F32Activation::silu);
        NativeCpuExperts native(loaded, argv[2]);
        std::vector<float> input(loaded.weights.hidden * 2);
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = std::sin(static_cast<float>(i) * 0.1f);
        const std::vector<ExpertJob> jobs{{0, 0, 2}, {1, 0, 0}, {0, 1, 1}};
        const auto first = native.execute(input, jobs);
        const auto second = native.execute(input, jobs);
        require(first == second && first.size() == jobs.size() * loaded.weights.hidden, "native expert repeat changed");
        const auto f32 = execute_f32_cpu(loaded.weights, input, jobs);
        for (std::size_t i = 0; i < f32.size(); ++i)
            require(std::abs(first[i] - f32[i]) < 1e-5f, "native F32 expert result differs");
        const auto activations = native.activations(input, jobs);
        require(activations.size() == jobs.size() * loaded.weights.intermediate,
                "native activation shape differs");
        require(native.execute_down(activations, jobs) == first, "separate native down changed output");
        const auto packed = native.pack_down(activations);
        require(!packed.empty(), "native down packing failed");
        require(native.execute_down_packed(packed, jobs) == first, "packed native down changed output");
        rejects([&] { native.execute_down_packed({0}, jobs); }, "invalid packed input accepted");
        rejects([&] { native.execute(input, {{2, 0, 0}}); }, "invalid input row accepted");
        rejects([&] { native.execute(input, {{0, 0, 3}}); }, "invalid expert accepted");
        rejects([&] { native.execute_down({1.0f}, jobs); }, "invalid down activation shape accepted");
        auto bad_input = input;
        bad_input[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { native.execute(bad_input, jobs); }, "non-finite input accepted");
        auto bad_weights = loaded;
        bad_weights.packed_weights[0].pop_back();
        rejects([&] { NativeCpuExperts invalid(bad_weights, argv[2]); }, "malformed packed weight accepted");
        std::puts("native expert execution, input guards and packed shape checks passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "native expert test: %s\n", error.what());
        return 1;
    }
}
