#include "quantized_cuda.hpp"
#include "native_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace vrambridge::experimental;

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    try {
        const auto loaded = load_gguf_experts(argv[1], 0, {2, 0, 1}, F32Activation::silu);
        NativeCpuExperts cpu(loaded, argv[2]);
        std::unique_ptr<Iq4NativeCudaDown> iq4;
        std::unique_ptr<Q6NativeCudaDown> q6;
        std::unique_ptr<Mxfp4NativeCudaDown> mxfp4;
        const bool q6_type = loaded.quant_types[2] == GGML_TYPE_Q6_K;
        const bool mxfp4_type = loaded.quant_types[2] == GGML_TYPE_MXFP4;
        if (q6_type) q6 = std::make_unique<Q6NativeCudaDown>(loaded, std::vector<std::uint8_t>{1, 0, 1});
        else if (mxfp4_type) mxfp4 = std::make_unique<Mxfp4NativeCudaDown>(loaded, std::vector<std::uint8_t>{1, 0, 1});
        else iq4 = std::make_unique<Iq4NativeCudaDown>(loaded, std::vector<std::uint8_t>{1, 0, 1});
        const auto gpu = [&](const std::vector<std::uint8_t> & packed, const std::vector<ExpertJob> & jobs) {
            return q6_type ? q6->execute(packed, jobs) : mxfp4_type ? mxfp4->execute(packed, jobs) : iq4->execute(packed, jobs);
        };
        const std::vector<ExpertJob> jobs{{0, 0, 0}, {1, 0, 2}, {0, 1, 0}};
        for (int sample = 0; sample < 12; ++sample) {
            std::vector<float> input(jobs.size() * loaded.weights.intermediate);
            for (std::size_t i = 0; i < input.size(); ++i)
                input[i] = sample == 0 ? 0.0f : std::sin(static_cast<float>(i * 13 + sample * 31) * 0.07f) * sample;
            auto packed = cpu.pack_down(input);
            if (sample == 11) {
                for (std::size_t i = q6_type ? 4 : 2; i < packed.size(); i += q6_type ? 292 : 34) packed[i] = 128;
            }
            const auto reference = cpu.execute_down_packed(packed, jobs);
            const auto actual = gpu(packed, jobs);
            if (reference.size() != actual.size() || std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float))) {
                std::size_t changed = 0;
                double maximum = 0;
                for (std::size_t i = 0; i < std::min(reference.size(), actual.size()); ++i) {
                    changed += std::memcmp(&reference[i], &actual[i], sizeof(float)) != 0;
                    maximum = std::max(maximum, std::abs(static_cast<double>(reference[i]) - actual[i]));
                }
                std::fprintf(stderr, "IQ4 native GPU differs: sample=%d changed=%zu max_abs=%.9g\n", sample, changed, maximum);
                for (std::size_t i = 0; i < std::min<std::size_t>(4, reference.size()); ++i)
                    std::fprintf(stderr, "row=%zu cpu=%.9g gpu=%.9g\n", i, reference[i], actual[i]);
                return 1;
            }
        }
        bool rejected = false;
        try { gpu(std::vector<std::uint8_t>(q6_type ? loaded.weights.intermediate / 256 * 292 :
                                                      loaded.weights.intermediate / 32 * 34), {{0, 0, 1}}); }
        catch (const std::exception &) { rejected = true; }
        if (!rejected) throw std::runtime_error("nonresident expert accepted");
        rejected = false;
        try { gpu({0}, {{0, 0, 0}}); }
        catch (const std::exception &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid packed input accepted");
        std::puts("native quantized GPU dot matches CPU bits on bounded inputs");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "IQ4 GPU test: %s\n", error.what());
        return 1;
    }
}
