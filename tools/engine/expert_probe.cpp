#include "expert_dispatch.hpp"
#include "f32_cuda.hpp"
#include "gguf_experts.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vrambridge::experimental;

namespace {

int number(const std::string & text) {
    int result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || result < 0)
        throw std::invalid_argument("invalid nonnegative integer");
    return result;
}

struct Error {
    double max_abs = 0, relative_l2 = 0, max_reference = 0;
    bool passed = false;
};

Error compare(const std::vector<float> & reference, const std::vector<float> & actual) {
    if (reference.size() != actual.size() || reference.empty()) throw std::runtime_error("output shapes differ");
    Error result;
    double square = 0, base = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double difference = static_cast<double>(actual[i]) - reference[i];
        result.max_abs = std::max(result.max_abs, std::abs(difference));
        result.max_reference = std::max(result.max_reference, std::abs(static_cast<double>(reference[i])));
        square += difference * difference;
        base += static_cast<double>(reference[i]) * reference[i];
    }
    result.relative_l2 = std::sqrt(square / std::max(base, 1e-30));
    result.passed = result.max_abs <= 1e-5 + 1e-4 * result.max_reference && result.relative_l2 <= 1e-4;
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: %s FIRST_GGUF LAYER ID,ID[,ID...] silu|openai\n", argv[0]);
        return 2;
    }
    try {
        const int layer = number(argv[2]);
        const std::string input_ids = argv[3];
        std::vector<std::uint32_t> ids;
        for (std::size_t start = 0; start < input_ids.size();) {
            const auto end = input_ids.find(',', start);
            ids.push_back(static_cast<std::uint32_t>(number(input_ids.substr(start, end - start))));
            if (end == std::string::npos) break;
            start = end + 1;
            if (start == input_ids.size()) throw std::invalid_argument("trailing expert separator");
        }
        if (ids.size() < 2) throw std::invalid_argument("at least two experts are required");
        const std::string kind = argv[4];
        if (kind != "silu" && kind != "openai") throw std::invalid_argument("unknown activation");
        const auto loaded = load_gguf_experts(argv[1], layer, ids,
            kind == "silu" ? F32Activation::silu : F32Activation::openai_swiglu);
        const auto & weights = loaded.weights;
        std::vector<float> input(weights.hidden * 2);
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = 0.1f * std::sin(static_cast<float>(i) * 0.013f);
        std::vector<std::vector<Route>> routes(2);
        std::vector<std::uint8_t> none(ids.size(), 0), mixed(ids.size()), all(ids.size(), 1);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            mixed[i] = static_cast<std::uint8_t>((i % 2) == 0);
            const float weight = 1.0f / static_cast<float>(ids.size());
            routes[0].push_back({static_cast<std::uint32_t>(i), weight});
            routes[1].push_back({static_cast<std::uint32_t>(ids.size() - i - 1), weight});
        }
        const auto cpu = [&](const std::vector<ExpertJob> & jobs) { return execute_f32_cpu(weights, input, jobs); };
        const auto reference = ExpertPlan(weights.hidden, routes, none).execute(cpu, {});
        F32CudaExperts mixed_cuda(weights, mixed);
        const auto mixed_gpu = [&](const std::vector<ExpertJob> & jobs) { return mixed_cuda.execute(input, jobs); };
        const auto split = ExpertPlan(weights.hidden, routes, mixed).execute(cpu, mixed_gpu);
        const auto split_error = compare(reference, split);
        F32CudaExperts full_cuda(weights, all);
        const auto full_gpu = [&](const std::vector<ExpertJob> & jobs) { return full_cuda.execute(input, jobs); };
        const auto full_error = compare(reference, ExpertPlan(weights.hidden, routes, all).execute({}, full_gpu));
        std::printf("{\"layer\":%d,\"selected_experts\":%zu,\"hidden\":%zu,\"intermediate\":%zu,"
                    "\"types\":[\"%s\",\"%s\",\"%s\"],\"activation\":\"%s\","
                    "\"input\":\"synthetic\",\"reference\":\"dequantized_f32_cpu\","
                    "\"mixed\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s},"
                    "\"all_gpu\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s}}\n",
                    layer, ids.size(), weights.hidden, weights.intermediate,
                    loaded.weight_types[0].c_str(), loaded.weight_types[1].c_str(), loaded.weight_types[2].c_str(),
                    kind.c_str(), split_error.max_abs, split_error.relative_l2, split_error.passed ? "true" : "false",
                    full_error.max_abs, full_error.relative_l2, full_error.passed ? "true" : "false");
        return split_error.passed && full_error.passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "expert probe: %s\n", error.what());
        return 2;
    }
}
