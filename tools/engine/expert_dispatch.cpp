#include "expert_dispatch.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <future>
#include <stdexcept>

namespace vrambridge::experimental {
namespace {

constexpr std::size_t max_floats = 32 * 1024 * 1024;

void validate_result(const std::vector<float> & values, std::size_t jobs, std::size_t width) {
    if (values.size() != jobs * width) throw std::runtime_error("expert backend returned an invalid output shape");
    if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("expert backend returned non-finite output");
}

} // namespace

ExpertPlan::ExpertPlan(std::size_t width, const std::vector<std::vector<Route>> & routes,
                       const std::vector<std::uint8_t> & gpu_resident)
    : width_(width), rows_(routes.size()) {
    if (!width || width > 65536 || routes.empty() || routes.size() > 4096 ||
        gpu_resident.empty() || gpu_resident.size() > 65536 || routes.size() > max_floats / width)
        throw std::invalid_argument("invalid expert plan dimensions");
    for (const auto resident : gpu_resident)
        if (resident > 1) throw std::invalid_argument("GPU residency entries must be zero or one");
    std::vector<std::uint32_t> seen(gpu_resident.size(), 0);
    for (std::size_t row = 0; row < routes.size(); ++row) {
        if (routes[row].empty() || routes[row].size() > gpu_resident.size())
            throw std::invalid_argument("invalid number of routed experts");
        const auto generation = static_cast<std::uint32_t>(row + 1);
        for (std::size_t rank = 0; rank < routes[row].size(); ++rank) {
            const auto & route = routes[row][rank];
            if (route.expert >= gpu_resident.size() || !std::isfinite(route.weight) ||
                seen[route.expert] == generation)
                throw std::invalid_argument("invalid or duplicate expert route");
            seen[route.expert] = generation;
            if (contributions_.size() >= max_floats / width)
                throw std::invalid_argument("expert output exceeds the bounded plan size");
            const bool gpu = gpu_resident[route.expert] != 0;
            auto & jobs = gpu ? gpu_jobs_ : cpu_jobs_;
            const auto offset = jobs.size();
            jobs.push_back({row, rank, route.expert});
            contributions_.push_back({row, offset, route.weight, gpu});
        }
    }
}

std::vector<float> ExpertPlan::execute(const ExpertRunner & cpu, const ExpertRunner & gpu) const {
    if ((!cpu_jobs_.empty() && !cpu) || (!gpu_jobs_.empty() && !gpu))
        throw std::invalid_argument("selected expert backend is unavailable");

    std::future<std::vector<float>> gpu_result;
    if (!gpu_jobs_.empty()) {
        gpu_result = std::async(std::launch::async, [&] { return gpu(gpu_jobs_); });
    }
    std::vector<float> cpu_values, gpu_values;
    std::exception_ptr cpu_error, gpu_error;
    try {
        if (!cpu_jobs_.empty()) cpu_values = cpu(cpu_jobs_);
    } catch (...) {
        cpu_error = std::current_exception();
    }
    try {
        if (gpu_result.valid()) gpu_values = gpu_result.get();
    } catch (...) {
        gpu_error = std::current_exception();
    }
    if (cpu_error) std::rethrow_exception(cpu_error);
    if (gpu_error) std::rethrow_exception(gpu_error);
    validate_result(cpu_values, cpu_jobs_.size(), width_);
    validate_result(gpu_values, gpu_jobs_.size(), width_);

    std::vector<float> output(rows_ * width_, 0.0f);
    for (const auto & contribution : contributions_) {
        const auto & values = contribution.gpu ? gpu_values : cpu_values;
        const std::size_t start = contribution.device_offset * width_;
        const std::size_t destination = contribution.row * width_;
        for (std::size_t i = 0; i < width_; ++i) {
            output[destination + i] += contribution.weight * values[start + i];
            if (!std::isfinite(output[destination + i]))
                throw std::runtime_error("expert reduction produced a non-finite output");
        }
    }
    return output;
}

} // namespace vrambridge::experimental
