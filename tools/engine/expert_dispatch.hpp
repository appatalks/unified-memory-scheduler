#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace vrambridge::experimental {

struct Route {
    std::uint32_t expert;
    float weight;
};

struct ExpertJob {
    std::size_t row;
    std::size_t rank;
    std::uint32_t expert;
};

using ExpertRunner = std::function<std::vector<float>(const std::vector<ExpertJob> &)>;

class ExpertPlan {
public:
    ExpertPlan(std::size_t width, const std::vector<std::vector<Route>> & routes,
               const std::vector<std::uint8_t> & gpu_resident);

    std::size_t width() const noexcept { return width_; }
    std::size_t rows() const noexcept { return rows_; }
    const std::vector<ExpertJob> & cpu_jobs() const noexcept { return cpu_jobs_; }
    const std::vector<ExpertJob> & gpu_jobs() const noexcept { return gpu_jobs_; }

    std::vector<float> execute(const ExpertRunner & cpu, const ExpertRunner & gpu) const;

private:
    struct Contribution {
        std::size_t row;
        std::size_t device_offset;
        float weight;
        bool gpu;
    };

    std::size_t width_;
    std::size_t rows_;
    std::vector<ExpertJob> cpu_jobs_;
    std::vector<ExpertJob> gpu_jobs_;
    std::vector<Contribution> contributions_;
};

} // namespace vrambridge::experimental
