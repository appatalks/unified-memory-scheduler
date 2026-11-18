#pragma once

#include "f32_experts.hpp"

#include <memory>
#include <vector>

namespace vrambridge::experimental {

class F32CudaExperts {
public:
    F32CudaExperts(const F32ExpertWeights & weights, const std::vector<std::uint8_t> & resident, int device = 0);
    ~F32CudaExperts();
    F32CudaExperts(const F32CudaExperts &) = delete;
    F32CudaExperts & operator=(const F32CudaExperts &) = delete;

    std::vector<float> execute(const std::vector<float> & input, const std::vector<ExpertJob> & jobs,
                               const F32Transforms & transforms = {});
    std::vector<float> execute_down(const std::vector<float> & activations, const std::vector<ExpertJob> & jobs,
                                    const F32Transforms & transforms = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vrambridge::experimental
