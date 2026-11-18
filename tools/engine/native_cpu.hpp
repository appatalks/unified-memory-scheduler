#pragma once

#include "gguf_experts.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vrambridge::experimental {

class NativeCpuExperts {
public:
    // The loaded expert weights must outlive this executor.
    NativeCpuExperts(const LoadedExperts & weights, const std::string & cpu_plugin);
    ~NativeCpuExperts();
    NativeCpuExperts(const NativeCpuExperts &) = delete;
    NativeCpuExperts & operator=(const NativeCpuExperts &) = delete;

    std::vector<float> execute(const std::vector<float> & input, const std::vector<ExpertJob> & jobs,
                               const F32Transforms & observations = {}) const;
    std::vector<float> activations(const std::vector<float> & input, const std::vector<ExpertJob> & jobs,
                                   const F32Transforms & observations = {}) const;
    std::vector<float> execute_down(const std::vector<float> & activations, const std::vector<ExpertJob> & jobs) const;
    std::vector<std::uint8_t> pack_down(const std::vector<float> & activations) const;
    std::vector<float> execute_down_packed(const std::vector<std::uint8_t> & packed, const std::vector<ExpertJob> & jobs) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vrambridge::experimental
