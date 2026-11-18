#pragma once

#include "gguf_experts.hpp"

#include <memory>
#include <vector>

namespace vrambridge::experimental {

class Iq4NativeCudaDown {
public:
    Iq4NativeCudaDown(const LoadedExperts & weights, const std::vector<std::uint8_t> & resident, int device = 0);
    ~Iq4NativeCudaDown();
    Iq4NativeCudaDown(const Iq4NativeCudaDown &) = delete;
    Iq4NativeCudaDown & operator=(const Iq4NativeCudaDown &) = delete;

    std::vector<float> execute(const std::vector<std::uint8_t> & packed_q8, const std::vector<ExpertJob> & jobs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Q6NativeCudaDown {
public:
    Q6NativeCudaDown(const LoadedExperts & weights, const std::vector<std::uint8_t> & resident, int device = 0);
    ~Q6NativeCudaDown();
    Q6NativeCudaDown(const Q6NativeCudaDown &) = delete;
    Q6NativeCudaDown & operator=(const Q6NativeCudaDown &) = delete;

    std::vector<float> execute(const std::vector<std::uint8_t> & packed_q8, const std::vector<ExpertJob> & jobs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Mxfp4NativeCudaDown {
public:
    Mxfp4NativeCudaDown(const LoadedExperts & weights, const std::vector<std::uint8_t> & resident, int device = 0);
    ~Mxfp4NativeCudaDown();
    Mxfp4NativeCudaDown(const Mxfp4NativeCudaDown &) = delete;
    Mxfp4NativeCudaDown & operator=(const Mxfp4NativeCudaDown &) = delete;

    std::vector<float> execute(const std::vector<std::uint8_t> & packed_q8, const std::vector<ExpertJob> & jobs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vrambridge::experimental
