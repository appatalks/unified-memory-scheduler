#include "vrambridge/info.hpp"

namespace vrambridge {

CudaInfo collect_cuda_info() {
    return {
        .available = false,
        .error = "VRAMBridge was built without CUDA support",
    };
}

std::vector<BandwidthResult> run_cuda_benchmarks(const BenchmarkOptions &) {
    return {};
}

} // namespace vrambridge