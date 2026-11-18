#pragma once

#include "f32_experts.hpp"
#include "ggml.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vrambridge::experimental {

struct LoadedExperts {
    F32ExpertWeights weights;
    std::vector<std::uint32_t> original_ids;
    std::vector<std::string> weight_types;
    std::array<ggml_type, 3> quant_types;
    std::array<std::vector<std::uint8_t>, 3> packed_weights;
};

LoadedExperts load_gguf_experts(const std::filesystem::path & first_shard, int layer,
                                const std::vector<std::uint32_t> & ids, F32Activation activation);

} // namespace vrambridge::experimental
