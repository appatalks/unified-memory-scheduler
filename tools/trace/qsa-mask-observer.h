#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace qsa_mask_observer {

struct Counts {
    uint64_t elements = 0;
    uint64_t bit_differences = 0;
    uint64_t value_differences = 0;
    uint64_t different_rows = 0;
};

inline Counts compare(const void * selected, const void * causal, ggml_type type, int cols, int rows) {
    if (!selected || !causal || cols <= 0 || rows <= 0 || (type != GGML_TYPE_F16 && type != GGML_TYPE_F32)) {
        throw std::runtime_error("invalid mask comparison");
    }
    Counts result;
    const size_t width = type == GGML_TYPE_F16 ? sizeof(ggml_fp16_t) : sizeof(float);
    const auto * a = static_cast<const uint8_t *>(selected);
    const auto * b = static_cast<const uint8_t *>(causal);
    for (int row = 0; row < rows; ++row) {
        bool different = false;
        for (int col = 0; col < cols; ++col) {
            const size_t offset = (size_t(row) * cols + col) * width;
            ++result.elements;
            if (std::memcmp(a + offset, b + offset, width) == 0) {
                uint32_t bits = 0;
                std::memcpy(&bits, a + offset, width);
                const bool nan = type == GGML_TYPE_F16 ? (bits & 0x7c00u) == 0x7c00u && (bits & 0x03ffu) != 0
                                                       : (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
                if (nan) { ++result.value_differences; different = true; }
                continue;
            }
            ++result.bit_differences;
            float av, bv;
            if (type == GGML_TYPE_F16) {
                ggml_fp16_t ah, bh;
                std::memcpy(&ah, a + offset, sizeof(ah));
                std::memcpy(&bh, b + offset, sizeof(bh));
                av = ggml_fp16_to_fp32(ah); bv = ggml_fp16_to_fp32(bh);
            } else {
                std::memcpy(&av, a + offset, sizeof(av));
                std::memcpy(&bv, b + offset, sizeof(bv));
            }
            if (std::isnan(av) || std::isnan(bv) || av != bv) {
                ++result.value_differences;
                different = true;
            }
        }
        result.different_rows += different;
    }
    return result;
}

struct Observer {
    bool active = false;
    std::string error;
    std::array<int, 48> frames{};
    std::array<int, 48> rows{};
    uint64_t same_storage = 0;
    uint64_t causal_differences = 0;
    int next_layer = 3;
    ggml_type causal_type = GGML_TYPE_COUNT;
    int causal_cols = 0, causal_rows = 0;
    std::vector<uint8_t> causal_reference;
    Counts total;
};

inline bool callback(ggml_tensor * tensor, bool ask, void * data) noexcept {
    auto & obs = *static_cast<Observer *>(data);
    constexpr char prefix[] = "kq_mask_qsa-";
    const bool match = std::strncmp(tensor->name, prefix, sizeof(prefix) - 1) == 0;
    if (ask) return obs.active && match;
    if (!obs.active || !match) return true;
    try {
        char * end = nullptr;
        const long layer = std::strtol(tensor->name + sizeof(prefix) - 1, &end, 10);
        if (*end || layer < 3 || layer > 47 || layer % 4 != 3) throw std::runtime_error("unexpected QSA layer");
        if (layer != obs.next_layer) throw std::runtime_error("unexpected QSA mask order");
        const auto * causal = tensor->src[1];
        if (!causal || !tensor->buffer || !causal->buffer || !ggml_is_contiguous(tensor) || !ggml_is_contiguous(causal) ||
            !ggml_are_same_shape(tensor, causal) || tensor->type != causal->type ||
            (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_F32) || ggml_nbytes(tensor) > 16u * 1024 * 1024) {
            throw std::runtime_error("unsupported QSA mask layout");
        }
        const size_t size = ggml_nbytes(tensor);
        const bool alias = tensor->data == causal->data;
        obs.same_storage += alias;
        if (layer == 3 && alias) throw std::runtime_error("first QSA mask aliases its causal input");
        const int rows = int(tensor->ne[1] * tensor->ne[2] * tensor->ne[3]);
        std::vector<uint8_t> selected(size);
        ggml_backend_tensor_get(tensor, selected.data(), 0, size);
        if (layer == 3) {
            obs.causal_type = tensor->type;
            obs.causal_cols = int(tensor->ne[0]);
            obs.causal_rows = rows;
            obs.causal_reference.resize(size);
            ggml_backend_tensor_get(causal, obs.causal_reference.data(), 0, size);
        } else if (obs.causal_reference.size() != size || obs.causal_type != tensor->type ||
                   obs.causal_cols != tensor->ne[0] || obs.causal_rows != rows) {
            throw std::runtime_error("QSA causal mask size changed within a batch");
        } else if (!alias) {
            std::vector<uint8_t> original(size);
            ggml_backend_tensor_get(causal, original.data(), 0, size);
            obs.causal_differences += std::memcmp(original.data(), obs.causal_reference.data(), size) != 0;
        }
        const auto result = compare(selected.data(), obs.causal_reference.data(), tensor->type, int(tensor->ne[0]), rows);
        ++obs.frames[layer];
        obs.rows[layer] += rows;
        obs.total.elements += result.elements;
        obs.total.bit_differences += result.bit_differences;
        obs.total.value_differences += result.value_differences;
        obs.total.different_rows += result.different_rows;
        obs.next_layer = layer == 47 ? 3 : int(layer) + 4;
        return true;
    } catch (const std::exception & exc) {
        obs.error = exc.what();
        return false;
    }
}

inline bool self_test() {
    const uint16_t a[] = {0, 0xfc00, 0, 0xfc00};
    const uint16_t b[] = {0x8000, 0xfc00, 0xfc00, 0xfc00};
    const auto c = compare(a, b, GGML_TYPE_F16, 2, 2);
    const uint16_t nan[] = {0x7e00};
    const auto invalid = compare(nan, nan, GGML_TYPE_F16, 1, 1);
    return c.elements == 4 && c.bit_differences == 2 && c.value_differences == 1 && c.different_rows == 1 &&
           invalid.value_differences == 1 && invalid.different_rows == 1;
}

} // namespace qsa_mask_observer
