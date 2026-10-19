#pragma once

#include "ggml.h"

#include <stdexcept>
#include <utility>

inline bool attention_window_eligible(int64_t tokens, int64_t outputs) {
    return tokens > 1 && tokens <= 32 && tokens == outputs;
}

inline ggml_tensor * attention_window_linear_column(ggml_context * ctx, ggml_tensor * input, int64_t position) {
    if (!ctx || !input || input->ne[2] != 1 || input->ne[3] != 1 ||
        position < 0 || position >= input->ne[1]) {
        throw std::runtime_error("linear columns require two-dimensional inputs and a valid column");
    }
    return ggml_view_2d(ctx, input, input->ne[0], 1, input->nb[1], position * input->nb[1]);
}

inline std::pair<ggml_tensor *, ggml_tensor *> attention_window_expert_column(
        ggml_context * ctx, ggml_tensor * input, ggml_tensor * ids, int64_t position) {
    if (!ctx || !input || !ids || input->ne[3] != 1 || ids->type != GGML_TYPE_I32 ||
        ids->ne[1] != input->ne[2] || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        position < 0 || position >= input->ne[2]) {
        throw std::runtime_error("expert columns require aligned token/index axes");
    }
    return {
        ggml_view_3d(ctx, input, input->ne[0], input->ne[1], 1,
                     input->nb[1], input->nb[2], position * input->nb[2]),
        ggml_view_2d(ctx, ids, ids->ne[0], 1, ids->nb[1], position * ids->nb[1])};
}

inline std::pair<ggml_tensor *, ggml_tensor *> attention_window_column(
        ggml_context * ctx, ggml_tensor * query, ggml_tensor * mask, int64_t position) {
    if (!ctx || !query || !mask || query->ne[3] != 1 || query->ne[2] < 1 || query->ne[2] > 32 ||
        mask->type != GGML_TYPE_F16 || mask->ne[1] < query->ne[2] || mask->ne[2] != 1 ||
        mask->ne[3] != 1 || !ggml_is_contiguous(mask) || position < 0 || position >= query->ne[2]) {
        throw std::runtime_error("attention columns require one stream, a bounded query axis and a contiguous F16 mask");
    }
    auto * q = ggml_view_4d(ctx, query, query->ne[0], query->ne[1], 1, 1,
                           query->nb[1], query->nb[2], query->nb[3], position * query->nb[2]);
    auto * m = ggml_view_4d(ctx, mask, mask->ne[0], 1, 1, 1,
                           mask->nb[1], mask->nb[2], mask->nb[3], position * mask->nb[1]);
    return {q, m};
}
