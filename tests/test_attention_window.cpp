#include "attention_window.h"

#include <cstdio>
#include <memory>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({1024 * 1024, nullptr, false}), ggml_free);
        require(bool(ctx), "context allocation failed");
        auto * query = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 64, 8, 4, 1);
        auto * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 128, 32, 1, 1);
        require(!attention_window_eligible(1, 1) && !attention_window_eligible(28, 1) &&
                !attention_window_eligible(33, 33) && attention_window_eligible(2, 2) &&
                attention_window_eligible(4, 4), "window/prefill eligibility changed");
        for (int64_t position = 0; position < 4; ++position) {
            auto [q, m] = attention_window_column(ctx.get(), query, mask, position);
            require(q->ne[0] == 64 && q->ne[1] == 8 && q->ne[2] == 1, "query shape changed");
            require(q->data == static_cast<char *>(query->data) + position * query->nb[2], "wrong query offset");
            require(m->data == static_cast<char *>(mask->data) + position * mask->nb[1], "wrong mask offset");
            require(m->ne[0] == 128 && m->ne[1] == 1 && ggml_is_contiguous(m), "mask column is not contiguous");
        }
        for (int64_t position : {-1, 4}) {
            bool rejected = false;
            try { attention_window_column(ctx.get(), query, mask, position); }
            catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "invalid query index was accepted");
        }
        auto * wrong_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 128, 32);
        bool rejected = false;
        try { attention_window_column(ctx.get(), query, wrong_mask, 0); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "non-F16 mask was accepted");
        auto * linear = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, 4);
        auto * experts = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 64, 4, 4);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, 4);
        for (int64_t position = 0; position < 4; ++position) {
            auto * column = attention_window_linear_column(ctx.get(), linear, position);
            require(column->ne[1] == 1 &&
                    column->data == static_cast<char *>(linear->data) + position * linear->nb[1],
                    "linear token column changed");
            const auto [expert_column, index_column] = attention_window_expert_column(ctx.get(), experts, ids, position);
            require(expert_column->ne[2] == 1 && index_column->ne[1] == 1 &&
                    expert_column->data == static_cast<char *>(experts->data) + position * experts->nb[2] &&
                    index_column->data == static_cast<char *>(ids->data) + position * ids->nb[1],
                    "expert token/index alignment changed");
        }
        std::puts("attention window column shapes, offsets and invalid-input gates passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "attention window test: %s\n", error.what());
        return 1;
    }
}
