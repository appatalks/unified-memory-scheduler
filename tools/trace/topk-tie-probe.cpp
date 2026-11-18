#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct Counts {
    uint64_t rows = 0;
    uint64_t changed_sets = 0;
    uint64_t order_only = 0;
    uint64_t cutoff_ties = 0;
    uint64_t invalid = 0;
    int max_replaced = 0;
};

Counts compare(const std::vector<float> & scores, const std::vector<int32_t> & reference,
               const std::vector<int32_t> & candidate, int cols, int rows, int k) {
    require(cols > 0 && rows > 0 && k > 0 && k <= cols && scores.size() == size_t(cols) * rows &&
            reference.size() == size_t(k) * rows && candidate.size() == reference.size(), "invalid comparison dimensions");
    Counts result;
    for (int row = 0; row < rows; ++row) {
        ++result.rows;
        const auto * values = scores.data() + size_t(row) * cols;
        const auto first = candidate.begin() + size_t(row) * k;
        const auto ref_first = reference.begin() + size_t(row) * k;
        std::vector<int32_t> selected(first, first + k), previous(ref_first, ref_first + k);
        std::sort(selected.begin(), selected.end());
        std::sort(previous.begin(), previous.end());
        const bool valid = selected.front() >= 0 && selected.back() < cols &&
                           std::adjacent_find(selected.begin(), selected.end()) == selected.end();
        if (!valid) {
            ++result.invalid;
            continue;
        }
        float cutoff = std::numeric_limits<float>::infinity();
        std::vector<bool> chosen(cols, false);
        for (const int index : selected) {
            chosen[index] = true;
            cutoff = std::min(cutoff, values[index]);
        }
        int equal_in = 0, equal_all = 0;
        float max_excluded = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < cols; ++col) {
            if (values[col] == cutoff) {
                ++equal_all;
                equal_in += chosen[col];
            }
            if (!chosen[col]) max_excluded = std::max(max_excluded, values[col]);
        }
        if (max_excluded > cutoff || !std::isfinite(cutoff)) ++result.invalid;
        if (std::isfinite(cutoff) && equal_all > equal_in) ++result.cutoff_ties;
        if (selected != previous) {
            ++result.changed_sets;
            std::vector<int32_t> diff;
            std::set_difference(selected.begin(), selected.end(), previous.begin(), previous.end(), std::back_inserter(diff));
            result.max_replaced = std::max(result.max_replaced, static_cast<int>(diff.size()));
        } else if (!std::equal(first, first + k, ref_first)) {
            ++result.order_only;
        }
    }
    return result;
}

bool self_test() {
    const std::vector<float> scores{2, 1, 1, 0};
    const std::vector<int32_t> ref{0, 1};
    const auto changed = compare(scores, ref, {0, 2}, 4, 1, 2);
    const auto order = compare(scores, ref, {1, 0}, 4, 1, 2);
    const auto invalid = compare(scores, ref, {0, 3}, 4, 1, 2);
    const auto duplicate = compare(scores, ref, {0, 0}, 4, 1, 2);
    return changed.changed_sets == 1 && changed.cutoff_ties == 1 && changed.invalid == 0 &&
           order.changed_sets == 0 && order.order_only == 1 && order.cutoff_ties == 1 &&
           invalid.invalid == 1 && duplicate.invalid == 1;
}

void run(ggml_backend_t backend, const char * name, int kind, int iterations, int cols, int rows, int k) {
    std::vector<float> scores(size_t(cols) * rows);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float value = 0;
            if (kind == 0) value = float(cols - col);
            if (kind == 1) value = col < k ? 1.0f : 0.0f;
            if (kind == 2) value = col < 256 ? 1.0f : 0.0f;
            if (kind == 3) value = col < 256 ? 1.0f : col < 1024 ? 0.0f : -std::numeric_limits<float>::infinity();
            if (kind == 4) value = float((col * 37 + row * 17) % 31);
            scores[size_t(row) * cols + col] = value;
        }
    }
    std::vector<int32_t> oracle;
    if (std::getenv("VRAMBRIDGE_TOPK_REQUIRE_CANONICAL")) {
        oracle.resize(size_t(k) * rows);
        std::vector<int32_t> indexes(cols);
        for (int row = 0; row < rows; ++row) {
            std::iota(indexes.begin(), indexes.end(), 0);
            const auto * values = scores.data() + size_t(row) * cols;
            std::partial_sort(indexes.begin(), indexes.begin() + k, indexes.end(), [values](int a, int b) {
                return values[a] != values[b] ? values[a] > values[b] : a < b;
            });
            std::copy(indexes.begin(), indexes.begin() + k, oracle.begin() + size_t(row) * k);
        }
    }
    ggml_init_params params{1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    require(ctx != nullptr, "context allocation failed");
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, rows);
    auto * output = ggml_top_k(ctx.get(), input, k);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "device allocation failed");
    ggml_backend_tensor_set(input, scores.data(), 0, scores.size() * sizeof(float));
    std::vector<int32_t> reference(size_t(k) * rows), candidate(reference.size());
    Counts total;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "top-k failed");
        ggml_backend_tensor_get(output, candidate.data(), 0, candidate.size() * sizeof(int32_t));
        if (!oracle.empty()) require(candidate == oracle, "top-k differs from canonical CPU oracle");
        if (iteration == 0) reference = candidate;
        const auto current = compare(scores, reference, candidate, cols, rows, k);
        require(current.invalid == 0, "top-k returned invalid selections");
        total.rows += current.rows;
        total.changed_sets += current.changed_sets;
        total.order_only += current.order_only;
        total.cutoff_ties += current.cutoff_ties;
        total.max_replaced = std::max(total.max_replaced, current.max_replaced);
    }
    std::printf("{\"case\":\"%s\",\"cols\":%d,\"rows\":%d,\"k\":%d,\"iterations\":%d,"
                "\"row_comparisons\":%llu,\"changed_sets\":%llu,\"order_only\":%llu,"
                "\"cutoff_ties\":%llu,\"max_replaced\":%d}\n", name, cols, rows, k, iterations,
                (unsigned long long) total.rows, (unsigned long long) total.changed_sets,
                (unsigned long long) total.order_only, (unsigned long long) total.cutoff_ties, total.max_replaced);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) return self_test() ? 0 : 1;
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s CUDA_PLUGIN | --self-test\n", argv[0]);
        return 2;
    }
    try {
        require(ggml_backend_load(argv[1]) != nullptr, "CUDA plugin load failed");
        const auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        require(device && std::strncmp(ggml_backend_dev_name(device), "CUDA", 4) == 0, "CUDA unavailable");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
        require(backend != nullptr, "CUDA backend initialization failed");
        for (int kind = 0; kind < 5; ++kind) {
            const char * names[] = {"unique", "within_selected_ties", "boundary_ties", "masked_boundary_ties", "mixed_ties"};
            run(backend.get(), names[kind], kind, 64, 2048, 512, 512);
        }
        run(backend.get(), "fallback_unique", 0, 8, 512, 16, 128);
        return 0;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "top-k tie probe failed: %s\n", exc.what());
        return 1;
    }
}
