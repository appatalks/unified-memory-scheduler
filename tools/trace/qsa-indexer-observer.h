#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qsa_observer {

struct Frame {
    int cols = 0;
    int rows = 0;
    int k = 0;
    bool score_device = false;
    bool ids_device = false;
    std::vector<float> scores;
    std::vector<int32_t> ids;
};

struct Summary {
    uint64_t rows = 0;
    uint64_t same_scores = 0;
    uint64_t changed_sets = 0;
    uint64_t changed_visible_sets = 0;
    uint64_t order_only = 0;
    uint64_t same_score_changed_sets = 0;
    uint64_t same_score_tie_changed_sets = 0;
    uint64_t first_finite_cutoff_ties = 0;
    uint64_t second_finite_cutoff_ties = 0;
    uint64_t invalid = 0;
};

inline bool valid_selection(const float * scores, int cols, const std::vector<int32_t> & ids) {
    std::vector<bool> selected(cols, false);
    float cutoff = std::numeric_limits<float>::infinity();
    for (int id : ids) {
        selected[id] = true;
        if (std::isnan(scores[id])) return false;
        cutoff = std::min(cutoff, scores[id]);
    }
    for (int col = 0; col < cols; ++col) {
        if (std::isnan(scores[col]) || (!selected[col] && scores[col] > cutoff)) return false;
    }
    return true;
}

inline bool selected_cutoff_tie(const float * scores, int cols, const std::vector<int32_t> & ids) {
    float cutoff = std::numeric_limits<float>::infinity();
    for (int id : ids) cutoff = std::min(cutoff, scores[id]);
    if (!std::isfinite(cutoff)) return false;
    int all = 0, selected = 0;
    for (int i = 0; i < cols; ++i) all += scores[i] == cutoff;
    for (int id : ids) selected += scores[id] == cutoff;
    return all > selected;
}

inline Summary compare(const std::vector<Frame> & first, const std::vector<Frame> & second) {
    if (first.size() != second.size() || first.empty()) throw std::runtime_error("QSA frame count mismatch");
    Summary summary;
    for (size_t frame = 0; frame < first.size(); ++frame) {
        const auto & a = first[frame];
        const auto & b = second[frame];
        if (a.cols != b.cols || a.rows != b.rows || a.k != b.k || a.cols <= 0 || a.rows <= 0 ||
            a.k <= 0 || a.k > a.cols || a.scores.size() != size_t(a.cols) * a.rows ||
            b.scores.size() != a.scores.size() || a.ids.size() != size_t(a.k) * a.rows || b.ids.size() != a.ids.size()) {
            throw std::runtime_error("QSA frame shape mismatch");
        }
        for (int row = 0; row < a.rows; ++row) {
            ++summary.rows;
            const auto * as = a.scores.data() + size_t(row) * a.cols;
            const auto * bs = b.scores.data() + size_t(row) * a.cols;
            const auto ai = a.ids.begin() + size_t(row) * a.k;
            const auto bi = b.ids.begin() + size_t(row) * a.k;
            std::vector<int32_t> aid(ai, ai + a.k), bid(bi, bi + a.k);
            std::sort(aid.begin(), aid.end());
            std::sort(bid.begin(), bid.end());
            if (aid.front() < 0 || aid.back() >= a.cols || bid.front() < 0 || bid.back() >= a.cols ||
                std::adjacent_find(aid.begin(), aid.end()) != aid.end() ||
                std::adjacent_find(bid.begin(), bid.end()) != bid.end()) {
                ++summary.invalid;
                continue;
            }
            if (!valid_selection(as, a.cols, aid) || !valid_selection(bs, b.cols, bid)) {
                ++summary.invalid;
                continue;
            }
            const bool same_score = std::memcmp(as, bs, size_t(a.cols) * sizeof(float)) == 0;
            summary.same_scores += same_score;
            const bool tie_a = selected_cutoff_tie(as, a.cols, aid);
            const bool tie_b = selected_cutoff_tie(bs, b.cols, bid);
            summary.first_finite_cutoff_ties += tie_a;
            summary.second_finite_cutoff_ties += tie_b;
            std::vector<int32_t> visible_a, visible_b;
            for (int id : aid) if (as[id] != -std::numeric_limits<float>::infinity()) visible_a.push_back(id);
            for (int id : bid) if (bs[id] != -std::numeric_limits<float>::infinity()) visible_b.push_back(id);
            summary.changed_visible_sets += visible_a != visible_b;
            if (aid != bid) {
                ++summary.changed_sets;
                summary.same_score_changed_sets += same_score;
                summary.same_score_tie_changed_sets += same_score && tie_a && tie_b;
            } else if (!std::equal(ai, ai + a.k, bi)) {
                ++summary.order_only;
            }
        }
    }
    return summary;
}

struct Observer {
    explicit Observer(int layer) : score_name("indexer_score-" + std::to_string(layer)),
                                   ids_name("indexer_top_k-" + std::to_string(layer)) {}
    std::string score_name, ids_name;
    bool active = false;
    bool waiting = false;
    size_t bytes = 0;
    std::string error;
    std::vector<Frame> frames;

    void start() {
        if (waiting) throw std::runtime_error("incomplete QSA tensor pair");
        frames.clear();
        bytes = 0;
        error.clear();
        active = true;
    }

    std::vector<Frame> finish() {
        active = false;
        if (!error.empty()) throw std::runtime_error(error);
        if (waiting || frames.empty()) throw std::runtime_error("incomplete QSA capture");
        return std::move(frames);
    }
};

inline bool callback(ggml_tensor * tensor, bool ask, void * data) noexcept {
    auto & obs = *static_cast<Observer *>(data);
    const bool score = obs.score_name == tensor->name;
    const bool ids = obs.ids_name == tensor->name;
    if (ask) return obs.active && (score || ids);
    if (!obs.active || (!score && !ids)) return true;
    try {
        if (!ggml_is_contiguous(tensor) || !tensor->buffer || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
            tensor->ne[0] <= 0 || tensor->ne[0] > 32768 || tensor->ne[1] <= 0 || tensor->ne[1] > 2048) {
            throw std::runtime_error("unsupported QSA tensor layout");
        }
        const size_t size = ggml_nbytes(tensor);
        if (size > 16u * 1024 * 1024 || obs.bytes + size > 128u * 1024 * 1024) {
            throw std::runtime_error("QSA capture exceeded memory bound");
        }
        obs.bytes += size;
        if (score) {
            if (obs.waiting || tensor->type != GGML_TYPE_F32) throw std::runtime_error("invalid QSA score sequence");
            obs.frames.emplace_back();
            auto & frame = obs.frames.back();
            frame.cols = int(tensor->ne[0]);
            frame.rows = int(tensor->ne[1]);
            frame.score_device = !ggml_backend_buffer_is_host(tensor->buffer);
            frame.scores.resize(size_t(frame.cols) * frame.rows);
            ggml_backend_tensor_get(tensor, frame.scores.data(), 0, size);
            obs.waiting = true;
        } else {
            if (!obs.waiting || tensor->type != GGML_TYPE_I32) throw std::runtime_error("invalid QSA top-k sequence");
            auto & frame = obs.frames.back();
            if (tensor->ne[1] != frame.rows || tensor->ne[0] > frame.cols) throw std::runtime_error("QSA top-k shape mismatch");
            frame.k = int(tensor->ne[0]);
            frame.ids_device = !ggml_backend_buffer_is_host(tensor->buffer);
            frame.ids.resize(size_t(frame.k) * frame.rows);
            ggml_backend_tensor_get(tensor, frame.ids.data(), 0, size);
            obs.waiting = false;
        }
        return true;
    } catch (const std::exception & exc) {
        obs.error = exc.what();
        return false;
    }
}

inline bool self_test() {
    Frame a;
    a.cols = 4; a.rows = 1; a.k = 2;
    a.scores = {2, 1, 1, 0}; a.ids = {0, 1};
    Frame b = a;
    b.ids = {0, 2};
    const auto changed = compare({a}, {b});
    b.ids = {1, 0};
    const auto order = compare({a}, {b});
    b.ids = {0, 0};
    const auto invalid = compare({a}, {b});
    a.scores = {3, 2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    a.k = 3; a.ids = {0, 1, 2};
    b = a; b.ids = {0, 1, 3};
    const auto hidden = compare({a}, {b});
    return changed.same_score_tie_changed_sets == 1 && changed.invalid == 0 && order.order_only == 1 &&
           order.changed_sets == 0 && invalid.invalid == 1 && hidden.changed_sets == 1 && hidden.changed_visible_sets == 0;
}

} // namespace qsa_observer
