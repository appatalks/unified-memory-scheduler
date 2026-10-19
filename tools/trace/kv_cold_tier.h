#pragma once

#include "ggml.h"

#include <cctype>
#include <string_view>

inline char kv_cold_role(std::string_view name) {
    if (!name.starts_with("cache_")) return 0;
    const auto marker = name.rfind("_l");
    if (marker == std::string_view::npos || marker < 7 || marker + 2 == name.size()) return 0;
    const auto role = marker - 1;
    if ((name[role] != 'k' && name[role] != 'v') || (role != 6 && name[role - 1] != '_')) return 0;
    for (size_t i = 6; i < role; ++i) {
        const unsigned char value = name[i];
        if (!std::isalnum(value) && value != '_') return 0;
    }
    for (size_t i = marker + 2; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') return 0;
    }
    return name[role];
}

inline bool kv_cold_context(const ggml_context * ctx) {
    if (!ctx) return false;
    bool key = false, value = false;
    for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (tensor->view_src) continue;
        const char role = kv_cold_role(tensor->name);
        if (!role || tensor->type != GGML_TYPE_F16 || tensor->op != GGML_OP_NONE ||
            tensor->ne[0] <= 0 || tensor->ne[1] < 32 || tensor->ne[3] != 1) return false;
        key |= role == 'k';
        value |= role == 'v';
    }
    return key && value;
}
