#include "kv_cold_tier.h"

#include <cstdio>
#include <memory>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static ggml_context * context() {
    return ggml_init({1024 * 1024, nullptr, true});
}

int main() {
    try {
        require(kv_cold_role("cache_k_l0") == 'k' && kv_cold_role("cache_swa_v_l17") == 'v',
                "native KV roles not recognized");
        for (const char * name : {"blk.0.attn_k.weight", "cache_k_l", "cache_k_l-1", "cache_badk_l0"}) {
            require(!kv_cold_role(name), "non-KV role accepted");
        }
        std::unique_ptr<ggml_context, decltype(&ggml_free)> good(context(), ggml_free), bad(context(), ggml_free);
        require(good && bad, "GGML context allocation failed");
        auto * key = ggml_new_tensor_3d(good.get(), GGML_TYPE_F16, 128, 512, 1);
        ggml_set_name(key, "cache_k_l0");
        auto * value = ggml_new_tensor_3d(good.get(), GGML_TYPE_F16, 128, 512, 1);
        ggml_set_name(value, "cache_v_l0");
        ggml_view_2d(good.get(), key, 128, 512, key->nb[1], 0);
        require(kv_cold_context(good.get()), "KV owner plus views was not recognized");
        auto * wrong = ggml_new_tensor_3d(bad.get(), GGML_TYPE_F32, 128, 512, 1);
        ggml_set_name(wrong, "cache_k_l0");
        require(!kv_cold_context(bad.get()), "unqualified precision accepted");
        auto * mixed = ggml_new_tensor_2d(good.get(), GGML_TYPE_F16, 128, 128);
        ggml_set_name(mixed, "blk.0.attn_k.weight");
        require(!kv_cold_context(good.get()), "mixed weights/KV context accepted");
        std::puts("cold KV context/role/precision/owner-view qualification passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "cold KV tier test: %s\n", error.what());
        return 1;
    }
}
