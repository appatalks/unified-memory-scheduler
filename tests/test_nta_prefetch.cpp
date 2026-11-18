#include "ggml.h"
#include "ggml-cpu/repack.h"

#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

using Dot = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
using Quantize = void (*)(const float *, void *, int64_t);

static void * symbol(void * library, const char * name) {
    void * value = dlsym(library, name);
    if (!value) throw std::runtime_error(name);
    return value;
}

template <typename W, typename A>
static size_t check(void * base, void * candidate, const char * dot_name, const char * quant_name,
                    int block_size, std::mt19937 & rng) {
    auto bdot = reinterpret_cast<Dot>(symbol(base, dot_name));
    auto cdot = reinterpret_cast<Dot>(symbol(candidate, dot_name));
    if (bdot == cdot) throw std::runtime_error("baseline and candidate resolve to the same kernel");
    auto quant = reinterpret_cast<Quantize>(symbol(base, quant_name));
    size_t count = 0;
    for (int n : {32, 64, 192, 256, 512, 1024, 2880, 4096}) {
        if (n % block_size) continue;
        std::vector<W> weights(n / block_size);
        std::vector<A> activations(n / block_size);
        std::vector<float> input(n);
        for (int iteration = 0; iteration < 100; ++iteration) {
            for (float & v : input) v = float(int(rng() % 16001) - 8000) / 1000;
            for (auto & block : weights) {
                auto * bytes = reinterpret_cast<unsigned char *>(&block);
                for (size_t j = 0; j < sizeof(block); ++j) bytes[j] = rng() & 255;
                if constexpr (std::is_same_v<W, block_mxfp4>) block.e = 115 + rng() % 16;
                if constexpr (std::is_same_v<W, block_q4_K>) { const uint16_t scale[2] = {0x3800, 0x3400}; std::memcpy(&block, scale, sizeof(scale)); }
                if constexpr (std::is_same_v<W, block_iq3_s>) block.d = 0x3800;
            }
            quant(input.data(), activations.data(), n);
            float a = 0, b = 0;
            bdot(n, &a, 0, weights.data(), 0, activations.data(), 0, 1);
            cdot(n, &b, 0, weights.data(), 0, activations.data(), 0, 1);
            if (!std::isfinite(a) || std::memcmp(&a, &b, sizeof(a))) {
                std::fprintf(stderr, "%s n=%d iteration=%d baseline=%a candidate=%a\n", dot_name, n, iteration, double(a), double(b));
                throw std::runtime_error("numeric mismatch");
            }
            ++count;
        }
    }
    std::printf("%s: %zu exact comparisons\n", dot_name, count);
    return count;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("baseline and candidate plugin paths required");
        ggml_context * context = ggml_init({1024 * 1024, nullptr, false});
        if (!context) throw std::runtime_error("ggml_init failed");
        void * base = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!base) throw std::runtime_error(dlerror());
        void * candidate = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
        if (!candidate) throw std::runtime_error(dlerror());
        reinterpret_cast<void (*)()>(symbol(base, "ggml_cpu_init"))();
        reinterpret_cast<void (*)()>(symbol(candidate, "ggml_cpu_init"))();
        std::mt19937 rng(417);
        check<block_mxfp4, block_q8_0>(base, candidate, "ggml_vec_dot_mxfp4_q8_0", "quantize_row_q8_0", 32, rng);
        check<block_q4_K, block_q8_K>(base, candidate, "ggml_vec_dot_q4_K_q8_K", "quantize_row_q8_K", 256, rng);
        check<block_iq3_s, block_q8_K>(base, candidate, "ggml_vec_dot_iq3_s_q8_K", "quantize_row_q8_K", 256, rng);
        ggml_free(context);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
