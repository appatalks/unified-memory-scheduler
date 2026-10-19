#include "exact_packed_cpu.h"

#include <dlfcn.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <memory>

using Dot = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static std::vector<block_mxfp4x8> pack_weights(const std::vector<block_mxfp4> & raw, int n, int columns) {
    const int blocks = n / 32;
    std::vector<block_mxfp4x8> packed(columns / 8 * blocks);
    for (int column = 0; column < columns; column += 8) {
        for (int block = 0; block < blocks; ++block) {
            auto & out = packed[column / 8 * blocks + block];
            for (int row = 0; row < 8; ++row) {
                const auto & input = raw[(column + row) * blocks + block];
                out.e[row] = input.e;
                for (int part = 0; part < 2; ++part) {
                    std::memcpy(out.qs + (part * 8 + row) * 8, input.qs + part * 8, 8);
                }
            }
        }
    }
    return packed;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "explicit native CPU library required");
        if (!__builtin_cpu_supports("avx512vnni") || !__builtin_cpu_supports("avx512bw") ||
            !__builtin_cpu_supports("f16c")) return 77;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({1024 * 1024, nullptr, false}), ggml_free);
        require(bool(context), "GGML reference initialization failed");
        void * library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        auto initialize = reinterpret_cast<void (*)()>(dlsym(library, "ggml_cpu_init"));
        require(initialize != nullptr, "native CPU initialization unavailable");
        initialize();
        auto native = reinterpret_cast<Dot>(dlsym(library, "ggml_vec_dot_mxfp4_q8_0"));
        auto quantize = reinterpret_cast<exact_packed_cpu::Quantize>(dlsym(library, "quantize_row_q8_0"));
        require(native && quantize, "native dot/quantizer unavailable");
        std::mt19937 generator(417);
        std::uniform_real_distribution<float> values(-8, 8);
        size_t checks = 0;
        for (int n : {64, 256, 2880}) {
            const int blocks = n / 32;
            for (int columns : {8, 16, 64}) {
                std::vector<block_mxfp4> weights(columns * blocks);
                for (auto & block : weights) {
                    block.e = 96 + generator() % 40;
                    for (auto & code : block.qs) code = uint8_t(generator());
                }
                weights[0].e = 0;
                weights[1].e = 1;
                auto packed = pack_weights(weights, n, columns);
                std::vector<float> floats(n * 8);
                for (auto & value : floats) value = values(generator);
                std::vector<block_q8_0> activations(blocks * 8);
                for (int row = 0; row < 8; ++row) {
                    quantize(floats.data() + row * n, activations.data() + row * blocks, n);
                }
                activations[0].qs[0] = -128;
                std::vector<float> expected(columns), actual(columns);
                for (int column = 0; column < columns; ++column) {
                    native(n, &expected[column], 0, weights.data() + column * blocks, 0, activations.data(), 0, 1);
                }
                exact_packed_cpu::gemv(n, actual.data(), 0, packed.data(), activations.data(), 1, columns);
                if (std::memcmp(expected.data(), actual.data(), columns * sizeof(float))) {
                    for (int column = 0; column < columns; ++column) {
                        if (std::memcmp(&expected[column], &actual[column], sizeof(float))) {
                            std::fprintf(stderr, "GEMV n=%d columns=%d first mismatch=%d native=%a packed=%a\n",
                                         n, columns, column, double(expected[column]), double(actual[column]));
                            break;
                        }
                    }
                }
                require(std::memcmp(expected.data(), actual.data(), columns * sizeof(float)) == 0,
                        "packed GEMV differs from native scalar");
                checks += columns;
                for (int rows : {4, 8}) {
                    std::vector<block_q8_0x4> packed_y(rows / 4 * blocks);
                    for (int group = 0; group < rows / 4; ++group) {
                        exact_packed_cpu::pack_activation(floats.data() + group * 4 * n,
                            packed_y.data() + group * blocks, n, quantize);
                    }
                    const size_t stride = columns + 3;
                    expected.assign(rows * stride, -123.25f);
                    actual.assign(rows * stride, -123.25f);
                    for (int row = 0; row < rows; ++row) {
                        quantize(floats.data() + row * n, activations.data() + row * blocks, n);
                        for (int column = 0; column < columns; ++column) {
                            native(n, &expected[row * stride + column], 0,
                                weights.data() + column * blocks, 0, activations.data() + row * blocks, 0, 1);
                        }
                    }
                    exact_packed_cpu::gemm(n, actual.data(), stride, packed.data(), packed_y.data(), rows, columns);
                    require(std::memcmp(expected.data(), actual.data(), rows * stride * sizeof(float)) == 0,
                            "packed GEMM/quantization differs from native scalar or overwrote stride padding");
                    checks += rows * columns;
                }
            }
        }
        std::printf("native-bit-exact packed output checks: %zu\n", checks);
        const int n = 2880, columns = 2880, blocks = n / 32;
        std::vector<block_mxfp4> weights(columns * blocks);
        for (auto & block : weights) {
            block.e = 120 + generator() % 8;
            for (auto & code : block.qs) code = uint8_t(generator());
        }
        auto packed = pack_weights(weights, n, columns);
        std::vector<float> input(n), output(columns);
        for (auto & value : input) value = values(generator);
        std::vector<block_q8_0> activation(blocks);
        quantize(input.data(), activation.data(), n);
        double raw_ms = 0, packed_ms = 0;
        for (int repetition = 0; repetition < 12; ++repetition) {
            for (int variant = 0; variant < 2; ++variant) {
                const bool candidate = (variant == 1) != (repetition % 2);
                const auto start = std::chrono::steady_clock::now();
                if (candidate) {
                    exact_packed_cpu::gemv(n, output.data(), 0, packed.data(), activation.data(), 1, columns);
                } else {
                    for (int column = 0; column < columns; ++column) {
                        native(n, &output[column], 0, weights.data() + column * blocks, 0, activation.data(), 0, 1);
                    }
                }
                asm volatile("" : : "r"(output.data()) : "memory");
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                if (repetition >= 2) (candidate ? packed_ms : raw_ms) += ms;
            }
        }
        std::printf("isolated warm single-thread raw %.4f ms, exact packed %.4f ms, ratio %.3f; not request throughput\n",
                    raw_ms / 10, packed_ms / 10, raw_ms / packed_ms);
        dlclose(library);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "exact packed CPU test: %s\n", error.what());
        return 1;
    }
}
