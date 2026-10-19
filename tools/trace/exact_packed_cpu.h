#pragma once

#include "ggml-cpu/repack.h"

#include <immintrin.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace exact_packed_cpu {

using Quantize = void (*)(const float *, void *, int64_t);

inline __m512 scales(const block_mxfp4x8 & weights, ggml_half activation_scale) {
    const __m256i e8 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(weights.e)));
    const __m512i duplicated = _mm512_permutexvar_epi32(
        _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7),
        _mm512_inserti64x4(_mm512_setzero_si512(), e8, 0));
    const __m512i normal = _mm512_slli_epi32(_mm512_sub_epi32(duplicated, _mm512_set1_epi32(1)), 23);
    const __m512i subnormal = _mm512_sllv_epi32(_mm512_set1_epi32(0x00200000), duplicated);
    const __mmask16 small = _mm512_cmplt_epi32_mask(duplicated, _mm512_set1_epi32(2));
    const __m512 exponent_half = _mm512_castsi512_ps(_mm512_mask_blend_epi32(small, normal, subnormal));
    return _mm512_mul_ps(_mm512_set1_ps(_cvtsh_ss(activation_scale)), exponent_half);
}

inline __m512i broadcast_eight(const int8_t * values) {
    int64_t bits;
    std::memcpy(&bits, values, sizeof(bits));
    return _mm512_set1_epi64(bits);
}

inline __m512i integer_dot(__m512i weights, __m512i activation) {
    const __mmask64 negative = _mm512_movepi8_mask(weights);
    const __m512i signed_activation = _mm512_mask_sub_epi8(
        activation, negative, _mm512_setzero_si512(), activation);
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(weights), signed_activation);
}

struct Dots {
    __m512i low, high;
};

inline Dots dots(const uint8_t * codes, const int8_t * low, const int8_t * high) {
    const __m512i table = _mm512_broadcast_i32x4(
        _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
    const __m512i bits = _mm512_loadu_si512(codes), mask = _mm512_set1_epi8(15);
    const __m512i wlow = _mm512_shuffle_epi8(table, _mm512_and_si512(bits, mask));
    const __m512i whigh = _mm512_shuffle_epi8(table, _mm512_and_si512(_mm512_srli_epi16(bits, 4), mask));
    return {integer_dot(wlow, broadcast_eight(low)), integer_dot(whigh, broadcast_eight(high))};
}

inline void tile(int blocks, float * output, const block_mxfp4x8 * weights, const block_q8_0 * activation) {
    __m512 e0 = _mm512_setzero_ps(), e1 = e0, e2 = e0, e3 = e0;
    __m512 o0 = e0, o1 = e0, o2 = e0, o3 = e0;
    // Preserve the native eight dot lanes and separate even/odd block FMA chains.
    for (int block = 0; block < blocks; block += 2) {
        const auto lo = dots(weights[block].qs, activation[block].qs, activation[block].qs + 16);
        const auto hi = dots(weights[block].qs + 64, activation[block].qs + 8, activation[block].qs + 24);
        const auto d = scales(weights[block], activation[block].d);
        e0 = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(lo.low), e0);
        e1 = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(hi.low), e1);
        e2 = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(lo.high), e2);
        e3 = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(hi.high), e3);
        const auto lo1 = dots(weights[block + 1].qs, activation[block + 1].qs, activation[block + 1].qs + 16);
        const auto hi1 = dots(weights[block + 1].qs + 64, activation[block + 1].qs + 8, activation[block + 1].qs + 24);
        const auto d1 = scales(weights[block + 1], activation[block + 1].d);
        o0 = _mm512_fmadd_ps(d1, _mm512_cvtepi32_ps(lo1.low), o0);
        o1 = _mm512_fmadd_ps(d1, _mm512_cvtepi32_ps(hi1.low), o1);
        o2 = _mm512_fmadd_ps(d1, _mm512_cvtepi32_ps(lo1.high), o2);
        o3 = _mm512_fmadd_ps(d1, _mm512_cvtepi32_ps(hi1.high), o3);
    }
    const auto a = _mm512_add_ps(e0, o0), b = _mm512_add_ps(e1, o1);
    const auto c = _mm512_add_ps(e2, o2), d = _mm512_add_ps(e3, o3);
    const auto pairs = _mm512_add_ps(_mm512_add_ps(c, a), _mm512_add_ps(d, b));
    const auto sum = _mm512_add_ps(pairs, _mm512_permute_ps(pairs, 0xb1));
    _mm512_mask_compressstoreu_ps(output, 0x5555, sum);
}

inline void gemv(int n, float * output, size_t, const void * vx, const void * vy, int nr, int nc) {
    if (n <= 0 || n % 64 || nr != 1 || nc <= 0 || nc % 8 || !output || !vx || !vy) {
        throw std::runtime_error("exact packed GEMV requires paired MXFP4 blocks, one activation and eight-column tiles");
    }
    const auto * weights = static_cast<const block_mxfp4x8 *>(vx);
    const auto * activation = static_cast<const block_q8_0 *>(vy);
    for (int column = 0; column < nc; column += 8) {
        tile(n / 32, output + column, weights + (column / 8) * (n / 32), activation);
    }
}

inline void pack_activation(const float * input, void * output, int64_t k, Quantize quantize) {
    if (!input || !output || !quantize || k <= 0 || k % 32) {
        throw std::runtime_error("exact activation packing requires native Q8 row quantization and complete blocks");
    }
    const int64_t blocks = k / 32;
    std::vector<block_q8_0> rows(4 * blocks);
    for (int row = 0; row < 4; ++row) quantize(input + row * k, rows.data() + row * blocks, k);
    auto * packed = static_cast<block_q8_0x4 *>(output);
    for (int64_t block = 0; block < blocks; ++block) {
        for (int row = 0; row < 4; ++row) {
            packed[block].d[row] = rows[row * blocks + block].d;
            for (int part = 0; part < 4; ++part) {
                std::memcpy(packed[block].qs + (part * 4 + row) * 8,
                            rows[row * blocks + block].qs + part * 8, 8);
            }
        }
    }
}

inline void gemm(int n, float * output, size_t stride, const void * vx, const void * vy, int nr, int nc) {
    if (n <= 0 || n % 64 || nr <= 0 || nr % 4 || nc <= 0 || nc % 8 || stride < static_cast<size_t>(nc) ||
        !output || !vx || !vy) {
        throw std::runtime_error("exact packed GEMM requires four-row activation groups and eight-column tiles");
    }
    const int blocks = n / 32;
    const auto * weights = static_cast<const block_mxfp4x8 *>(vx);
    const auto * packed = static_cast<const block_q8_0x4 *>(vy);
    std::vector<block_q8_0> rows(4 * blocks);
    for (int group = 0; group < nr / 4; ++group) {
        for (int block = 0; block < blocks; ++block) {
            for (int row = 0; row < 4; ++row) {
                auto & destination = rows[row * blocks + block];
                destination.d = packed[group * blocks + block].d[row];
                for (int part = 0; part < 4; ++part) {
                    std::memcpy(destination.qs + part * 8,
                        packed[group * blocks + block].qs + (part * 4 + row) * 8, 8);
                }
            }
        }
        for (int column = 0; column < nc; column += 8) {
            for (int row = 0; row < 4; ++row) {
                tile(blocks, output + (group * 4 + row) * stride + column,
                     weights + (column / 8) * blocks, rows.data() + row * blocks);
            }
        }
    }
}

} // namespace exact_packed_cpu
