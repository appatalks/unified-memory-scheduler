#include <immintrin.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

struct MxBlock {
    uint8_t exponent;
    uint8_t codes[16];
};

struct Q8Block {
    uint16_t scale;
    int8_t codes[32];
};

static_assert(sizeof(MxBlock) == 17 && sizeof(Q8Block) == 34);

constexpr float exponent_half(uint8_t exponent) {
    return std::bit_cast<float>(exponent < 2 ? (0x00200000u << exponent) : (uint32_t(exponent - 1) << 23));
}

struct Exponents {
    float values[256]{};
    constexpr Exponents() {
        for (int i = 0; i < 256; ++i) values[i] = exponent_half(uint8_t(i));
    }
};

constexpr Exponents exponents;

__m256i integer_dot(const MxBlock & x, const Q8Block & y) {
    const __m128i table = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x.codes));
    const __m128i low = _mm_shuffle_epi8(table, _mm_and_si128(bits, mask));
    const __m128i high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bits, 4), mask));
    const __m256i a = _mm256_set_m128i(high, low);
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y.codes));
    // VPSIGNB preserves native signed-byte behavior, including -128.
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(a, a), _mm256_sign_epi8(b, a));
}

} // namespace

extern "C" void ggml_vec_dot_mxfp4_q8_0(int n, float * result, size_t, const void * vx, size_t,
                                       const void * vy, size_t, int nrc) {
    if (n <= 0 || n % 64 != 0 || nrc != 1) {
        std::fprintf(stderr, "CPU VNNI research supports positive paired MXFP4/Q8 blocks and nrc=1 only\n");
        std::abort();
    }
    const auto * x = static_cast<const MxBlock *>(vx);
    const auto * y = static_cast<const Q8Block *>(vy);
    __m256 even = _mm256_setzero_ps(), odd = _mm256_setzero_ps();
    for (int block = 0; block < n / 32; block += 2) {
        const float d0 = _cvtsh_ss(y[block].scale) * exponents.values[x[block].exponent];
        const float d1 = _cvtsh_ss(y[block + 1].scale) * exponents.values[x[block + 1].exponent];
        even = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(integer_dot(x[block], y[block])), even);
        odd = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(integer_dot(x[block + 1], y[block + 1])), odd);
    }
    const __m256 lanes = _mm256_add_ps(even, odd);
    __m128 sum = _mm_add_ps(_mm256_extractf128_ps(lanes, 1), _mm256_castps256_ps128(lanes));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    *result = _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum)));
}
