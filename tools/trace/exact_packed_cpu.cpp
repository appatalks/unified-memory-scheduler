#include "exact_packed_cpu.h"
#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace {

using Matrix = void (*)(int, float *, size_t, const void *, const void *, int, int);

struct alignas(64) Counters {
    std::atomic<uint64_t> gemv{0}, gemm{0}, pack{0};
};

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "Exact packed CPU: %s\n", message);
    std::abort();
}

struct State {
    bool enabled = false;
    Matrix gemv = nullptr, gemm = nullptr;
    exact_packed_cpu::Quantize pack = nullptr, quantize = nullptr;
    FILE * output = nullptr;
    std::array<Counters, 256> counters;
    std::atomic<size_t> next_counter{0};

    State() {
        const char * mode = std::getenv("VRAMBRIDGE_CPU_PACKED_MODE");
        if (mode && std::strcmp(mode, "off") && std::strcmp(mode, "exact")) {
            throw std::runtime_error("mode must be off or exact");
        }
        enabled = mode && !std::strcmp(mode, "exact");
        const char * library = std::getenv("VRAMBRIDGE_CPU_PACKED_LIBRARY");
        if (!library) throw std::runtime_error("explicit matching native CPU library required");
        void * handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
        if (!handle) throw std::runtime_error(dlerror());
        gemv = reinterpret_cast<Matrix>(dlsym(handle, "ggml_gemv_mxfp4_8x8_q8_0"));
        gemm = reinterpret_cast<Matrix>(dlsym(handle, "ggml_gemm_mxfp4_8x8_q8_0"));
        pack = reinterpret_cast<exact_packed_cpu::Quantize>(dlsym(handle, "ggml_quantize_mat_q8_0_4x8"));
        quantize = reinterpret_cast<exact_packed_cpu::Quantize>(dlsym(handle, "quantize_row_q8_0"));
        if (!gemv || !gemm || !pack || !quantize) throw std::runtime_error("native packed entry points unavailable");
        if (!enabled) return;
        if (!__builtin_cpu_supports("avx512vnni") || !__builtin_cpu_supports("avx512bw") ||
            !__builtin_cpu_supports("f16c")) {
            throw std::runtime_error("exact packed research requires AVX512 VNNI/BW and F16C");
        }
        const char * path = std::getenv("VRAMBRIDGE_CPU_PACKED_OUT");
        if (!path) throw std::runtime_error("explicit private output required");
        struct stat info{};
        const auto parent = std::filesystem::path(path).parent_path();
        if (parent.empty() || stat(parent.c_str(), &info) || !S_ISDIR(info.st_mode) ||
            info.st_uid != getuid() || (info.st_mode & 077)) {
            throw std::runtime_error("report parent must be owned and private");
        }
        output = std::fopen(path, "wx");
        if (!output || fchmod(fileno(output), 0600)) throw std::runtime_error("cannot create exclusive private output");
        std::fprintf(stderr, "Exact packed CPU active: eight-row MXFP4 with native Q8 quantization and FMA lanes\n");
    }

    ~State() {
        if (!enabled) return;
        uint64_t gemv_calls = 0, gemm_calls = 0, pack_calls = 0;
        for (const auto & counter : counters) {
            gemv_calls += counter.gemv.load(std::memory_order_relaxed);
            gemm_calls += counter.gemm.load(std::memory_order_relaxed);
            pack_calls += counter.pack.load(std::memory_order_relaxed);
        }
        const nlohmann::json report = {{"schema_version", 1}, {"kind", "exact_packed_cpu"}, {"complete", true},
            {"pid", getpid()}, {"source_sha256", VRAMBRIDGE_EXACT_PACKED_SOURCE_SHA256},
            {"header_sha256", VRAMBRIDGE_EXACT_PACKED_HEADER_SHA256}, {"gemv_calls", gemv_calls},
            {"gemm_calls", gemm_calls}, {"pack_calls", pack_calls},
            {"target_weights_requantized", false}, {"full_model_acceleration_claim", false}};
        const auto text = report.dump(2) + "\n";
        if (std::fwrite(text.data(), 1, text.size(), output) != text.size() || std::fclose(output)) {
            fail("cannot finish report");
        }
    }
};

State & state() {
    static State value;
    return value;
}

Counters & local_counters(State & control) {
    static thread_local const size_t slot = control.next_counter.fetch_add(1, std::memory_order_relaxed);
    if (slot >= control.counters.size()) fail("bounded thread-counter slots exhausted");
    return control.counters[slot];
}

} // namespace

extern "C" void ggml_gemv_mxfp4_8x8_q8_0(
        int n, float * output, size_t stride, const void * weights, const void * activation, int nr, int nc) {
    try {
        auto & control = state();
        if (!control.enabled) return control.gemv(n, output, stride, weights, activation, nr, nc);
        local_counters(control).gemv.fetch_add(1, std::memory_order_relaxed);
        exact_packed_cpu::gemv(n, output, stride, weights, activation, nr, nc);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

extern "C" void ggml_gemm_mxfp4_8x8_q8_0(
        int n, float * output, size_t stride, const void * weights, const void * activation, int nr, int nc) {
    try {
        auto & control = state();
        if (!control.enabled) return control.gemm(n, output, stride, weights, activation, nr, nc);
        local_counters(control).gemm.fetch_add(1, std::memory_order_relaxed);
        exact_packed_cpu::gemm(n, output, stride, weights, activation, nr, nc);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

extern "C" void ggml_quantize_mat_q8_0_4x8(const float * input, void * output, int64_t k) {
    try {
        auto & control = state();
        if (!control.enabled) return control.pack(input, output, k);
        local_counters(control).pack.fetch_add(1, std::memory_order_relaxed);
        exact_packed_cpu::pack_activation(input, output, k, control.quantize);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}
