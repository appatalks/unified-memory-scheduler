#include "kv_cold_tier.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace {

using Allocate = ggml_backend_buffer_t (*)(ggml_context *, ggml_backend_buffer_type_t);
using Clear = void (*)(ggml_backend_buffer_t, uint8_t);
using Free = void (*)(ggml_backend_buffer_t);
using Malloc = cudaError_t (*)(void **, size_t);

thread_local bool kv_allocation = false;

struct AllocationScope {
    bool previous;
    explicit AllocationScope(bool enabled) : previous(kv_allocation) { kv_allocation = enabled; }
    ~AllocationScope() { kv_allocation = previous; }
};

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "Cold KV tier: %s\n", message);
    std::abort();
}

void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
    void * base;
    size_t bytes;
    int device;
};

struct State {
    bool enabled = false;
    Allocate allocate = nullptr;
    Clear clear = nullptr;
    Free release = nullptr;
    Malloc device_malloc = nullptr;
    FILE * output = nullptr;
    std::mutex mutex;
    std::unordered_map<ggml_backend_buffer_t, Buffer> buffers;
    std::atomic<uint64_t> allocations{0}, logical_bytes{0}, stage_requests{0}, requested_stage_bytes{0};
    std::atomic<uint64_t> managed_allocations{0};

    State() {
        const char * mode = std::getenv("VRAMBRIDGE_KV_COLD_MODE");
        if (mode && std::strcmp(mode, "off") && std::strcmp(mode, "stage")) {
            throw std::runtime_error("mode must be off or stage");
        }
        enabled = mode && !std::strcmp(mode, "stage");
        const char * library = std::getenv("VRAMBRIDGE_KV_COLD_BASE_LIBRARY");
        if (!library) throw std::runtime_error("explicit matching GGML base library required");
        void * handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
        if (!handle) throw std::runtime_error(dlerror());
        allocate = reinterpret_cast<Allocate>(dlsym(handle, "ggml_backend_alloc_ctx_tensors_from_buft"));
        clear = reinterpret_cast<Clear>(dlsym(handle, "ggml_backend_buffer_clear"));
        release = reinterpret_cast<Free>(dlsym(handle, "ggml_backend_buffer_free"));
        if (!allocate || !clear || !release) throw std::runtime_error("native buffer entry points unavailable");
        void * runtime = dlopen(VRAMBRIDGE_KV_CUDA_RUNTIME_LIBRARY, RTLD_NOW | RTLD_NOLOAD);
        if (!runtime) throw std::runtime_error(dlerror());
        device_malloc = reinterpret_cast<Malloc>(dlsym(runtime, "cudaMalloc"));
        if (!device_malloc) throw std::runtime_error("native CUDA allocator unavailable");
        if (!enabled) return;
        const char * managed = std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
        if (managed && std::strcmp(managed, "0")) {
            throw std::runtime_error("selective KV staging requires global unified-memory mode off");
        }
        const char * path = std::getenv("VRAMBRIDGE_KV_COLD_OUT");
        if (!path) throw std::runtime_error("explicit private output required");
        struct stat info{};
        const auto parent = std::filesystem::path(path).parent_path();
        if (parent.empty() || stat(parent.c_str(), &info) || !S_ISDIR(info.st_mode) ||
            info.st_uid != getuid() || (info.st_mode & 077)) throw std::runtime_error("owned private parent required");
        output = std::fopen(path, "wx");
        if (!output || fchmod(fileno(output), 0600)) throw std::runtime_error("cannot create private output");
    }

    ~State() {
        if (!enabled) return;
        const nlohmann::json report = {{"schema_version", 1}, {"kind", "cold_kv_tier"}, {"complete", true},
            {"pid", getpid()}, {"source_sha256", VRAMBRIDGE_KV_COLD_SOURCE_SHA256},
            {"header_sha256", VRAMBRIDGE_KV_COLD_HEADER_SHA256}, {"kv_allocations", allocations.load()},
            {"logical_buffer_bytes", logical_bytes.load()}, {"cpu_stage_requests", stage_requests.load()},
            {"managed_allocation_calls", managed_allocations.load()}, {"weights_remain_device_allocations", true},
            {"requested_stage_bytes_not_measured_transfers", requested_stage_bytes.load()},
            {"buffers_still_tracked", buffers.size()}, {"configured_context_changed", false},
            {"kv_precision_changed", false}, {"active_pages_can_migrate_to_gpu", true},
            {"full_model_acceleration_claim", false}};
        const auto text = report.dump(2) + "\n";
        if (std::fwrite(text.data(), 1, text.size(), output) != text.size() || std::fclose(output)) fail("cannot finish report");
    }
};

State & state() {
    static State value;
    return value;
}

void stage_cold(State & control, const Buffer & buffer) {
    std::ifstream input("/proc/meminfo");
    std::string key, unit;
    uint64_t value = 0, available = 0;
    while (input >> key >> value >> unit) {
        if (key == "MemAvailable:") { available = value * 1024; break; }
    }
    if (available < buffer.bytes + 16ull * 1024 * 1024 * 1024) {
        throw std::runtime_error("cold KV staging requires buffer bytes plus16GiB host headroom");
    }
    checked(cudaMemPrefetchAsync(buffer.base, buffer.bytes, cudaCpuDeviceId, nullptr));
    checked(cudaStreamSynchronize(nullptr));
    ++control.stage_requests;
    control.requested_stage_bytes += buffer.bytes;
}

} // namespace

extern "C" ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors_from_buft(
        ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    try {
        auto & control = state();
        const bool selected = control.enabled && buft && kv_cold_context(ctx) && !ggml_backend_buft_is_host(buft) &&
                              !std::strncmp(ggml_backend_buft_name(buft), "CUDA", 4);
        ggml_backend_buffer_t result;
        {
            const AllocationScope scope(selected);
            result = control.allocate(ctx, buft);
        }
        if (!selected) return result;
        if (!result) throw std::runtime_error("native KV allocation failed");
        for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            if (!tensor->view_src && tensor->buffer != result) {
                throw std::runtime_error("cold KV staging currently requires one backing buffer per context");
            }
        }
        void * base = ggml_backend_buffer_get_base(result);
        const size_t bytes = ggml_backend_buffer_get_size(result);
        cudaPointerAttributes attributes{};
        checked(cudaPointerGetAttributes(&attributes, base));
        if (attributes.type != cudaMemoryTypeManaged || !bytes) throw std::runtime_error("KV allocation is not managed memory");
        int device = 0;
        checked(cudaGetDevice(&device));
        checked(cudaMemAdvise(base, bytes, cudaMemAdviseSetPreferredLocation, device));
        {
            std::lock_guard lock(control.mutex);
            if (!control.buffers.emplace(result, Buffer{base, bytes, device}).second) throw std::runtime_error("duplicate KV owner");
        }
        ++control.allocations;
        control.logical_bytes += bytes;
        return result;
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

extern "C" void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    try {
        auto & control = state();
        control.clear(buffer, value);
        if (!control.enabled) return;
        Buffer tracked{};
        {
            std::lock_guard lock(control.mutex);
            const auto found = control.buffers.find(buffer);
            if (found == control.buffers.end()) return;
            tracked = found->second;
        }
        // Bulk initialization must not leave every inactive context page resident in VRAM.
        stage_cold(control, tracked);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

extern "C" void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    try {
        auto & control = state();
        {
            std::lock_guard lock(control.mutex);
            control.buffers.erase(buffer);
        }
        control.release(buffer);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

extern "C" cudaError_t CUDARTAPI cudaMalloc(void ** pointer, size_t bytes) {
    try {
        auto & control = state();
        if (!control.enabled || !kv_allocation) return control.device_malloc(pointer, bytes);
        const AllocationScope internal(false);
        const auto result = cudaMallocManaged(pointer, bytes, cudaMemAttachGlobal);
        if (result == cudaSuccess) ++control.managed_allocations;
        return result;
    } catch (const std::exception & error) {
        fail(error.what());
    }
}
