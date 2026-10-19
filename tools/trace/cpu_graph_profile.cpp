#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <vector>
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

using json = nlohmann::json;
using Compute = ggml_status (*)(ggml_cgraph *, ggml_cplan *);

namespace {

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "CPU graph research profiler: %s\n", message);
    std::abort();
}

uint64_t now_ns() {
    timespec time{};
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) fail("clock_gettime failed");
    return uint64_t(time.tv_sec) * 1000000000 + time.tv_nsec;
}

struct State {
    Compute original = nullptr;
    FILE * file = nullptr;
    bool expert_ids = false;
    std::atomic<uint64_t> sequence{0};
    std::mutex mutex;
    std::vector<json> records;
    ~State() {
        for (const auto & record : records) {
            const auto text = record.dump() + "\n";
            if (std::fwrite(text.data(), 1, text.size(), file) != text.size()) fail("profile write failed");
        }
        if (std::fclose(file) != 0) fail("profile close failed");
    }
};

State & state() {
    static State value;
    static const bool initialized = [&] {
        const char * library = std::getenv("VRAMBRIDGE_CPU_PROFILE_LIBRARY");
        const char * output = std::getenv("VRAMBRIDGE_CPU_PROFILE_OUT");
        if (!library || !output) fail("explicit library and output paths are required");
        if (const char * ids = std::getenv("VRAMBRIDGE_CPU_PROFILE_EXPERT_IDS")) {
            if (std::strcmp(ids, "0") != 0 && std::strcmp(ids, "1") != 0) fail("expert-ID mode must be 0 or 1");
            value.expert_ids = std::strcmp(ids, "1") == 0;
        }
        if (value.expert_ids) {
            const auto parent = std::filesystem::path(output).parent_path();
            struct stat info{};
            if (parent.empty() || stat(parent.c_str(), &info) != 0 || !S_ISDIR(info.st_mode) ||
                info.st_uid != getuid() || (info.st_mode & 077) != 0) {
                fail("expert-ID recording requires an owned private directory");
            }
        }
        void * handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
        if (!handle) fail(dlerror());
        value.original = reinterpret_cast<Compute>(dlsym(handle, "ggml_graph_compute"));
        if (!value.original || value.original == &ggml_graph_compute) fail("cannot resolve original CPU function");
        value.file = std::fopen(output, "wx");
        if (!value.file || fchmod(fileno(value.file), S_IRUSR | S_IWUSR) != 0) fail("cannot create private exclusive output");
        value.records.reserve(32768);
        std::fprintf(stderr, "CPU graph research profiler active; no tensor observer or graph changes\n");
        return true;
    }();
    (void) initialized;
    return value;
}

bool external_ids(ggml_cgraph * graph, const ggml_tensor * ids) {
    for (const ggml_tensor * tensor = ids; tensor; tensor = tensor->view_src) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (ggml_graph_node(graph, i) == tensor) return false;
        }
    }
    return true;
}

json pre_dispatch_hint(ggml_cgraph * graph, const ggml_tensor * node) {
    const auto * ids = node->src[0];
    const auto * map = node->src[1];
    if (node->op != GGML_OP_CUSTOM || !ids || !map ||
        std::strncmp(map->name, "moe_cache.slot_map-", 19) != 0) return nullptr;
    if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        ids->ne[0] < 1 || ids->ne[0] > 1024 || map->type != GGML_TYPE_I32 ||
        map->ne[0] != 1 || map->ne[1] < ids->ne[0] || map->ne[1] > 65536 ||
        !ids->buffer || !map->buffer || !ggml_backend_buffer_is_host(ids->buffer) ||
        !ggml_backend_buffer_is_host(map->buffer) || !external_ids(graph, ids) || !external_ids(graph, map)) {
        return {{"observable", false}, {"reason", "unsupported or internally produced dispatcher input"}};
    }
    json selected = json::array(), misses = json::array();
    std::vector<bool> seen(map->ne[1], false);
    for (int rank = 0; rank < ids->ne[0]; ++rank) {
        int32_t id, slot;
        std::memcpy(&id, static_cast<const char *>(ids->data) + rank * ids->nb[0], sizeof(id));
        if (id < 0 || id >= map->ne[1] || seen[id]) fail("invalid early dispatcher expert IDs");
        seen[id] = true;
        std::memcpy(&slot, static_cast<const char *>(map->data) + id * map->nb[1], sizeof(slot));
        if (slot < -1) fail("invalid early dispatcher residency map");
        selected.push_back(id);
        if (slot == -1) misses.push_back(id);
    }
    return {{"observable", true}, {"selected_ids", selected}, {"pre_admission_misses", misses}};
}

} // namespace

extern "C" ggml_status ggml_graph_compute(ggml_cgraph * graph, ggml_cplan * plan) {
    auto & profile = state();
    const auto sequence = profile.sequence.fetch_add(1, std::memory_order_relaxed);
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
    char label[64];
    std::snprintf(label, sizeof(label), "vrambridge.cpu_meta:%llu", static_cast<unsigned long long>(sequence));
    nvtxRangePushA(label);
#endif
    json operations = json::object(), matmuls = json::array(), nodes = json::array();
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        const auto key = ggml_op_name(node->op);
        operations[key] = operations.value(key, 0) + 1;
        if (profile.expert_ids) {
            json metadata = {{"name", node->name}, {"op", key},
                {"tensor_key", reinterpret_cast<uintptr_t>(node)}};
            const auto early = pre_dispatch_hint(graph, node);
            if (!early.is_null()) metadata["early_route"] = early;
            nodes.push_back(std::move(metadata));
        }
        if (node->op != GGML_OP_MUL_MAT_ID) continue;
        const auto * weights = node->src[0];
        const auto * ids = node->src[2];
        json item = {{"name", node->name}, {"weight_type", ggml_type_name(weights->type)},
            {"weight_name", weights->name}, {"expert_bytes", ggml_nbytes(weights) / weights->ne[2]},
            {"input_width", weights->ne[0]}, {"output_width", weights->ne[1]},
            {"expert_bank_size", weights->ne[2]}, {"rows", ids->ne[1]},
            {"active_slots", ids->ne[0]}, {"cold_experts", nullptr}, {"logical_weight_bytes", nullptr}};
        if (profile.expert_ids) {
            json sources = json::array();
            json source_keys = json::array();
            for (const auto * tensor = ids; tensor; tensor = tensor->view_src) {
                if (sources.size() >= 128) fail("expert-ID view ancestry exceeds bound");
                sources.push_back(tensor->name);
                source_keys.push_back(reinterpret_cast<uintptr_t>(tensor));
            }
            item["id_sources"] = sources;
            item["id_source_keys"] = source_keys;
        }
        if (ids->ne[1] == 1 && external_ids(graph, ids) && ids->type == GGML_TYPE_I32 &&
            ids->buffer && ggml_backend_buffer_is_host(ids->buffer)) {
            int cold = 0;
            json cold_ids = json::array();
            std::vector<bool> seen(weights->ne[2], false);
            for (int rank = 0; rank < ids->ne[0]; ++rank) {
                int32_t id;
                std::memcpy(&id, static_cast<const char *>(ids->data) + rank * ids->nb[0], sizeof(id));
                if (id < -1 || id >= weights->ne[2]) fail("external expert ID outside native domain");
                if (id >= 0 && !seen[id]) {
                    seen[id] = true;
                    ++cold;
                    if (profile.expert_ids) cold_ids.push_back(id);
                }
            }
            item["cold_experts"] = cold;
            item["logical_weight_bytes"] = cold * (ggml_nbytes(weights) / weights->ne[2]);
            if (profile.expert_ids) item["cold_ids"] = cold_ids;
        }
        matmuls.push_back(std::move(item));
    }
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
    nvtxRangePop();
    std::snprintf(label, sizeof(label), "vrambridge.cpu_graph:%llu", static_cast<unsigned long long>(sequence));
#endif
    const auto start = now_ns();
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
    nvtxRangePushA(label);
#endif
    const auto result = profile.original(graph, plan);
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
    nvtxRangePop();
#endif
    const auto end = now_ns();
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
    std::snprintf(label, sizeof(label), "vrambridge.cpu_record:%llu", static_cast<unsigned long long>(sequence));
    nvtxRangePushA(label);
#endif
    {
        std::lock_guard lock(profile.mutex);
        if (profile.records.size() >= 100000) fail("bounded profile record count exceeded");
        profile.records.push_back({{"schema_version", 3}, {"sequence", sequence},
            {"contains_sensitive_routing", profile.expert_ids},
            {"start_ns", start}, {"end_ns", end}, {"elapsed_ns", end - start},
            {"threads", plan->n_threads}, {"status", int(result)}, {"ops", operations},
            {"nodes", nodes}, {"matmuls", matmuls}});
    }
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
    nvtxRangePop();
#endif
    return result;
}
