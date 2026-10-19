#include "ggml.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>

using Compute = ggml_status (*)(ggml_cgraph *, ggml_cplan *);

namespace {

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "CPU control threads: %s\n", message);
    std::abort();
}

bool eligible(ggml_cgraph * graph) {
    bool dispatcher = false, copy = false;
    if (ggml_graph_n_nodes(graph) > 32) return false;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
                break;
            case GGML_OP_CPY:
                if (ggml_nbytes(node) > 64 * 1024) return false;
                copy = true;
                break;
            case GGML_OP_CUSTOM:
                if (ggml_nbytes(node) > 64 * 1024) return false;
                if (node->src[1] && std::strncmp(node->src[1]->name, "moe_cache.slot_map-", 19) == 0) {
                    if (!node->src[0] || node->src[0]->ne[1] != 1) return false;
                    dispatcher = true;
                }
                break;
            default:
                return false;
        }
    }
    return dispatcher && copy;
}

struct State {
    Compute original = nullptr;
    bool enabled = false;
    FILE * output = nullptr;
    std::mutex mutex;
    uint64_t calls = 0, eligible_calls = 0, changed_calls = 0, other_calls = 0;
    int maximum_original_threads = 0;

    State() {
        const char * mode = std::getenv("VRAMBRIDGE_CPU_CONTROL_THREADS");
        if (mode && std::strcmp(mode, "off") != 0 && std::strcmp(mode, "one") != 0) {
            throw std::runtime_error("mode must be off or one");
        }
        enabled = mode && std::strcmp(mode, "one") == 0;
        const char * library = std::getenv("VRAMBRIDGE_CPU_CONTROL_LIBRARY");
        if (!library) throw std::runtime_error("explicit native CPU library required");
        void * handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
        if (!handle) throw std::runtime_error(dlerror());
        original = reinterpret_cast<Compute>(dlsym(handle, "ggml_graph_compute"));
        if (!original || original == &ggml_graph_compute) throw std::runtime_error("cannot resolve native graph compute");
        if (!enabled) return;
        if (std::getenv("VRAMBRIDGE_CPU_WARM_MODE") || std::getenv("VRAMBRIDGE_CPU_PROFILE_OUT")) {
            throw std::runtime_error("control experiment must not be combined with warmer/profiler");
        }
        const char * path = std::getenv("VRAMBRIDGE_CPU_CONTROL_OUT");
        if (!path) throw std::runtime_error("explicit private report required");
        struct stat info{};
        const auto parent = std::filesystem::path(path).parent_path();
        if (parent.empty() || stat(parent.c_str(), &info) || !S_ISDIR(info.st_mode) ||
            info.st_uid != getuid() || (info.st_mode & 077)) throw std::runtime_error("report parent must be owned and private");
        output = std::fopen(path, "wx");
        if (!output || fchmod(fileno(output), 0600)) throw std::runtime_error("cannot create exclusive private output");
        std::fprintf(stderr, "CPU control threads active: scalar dispatch/copy graphs one worker; other plans unchanged\n");
    }

    ~State() {
        if (!enabled) return;
        const nlohmann::json report = {{"schema_version", 1}, {"kind", "cpu_control_threads"}, {"complete", true},
            {"pid", getpid()}, {"source_sha256", VRAMBRIDGE_CONTROL_THREADS_SOURCE_SHA256},
            {"calls", calls}, {"eligible_calls", eligible_calls}, {"changed_calls", changed_calls},
            {"other_calls", other_calls}, {"maximum_original_threads", maximum_original_threads},
            {"control_threads", 1}, {"expert_plan_threads_changed", false},
            {"weights_changed", false}, {"routing_changed", false}, {"full_model_acceleration_claim", false}};
        const auto text = report.dump(2) + "\n";
        if (std::fwrite(text.data(), 1, text.size(), output) != text.size() || std::fclose(output)) {
            fail("cannot finish report");
        }
    }
};

} // namespace

extern "C" ggml_status ggml_graph_compute(ggml_cgraph * graph, ggml_cplan * plan) {
    try {
        static State state;
        if (!state.enabled) return state.original(graph, plan);
        const bool control = eligible(graph);
        {
            std::lock_guard lock(state.mutex);
            ++state.calls;
            state.eligible_calls += control;
            state.other_calls += !control;
            if (control && plan->n_threads > 1) {
                ++state.changed_calls;
                state.maximum_original_threads = std::max(state.maximum_original_threads, plan->n_threads);
            }
        }
        if (!control || plan->n_threads <= 1) return state.original(graph, plan);
        auto scalar_plan = *plan;
        scalar_plan.n_threads = 1;
        return state.original(graph, &scalar_plan);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}
