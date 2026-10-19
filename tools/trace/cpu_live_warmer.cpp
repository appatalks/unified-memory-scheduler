#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Compute = ggml_status (*)(ggml_cgraph *, ggml_cplan *);
using FreeBuffer = void (*)(ggml_backend_buffer_t);

namespace {

[[noreturn]] void fail(const std::string & message) {
    std::fprintf(stderr, "CPU live warmer: %s\n", message.c_str());
    std::abort();
}

int number(const char * name, int fallback, int minimum, int maximum) {
    const char * value = std::getenv(name);
    if (!value) return fallback;
    int result;
    const auto parsed = std::from_chars(value, value + std::strlen(value), result);
    if (parsed.ec != std::errc{} || *parsed.ptr || result < minimum || result > maximum) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return result;
}

int suffix_number(const char * value) {
    int result;
    const auto parsed = std::from_chars(value, value + std::strlen(value), result);
    if (parsed.ec != std::errc{} || *parsed.ptr || result < 0 || result > 255) {
        throw std::runtime_error("invalid native layer suffix");
    }
    return result;
}

std::optional<std::pair<int, int>> projection_name(const char * name) {
    if (std::strncmp(name, "blk.", 4) != 0) return {};
    int layer;
    const char * end = name + std::strlen(name);
    const auto parsed = std::from_chars(name + 4, end, layer);
    if (parsed.ec != std::errc{} || layer < 0 || layer > 255) return {};
    const char * parts[]{".ffn_gate_exps.weight", ".ffn_up_exps.weight", ".ffn_down_exps.weight"};
    for (int part = 0; part < 3; ++part) {
        if (std::strcmp(parsed.ptr, parts[part]) == 0) return std::pair{layer, part};
    }
    return {};
}

bool external(ggml_cgraph * graph, const ggml_tensor * tensor) {
    int depth = 0;
    for (; tensor; tensor = tensor->view_src) {
        if (++depth > 128) throw std::runtime_error("tensor ancestry exceeds bound");
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (ggml_graph_node(graph, i) == tensor) return false;
        }
    }
    return true;
}

struct Source {
    const unsigned char * data = nullptr;
    ggml_backend_buffer_t owner = nullptr;
    size_t expert_bytes = 0;
    int experts = 0;
};

struct Span {
    const unsigned char * data;
    size_t bytes;
    int part, expert;
};

struct Progress {
    int layer;
    std::array<std::atomic<uint64_t>, 12> completed{};
    std::vector<Span> spans;
};

struct Job {
    uint64_t generation;
    Clock::time_point submitted;
    std::vector<Span> spans;
    size_t bytes = 0;
    std::shared_ptr<Progress> progress;
};

struct State {
    Compute original = nullptr;
    std::string mode = "off";
    FILE * output = nullptr;
    size_t budget = 0;
    int helper_cpu = 0, maximum_us = 0;
    std::vector<int> compute_affinity;
    std::recursive_mutex graph_mutex;
    std::unordered_map<int, std::array<Source, 3>> sources;
    std::atomic<uint64_t> generation{0}, sink{0};
    std::atomic<bool> reading{false};
    std::mutex mutex;
    std::condition_variable condition;
    std::optional<Job> pending;
    std::shared_ptr<Progress> last_progress;
    std::thread helper;
    bool stopping = false, initialized = false;
    int affinity_error = 0;
    uint64_t graphs = 0, dispatches = 0, no_misses = 0, missing_sources = 0;
    uint64_t submitted = 0, planned_bytes = 0, overwritten = 0, buffer_drains = 0;
    uint64_t expert_calls_with_reader = 0, invalidations = 0;
    uint64_t jobs_started = 0, jobs_finished = 0, jobs_canceled = 0, jobs_expired = 0;
    uint64_t read_bytes = 0, before_deadline_bytes = 0, cancellation_tail_bytes = 0;
    uint64_t reader_ns = 0, queue_ns = 0;
    uint64_t needed_snapshot_bytes = 0, admitted_snapshot_bytes = 0, snapshot_jobs = 0;
    uint64_t replaced_snapshots = 0;

    void cancel() { generation.fetch_add(1, std::memory_order_acq_rel); }

    void worker() {
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(helper_cpu, &affinity);
        {
            std::lock_guard lock(mutex);
            affinity_error = pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            initialized = true;
            condition.notify_all();
        }
        if (affinity_error) return;
        for (;;) {
            std::unique_lock lock(mutex);
            condition.wait(lock, [&] { return stopping || pending.has_value(); });
            if (stopping) return;
            Job job = std::move(*pending);
            pending.reset();
            reading.store(true, std::memory_order_release);
            lock.unlock();
            const auto start = Clock::now();
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
            nvtxRangePushA("vrambridge.live_warm_job");
#endif
            queue_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(start - job.submitted).count();
            ++jobs_started;
            bool canceled = false, expired = false;
            uint64_t checksum = 0;
            for (size_t span_index = 0; span_index < job.spans.size(); ++span_index) {
                const auto & span = job.spans[span_index];
                for (size_t offset = 0; offset < span.bytes; offset += 64 * 1024) {
                    if (generation.load(std::memory_order_acquire) != job.generation) {
                        canceled = true;
                        break;
                    }
                    if (Clock::now() - job.submitted > std::chrono::microseconds(maximum_us)) {
                        expired = true;
                        break;
                    }
                    const size_t count = std::min<size_t>(64 * 1024, span.bytes - offset);
                    if (mode == "read") {
                        const volatile unsigned char * data = span.data + offset;
                        for (size_t index = 0; index < count; index += 64) checksum += data[index];
                        read_bytes += count;
                        if (generation.load(std::memory_order_acquire) == job.generation) {
                            before_deadline_bytes += count;
                            job.progress->completed[span_index].fetch_add(count, std::memory_order_release);
                        } else {
                            cancellation_tail_bytes += count;
                        }
                    }
                }
                if (canceled || expired) break;
            }
            sink.fetch_xor(checksum, std::memory_order_relaxed);
            reader_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
            nvtxRangePop();
#endif
            jobs_canceled += canceled;
            jobs_expired += expired;
            jobs_finished += !canceled && !expired;
            lock.lock();
            reading.store(false, std::memory_order_release);
            condition.notify_all();
        }
    }

    State() {
        if (const char * value = std::getenv("VRAMBRIDGE_CPU_WARM_MODE")) mode = value;
        if (mode != "off" && mode != "noop" && mode != "read") throw std::runtime_error("mode must be off, noop or read");
        const char * library = std::getenv("VRAMBRIDGE_CPU_WARM_LIBRARY");
        if (!library) throw std::runtime_error("explicit native CPU library required");
        void * handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
        if (!handle) throw std::runtime_error(dlerror());
        original = reinterpret_cast<Compute>(dlsym(handle, "ggml_graph_compute"));
        if (!original || original == &ggml_graph_compute) throw std::runtime_error("cannot resolve native CPU graph entry");
        if (mode == "off") return;
        if (std::getenv("VRAMBRIDGE_CPU_PROFILE_OUT")) throw std::runtime_error("do not combine with the graph profiler");
        if (sysconf(_SC_LEVEL1_DCACHE_LINESIZE) != 64) throw std::runtime_error("research requires 64-byte cachelines");
        helper_cpu = number("VRAMBRIDGE_CPU_WARM_CPU", 0, 0, CPU_SETSIZE - 1);
        maximum_us = number("VRAMBRIDGE_CPU_WARM_MAX_US", 200, 1, 5000);
        budget = size_t(number("VRAMBRIDGE_CPU_WARM_KIB", 1024, 0, 16384)) * 1024;
        cpu_set_t affinity;
        if (sched_getaffinity(0, sizeof(affinity), &affinity)) throw std::runtime_error("cannot inspect compute affinity");
        for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &affinity)) compute_affinity.push_back(i);
        const char * path = std::getenv("VRAMBRIDGE_CPU_WARM_OUT");
        if (!path) throw std::runtime_error("explicit private report path required");
        struct stat info{};
        const auto parent = std::filesystem::path(path).parent_path();
        if (parent.empty() || stat(parent.c_str(), &info) || !S_ISDIR(info.st_mode) ||
            info.st_uid != getuid() || (info.st_mode & 077)) throw std::runtime_error("report parent must be owned and private");
        output = std::fopen(path, "wx");
        if (!output || fchmod(fileno(output), 0600)) throw std::runtime_error("cannot create exclusive private report");
        helper = std::thread([this] { worker(); });
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return initialized; });
        if (affinity_error) {
            lock.unlock();
            helper.join();
            throw std::runtime_error("helper affinity failed: " + std::string(std::strerror(affinity_error)));
        }
        std::fprintf(stderr, "CPU live warmer active: mode=%s budget=%zu CPU=%d max_us=%d\n",
                     mode.c_str(), budget, helper_cpu, maximum_us);
    }

    ~State() {
        if (mode == "off") return;
        cancel();
        {
            std::lock_guard lock(mutex);
            stopping = true;
            pending.reset();
            condition.notify_all();
        }
        helper.join();
        json report = {{"schema_version", 1}, {"kind", "live_cpu_cache_warmer"}, {"complete", true},
            {"pid", getpid()}, {"initial_compute_affinity", compute_affinity},
            {"source_sha256", VRAMBRIDGE_LIVE_WARMER_SOURCE_SHA256},
            {"mode", mode}, {"warm_budget_bytes", budget}, {"helper_cpu", helper_cpu}, {"maximum_job_us", maximum_us},
            {"full_model_acceleration_claim", false}, {"weights_modified", false}, {"routing_modified", false},
            {"raw_ids_or_addresses_recorded", false}, {"graph_calls", graphs}, {"dispatches", dispatches},
            {"dispatches_without_misses", no_misses}, {"dispatches_without_sources", missing_sources},
            {"jobs_submitted", submitted}, {"planned_bytes", planned_bytes}, {"pending_jobs_overwritten", overwritten},
            {"jobs_started", jobs_started}, {"jobs_finished", jobs_finished}, {"jobs_canceled", jobs_canceled},
            {"jobs_expired", jobs_expired}, {"read_payload_bytes", read_bytes},
            {"bytes_completed_before_observed_deadline", before_deadline_bytes},
            {"cancellation_tail_payload_bytes", cancellation_tail_bytes},
            {"expert_calls_with_active_reader_at_entry", expert_calls_with_reader},
            {"buffer_release_drains", buffer_drains}, {"source_invalidations", invalidations},
            {"reader_ns", reader_ns}, {"queue_ns", queue_ns},
            {"post_admission_snapshot_jobs", snapshot_jobs}, {"replaced_snapshot_jobs", replaced_snapshots},
            {"read_snapshot_bytes_for_actual_cpu_experts", needed_snapshot_bytes},
            {"read_snapshot_bytes_for_admitted_gpu_experts", admitted_snapshot_bytes},
            {"snapshot_scope", "Completed chunks sampled at the next CPU expert graph entry; this is not cache-hit or time-saved accounting."},
            {"scope", "Current pre-admission misses; read-only bounded prefix; cancel at any CPU expert graph; no guarantee of cache hits or GPU overlap."}};
        const auto text = report.dump(2) + "\n";
        if (std::fwrite(text.data(), 1, text.size(), output) != text.size() || std::fclose(output)) {
            fail("cannot finish report");
        }
    }

    void learn(const ggml_tensor * tensor) {
        const auto key = projection_name(tensor->name);
        if (!key) return;
        if (tensor->type != GGML_TYPE_MXFP4 || !tensor->buffer ||
            !ggml_backend_buffer_is_host(tensor->buffer) || !ggml_is_contiguous(tensor) ||
            ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            tensor->ne[2] <= 0 || tensor->ne[2] > 128 || tensor->ne[3] != 1) {
            throw std::runtime_error("unsupported native expert source layout/ownership");
        }
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(tensor->buffer));
        const auto data = reinterpret_cast<uintptr_t>(tensor->data);
        const size_t bytes = ggml_nbytes(tensor), size = ggml_backend_buffer_get_size(tensor->buffer);
        if (data < base || data - base > size || bytes > size - (data - base)) {
            throw std::runtime_error("native expert source exceeds owner buffer");
        }
        auto & source = sources[key->first][key->second];
        if (source.data && source.data != tensor->data) throw std::runtime_error("overlapping models/layer identities unsupported");
        source = {static_cast<const unsigned char *>(tensor->data), tensor->buffer,
                  bytes / size_t(tensor->ne[2]), int(tensor->ne[2])};
    }

    void dispatch(ggml_cgraph * graph, const ggml_tensor * node) {
        const auto * ids = node->src[0];
        const auto * map = node->src[1];
        if (node->op != GGML_OP_CUSTOM || !ids || !map ||
            std::strncmp(map->name, "moe_cache.slot_map-", 19)) return;
        if (ids->ne[1] != 1) return;
        if (ids->type != GGML_TYPE_I32 || ids->ne[0] < 1 || ids->ne[0] > 4 ||
            map->type != GGML_TYPE_I32 || map->ne[0] != 1 || map->ne[1] < ids->ne[0] ||
            map->ne[1] > 128 || !ids->buffer || !map->buffer ||
            !ggml_backend_buffer_is_host(ids->buffer) || !ggml_backend_buffer_is_host(map->buffer) ||
            !external(graph, ids) || !external(graph, map)) throw std::runtime_error("unsupported dispatcher inputs");
        ++dispatches;
        const int layer = suffix_number(map->name + 19);
        std::vector<int32_t> misses;
        std::array<bool, 128> seen{};
        for (int rank = 0; rank < ids->ne[0]; ++rank) {
            int32_t id, slot;
            std::memcpy(&id, static_cast<const char *>(ids->data) + rank * ids->nb[0], sizeof(id));
            if (id < 0 || id >= map->ne[1] || seen[id]) throw std::runtime_error("invalid current router ID");
            seen[id] = true;
            std::memcpy(&slot, static_cast<const char *>(map->data) + id * map->nb[1], sizeof(slot));
            if (slot < -1) throw std::runtime_error("invalid current slot map");
            if (slot == -1) misses.push_back(id);
        }
        if (misses.empty()) { ++no_misses; return; }
        const auto found = sources.find(layer);
        if (found == sources.end() || std::any_of(found->second.begin(), found->second.end(),
                [&](const Source & source) { return !source.data || source.experts != map->ne[1]; })) {
            ++missing_sources;
            return;
        }
        Job job{generation.fetch_add(1, std::memory_order_acq_rel) + 1, Clock::now(), {}};
        job.progress = std::make_shared<Progress>();
        job.progress->layer = layer;
        job.spans.reserve(12);
        for (int part = 0; part < 3; ++part) {
            const auto & source = found->second[part];
            for (int32_t id : misses) {
                const size_t count = std::min(budget - job.bytes, source.expert_bytes) / 64 * 64;
                if (count) job.spans.push_back({source.data + size_t(id) * source.expert_bytes, count, part, id});
                job.bytes += count;
                if (job.bytes == budget) break;
            }
            if (job.bytes == budget) break;
        }
        ++submitted;
        planned_bytes += job.bytes;
        job.progress->spans = job.spans;
        replaced_snapshots += last_progress != nullptr;
        last_progress = job.progress;
        {
            std::lock_guard lock(mutex);
            overwritten += pending.has_value();
            pending = std::move(job);
            condition.notify_all();
        }
    }

    void snapshot(ggml_cgraph * graph) {
        if (!last_progress) return;
        const auto progress = std::move(last_progress);
        std::array<std::array<bool, 128>, 3> selected{};
        std::array<bool, 3> found{};
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto * node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_MUL_MAT_ID) continue;
            const auto key = projection_name(node->src[0]->name);
            if (!key || key->first != progress->layer) continue;
            const auto * ids = node->src[2];
            if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || !ids->buffer ||
                !ggml_backend_buffer_is_host(ids->buffer) || !external(graph, ids)) {
                throw std::runtime_error("cannot observe post-admission CPU IDs");
            }
            for (int rank = 0; rank < ids->ne[0]; ++rank) {
                int32_t id;
                std::memcpy(&id, static_cast<const char *>(ids->data) + rank * ids->nb[0], sizeof(id));
                if (id < -1 || id >= 128) throw std::runtime_error("post-admission CPU ID outside domain");
                if (id >= 0) selected[key->second][id] = true;
            }
            found[key->second] = true;
        }
        if (!std::all_of(found.begin(), found.end(), [](bool value) { return value; })) {
            throw std::runtime_error("prefetch snapshot does not match the next native CPU expert graph");
        }
        ++snapshot_jobs;
        for (size_t index = 0; index < progress->spans.size(); ++index) {
            const auto & span = progress->spans[index];
            const uint64_t bytes = progress->completed[index].load(std::memory_order_acquire);
            if (selected[span.part][span.expert]) needed_snapshot_bytes += bytes;
            else admitted_snapshot_bytes += bytes;
        }
    }

    void invalidate(ggml_backend_buffer_t owner) {
        std::lock_guard lock(graph_mutex);
        bool relevant = false;
        for (auto & [_, projections] : sources) {
            for (auto & source : projections) {
                if (source.owner == owner) { source = {}; ++invalidations; relevant = true; }
            }
        }
        if (!relevant) return;
        cancel();
        last_progress.reset();
        std::unique_lock pending_lock(mutex);
        pending.reset();
        condition.wait(pending_lock, [&] { return !reading.load(std::memory_order_acquire); });
        ++buffer_drains;
    }
};

std::atomic<State *> active{nullptr};

State & state() {
    static State * value = [] {
        auto * created = new State;
        active.store(created, std::memory_order_release);
        std::atexit([] {
            auto * current = active.exchange(nullptr, std::memory_order_acq_rel);
            delete current;
        });
        return created;
    }();
    return *value;
}

} // namespace

extern "C" ggml_status ggml_graph_compute(ggml_cgraph * graph, ggml_cplan * plan) {
    try {
        auto & warmer = state();
        if (warmer.mode == "off") return warmer.original(graph, plan);
        std::lock_guard lock(warmer.graph_mutex);
        ++warmer.graphs;
        bool experts = false;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT_ID) {
                experts = true;
                warmer.learn(node->src[0]);
            }
        }
        if (experts) {
            warmer.cancel();
            warmer.expert_calls_with_reader += warmer.reading.load(std::memory_order_acquire);
            warmer.snapshot(graph);
        } else {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) warmer.dispatch(graph, ggml_graph_node(graph, i));
        }
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
        nvtxRangePushA(experts ? "vrambridge.live_expert" : "vrambridge.live_control");
#endif
        const auto status = warmer.original(graph, plan);
#ifdef VRAMBRIDGE_CPU_PROFILE_NVTX
        nvtxRangePop();
#endif
        return status;
    } catch (const std::exception & error) {
        fail(error.what());
    }
}

extern "C" void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    try {
        static const auto original = reinterpret_cast<FreeBuffer>(dlsym(RTLD_NEXT, "ggml_backend_buffer_free"));
        if (!original || original == &ggml_backend_buffer_free) fail("cannot resolve native buffer release");
        if (auto * current = active.load(std::memory_order_acquire)) current->invalidate(buffer);
        original(buffer);
    } catch (const std::exception & error) {
        fail(error.what());
    }
}
