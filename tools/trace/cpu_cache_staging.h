#pragma once

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

struct CacheWarmWorker {
    std::vector<std::pair<const char *, size_t>> spans;
    size_t bytes = 0, line_bytes;
    std::mutex mutex;
    std::condition_variable condition;
    bool initialized = false, requested = false, completed = false, stopping = false;
    int affinity_error = 0;
    double touch_ms = 0;
    std::atomic<uint64_t> sink{0};
    std::thread worker;

    CacheWarmWorker(const std::vector<ggml_tensor *> & tensors, size_t budget, int cpu, size_t line) :
            line_bytes(line) {
        for (const auto * tensor : tensors) {
            const size_t current = std::min(budget - bytes, ggml_nbytes(tensor)) / line_bytes * line_bytes;
            require(reinterpret_cast<uintptr_t>(tensor->data) % alignof(uint64_t) == 0,
                    "Unaligned CPU warming source.");
            spans.emplace_back(static_cast<const char *>(tensor->data), current);
            bytes += current;
            if (bytes == budget) break;
        }
        worker = std::thread([&, cpu] {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(cpu, &mask);
            {
                std::lock_guard lock(mutex);
                affinity_error = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
                initialized = true;
                condition.notify_all();
            }
            if (affinity_error) return;
            for (;;) {
                std::unique_lock lock(mutex);
                condition.wait(lock, [&] { return requested || stopping; });
                if (stopping) return;
                requested = false;
                lock.unlock();
                const auto start = Clock::now();
                uint64_t checksum = 0;
                for (const auto & [source, count] : spans) {
                    for (size_t offset = 0; offset < count; offset += line_bytes) {
                        uint64_t value;
                        std::memcpy(&value, source + offset, sizeof(value));
                        checksum ^= value;
                    }
                }
                sink.fetch_xor(checksum, std::memory_order_relaxed);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                const auto end = Clock::now();
                lock.lock();
                touch_ms = std::chrono::duration<double, std::milli>(end - start).count();
                completed = true;
                condition.notify_all();
            }
        });
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return initialized; });
        if (affinity_error) {
            lock.unlock();
            worker.join();
            throw std::runtime_error("Cannot pin cache warmer: " + std::string(std::strerror(affinity_error)));
        }
    }

    ~CacheWarmWorker() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            condition.notify_all();
        }
        worker.join();
    }

    double warm() {
        const auto start = Clock::now();
        std::unique_lock lock(mutex);
        completed = false;
        requested = true;
        condition.notify_all();
        condition.wait(lock, [&] { return completed; });
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
};

static json check_cpu_cache_staging(ggml_backend_t cpu, const Weights & weights, int repetitions,
                                   size_t warm_bytes, int warm_cpu, bool self_test) {
    const long cache_line = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    require(cache_line >= long(sizeof(uint64_t)) && (cache_line & (cache_line - 1)) == 0,
            "Cannot determine a supported CPU cache-line size.");
    cpu_set_t affinity;
    require(sched_getaffinity(0, sizeof(affinity), &affinity) == 0, "Cannot inspect compute affinity.");
    std::vector<int> compute_cpus;
    for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &affinity)) compute_cpus.push_back(i);
    const std::vector<float> routing(weights.selected, 1.0f / weights.selected);
    Branch branch(cpu, weights, 0, weights.selected, routing, true, 1);
    const auto input = activation_input(weights, 1, 0, self_test, "sine");
    branch.set_input(input);
    branch.compute(false);
    const auto expected = branch.result();
    const auto finite_check = compare(expected, expected);
    std::vector<ggml_tensor *> tensors;
    std::set<ggml_tensor *> seen;
    for (int i = 0; i < ggml_graph_n_nodes(branch.graph); ++i) {
        auto * node = ggml_graph_node(branch.graph, i);
        if (node->op == GGML_OP_MUL_MAT_ID && seen.insert(node->src[0]).second) {
            require(node->src[0]->buffer && ggml_backend_buffer_is_host(node->src[0]->buffer),
                    "Cache staging requires owned host projection buffers.");
            tensors.push_back(node->src[0]);
        }
    }
    require(tensors.size() == 3, "Cache staging requires the original three-projection FFN.");
    CacheWarmWorker warmer(tensors, warm_bytes, warm_cpu, static_cast<size_t>(cache_line));
    const auto flush = [&] {
        for (const auto * tensor : tensors) {
            const auto * data = static_cast<const char *>(tensor->data);
            for (size_t offset = 0; offset < ggml_nbytes(tensor); offset += cache_line) _mm_clflush(data + offset);
        }
        _mm_mfence();
    };
    const auto compute = [&] {
        const auto start = Clock::now();
        branch.compute(false);
        const auto actual = branch.result();
        const auto end = Clock::now();
        require(expected.size() == actual.size() &&
                std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0,
                "Cache staging changed native FFN output bits.");
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    std::vector<double> cold, warm, handoff, touch, serial;
    for (int iteration = -2; iteration < repetitions; ++iteration) {
        double cold_ms, warm_ms, handoff_ms;
        const auto run_cold = [&] { flush(); cold_ms = compute(); };
        const auto run_warm = [&] { flush(); handoff_ms = warmer.warm(); warm_ms = compute(); };
        if (iteration % 2 == 0) { run_cold(); run_warm(); }
        else { run_warm(); run_cold(); }
        if (iteration >= 0) {
            cold.push_back(cold_ms); warm.push_back(warm_ms); handoff.push_back(handoff_ms);
            touch.push_back(warmer.touch_ms); serial.push_back(handoff_ms + warm_ms);
        }
    }
    size_t total_bytes = 0;
    for (const auto * tensor : tensors) total_bytes += ggml_nbytes(tensor);
    return {{"schema_version", 1}, {"kind", "cpu_cache_staging_microbench"},
        {"probe_source_sha256", VRAMBRIDGE_HYBRID_SOURCE_SHA256},
        {"cache_source_sha256", VRAMBRIDGE_CPU_CACHE_SOURCE_SHA256},
        {"backend_commit", llama_commit()}, {"reference_check", finite_check},
        {"all_checks_passed", true}, {"bitwise_equal_to_native_cpu", true},
        {"full_model_acceleration_claim", false}, {"actual_gpu_window_execution", false},
        {"activation_source", "deterministic synthetic"}, {"raw_activations_recorded", false},
        {"kernel_math_changed", false}, {"model_weights_changed", false},
        {"expert_count", weights.selected}, {"projection_weight_bytes", total_bytes},
        {"cache_line_bytes", cache_line}, {"warm_budget_bytes", warm_bytes},
        {"warmed_prefix_bytes", warmer.bytes}, {"warming_cpu", warm_cpu}, {"compute_affinity", compute_cpus},
        {"warming_method", "read one word per cache line of owned native projection buffers, in graph weight order"},
        {"cold_method", "CLFLUSH on those owned buffers only; no OS cache drop or original-file modification"},
        {"timing_excludes", {"allocation", "initial weight loading", "helper creation", "owned-buffer cache flush"}},
        {"timing_includes", "Compute includes native quantization/FFN and CPU result readback; warming includes per-job wake/ack"},
        {"cold_compute_ms", cold}, {"warm_compute_ms", warm}, {"warm_handoff_ms", handoff},
        {"warm_touch_ms", touch}, {"serial_warm_plus_compute_ms", serial},
        {"median_cold_compute_ms", median(cold)}, {"median_warm_compute_ms", median(warm)},
        {"median_warm_handoff_ms", median(handoff)}, {"median_warm_touch_ms", median(touch)},
        {"median_serial_warm_plus_compute_ms", median(serial)},
        {"warm_compute_speedup", median(cold) / median(warm)},
        {"ideal_hidden_warm_compute_saving_ms", median(cold) - median(warm)},
        {"measured_touch_gbps", warmer.bytes / (median(touch) * 1e6)},
        {"limitations", "Isolated cold/warm cache calibration, not a predictor, real decode speedup or demonstrated GPU overlap."}};
}
