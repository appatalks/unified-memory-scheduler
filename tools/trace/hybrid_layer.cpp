#include "build-info.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
struct CloseFile {
    void operator()(FILE * file) const noexcept { std::fclose(file); }
};

static void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static Context context() {
    ggml_init_params params{1024 * 1024, nullptr, true};
    Context result(ggml_init(params), ggml_free);
    require(result != nullptr, "Cannot allocate tensor metadata.");
    return result;
}

struct Weights {
    int hidden = 0;
    int intermediate = 0;
    int selected = 4;
    ggml_type type = GGML_TYPE_F32;
    std::vector<int> ids{0, 7, 42, 113};
    std::map<std::string, std::vector<uint8_t>> tensors;
    uint64_t bytes = 0;
};

static Weights read_weights(const fs::path & first, int layer) {
    std::vector<fs::path> shards{first};
    std::smatch match;
    const std::string name = first.filename().string();
    if (std::regex_match(name, match, std::regex(R"((.*)-00001-of-([0-9]{5})\.gguf)"))) {
        const int count = std::stoi(match[2]);
        require(count >= 1 && count <= 100, "Unsupported shard count.");
        for (int index = 2; index <= count; ++index) {
            char suffix[64];
            std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", index, count);
            shards.push_back(first.parent_path() / (match[1].str() + suffix));
        }
    }
    Weights result;
    const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
    for (const auto & shard : shards) {
        require(fs::is_regular_file(shard), "Missing GGUF shard: " + shard.filename().string());
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(
            gguf_init_from_file(shard.c_str(), {true, &raw}), gguf_free);
        Context metadata(raw, ggml_free);
        require(file != nullptr && metadata != nullptr, "Cannot read GGUF metadata.");
        const int64_t arch = gguf_find_key(file.get(), "general.architecture");
        if (shard == first || arch >= 0) {
            require(arch >= 0 && gguf_get_kv_type(file.get(), arch) == GGUF_TYPE_STRING &&
                    std::string(gguf_get_val_str(file.get(), arch)) == "gpt-oss",
                    "Real-weight probe currently supports GPT-OSS only.");
        }
        for (const std::string part : {"gate", "up", "down"}) {
            for (const std::string kind : {"weight", "bias"}) {
                const std::string key = part + "." + kind;
                const std::string tensor_name = prefix + part + "_exps." + kind;
                const int64_t id = gguf_find_tensor(file.get(), tensor_name.c_str());
                if (id < 0) {
                    continue;
                }
                require(!result.tensors.count(key), "Duplicate expert component across shards.");
                const auto * tensor = ggml_get_tensor(metadata.get(), tensor_name.c_str());
                require(tensor != nullptr && tensor->ne[3] == 1, "Missing expert shape.");
                if (kind == "weight") {
                    require(tensor->ne[0] == 2880 && tensor->ne[1] == 2880 && tensor->ne[2] == 128 &&
                            tensor->type == GGML_TYPE_MXFP4, "Unsupported GPT-OSS MXFP4 weight shape/type.");
                    result.hidden = result.intermediate = 2880;
                    result.type = tensor->type;
                } else {
                    require(tensor->ne[0] == 2880 && tensor->ne[1] == 128 && tensor->ne[2] == 1 &&
                            tensor->type == GGML_TYPE_F32, "Unsupported GPT-OSS expert bias shape/type.");
                }
                const size_t stride = ggml_nbytes(tensor) / 128;
                require(stride * 128 == ggml_nbytes(tensor), "Nonuniform expert tensor storage.");
                const uint64_t start = gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), id);
                std::ifstream input(shard, std::ios::binary);
                require(input.good(), "Cannot open expert weight shard.");
                auto & bytes = result.tensors[key];
                bytes.resize(stride * result.selected);
                for (int local = 0; local < result.selected; ++local) {
                    const uint64_t offset = start + stride * result.ids[local];
                    require(offset <= fs::file_size(shard) && stride <= fs::file_size(shard) - offset,
                            "Expert slice exceeds shard storage.");
                    input.seekg(static_cast<std::streamoff>(offset));
                    input.read(reinterpret_cast<char *>(bytes.data() + local * stride), stride);
                    require(input.good(), "Failed to read a complete expert slice.");
                }
                result.bytes += bytes.size();
            }
        }
    }
    require(result.tensors.size() == 6 && result.hidden > 0, "Missing selected layer expert weights/biases.");
    return result;
}

static Weights synthetic_weights() {
    Weights result;
    result.hidden = 64;
    result.intermediate = 96;
    result.ids = {0, 1, 2, 3};
    for (const std::string part : {"gate", "up", "down"}) {
        for (const std::string kind : {"weight", "bias"}) {
            const size_t elements = kind == "bias"
                ? (part == "down" ? result.hidden : result.intermediate)
                : result.hidden * result.intermediate;
            std::vector<float> values(elements * result.selected);
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = 0.03f * std::sin(float(i * 17 + part.size() * 31 + kind.size()));
            }
            auto & bytes = result.tensors[part + "." + kind];
            bytes.resize(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            result.bytes += bytes.size();
        }
    }
    return result;
}

struct Branch {
    ggml_backend_t backend;
    Context ctx = context();
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    int count;

    Branch(ggml_backend_t device, const Weights & weights, int begin, int end,
           const std::vector<float> & routing, bool reduce = true) : backend(device), count(end - begin) {
        require(begin >= 0 && end <= weights.selected && count > 0, "Invalid expert partition.");
        std::map<std::string, ggml_tensor *> tensors;
        for (const std::string part : {"gate", "up", "down"}) {
            tensors[part + ".weight"] = ggml_new_tensor_3d(ctx.get(), weights.type,
                part == "down" ? weights.intermediate : weights.hidden,
                part == "down" ? weights.hidden : weights.intermediate, count);
            tensors[part + ".bias"] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32,
                part == "down" ? weights.hidden : weights.intermediate, count);
        }
        input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, weights.hidden, 1, 1);
        ggml_set_input(input);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, count, 1);
        auto * probabilities = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, count, 1);
        auto * gate = ggml_mul_mat_id(ctx.get(), tensors.at("gate.weight"), input, ids);
        gate = ggml_add_id(ctx.get(), gate, tensors.at("gate.bias"), ids);
        auto * up = ggml_mul_mat_id(ctx.get(), tensors.at("up.weight"), input, ids);
        up = ggml_add_id(ctx.get(), up, tensors.at("up.bias"), ids);
        auto * activated = ggml_swiglu_oai(ctx.get(), gate, up, 1.702f, 7.0f);
        auto * down = ggml_mul_mat_id(ctx.get(), tensors.at("down.weight"), activated, ids);
        down = ggml_add_id(ctx.get(), down, tensors.at("down.bias"), ids);
        auto * weighted = ggml_mul(ctx.get(), down, probabilities);
        auto * transposed = ggml_cont(ctx.get(), ggml_transpose(ctx.get(),
            ggml_reshape_2d(ctx.get(), weighted, weights.hidden, count)));
        output = ggml_reshape_1d(ctx.get(), ggml_sum_rows(ctx.get(), transposed), weights.hidden);
        if (!reduce) {
            output = ggml_reshape_2d(ctx.get(), down, weights.hidden, count);
        }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(graph, output);
        for (int node = 0; node < ggml_graph_n_nodes(graph); ++node) {
            require(ggml_backend_supports_op(backend, ggml_graph_node(graph, node)),
                    "Backend does not support an expert graph operation.");
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(buffer != nullptr, "Cannot allocate expert branch.");
        for (const auto & [key, tensor] : tensors) {
            const auto & source = weights.tensors.at(key);
            const size_t stride = source.size() / weights.selected;
            require(ggml_nbytes(tensor) == stride * count, "Gathered expert slice size mismatch.");
            ggml_backend_tensor_set(tensor, source.data() + stride * begin, 0, stride * count);
        }
        std::vector<int32_t> local_ids(count);
        std::iota(local_ids.begin(), local_ids.end(), 0);
        ggml_backend_tensor_set(ids, local_ids.data(), 0, local_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(probabilities, routing.data() + begin, 0, count * sizeof(float));
    }

    void set_input(const std::vector<float> & values) {
        require(values.size() * sizeof(float) == ggml_nbytes(input), "Activation size mismatch.");
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    }

    void compute(bool asynchronous) {
        const auto status = asynchronous ? ggml_backend_graph_compute_async(backend, graph)
                                         : ggml_backend_graph_compute(backend, graph);
        require(status == GGML_STATUS_SUCCESS, "Expert graph execution failed.");
    }

    std::vector<float> result() {
        std::vector<float> values(ggml_nelements(output));
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
        return values;
    }
};

struct Merge {
    ggml_backend_t backend;
    Context ctx = context();
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    Buffer staging{nullptr, ggml_backend_buffer_free};
    ggml_tensor * cpu;
    ggml_tensor * gpu;
    ggml_tensor * output;
    ggml_cgraph * graph;

    Merge(ggml_backend_t stream, ggml_backend_dev_t device, int hidden, int resident_count,
          const std::vector<float> & routing, bool gpu_reduce, bool self_test) : backend(stream) {
        cpu = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden, gpu_reduce ? std::max(1, 4 - resident_count) : 1);
        gpu = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden, gpu_reduce ? std::max(1, resident_count) : 1);
        ggml_set_input(cpu);
        ggml_set_input(gpu);
        output = nullptr;
        ggml_tensor * probabilities = nullptr;
        if (gpu_reduce) {
            auto * combined = resident_count == 0 ? cpu : resident_count == 4 ? gpu :
                ggml_concat(ctx.get(), gpu, cpu, 1);
            probabilities = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, 4);
            auto * weighted = ggml_mul(ctx.get(), combined, probabilities);
            auto * transposed = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), weighted));
            output = ggml_reshape_1d(ctx.get(), ggml_sum_rows(ctx.get(), transposed), hidden);
        } else {
            output = ggml_add(ctx.get(), cpu, gpu);
        }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, output);
        for (int node = 0; node < ggml_graph_n_nodes(graph); ++node) {
            require(ggml_backend_supports_op(backend, ggml_graph_node(graph, node)),
                    "A reconciliation operation is unsupported.");
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(buffer != nullptr, "Cannot allocate GPU merge tensors.");
        if (probabilities) {
            ggml_backend_tensor_set(probabilities, routing.data(), 0, routing.size() * sizeof(float));
        }
        const auto pinned = self_test ? ggml_backend_get_default_buffer_type(backend)
                                      : ggml_backend_dev_host_buffer_type(device);
        require(pinned != nullptr, "GPU does not expose a bounded pinned staging buffer type.");
        staging.reset(ggml_backend_buft_alloc_buffer(pinned, ggml_nbytes(cpu)));
        require(staging != nullptr && ggml_backend_buffer_is_host(staging.get()), "Cannot allocate merge staging.");
    }

    void compute(Branch * host, Branch * resident) {
        if (host) {
            require(ggml_nbytes(host->output) == ggml_nbytes(cpu), "CPU partial output shape mismatch.");
            void * data = ggml_backend_buffer_get_base(staging.get());
            ggml_backend_tensor_get(host->output, data, 0, ggml_nbytes(cpu));
            ggml_backend_tensor_set_async(backend, cpu, data, 0, ggml_nbytes(cpu));
        } else {
            ggml_backend_tensor_memset(cpu, 0, 0, ggml_nbytes(cpu));
        }
        if (resident) {
            require(ggml_nbytes(resident->output) == ggml_nbytes(gpu), "GPU partial output shape mismatch.");
            ggml_backend_tensor_copy_async(resident->backend, backend, resident->output, gpu);
        } else {
            ggml_backend_tensor_memset(gpu, 0, 0, ggml_nbytes(gpu));
        }
        require(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS, "GPU merge failed.");
        ggml_backend_synchronize(backend);
    }

    std::vector<float> result() {
        std::vector<float> values(output->ne[0]);
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
        return values;
    }
};

static json compare(const std::vector<float> & reference, const std::vector<float> & actual) {
    require(reference.size() == actual.size(), "Output shape mismatch.");
    double error2 = 0, reference2 = 0, maximum = 0, scale = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(actual[i]), "Nonfinite expert output.");
        const double error = double(actual[i]) - reference[i];
        error2 += error * error;
        reference2 += double(reference[i]) * reference[i];
        maximum = std::max(maximum, std::abs(error));
        scale = std::max(scale, std::abs(double(reference[i])));
    }
    const double relative_l2 = std::sqrt(error2 / std::max(reference2, 1e-30));
    return {{"max_abs_error", maximum}, {"reference_max_abs", scale}, {"relative_l2_error", relative_l2},
            {"passed", maximum <= 0.00001 + 0.0001 * scale && relative_l2 <= 0.0001},
            {"tolerance", "max_abs <= 0.00001 + 0.0001 * reference_max_abs; relative_l2 <= 0.0001"}};
}

static std::vector<float> scalar_reference(const Weights & weights, const std::vector<float> & input,
                                           const std::vector<float> & routing) {
    require(weights.type == GGML_TYPE_F32, "Scalar reference is for the synthetic F32 self-test only.");
    const auto read = [&](const std::string & key, size_t index) {
        float value;
        const auto & bytes = weights.tensors.at(key);
        require((index + 1) * sizeof(float) <= bytes.size(), "Scalar reference tensor bounds error.");
        std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(value));
        return double(value);
    };
    std::vector<double> output(weights.hidden, 0);
    for (int expert = 0; expert < weights.selected; ++expert) {
        std::vector<double> activated(weights.intermediate);
        for (int row = 0; row < weights.intermediate; ++row) {
            double gate = read("gate.bias", expert * weights.intermediate + row);
            double up = read("up.bias", expert * weights.intermediate + row);
            for (int col = 0; col < weights.hidden; ++col) {
                const size_t index = (expert * weights.intermediate + row) * weights.hidden + col;
                gate += read("gate.weight", index) * input[col];
                up += read("up.weight", index) * input[col];
            }
            gate = std::min(gate, 7.0);
            up = std::clamp(up, -7.0, 7.0);
            activated[row] = gate / (1.0 + std::exp(-1.702 * gate)) * (up + 1.0);
        }
        for (int row = 0; row < weights.hidden; ++row) {
            double down = read("down.bias", expert * weights.hidden + row);
            for (int col = 0; col < weights.intermediate; ++col) {
                const size_t index = (expert * weights.hidden + row) * weights.intermediate + col;
                down += read("down.weight", index) * activated[col];
            }
            output[row] += routing[expert] * down;
        }
    }
    return std::vector<float>(output.begin(), output.end());
}

static double median(std::vector<double> values) {
    require(!values.empty(), "Cannot summarize empty timings.");
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}

static int run(int argc, char ** argv) {
    fs::path model, destination;
    int layer = 0, repetitions = 20, samples = 3, threads = 4;
    bool self_test = false, gpu_reduce = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "Usage: vrambridge-hybrid-layer --model first-shard.gguf --output report.json "
                         "[--layer N] [--repetitions N] [--samples N] [--threads N] [--gpu-reduce]\n"
                         "Or --self-test (CPU F32 only). Real probe uses GPT-OSS MXFP4 weights, synthetic "
                         "activations and fixed expert IDs. Warm isolated layer timings are not model tok/s.\n";
            return 0;
        }
        if (arg == "--self-test") {
            self_test = true;
            continue;
        }
        if (arg == "--gpu-reduce") {
            gpu_reduce = true;
            continue;
        }
        require(i + 1 < argc, "Missing option value: " + arg);
        const std::string value = argv[++i];
        if (arg == "--model") model = value;
        else if (arg == "--output") destination = value;
        else {
            size_t consumed = 0;
            const int number = std::stoi(value, &consumed);
            require(consumed == value.size(), "Invalid numeric option.");
            if (arg == "--layer") layer = number;
            else if (arg == "--repetitions") repetitions = number;
            else if (arg == "--samples") samples = number;
            else if (arg == "--threads") threads = number;
            else throw std::runtime_error("Unknown option: " + arg);
        }
    }
    require(layer >= 0 && layer < 36 && repetitions >= samples && repetitions <= 1000 &&
            samples >= 1 && samples <= 20 && threads >= 1 && threads <= 32, "Option out of supported range.");
    require(self_test || (!model.empty() && !destination.empty()), "Real probe requires --model and --output.");
    require(destination.empty() || (!fs::exists(destination) && !fs::exists(destination.string() + ".part")),
            "Refusing to overwrite existing result or partial result.");
    const std::string identity = llama_commit();
    require(identity.size() >= 7 &&
            std::string("620d2225cc436ab0b001e3200e590e3b286ef9ab").rfind(identity, 0) == 0,
            "Pinned backend version mismatch.");
    ggml_backend_load_all_from_path(VRAMBRIDGE_LLAMA_LIBRARY_DIR);
    ggml_backend_dev_t cpu_device = nullptr, gpu_device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && !cpu_device) cpu_device = dev;
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU && !gpu_device) gpu_device = dev;
    }
    require(cpu_device != nullptr, "No CPU backend.");
    require(self_test || gpu_device != nullptr, "No usable GPU; no CPU-only fallback for a real hybrid probe.");
    Backend cpu(ggml_backend_dev_init(cpu_device, nullptr), ggml_backend_free);
    require(cpu != nullptr, "CPU backend initialization failed.");
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_device), "ggml_backend_set_n_threads"));
    require(set_threads != nullptr, "CPU backend thread control unavailable.");
    set_threads(cpu.get(), threads);
    Weights weights = self_test ? synthetic_weights() : read_weights(model, layer);
    const std::vector<float> routing{0.05f, 0.15f, 0.30f, 0.50f};
    Branch reference(cpu.get(), weights, 0, 4, routing);
    Backend gpu(nullptr, ggml_backend_free), reconciliation(nullptr, ggml_backend_free);
    if (!self_test) {
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(gpu_device, &props);
        require(props.memory_free >= 256 * 1024 * 1024, "Less than 256 MiB free GPU memory; unload the app model.");
        gpu.reset(ggml_backend_dev_init(gpu_device, nullptr));
        reconciliation.reset(ggml_backend_dev_init(gpu_device, nullptr));
        require(gpu != nullptr && reconciliation != nullptr, "GPU compute/merge backend initialization failed.");
    }
    json report = {
        {"schema_version", 1}, {"backend_commit", llama_commit()}, {"layer", layer},
        {"probe_source_sha256", VRAMBRIDGE_HYBRID_SOURCE_SHA256},
        {"weights", self_test ? "synthetic_f32" : "real_gptoss_mxfp4"},
        {"independent_scalar_reference", self_test},
        {"model", model.filename().string()}, {"selected_expert_ids", weights.ids},
        {"selected_weight_bytes", weights.bytes}, {"activation_source", "deterministic synthetic"},
        {"routing_weights", routing}, {"cpu_threads", threads}, {"repetitions", repetitions}, {"samples", samples},
        {"full_model_acceleration_claim", false}, {"warm_cache_microbenchmark", true},
        {"separate_gpu_reconciliation_stream", !self_test}, {"hardware_gpu_partition", false},
        {"reduction_strategy", gpu_reduce ? "per_expert_outputs_weighted_and_reduced_on_merge_backend"
                                         : "weighted_partial_vectors_then_merge_add"},
        {"timing_order", "serial/concurrent interleaved; first schedule alternates each repetition"},
        {"partitions", json::array()},
        {"limitations", {"Fixed selected weights may fit CPU/GPU caches; this is not a full-model bandwidth roofline.",
                          "Synthetic activations and chosen experts are not actual model hidden states or router decisions.",
                          "No attention, router, shared expert, residual, prefill, disk tier, or dynamic cache is executed.",
                          "Correctness tolerance is a microkernel gate, not full-model token/logit equivalence.",
                          "Cross-backend async copies may internally fall back to synchronization.",
                          "Timing includes input upload, dispatch, CPU execution, output staging, dependency copy, merge and final synchronization.",
                          "Weight loading, graph setup, and verification readback are outside steady-state timing."}},
    };
    bool passed = true;
    for (int resident_count = 0; resident_count <= 4; ++resident_count) {
        std::unique_ptr<Branch> host, resident;
        if (resident_count < 4) host = std::make_unique<Branch>(
            cpu.get(), weights, resident_count, 4, routing, !gpu_reduce);
        if (resident_count > 0) resident = std::make_unique<Branch>(
            self_test ? cpu.get() : gpu.get(), weights, 0, resident_count, routing, !gpu_reduce);
        std::unique_ptr<Merge> merge;
        if (!self_test || gpu_reduce) {
            merge = std::make_unique<Merge>(self_test ? cpu.get() : reconciliation.get(),
                self_test ? cpu_device : gpu_device, weights.hidden, resident_count, routing, gpu_reduce, self_test);
        }
        json checks = json::array();
        std::vector<std::vector<float>> inputs, references;
        for (int sample = 0; sample < samples; ++sample) {
            std::vector<float> input(weights.hidden);
            const float magnitude = self_test ? (sample % 3 == 0 ? 0.2f : sample % 3 == 1 ? 50.0f : 500.0f) : 0.2f;
            for (int i = 0; i < weights.hidden; ++i) {
                input[i] = magnitude * std::sin(float((i + 1) * (sample + 3)) * 0.017f);
            }
            reference.set_input(input);
            reference.compute(false);
            auto expected = reference.result();
            if (self_test) {
                const auto scalar = scalar_reference(weights, input, routing);
                require(compare(scalar, expected).at("passed").get<bool>(),
                        "Unsplit graph differs from independent scalar reference.");
                expected = scalar;
            }
            references.push_back(std::move(expected));
            inputs.push_back(std::move(input));
        }
        json modes = json::object();
        std::map<std::string, std::vector<double>> timings, host_times, dispatch_times, merge_times;
        std::map<std::string, double> worst_abs, worst_relative;
        for (int index = -1; index < repetitions; ++index) {
            std::vector<std::string> schedules{"serial"};
            if (!self_test) {
                schedules.push_back("concurrent");
                if (index >= 0 && index % 2) std::reverse(schedules.begin(), schedules.end());
            }
            for (const auto & mode : schedules) {
                const int sample = index < 0 ? 0 : index % samples;
                const auto started = Clock::now();
                if (resident) {
                    resident->set_input(inputs[sample]);
                    resident->compute(!self_test && mode == "concurrent");
                }
                const auto submitted = Clock::now();
                if (host) {
                    host->set_input(inputs[sample]);
                    host->compute(false);
                }
                const auto host_done = Clock::now();
                std::vector<float> actual;
                if (self_test && !gpu_reduce) {
                    actual.assign(weights.hidden, 0);
                    for (Branch * branch : {host.get(), resident.get()}) {
                        if (!branch) continue;
                        const auto values = branch->result();
                        for (int i = 0; i < weights.hidden; ++i) actual[i] += values[i];
                    }
                } else {
                    merge->compute(host.get(), resident.get());
                }
                const auto completed = Clock::now();
                if (index >= 0) {
                    timings[mode].push_back(std::chrono::duration<double, std::milli>(completed - started).count());
                    dispatch_times[mode].push_back(std::chrono::duration<double, std::milli>(submitted - started).count());
                    host_times[mode].push_back(std::chrono::duration<double, std::milli>(host_done - submitted).count());
                    merge_times[mode].push_back(std::chrono::duration<double, std::milli>(completed - host_done).count());
                }
                if (!self_test || gpu_reduce) actual = merge->result();
                auto check = compare(references[sample], actual);
                check["mode"] = mode;
                check["sample"] = sample;
                passed = passed && check.at("passed").get<bool>();
                worst_abs[mode] = std::max(worst_abs[mode], check.at("max_abs_error").get<double>());
                worst_relative[mode] = std::max(worst_relative[mode], check.at("relative_l2_error").get<double>());
                if (index >= 0 && index < samples) checks.push_back(check);
            }
        }
        for (const auto & [mode, values] : timings) {
            modes[mode] = {
                {"median_ms", median(values)}, {"min_ms", *std::min_element(values.begin(), values.end())},
                {"samples_ms", values}, {"worst_max_abs_error", worst_abs[mode]},
                {"worst_relative_l2_error", worst_relative[mode]},
                {"median_gpu_input_dispatch_ms", median(dispatch_times[mode])},
                {"median_cpu_input_compute_ms", median(host_times[mode])},
                {"median_reconciliation_and_dependency_wait_ms", median(merge_times[mode])},
            };
        }
        report["partitions"].push_back({
            {"gpu_experts", self_test ? 0 : resident_count}, {"cpu_experts", self_test ? 4 : 4 - resident_count},
            {"split_index", resident_count}, {"modes", modes}, {"correctness", checks},
            {"cpu_partial_output_bytes", !self_test && host ? ggml_nbytes(host->output) : 0},
            {"pinned_staging_bytes", !self_test ? ggml_nbytes(merge->cpu) : 0},
        });
    }
    report["all_checks_passed"] = passed;
    if (!destination.empty()) {
        if (!destination.parent_path().empty()) fs::create_directories(destination.parent_path());
        const fs::path partial = destination.string() + ".part";
        std::unique_ptr<FILE, CloseFile> file(std::fopen(partial.c_str(), "wx"));
        require(file != nullptr, "Cannot exclusively create result.");
        const std::string serialized = report.dump(2) + '\n';
        require(std::fwrite(serialized.data(), 1, serialized.size(), file.get()) == serialized.size(),
                "Failed to write complete result.");
        require(std::fclose(file.release()) == 0, "Failed to close result.");
        fs::create_hard_link(partial, destination);
        fs::remove(partial);
        std::cout << destination.string() << '\n';
    } else {
        std::cout << report.dump(2) << '\n';
    }
    return passed ? 0 : 1;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & exc) {
        std::cerr << "vrambridge-hybrid-layer: " << exc.what() << '\n';
        return 2;
    }
}
