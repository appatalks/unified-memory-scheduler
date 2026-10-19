#include "build-info.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <nlohmann/json.hpp>
#ifdef VRAMBRIDGE_COMPATIBLE_DOWN
#include "compatible_down.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <set>
#include <sstream>
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

static Context context(size_t bytes = 1024 * 1024) {
    ggml_init_params params{bytes, nullptr, true};
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

struct RoutedInput {
    int rows = 0;
    int hidden = 0;
    int layer = 0;
    std::vector<float> input;
    std::vector<float> probabilities;
    std::vector<float> expected;
    std::vector<int> experts;
    std::vector<int32_t> ids;
    json metadata;
};

static RoutedInput read_routed_input(const fs::path & path, int index, bool self_test, const fs::path & model) {
    require(fs::is_regular_file(path) && fs::file_size(path) <= 128 * 1024 * 1024,
            "Captured replay input must be a regular file of at most 128 MiB.");
    std::ifstream stream(path);
    const auto capture = json::parse(stream);
    require(capture.at("schema_version") == 1 && capture.at("kind") ==
            (self_test ? "synthetic_routed_fixture" : "actual_prefill_ffn_capture"),
            "Captured replay kind/version mismatch.");
    const auto & records = capture.at("records");
    require(records.is_array() && !records.empty() && records.size() <= 9 && index >= 0 &&
            static_cast<size_t>(index) < records.size(), "Captured replay record index/count invalid.");
    if (!self_test) {
        require(capture.at("observer_full_logit_checks_passed") == true,
                "Actual routed replay requires a capture with passing full-logit observer controls.");
        require(capture.at("contains_sensitive_activations") == true &&
                capture.at("model") == model.filename().string() &&
                capture.at("backend_commit") == llama_commit(), "Captured replay model/backend mismatch.");
    }
    const auto & record = records.at(index);
    require(record.at("request").is_string() && !record.at("request").get<std::string>().empty() &&
            record.at("request").get<std::string>().size() <= 128 &&
            (record.at("category") == "coding" || record.at("category") == "prose" ||
             record.at("category") == "reasoning"), "Invalid captured request/category metadata.");
    require(self_test || (record.at("observed_empty_batches").is_number_integer() &&
                         record.at("observed_empty_batches") >= 0),
            "Invalid captured pruning metadata.");
    for (const std::string key : {"rows", "hidden", "layer", "native_rows", "row_offset"}) {
        require(record.at(key).is_number_integer(), "Captured replay shape metadata must be integer.");
    }
    RoutedInput result;
    require(record.at("rows") >= 1 && record.at("rows") <= 64 &&
            record.at("hidden") == (self_test ? 64 : 2880) && record.at("layer") >= 0 &&
            record.at("layer") < 36 && record.at("native_rows") >= record.at("rows") &&
            record.at("native_rows") <= 2048 && record.at("row_offset") >= 0 &&
            record.at("row_offset") <= record.at("native_rows").get<int>() - record.at("rows").get<int>(),
            "Captured replay shape outside bounded layout.");
    result.rows = record.at("rows");
    result.hidden = record.at("hidden");
    result.layer = record.at("layer");
    result.input = record.at("input").get<std::vector<float>>();
    result.probabilities = record.at("probabilities").get<std::vector<float>>();
    const auto & ids = record.at("expert_ids");
    require(result.input.size() == size_t(result.hidden * result.rows) &&
            result.probabilities.size() == size_t(4 * result.rows) && ids.is_array() &&
            ids.size() == size_t(4 * result.rows), "Captured replay payload size mismatch.");
    if (!self_test) {
        result.expected = record.at("output").get<std::vector<float>>();
        require(result.expected.size() == result.input.size(), "Captured replay output size mismatch.");
    } else {
        require(!record.contains("output"), "Synthetic routed fixtures use an independent scalar reference.");
    }
    std::set<int> unique;
    for (int row = 0; row < result.rows; ++row) {
        std::set<int> selected;
        double total = 0;
        for (int rank = 0; rank < 4; ++rank) {
            const size_t offset = 4 * row + rank;
            require(ids.at(offset).is_number_integer() && ids.at(offset) >= 0 && ids.at(offset) < 128,
                    "Captured expert IDs must be integers within the original expert bank.");
            const int id = ids.at(offset);
            require(id >= 0 && id < 128 && selected.insert(id).second,
                    "Captured expert ID out of range or duplicated in a row.");
            require(std::isfinite(result.probabilities[offset]) && result.probabilities[offset] >= 0,
                    "Captured routing probability is invalid.");
            total += result.probabilities[offset];
            unique.insert(id);
        }
        require(total > 0, "Captured routing row has no positive probability.");
    }
    require(std::all_of(result.input.begin(), result.input.end(), [](float x) { return std::isfinite(x); }) &&
            std::all_of(result.expected.begin(), result.expected.end(), [](float x) { return std::isfinite(x); }),
            "Captured replay contains nonfinite activations.");
    result.experts.assign(unique.begin(), unique.end());
    for (const auto & id : ids) {
        result.ids.push_back(static_cast<int32_t>(
            std::lower_bound(result.experts.begin(), result.experts.end(), id.get<int>()) - result.experts.begin()));
    }
    result.metadata = {{"request", record.at("request")}, {"category", record.at("category")},
        {"native_rows", record.at("native_rows")}, {"rows", result.rows}, {"row_offset", record.at("row_offset")},
        {"layer", result.layer},
        {"observed_empty_batches", self_test ? json(0) : record.at("observed_empty_batches")}};
    if (!self_test) {
        const auto & controls = capture.at("execution_controls");
        const auto & fusion = controls.at("cuda_disable_fusion");
        require((fusion.is_null() || fusion == "0" || fusion == "1") &&
                controls.at("cpu_reference").is_boolean(), "Unsupported captured execution controls.");
        result.metadata["capture_execution_controls"] = {
            {"cuda_disable_fusion", fusion}, {"cpu_reference", controls.at("cpu_reference")}};
    }
    return result;
}

static Weights read_weights(const fs::path & first, int layer, const std::vector<int> & ids) {
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
    result.ids = ids;
    result.selected = static_cast<int>(ids.size());
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

static Weights synthetic_weights(int experts = 4) {
    Weights result;
    result.hidden = 64;
    result.intermediate = 96;
    result.selected = experts;
    result.ids.resize(experts);
    std::iota(result.ids.begin(), result.ids.end(), 0);
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

enum class BranchStage { Full, CpuActivation, Down };

struct Branch {
    ggml_backend_t backend;
    Context ctx;
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    std::map<std::string, ggml_tensor *> tensors;
    std::vector<std::pair<std::string, ggml_tensor *>> observations;
    int count;

    Branch(ggml_backend_t device, const Weights & weights, int begin, int end,
           const std::vector<float> & routing, bool reduce = true, int rows = 1, int row_limit = 0,
           bool capture_operators = false, BranchStage stage = BranchStage::Full,
           const RoutedInput * selection = nullptr) :
            backend(device), ctx(context(row_limit ? 8 * 1024 * 1024 : 1024 * 1024)), count(end - begin) {
        require(begin >= 0 && end <= weights.selected && count > 0, "Invalid expert partition.");
        require(!selection || (begin == 0 && end == weights.selected && rows == selection->rows),
                "Routed replay requires the complete captured expert bank.");
        require(selection || routing.size() == size_t(weights.selected), "Routing probability count mismatch.");
        const int active = selection ? 4 : count;
        for (const std::string part : {"gate", "up", "down"}) {
            if ((stage == BranchStage::CpuActivation && part == "down") ||
                (stage == BranchStage::Down && part != "down")) continue;
            tensors[part + ".weight"] = ggml_new_tensor_3d(ctx.get(), weights.type,
                part == "down" ? weights.intermediate : weights.hidden,
                part == "down" ? weights.hidden : weights.intermediate, count);
            tensors[part + ".bias"] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32,
                part == "down" ? weights.hidden : weights.intermediate, count);
        }
        input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32,
            stage == BranchStage::Down ? weights.intermediate : weights.hidden,
            stage == BranchStage::Down ? active : 1, rows);
        ggml_set_input(input);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, active, rows);
        auto * probabilities = stage == BranchStage::CpuActivation ? nullptr :
            ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, active, rows);
        const int chunk_size = row_limit ? std::min(rows, row_limit) : rows;
        const int chunks = (rows + chunk_size - 1) / chunk_size;
        const auto observe = [&](const std::string & name, ggml_tensor * tensor) {
            if (capture_operators) {
                ggml_set_output(tensor);
                observations.emplace_back(name, tensor);
            }
        };
        for (int offset = 0; offset < rows; offset += chunk_size) {
            const int current_rows = std::min(chunk_size, rows - offset);
            auto * chunk_input = chunks == 1 ? input : ggml_view_3d(ctx.get(), input,
                input->ne[0], input->ne[1], current_rows, input->nb[1], input->nb[2], offset * input->nb[2]);
            auto * chunk_ids = chunks == 1 ? ids : ggml_view_2d(ctx.get(), ids,
                active, current_rows, ids->nb[1], offset * ids->nb[1]);
            auto * chunk_probabilities = !probabilities || chunks == 1 ? probabilities : ggml_view_3d(ctx.get(), probabilities,
                1, active, current_rows, probabilities->nb[1], probabilities->nb[2], offset * probabilities->nb[2]);
            auto * activated = chunk_input;
            if (stage != BranchStage::Down) {
                auto * gate = ggml_mul_mat_id(ctx.get(), tensors.at("gate.weight"), chunk_input, chunk_ids);
                observe("gate_projection", gate);
                gate = ggml_add_id(ctx.get(), gate, tensors.at("gate.bias"), chunk_ids);
                observe("gate_bias", gate);
                auto * up = ggml_mul_mat_id(ctx.get(), tensors.at("up.weight"), chunk_input, chunk_ids);
                observe("up_projection", up);
                up = ggml_add_id(ctx.get(), up, tensors.at("up.bias"), chunk_ids);
                observe("up_bias", up);
                activated = ggml_swiglu_oai(ctx.get(), gate, up, 1.702f, 7.0f);
                observe("swiglu", activated);
            }
            ggml_tensor * chunk_output;
            if (stage == BranchStage::CpuActivation) {
                chunk_output = weights.type == GGML_TYPE_MXFP4
                    ? ggml_cast(ctx.get(), ggml_cast(ctx.get(), activated, GGML_TYPE_Q8_0), GGML_TYPE_F32)
                    : activated;
            } else {
                auto * down = ggml_mul_mat_id(ctx.get(), tensors.at("down.weight"), activated, chunk_ids);
                if (stage == BranchStage::Down && weights.type == GGML_TYPE_MXFP4) {
                    require(ggml_prec_set_src(down, GGML_PREC_Q8, 1), "Cannot request Q8 GPU-down inputs.");
                }
                observe("down_projection", down);
                down = ggml_add_id(ctx.get(), down, tensors.at("down.bias"), chunk_ids);
                observe("down_bias", down);
                if (reduce) {
                    auto * weighted = ggml_mul(ctx.get(), down, chunk_probabilities);
                    observe("weighting", weighted);
                    auto * transposed = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), weighted));
                    chunk_output = ggml_reshape_3d(ctx.get(), ggml_sum_rows(ctx.get(), transposed),
                        weights.hidden, 1, current_rows);
                } else {
                    chunk_output = ggml_reshape_3d(ctx.get(), down, weights.hidden, active, current_rows);
                }
            }
            observe("reduction", chunk_output);
            output = output ? ggml_concat(ctx.get(), output, chunk_output, 2) : chunk_output;
        }
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 128 + 32 * (chunks - 1), false);
        ggml_build_forward_expand(graph, output);
        for (int node = 0; node < ggml_graph_n_nodes(graph); ++node) {
            require(ggml_backend_supports_op(backend, ggml_graph_node(graph, node)),
                    "Backend does not support an expert graph operation.");
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(buffer != nullptr, "Cannot allocate expert branch.");
        upload_weights(weights, begin);
        std::vector<int32_t> local_ids(active * rows);
        std::vector<float> probabilities_data(active * rows);
        if (selection) {
            local_ids = selection->ids;
            probabilities_data = selection->probabilities;
        } else {
            for (int row = 0; row < rows; ++row) {
                std::iota(local_ids.begin() + row * active, local_ids.begin() + (row + 1) * active, 0);
                std::copy_n(routing.begin() + begin, active, probabilities_data.begin() + row * active);
            }
        }
        ggml_backend_tensor_set(ids, local_ids.data(), 0, local_ids.size() * sizeof(int32_t));
        if (probabilities) {
            ggml_backend_tensor_set(probabilities, probabilities_data.data(), 0, probabilities_data.size() * sizeof(float));
        }
    }

    void upload_weights(const Weights & weights, int begin) {
        for (const auto & [key, tensor] : tensors) {
            const auto & source = weights.tensors.at(key);
            const size_t stride = source.size() / weights.selected;
            require(ggml_nbytes(tensor) == stride * count, "Gathered expert slice size mismatch.");
            ggml_backend_tensor_set(tensor, source.data() + stride * begin, 0, stride * count);
        }
    }

    void set_input(const std::vector<float> & values) {
        require(values.size() * sizeof(float) == ggml_nbytes(input), "Activation size mismatch.");
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    }

    void copy_input(const ggml_tensor * source) {
        require(source->type == input->type && ggml_are_same_shape(source, input) &&
                ggml_are_same_stride(source, input), "CPU activation/GPU-down input layout mismatch.");
        ggml_backend_tensor_copy(source, input);
    }

    size_t weight_bytes() const {
        size_t bytes = 0;
        for (const auto & entry : tensors) bytes += ggml_nbytes(entry.second);
        return bytes;
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

    std::map<std::string, std::vector<float>> operator_results() {
        std::map<std::string, std::vector<float>> result;
        for (const auto & [name, tensor] : observations) {
            require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                    ggml_nelements(tensor) <= 1000000, "Unsupported or unbounded operator observation.");
            std::vector<float> values(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
            auto & combined = result[name];
            require(combined.size() + values.size() <= 1000000, "Operator observation exceeds its capture bound.");
            combined.insert(combined.end(), values.begin(), values.end());
        }
        return result;
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
          const std::vector<float> & routing, bool gpu_reduce, bool self_test, int rows = 1) : backend(stream) {
        cpu = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hidden,
            gpu_reduce ? std::max(1, 4 - resident_count) : 1, rows);
        gpu = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, hidden,
            gpu_reduce ? std::max(1, resident_count) : 1, rows);
        ggml_set_input(cpu);
        ggml_set_input(gpu);
        output = nullptr;
        ggml_tensor * probabilities = nullptr;
        if (gpu_reduce) {
            auto * combined = resident_count == 0 ? cpu : resident_count == 4 ? gpu :
                ggml_concat(ctx.get(), gpu, cpu, 1);
            probabilities = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 4, rows);
            auto * weighted = ggml_mul(ctx.get(), combined, probabilities);
            auto * transposed = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), weighted));
            output = ggml_reshape_2d(ctx.get(), ggml_sum_rows(ctx.get(), transposed), hidden, rows);
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
            std::vector<float> values(routing.size() * rows);
            for (int row = 0; row < rows; ++row) {
                std::copy(routing.begin(), routing.end(), values.begin() + row * routing.size());
            }
            ggml_backend_tensor_set(probabilities, values.data(), 0, values.size() * sizeof(float));
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
            require(host->output->type == cpu->type && ggml_are_same_shape(host->output, cpu) &&
                    ggml_are_same_stride(host->output, cpu), "CPU partial output layout mismatch.");
            void * data = ggml_backend_buffer_get_base(staging.get());
            ggml_backend_tensor_get(host->output, data, 0, ggml_nbytes(cpu));
            ggml_backend_tensor_set_async(backend, cpu, data, 0, ggml_nbytes(cpu));
        } else {
            ggml_backend_tensor_memset(cpu, 0, 0, ggml_nbytes(cpu));
        }
        if (resident) {
            require(ggml_nbytes(resident->output) == ggml_nbytes(gpu), "GPU partial output shape mismatch.");
            require(resident->output->type == gpu->type && ggml_are_same_shape(resident->output, gpu) &&
                    ggml_are_same_stride(resident->output, gpu), "GPU partial output layout mismatch.");
            ggml_backend_tensor_copy_async(resident->backend, backend, resident->output, gpu);
        } else {
            ggml_backend_tensor_memset(gpu, 0, 0, ggml_nbytes(gpu));
        }
        require(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS, "GPU merge failed.");
        ggml_backend_synchronize(backend);
    }

    std::vector<float> result() {
        std::vector<float> values(ggml_nelements(output));
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
    require(!input.empty() && input.size() % weights.hidden == 0, "Scalar reference activation shape mismatch.");
    if (input.size() != static_cast<size_t>(weights.hidden)) {
        std::vector<float> output;
        output.reserve(input.size());
        for (size_t offset = 0; offset < input.size(); offset += weights.hidden) {
            const auto row = scalar_reference(weights,
                std::vector<float>(input.begin() + offset, input.begin() + offset + weights.hidden), routing);
            output.insert(output.end(), row.begin(), row.end());
        }
        return output;
    }
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

static std::vector<float> activation_input(const Weights & weights, int rows, int sample, bool self_test,
                                         const std::string & pattern) {
    std::vector<float> input(weights.hidden * rows);
    const float magnitude = self_test ? (sample % 3 == 0 ? 0.2f : sample % 3 == 1 ? 50.0f : 500.0f) : 0.2f;
    for (size_t i = 0; i < input.size(); ++i) {
        if (pattern == "zero") input[i] = 0;
        else if (pattern == "q8-ties") {
            input[i] = i % 32 == 0 ? 127.0f : i % 32 == 1 ? -127.0f :
                (i % 2 ? -1.0f : 1.0f) * (0.5f + float((i + 7 * sample) % 31));
        } else input[i] = magnitude * std::sin(float((i + 1) * (sample + 3)) * 0.017f);
    }
    return input;
}

enum class ReplayKind { Matmul, Bias, Swiglu, Weighting, Reduction, CpuQ8Roundtrip };

static std::string output_fingerprint(const std::vector<float> & values) {
    uint64_t hash = 14695981039346656037ULL;
    for (float value : values) {
        uint8_t bytes[sizeof(value)];
        std::memcpy(bytes, &value, sizeof(value));
        for (uint8_t byte : bytes) hash = (hash ^ byte) * 1099511628211ULL;
    }
    return std::to_string(hash);
}

static std::vector<float> replay_operator(ggml_backend_t backend, ReplayKind kind,
                                         const std::vector<float> & values, int width, int active, int rows,
                                         ggml_tensor * weights = nullptr, const std::vector<float> & second = {},
                                         std::vector<uint8_t> * quantized = nullptr) {
    auto ctx = context();
    auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, active, rows);
    ggml_set_input(input);
    ggml_tensor * other = nullptr, * ids = nullptr, * output = nullptr, * packed = nullptr;
    require(!quantized || kind == ReplayKind::CpuQ8Roundtrip, "Packed replay capture requires CPU Q8.");
    if (kind == ReplayKind::Matmul || kind == ReplayKind::Bias) {
        require(weights != nullptr, "Projection/bias replay requires original weights.");
        const int experts = static_cast<int>(weights->ne[kind == ReplayKind::Bias ? 1 : 2]);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, experts, rows);
        if (kind == ReplayKind::Matmul) {
            output = ggml_mul_mat_id(ctx.get(), weights, input, ids);
            require(ggml_prec_set_src(output, GGML_PREC_Q8, 1), "Cannot request Q8 projection replay.");
        } else {
            output = ggml_add_id(ctx.get(), input, weights, ids);
        }
    } else if (kind == ReplayKind::Swiglu || kind == ReplayKind::Weighting) {
        other = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32,
            kind == ReplayKind::Weighting ? 1 : width, active, rows);
        ggml_set_input(other);
        output = kind == ReplayKind::Swiglu ? ggml_swiglu_oai(ctx.get(), input, other, 1.702f, 7.0f)
                                            : ggml_mul(ctx.get(), input, other);
    } else if (kind == ReplayKind::Reduction) {
        auto * transposed = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), input));
        output = ggml_reshape_3d(ctx.get(), ggml_sum_rows(ctx.get(), transposed), width, 1, rows);
    } else {
        require(width % 32 == 0, "CPU Q8 replay requires block-aligned rows.");
        packed = ggml_cast(ctx.get(), input, GGML_TYPE_Q8_0);
        ggml_set_output(packed);
        output = ggml_cast(ctx.get(), packed, GGML_TYPE_F32);
    }
    require(values.size() * sizeof(float) == ggml_nbytes(input), "Replay input shape mismatch.");
    require(!other || second.size() * sizeof(float) == ggml_nbytes(other), "Replay secondary input shape mismatch.");
    require(ggml_nelements(output) <= 1000000, "Replay output exceeds capture bound.");
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    for (int node = 0; node < ggml_graph_n_nodes(graph); ++node) {
        require(ggml_backend_supports_op(backend, ggml_graph_node(graph, node)), "Replay operation unsupported.");
    }
    Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "Cannot allocate operator replay.");
    ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
    if (other) ggml_backend_tensor_set(other, second.data(), 0, ggml_nbytes(other));
    if (ids) {
        std::vector<int32_t> selected(ggml_nelements(ids));
        for (int row = 0; row < rows; ++row) {
            std::iota(selected.begin() + row * ids->ne[0], selected.begin() + (row + 1) * ids->ne[0], 0);
        }
        ggml_backend_tensor_set(ids, selected.data(), 0, ggml_nbytes(ids));
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Operator replay failed.");
    std::vector<float> result(ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
    if (quantized) {
        quantized->resize(ggml_nbytes(packed));
        ggml_backend_tensor_get(packed, quantized->data(), 0, quantized->size());
    }
    return result;
}

struct Q8Snapshot {
    std::vector<float> effective;
    std::vector<uint8_t> packed;
};

static Q8Snapshot snapshot_q8(ggml_backend_t cpu, const std::vector<float> & values,
                             int width, int active, int rows) {
    Q8Snapshot snapshot;
    snapshot.effective = replay_operator(cpu, ReplayKind::CpuQ8Roundtrip, values, width, active, rows,
                                        nullptr, {}, &snapshot.packed);
    return snapshot;
}

static json compare_q8(const Q8Snapshot & expected, const Q8Snapshot & actual) {
    const size_t block_bytes = ggml_type_size(GGML_TYPE_Q8_0);
    const size_t codes_per_block = ggml_blck_size(GGML_TYPE_Q8_0);
    require(block_bytes == sizeof(ggml_fp16_t) + codes_per_block &&
            expected.packed.size() == actual.packed.size() && expected.packed.size() % block_bytes == 0,
            "Unexpected CPU Q8 block layout.");
    size_t scales = 0, codes = 0, code_blocks = 0;
    for (size_t start = 0; start < expected.packed.size(); start += block_bytes) {
        scales += std::memcmp(expected.packed.data() + start, actual.packed.data() + start, sizeof(ggml_fp16_t)) != 0;
        size_t changed = 0;
        for (size_t index = sizeof(ggml_fp16_t); index < block_bytes; ++index) {
            changed += expected.packed[start + index] != actual.packed[start + index];
        }
        codes += changed;
        code_blocks += changed > 0;
    }
    return {{"blocks", expected.packed.size() / block_bytes}, {"different_scale_blocks", scales},
            {"different_code_blocks", code_blocks}, {"different_code_values", codes},
            {"effective_value_error", compare(expected.effective, actual.effective)}};
}

static json check_cpu_activation_split(ggml_backend_t cpu, ggml_backend_t gpu, const Weights & weights,
                                      const std::vector<float> & routing, Branch & reference, int rows,
                                      int gpu_row_limit, int cpu_row_limit, int samples, bool self_test,
                                      const std::string & pattern, const RoutedInput * selection = nullptr) {
    Branch cpu_observed(cpu, weights, 0, weights.selected, routing, true, rows, 0, true, BranchStage::Full, selection);
    Branch activation_plain(cpu, weights, 0, weights.selected, routing, false, rows, cpu_row_limit, false, BranchStage::CpuActivation, selection);
    Branch activation_observed(cpu, weights, 0, weights.selected, routing, false, rows, cpu_row_limit, true, BranchStage::CpuActivation, selection);
    Branch gpu_plain(gpu, weights, 0, weights.selected, routing, true, rows, gpu_row_limit, false, BranchStage::Down, selection);
    Branch gpu_observed(gpu, weights, 0, weights.selected, routing, true, rows, gpu_row_limit, true, BranchStage::Down, selection);
    json result = {{"raw_activations_recorded", false}, {"performance_measurement", false},
                   {"candidate_backend", self_test ? "cpu_self_test" : "gpu"}, {"samples", json::array()}};
    bool passed = true;
    for (int sample = 0; sample < samples; ++sample) {
        const auto input = selection ? selection->input : activation_input(weights, rows, sample, self_test, pattern);
        for (Branch * branch : {&reference, &cpu_observed, &activation_plain, &activation_observed}) {
            branch->set_input(input);
            branch->compute(false);
        }
        gpu_plain.copy_input(activation_plain.output);
        gpu_plain.compute(false);
        gpu_observed.copy_input(activation_observed.output);
        gpu_observed.compute(false);
        const auto cpu_observer = compare(reference.result(), cpu_observed.result());
        const auto activation_observer = compare(activation_plain.result(), activation_observed.result());
        const auto gpu_observer = compare(gpu_plain.result(), gpu_observed.result());
        const auto final = compare(reference.result(), gpu_plain.result());
        const auto cpu_values = cpu_observed.operator_results();
        const auto activation_values = activation_observed.operator_results();
        const auto gpu_values = gpu_observed.operator_results();
        json checks = json::array(), first_nonzero = nullptr, first_failing = nullptr;
        for (const std::string name : {"gate_projection", "gate_bias", "up_projection", "up_bias",
                                      "swiglu", "down_projection", "down_bias", "weighting", "reduction"}) {
            const bool cpu_stage = name == "gate_projection" || name == "gate_bias" ||
                name == "up_projection" || name == "up_bias" || name == "swiglu";
            auto check = compare(cpu_values.at(name), cpu_stage ? activation_values.at(name) : gpu_values.at(name));
            check["operator"] = name;
            check["backend"] = cpu_stage ? "cpu" : "gpu_down_stage";
            if (first_nonzero.is_null() && check.at("max_abs_error").get<double>() > 0) first_nonzero = name;
            if (first_failing.is_null() && !check.at("passed").get<bool>()) first_failing = name;
            passed = passed && check.at("passed").get<bool>();
            checks.push_back(check);
        }
        passed = passed && cpu_observer.at("passed").get<bool>() && activation_observer.at("passed").get<bool>() &&
            gpu_observer.at("passed").get<bool>() && final.at("passed").get<bool>();
        result["samples"].push_back({{"sample", sample}, {"cpu_observer_control", cpu_observer},
            {"cpu_activation_observer_control", activation_observer}, {"gpu_observer_control", gpu_observer},
            {"uninstrumented_cpu_gpu", final}, {"cpu_output_fnv1a64", output_fingerprint(reference.result())},
            {"gpu_output_fnv1a64", output_fingerprint(gpu_plain.result())},
            {"first_nonzero_operator", first_nonzero}, {"first_failing_operator", first_failing},
            {"chained_operator_checks", checks}});
    }
    result["all_checks_passed"] = passed;
    return result;
}

static json check_operators(ggml_backend_t cpu, ggml_backend_t gpu, const Weights & weights,
                           const std::vector<float> & routing, Branch & reference,
                           int rows, int gpu_row_limit, int samples, bool self_test, const std::string & pattern) {
    Branch cpu_observed(cpu, weights, 0, 4, routing, true, rows, 0, true);
    Branch gpu_plain(gpu, weights, 0, 4, routing, true, rows, gpu_row_limit);
    Branch gpu_observed(gpu, weights, 0, 4, routing, true, rows, gpu_row_limit, true);
    json result = {{"raw_activations_recorded", false}, {"performance_measurement", false},
                   {"candidate_backend", self_test ? "cpu_self_test" : "gpu"},
                   {"samples", json::array()}};
    bool passed = true;
    for (int sample = 0; sample < samples; ++sample) {
        const auto input = activation_input(weights, rows, sample, self_test, pattern);
        for (auto * branch : {&reference, &cpu_observed, &gpu_plain, &gpu_observed}) {
            branch->set_input(input);
            branch->compute(false);
        }
        const auto cpu_observer = compare(reference.result(), cpu_observed.result());
        const auto gpu_observer = compare(gpu_plain.result(), gpu_observed.result());
        const auto cpu_values = cpu_observed.operator_results();
        const auto gpu_values = gpu_observed.operator_results();
        const auto input_q8 = snapshot_q8(cpu, input, weights.hidden, 1, rows);
        const auto input_requantized = snapshot_q8(cpu, input_q8.effective, weights.hidden, 1, rows);
        const auto quantizer_self_check = compare_q8(input_q8, input_requantized);
        require(quantizer_self_check.at("different_scale_blocks") == 0 &&
                quantizer_self_check.at("different_code_values") == 0, "CPU Q8 roundtrip is not idempotent.");
        json independent = json::array();
        const auto record = [&](const std::string & name, const std::string & control,
                                const std::vector<float> & actual) {
            auto check = compare(cpu_values.at(name), actual);
            check["operator"] = name;
            check["input_control"] = control;
            passed = passed && check.at("passed").get<bool>();
            independent.push_back(check);
        };
        for (const std::string part : {"gate", "up"}) {
            record(part + "_projection", "identical_original_input_q8_requested",
                replay_operator(gpu, ReplayKind::Matmul, input, weights.hidden, 1, rows,
                                gpu_observed.tensors.at(part + ".weight")));
            record(part + "_bias", "identical_cpu_projection",
                replay_operator(gpu, ReplayKind::Bias, cpu_values.at(part + "_projection"),
                                weights.intermediate, 4, rows, gpu_observed.tensors.at(part + ".bias")));
        }
        const auto same_input_activation = replay_operator(gpu, ReplayKind::Swiglu,
            cpu_values.at("gate_bias"), weights.intermediate, 4, rows, nullptr, cpu_values.at("up_bias"));
        record("swiglu", "identical_cpu_biased_gate_up", same_input_activation);
        record("down_projection", "identical_cpu_swiglu_q8_requested",
            replay_operator(gpu, ReplayKind::Matmul, cpu_values.at("swiglu"), weights.intermediate, 4, rows,
                            gpu_observed.tensors.at("down.weight")));
        record("down_bias", "identical_cpu_down_projection",
            replay_operator(gpu, ReplayKind::Bias, cpu_values.at("down_projection"), weights.hidden, 4, rows,
                            gpu_observed.tensors.at("down.bias")));
        std::vector<float> probabilities;
        for (int row = 0; row < rows; ++row) probabilities.insert(probabilities.end(), routing.begin(), routing.end());
        record("weighting", "identical_cpu_biased_down",
            replay_operator(gpu, ReplayKind::Weighting, cpu_values.at("down_bias"), weights.hidden, 4, rows,
                            nullptr, probabilities));
        record("reduction", "identical_cpu_weighted_outputs",
            replay_operator(gpu, ReplayKind::Reduction, cpu_values.at("weighting"), weights.hidden, 4, rows));
        json propagation = nullptr;
        if (weights.type == GGML_TYPE_MXFP4) {
            const auto canonical_input = replay_operator(cpu, ReplayKind::CpuQ8Roundtrip,
                input, weights.hidden, 1, rows);
            for (const std::string part : {"gate", "up"}) {
                record(part + "_projection", "cpu_effective_q8_input_q8_requested",
                    replay_operator(gpu, ReplayKind::Matmul, canonical_input, weights.hidden, 1, rows,
                                    gpu_observed.tensors.at(part + ".weight")));
            }
            const auto cpu_activation = snapshot_q8(cpu, cpu_values.at("swiglu"), weights.intermediate, 4, rows);
            const auto gpu_chain_activation = snapshot_q8(cpu, gpu_values.at("swiglu"), weights.intermediate, 4, rows);
            const auto gpu_same_activation = snapshot_q8(cpu, same_input_activation, weights.intermediate, 4, rows);
            record("down_projection", "cpu_effective_q8_swiglu_q8_requested",
                replay_operator(gpu, ReplayKind::Matmul, cpu_activation.effective, weights.intermediate, 4, rows,
                                gpu_observed.tensors.at("down.weight")));
            const auto down_input_lesion = replay_operator(cpu, ReplayKind::Matmul, gpu_chain_activation.effective,
                weights.intermediate, 4, rows, cpu_observed.tensors.at("down.weight"));
            const auto activation_only_lesion = replay_operator(cpu, ReplayKind::Matmul, gpu_same_activation.effective,
                weights.intermediate, 4, rows, cpu_observed.tensors.at("down.weight"));
            propagation = {
                {"chain_vs_cpu", compare_q8(cpu_activation, gpu_chain_activation)},
                {"activation_only_vs_cpu", compare_q8(cpu_activation, gpu_same_activation)},
                {"cpu_down_with_chain_effective_input", compare(cpu_values.at("down_projection"), down_input_lesion)},
                {"cpu_down_with_same_input_activation", compare(cpu_values.at("down_projection"), activation_only_lesion)},
                {"gpu_down_vs_same_effective_cpu_input", compare(down_input_lesion, gpu_values.at("down_projection"))}};
        }
        json checks = json::array(), first_nonzero = nullptr, first_failing = nullptr;
        for (const std::string name : {"gate_projection", "gate_bias", "up_projection", "up_bias",
                                      "swiglu", "down_projection", "down_bias", "weighting", "reduction"}) {
            auto check = compare(cpu_values.at(name), gpu_values.at(name));
            check["operator"] = name;
            if (first_nonzero.is_null() && check.at("max_abs_error").get<double>() > 0) first_nonzero = name;
            if (first_failing.is_null() && !check.at("passed").get<bool>()) first_failing = name;
            passed = passed && check.at("passed").get<bool>();
            checks.push_back(check);
        }
        passed = passed && cpu_observer.at("passed").get<bool>() && gpu_observer.at("passed").get<bool>();
        result["samples"].push_back({{"sample", sample}, {"cpu_observer_control", cpu_observer},
            {"gpu_observer_control", gpu_observer}, {"uninstrumented_cpu_gpu", compare(reference.result(), gpu_plain.result())},
            {"cpu_output_fnv1a64", output_fingerprint(reference.result())},
            {"gpu_output_fnv1a64", output_fingerprint(gpu_plain.result())},
            {"first_nonzero_operator", first_nonzero}, {"first_failing_operator", first_failing},
            {"chained_operator_checks", checks}, {"identical_input_replays", independent},
            {"cpu_q8_roundtrip_self_check", quantizer_self_check}, {"effective_q8_propagation", propagation},
            {"cpu_effective_q8_replays_applicable", weights.type == GGML_TYPE_MXFP4}});
    }
    result["all_checks_passed"] = passed;
    return result;
}

static void write_report(const json & report, const fs::path & destination) {
    if (destination.empty()) {
        std::cout << report.dump(2) << '\n';
        return;
    }
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
}

static json replay_routed_input(ggml_backend_t cpu, ggml_backend_t gpu, const Weights & weights,
                               const RoutedInput & input, bool self_test) {
    const std::vector<float> unused_routing;
    Branch reference(cpu, weights, 0, weights.selected, unused_routing, true, input.rows,
                     0, false, BranchStage::Full, &input);
    Branch activation(cpu, weights, 0, weights.selected, unused_routing, false, input.rows,
                      0, false, BranchStage::CpuActivation, &input);
    Branch down(gpu, weights, 0, weights.selected, unused_routing, true, input.rows,
                0, false, BranchStage::Down, &input);
    reference.set_input(input.input);
    reference.compute(false);
    auto expected = input.expected;
    if (self_test) {
        for (int row = 0; row < input.rows; ++row) {
            std::vector<float> probabilities(weights.selected, 0);
            for (int rank = 0; rank < 4; ++rank) {
                probabilities[input.ids[4 * row + rank]] = input.probabilities[4 * row + rank];
            }
            const auto scalar = scalar_reference(weights, std::vector<float>(
                input.input.begin() + row * weights.hidden, input.input.begin() + (row + 1) * weights.hidden),
                probabilities);
            expected.insert(expected.end(), scalar.begin(), scalar.end());
        }
    }
    activation.set_input(input.input);
    activation.compute(false);
    down.copy_input(activation.output);
    down.compute(false);
    const auto cpu_reference = compare(expected, reference.result());
    const auto candidate_reference = compare(expected, down.result());
    const auto candidate_cpu = compare(reference.result(), down.result());
    const auto diagnostics = check_cpu_activation_split(cpu, gpu, weights, unused_routing, reference,
        input.rows, 0, 0, 1, self_test, "sine", &input);
    std::set<std::vector<int32_t>> routes;
    std::vector<int> histogram(weights.selected, 0);
    for (int row = 0; row < input.rows; ++row) {
        routes.emplace(input.ids.begin() + 4 * row, input.ids.begin() + 4 * (row + 1));
        for (int rank = 0; rank < 4; ++rank) ++histogram[input.ids[4 * row + rank]];
    }
    return {{"schema_version", 1}, {"kind", "routed_ffn_replay"}, {"backend_commit", llama_commit()},
        {"probe_source_sha256", VRAMBRIDGE_HYBRID_SOURCE_SHA256}, {"capture", input.metadata},
        {"activation_source", self_test ? "synthetic_routed_fixture" : "actual_prefill_capture"},
        {"independent_scalar_reference", self_test}, {"raw_activations_recorded", false},
        {"raw_routing_recorded", false}, {"timings_qualified", false}, {"performance_measurement", false},
        {"full_model_acceleration_claim", false}, {"general_model_quality_established", false},
        {"expert_bank_size", weights.selected}, {"active_experts_per_row", 4},
        {"distinct_ordered_routes", routes.size()}, {"expert_row_counts", histogram},
        {"routing_probabilities_modified", false}, {"cpu_vs_reference", cpu_reference},
        {"candidate_vs_reference", candidate_reference}, {"candidate_vs_cpu", candidate_cpu},
        {"operator_diagnostics", diagnostics},
        {"all_checks_passed", cpu_reference.at("passed").get<bool>() && candidate_reference.at("passed").get<bool>() &&
            candidate_cpu.at("passed").get<bool>() && diagnostics.at("all_checks_passed").get<bool>()}};
}

#ifdef VRAMBRIDGE_COMPATIBLE_DOWN
static json compare_bits(const std::vector<float> & expected, const std::vector<float> & actual) {
    auto result = compare(expected, actual);
    size_t different = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        different += std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0;
    }
    result["different_bits_values"] = different;
    result["bitwise_equal"] = different == 0;
    return result;
}

static Weights compatible_fixture_weights() {
    Weights weights;
    weights.hidden = 64;
    weights.intermediate = 128;
    weights.type = GGML_TYPE_MXFP4;
    constexpr uint8_t exponents[]{0, 1, 2, 120, 126, 127, 128, 130};
    auto & down = weights.tensors["down.weight"];
    down.resize(size_t(weights.selected) * weights.hidden * (weights.intermediate / 32) * 17);
    for (size_t block = 0; block < down.size() / 17; ++block) {
        down[block * 17] = exponents[(block / (weights.intermediate / 32)) % 8];
        for (int j = 0; j < 16; ++j) {
            down[block * 17 + 1 + j] = uint8_t(((block + j) % 16) | (((block * 7 + j * 3) % 16) << 4));
        }
    }
    return weights;
}

static json check_compatible_down(ggml_backend_t cpu, const Weights & weights, int rows,
                                 int repetitions, int samples, const std::string & pattern,
                                 bool upload_weights, bool fixture, bool gpu_first, bool packed_dot,
                                 const RoutedInput * selection) {
    require(weights.type == GGML_TYPE_MXFP4 && weights.intermediate % 64 == 0,
            "Compatible-down requires MXFP4 weights and paired 32-value blocks.");
    const int active = selection ? 4 : weights.selected;
    const int width = weights.intermediate, outputs = weights.hidden;
    std::vector<int32_t> ids(active * rows);
    for (int row = 0; row < rows; ++row) {
        std::iota(ids.begin() + row * active, ids.begin() + (row + 1) * active, 0);
    }
    if (selection) ids = selection->ids;
    const std::vector<float> routing{0.05f, 0.15f, 0.30f, 0.50f};
    std::unique_ptr<Branch> observed, plain;
    if (!fixture) {
        observed = std::make_unique<Branch>(cpu, weights, 0, weights.selected, routing,
            true, rows, 0, true, BranchStage::Full, selection);
        plain = std::make_unique<Branch>(cpu, weights, 0, weights.selected, routing,
            true, rows, 0, false, BranchStage::Full, selection);
    }
    auto ctx = context();
    auto * matrix = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_MXFP4, width, outputs, weights.selected);
    auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, active, rows);
    auto * route_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, active, rows);
    auto * packed = ggml_cast(ctx.get(), input, GGML_TYPE_Q8_0);
    auto * raw = ggml_mul_mat_id(ctx.get(), matrix, input, route_ids);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    auto * quant_graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, raw);
    ggml_build_forward_expand(quant_graph, packed);
    ggml_tensor * post_input = nullptr, * bias = nullptr, * probabilities = nullptr, * post_output = nullptr;
    ggml_cgraph * post_graph = nullptr;
    if (!fixture) {
        post_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, outputs, active, rows);
        bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, outputs, weights.selected);
        probabilities = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, active, rows);
        auto * weighted = ggml_mul(ctx.get(), ggml_add_id(ctx.get(), post_input, bias, route_ids), probabilities);
        post_output = ggml_reshape_3d(ctx.get(),
            ggml_sum_rows(ctx.get(), ggml_cont(ctx.get(), ggml_transpose(ctx.get(), weighted))), outputs, 1, rows);
        post_graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(post_graph, post_output);
    }
    Buffer buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu), ggml_backend_buffer_free);
    require(buffer != nullptr, "Cannot allocate compatible-down CPU controls.");
    ggml_backend_tensor_set(matrix, weights.tensors.at("down.weight").data(), 0, ggml_nbytes(matrix));
    ggml_backend_tensor_set(route_ids, ids.data(), 0, ggml_nbytes(route_ids));
    if (bias) {
        ggml_backend_tensor_set(bias, weights.tensors.at("down.bias").data(), 0, ggml_nbytes(bias));
        std::vector<float> probs(active * rows);
        for (int row = 0; row < rows; ++row) std::copy(routing.begin(), routing.end(), probs.begin() + row * active);
        if (selection) probs = selection->probabilities;
        ggml_backend_tensor_set(probabilities, probs.data(), 0, ggml_nbytes(probabilities));
    }
    json report = {{"schema_version", 1}, {"kind", "cpu_compatible_packed_down_probe"},
        {"backend_commit", llama_commit()}, {"probe_source_sha256", VRAMBRIDGE_HYBRID_SOURCE_SHA256},
        {"cuda_source_sha256", VRAMBRIDGE_COMPATIBLE_CUDA_SHA256},
        {"cuda_header_sha256", VRAMBRIDGE_COMPATIBLE_HEADER_SHA256},
        {"cpu_reference", "loaded native CPU MXFP4 mul_mat_id, including native Q8 quantization"},
        {"gpu_contract", "paired eight-lane even/odd FMA and native AVX2 horizontal sum"},
        {"integer_dot", packed_dot ? "byte_permute_dp4a" : "scalar_quad"},
        {"cuda_device", 0}, {"raw_activations_recorded", false}, {"raw_routing_recorded", false},
        {"full_model_acceleration_claim", false}, {"full_model_qualification", false},
        {"performance_scope", "raw down projection after shared CPU activation, before shared CPU post-ops"},
        {"timing_includes", {"CPU native Q8 cast and packed readback", "pageable H2D packed input and IDs",
                            "GPU raw projection", "D2H all raw per-rank outputs", "device synchronization"}},
        {"timing_excludes", {"allocation", "shared CPU gate/up/SwiGLU", "shared CPU bias/weight/reduction"}},
        {"weight_uploads_in_timing", upload_weights}, {"rows", rows}, {"active_experts_per_row", active},
        {"timing_order", gpu_first ? "gpu_then_cpu" : "cpu_then_gpu"},
        {"expert_bank_size", weights.selected}, {"input_width", width}, {"output_width", outputs},
        {"down_weight_bytes", ggml_nbytes(matrix)}, {"packed_input_bytes", ggml_nbytes(packed)},
        {"f32_input_bytes", ggml_nbytes(input)}, {"raw_output_bytes", ggml_nbytes(raw)},
        {"activation_source", fixture ? "adversarial raw-dot fixture" :
            selection ? "observer-qualified actual routed CPU SwiGLU" : "real weights, synthetic FFN input"},
        {"activation_pattern", pattern}, {"repetitions", repetitions}, {"samples", json::array()}};
    bool passed = true;
    for (int sample = 0; sample < samples; ++sample) {
        std::vector<float> values;
        std::map<std::string, std::vector<float>> observations;
        json observer = nullptr;
        if (fixture) {
            Weights shape;
            shape.hidden = width * active;
            values = activation_input(shape, rows, sample, true, pattern);
            if (pattern == "sine") {
                const float magnitudes[]{1.0f, 0.001f, 0.00000001f, 16384.0f};
                for (size_t i = 0; i < values.size(); ++i) values[i] *= magnitudes[(i / 32 + sample) % 4];
            }
        } else {
            const auto hidden = selection ? selection->input : activation_input(weights, rows, sample, false, pattern);
            for (auto * branch : {plain.get(), observed.get()}) {
                branch->set_input(hidden);
                branch->compute(false);
            }
            observer = compare_bits(plain->result(), observed->result());
            passed &= observer.at("bitwise_equal").get<bool>();
            observations = observed->operator_results();
            values = observations.at("swiglu");
        }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        std::vector<float> expected(ggml_nelements(raw)), first(expected.size());
        std::vector<double> cpu_times;
        bool cpu_stable = true;
        const auto run_cpu = [&] {
            for (int iteration = -2; iteration < repetitions; ++iteration) {
                const auto start = Clock::now();
                require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "Native raw-down control failed.");
                ggml_backend_tensor_get(raw, expected.data(), 0, ggml_nbytes(raw));
                const auto end = Clock::now();
                if (iteration == -2) first = expected;
                else cpu_stable &= std::memcmp(first.data(), expected.data(), ggml_nbytes(raw)) == 0;
                if (iteration >= 0) cpu_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
        };
        const auto run_gpu = [&] {
            return compatible_down(weights.tensors.at("down.weight"), ids, width, outputs,
                weights.selected, repetitions, upload_weights, packed_dot, [&](std::vector<uint8_t> & bytes) {
                    require(ggml_backend_graph_compute(cpu, quant_graph) == GGML_STATUS_SUCCESS, "Native Q8 cast failed.");
                    require(bytes.size() == ggml_nbytes(packed), "Packed Q8 size mismatch.");
                    ggml_backend_tensor_get(packed, bytes.data(), 0, bytes.size());
                });
        };
        CompatibleDownResult candidate;
        if (gpu_first) {
            candidate = run_gpu();
            run_cpu();
        } else {
            run_cpu();
            candidate = run_gpu();
        }
        const auto raw_check = compare_bits(expected, candidate.output);
        json original_raw = nullptr, post_check = nullptr, captured_check = nullptr;
        if (!fixture) {
            original_raw = compare_bits(observations.at("down_projection"), expected);
            ggml_backend_tensor_set(post_input, candidate.output.data(), 0, ggml_nbytes(post_input));
            require(ggml_backend_graph_compute(cpu, post_graph) == GGML_STATUS_SUCCESS, "CPU original post-ops failed.");
            std::vector<float> final(ggml_nelements(post_output));
            ggml_backend_tensor_get(post_output, final.data(), 0, ggml_nbytes(post_output));
            post_check = compare_bits(plain->result(), final);
            passed &= original_raw.at("bitwise_equal").get<bool>() && post_check.at("bitwise_equal").get<bool>();
            if (selection) {
                captured_check = compare(selection->expected, final);
                passed &= captured_check.at("passed").get<bool>();
            }
        }
        const bool sample_passed = raw_check.at("bitwise_equal").get<bool>() && cpu_stable &&
            candidate.all_iterations_identical && (fixture ||
                (observer.at("bitwise_equal").get<bool>() && original_raw.at("bitwise_equal").get<bool>() &&
                 post_check.at("bitwise_equal").get<bool>())) &&
            (!selection || captured_check.at("passed").get<bool>());
        passed &= sample_passed;
        report["samples"].push_back({{"sample", sample}, {"observer_control", observer},
            {"cpu_raw_vs_original", original_raw}, {"gpu_raw_vs_cpu", raw_check},
            {"cpu_postops_vs_original", post_check}, {"candidate_vs_captured_boundary", captured_check},
            {"cpu_all_iterations_identical", cpu_stable},
            {"gpu_all_iterations_identical", candidate.all_iterations_identical},
            {"cpu_samples_ms", cpu_times}, {"gpu_inclusive_samples_ms", candidate.samples_ms},
            {"gpu_cpu_quantize_samples_ms", candidate.quantize_ms}, {"cpu_median_ms", median(cpu_times)},
            {"gpu_inclusive_median_ms", median(candidate.samples_ms)},
            {"raw_stage_speedup", median(cpu_times) / median(candidate.samples_ms)},
            {"timings_numerically_qualified", sample_passed}});
    }
    report["all_checks_passed"] = passed;
    return report;
}
#endif

#ifdef VRAMBRIDGE_CPU_CACHE_STAGING
#include "cpu_cache_staging.h"
#endif

static int run(int argc, char ** argv) {
    fs::path model, destination, activation_capture;
    std::string activation_pattern = "sine";
    std::vector<int> expert_ids{0, 7, 42, 113};
    bool explicit_experts = false;
    int layer = 0, repetitions = 20, samples = 3, threads = 4, rows = 1, gpu_row_limit = 0, cpu_row_limit = 0;
    bool self_test = false, gpu_reduce = false, upload_weights = false, reverse_splits = false, operator_check = false;
    bool cpu_activation = false, compatible = false, compatible_gpu_first = false, compatible_packed_dot = false;
    bool cpu_cache = false;
    int cache_experts = 4, warm_mib = 8, warm_cpu = 0;
    int capture_record = 0;
    std::set<std::string> value_options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "Usage: vrambridge-hybrid-layer --model first-shard.gguf --output report.json "
                         "[--layer N] [--repetitions N] [--samples N] [--threads N] [--gpu-reduce] "
                         "[--rows N] [--gpu-row-limit N] [--cpu-row-limit N] [--upload-weights] [--reverse-splits] "
                         "[--operator-check] [--cpu-activation] [--expert-ids A,B,C,D] [--activation-pattern sine|zero|q8-ties]\n"
                         "Or --cpu-activation --activation-capture private.json [--capture-record N] "
                         "(bounded actual routed replay; always untimed and operator-checked).\n"
                         "Or --self-test (CPU F32 only). Real probe uses GPT-OSS MXFP4 weights, synthetic "
                         "activations and fixed expert IDs. --rows (1-256) reuses each expert across rows; "
                         "--upload-weights includes synchronous pageable weight copies in every timed iteration. "
                         "Isolated expert timings are not prefill or model tok/s.\n"
                         "Optional CUDA research build: --compatible-down checks packed native-Q8 raw down "
                         "bits and CPU post-ops. --self-test --compatible-down uses adversarial MXFP4/Q8 "
                         "GPU fixtures (not the CPU F32 self-test). No full-model speed claim.\n";
            std::cout << "--compatible-gpu-first reverses the raw-stage timing order.\n";
            std::cout << "--compatible-dp4a uses packed integer dot products with the same FP32 contract.\n";
            std::cout << "Optional CPU research build: --cpu-cache-staging [--cpu-experts 1-4] "
                         "[--cache-warm-mib 0-64] [--warm-cpu N] [--repetitions N]. "
                         "One-row owned-buffer CLFLUSH/cache-warming calibration, not model tok/s.\n";
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
        if (arg == "--upload-weights") {
            upload_weights = true;
            continue;
        }
        if (arg == "--reverse-splits") {
            reverse_splits = true;
            continue;
        }
        if (arg == "--operator-check") {
            operator_check = true;
            continue;
        }
        if (arg == "--cpu-activation") {
            cpu_activation = true;
            continue;
        }
        if (arg == "--compatible-down") {
            compatible = true;
            continue;
        }
        if (arg == "--compatible-gpu-first") {
            compatible_gpu_first = true;
            continue;
        }
        if (arg == "--compatible-dp4a") {
            compatible_packed_dot = true;
            continue;
        }
        if (arg == "--cpu-cache-staging") {
            cpu_cache = true;
            continue;
        }
        require(i + 1 < argc, "Missing option value: " + arg);
        const std::string value = argv[++i];
        value_options.insert(arg);
        if (arg == "--model") model = value;
        else if (arg == "--output") destination = value;
        else if (arg == "--activation-capture") activation_capture = value;
        else if (arg == "--activation-pattern") activation_pattern = value;
        else if (arg == "--expert-ids") {
            explicit_experts = true;
            expert_ids.clear();
            std::istringstream selected(value);
            std::string item;
            while (std::getline(selected, item, ',')) {
                size_t consumed = 0;
                const int id = std::stoi(item, &consumed);
                require(consumed == item.size() && id >= 0 && id < 128, "Expert ID out of supported range.");
                require(std::find(expert_ids.begin(), expert_ids.end(), id) == expert_ids.end(), "Duplicate expert ID.");
                expert_ids.push_back(id);
            }
            require(expert_ids.size() == 4 && !value.empty() && value.back() != ',',
                    "Supply exactly four distinct expert IDs.");
        }
        else {
            size_t consumed = 0;
            const int number = std::stoi(value, &consumed);
            require(consumed == value.size(), "Invalid numeric option.");
            if (arg == "--layer") layer = number;
            else if (arg == "--repetitions") repetitions = number;
            else if (arg == "--samples") samples = number;
            else if (arg == "--threads") threads = number;
            else if (arg == "--rows") rows = number;
            else if (arg == "--gpu-row-limit") gpu_row_limit = number;
            else if (arg == "--cpu-row-limit") cpu_row_limit = number;
            else if (arg == "--capture-record") capture_record = number;
            else if (arg == "--cpu-experts") cache_experts = number;
            else if (arg == "--cache-warm-mib") warm_mib = number;
            else if (arg == "--warm-cpu") warm_cpu = number;
            else throw std::runtime_error("Unknown option: " + arg);
        }
    }
    require(layer >= 0 && layer < 36 && repetitions >= (cpu_cache ? 1 : samples) && repetitions <= 1000 &&
            samples >= 1 && samples <= 20 && threads >= 1 && threads <= 32 &&
            rows >= 1 && rows <= 256 && gpu_row_limit >= 0 && gpu_row_limit <= 256 &&
            cpu_row_limit >= 0 && cpu_row_limit <= 256,
            "Option out of supported range.");
    require(self_test || (!model.empty() && !destination.empty()), "Real probe requires --model and --output.");
    require(!self_test || !explicit_experts, "Explicit expert IDs require real model weights.");
    require(activation_pattern == "sine" || activation_pattern == "zero" || activation_pattern == "q8-ties",
            "Unsupported activation pattern.");
    require(!operator_check || rows <= 64, "Operator checks are bounded to at most 64 rows.");
#ifndef VRAMBRIDGE_CPU_CACHE_STAGING
    require(!cpu_cache, "--cpu-cache-staging requires the opt-in CPU research build.");
#endif
    require(cache_experts >= 1 && cache_experts <= 4 && warm_mib >= 0 && warm_mib <= 64 &&
            warm_cpu >= 0 && warm_cpu < 1024, "CPU cache parameters out of supported range.");
    require(cpu_cache || (!value_options.count("--cpu-experts") && !value_options.count("--cache-warm-mib") &&
                          !value_options.count("--warm-cpu")), "CPU cache parameters require --cpu-cache-staging.");
    if (cpu_cache) {
        require(!compatible && !cpu_activation && !gpu_reduce && !upload_weights && !reverse_splits &&
                !operator_check && !gpu_row_limit && !cpu_row_limit && rows == 1 && activation_capture.empty(),
                "CPU cache calibration requires the unsplit original one-row CPU FFN.");
        const std::set<std::string> allowed{"--model", "--output", "--layer", "--threads", "--repetitions",
            "--cpu-experts", "--cache-warm-mib", "--warm-cpu", "--expert-ids"};
        for (const auto & option : value_options) require(allowed.count(option), "CPU cache mode does not accept " + option);
#ifdef VRAMBRIDGE_CPU_CACHE_STAGING
        require(warm_cpu < sysconf(_SC_NPROCESSORS_CONF), "Warming CPU is not present.");
#endif
    }
#ifndef VRAMBRIDGE_COMPATIBLE_DOWN
    require(!compatible, "--compatible-down requires an explicit CUDA compatible-down research build.");
#endif
    require(!compatible_gpu_first || compatible, "--compatible-gpu-first requires --compatible-down.");
    require(!compatible_packed_dot || compatible, "--compatible-dp4a requires --compatible-down.");
    require(!compatible || (!gpu_reduce && !reverse_splits && !operator_check &&
                            !gpu_row_limit && !cpu_row_limit && rows <= 64),
            "Compatible-down checks require <=64 rows and no split/reduction/operator overrides.");
    require(destination.empty() || (!fs::exists(destination) && !fs::exists(destination.string() + ".part")),
            "Refusing to overwrite existing result or partial result.");
    std::unique_ptr<RoutedInput> routed;
    require(!value_options.count("--capture-record") || !activation_capture.empty(),
            "--capture-record requires --activation-capture.");
    if (!activation_capture.empty()) {
        require(cpu_activation && !gpu_reduce && (!upload_weights || compatible) && !reverse_splits,
                "Captured replay requires --cpu-activation and no timing/partition overrides.");
        std::set<std::string> allowed{"--model", "--output", "--threads", "--activation-capture", "--capture-record"};
        if (compatible) allowed.insert("--repetitions");
        for (const auto & option : value_options) {
            require(allowed.count(option) != 0, "Captured replay does not accept " + option);
        }
        routed = std::make_unique<RoutedInput>(read_routed_input(activation_capture, capture_record, self_test, model));
    }
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
    require((self_test && !compatible) || cpu_cache || gpu_device != nullptr, "No usable GPU; no CPU-only fallback for this probe.");
    Backend cpu(ggml_backend_dev_init(cpu_device, nullptr), ggml_backend_free);
    require(cpu != nullptr, "CPU backend initialization failed.");
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu_device), "ggml_backend_set_n_threads"));
    require(set_threads != nullptr, "CPU backend thread control unavailable.");
    set_threads(cpu.get(), threads);
#ifdef VRAMBRIDGE_CPU_CACHE_STAGING
    if (cpu_cache) {
        const auto weights = self_test ? synthetic_weights(cache_experts) :
            read_weights(model, layer, std::vector<int>(expert_ids.begin(), expert_ids.begin() + cache_experts));
        auto report = check_cpu_cache_staging(cpu.get(), weights, repetitions,
            size_t(warm_mib) * 1024 * 1024, warm_cpu, self_test);
        report["model"] = model.filename().string();
        report["weights"] = self_test ? "synthetic_f32" : "real_gptoss_mxfp4";
        report["layer"] = layer;
        report["cpu_threads"] = threads;
        write_report(report, destination);
        return 0;
    }
#endif
#ifdef VRAMBRIDGE_COMPATIBLE_DOWN
    if (compatible) {
        require(!routed || !self_test, "Compatible-down routed replay requires actual captures, not F32 fixtures.");
        const auto weights = self_test ? compatible_fixture_weights() :
            read_weights(model, routed ? routed->layer : layer, routed ? routed->experts : expert_ids);
        auto report = check_compatible_down(cpu.get(), weights, routed ? routed->rows : rows,
            repetitions, routed ? 1 : samples, activation_pattern, upload_weights, self_test,
            compatible_gpu_first, compatible_packed_dot, routed.get());
        report["layer"] = routed ? routed->layer : layer;
        report["cpu_threads"] = threads;
        report["cpu_backend"] = ggml_backend_dev_description(cpu_device);
        report["gpu_backend"] = ggml_backend_dev_description(gpu_device);
        report["model"] = model.filename().string();
        report["capture_source_file"] = activation_capture.filename().string();
        report["capture_record"] = capture_record;
        report["capture"] = routed ? routed->metadata : json(nullptr);
        write_report(report, destination);
        return report.at("all_checks_passed").get<bool>() ? 0 : 1;
    }
#endif
    if (routed) {
        Weights captured_weights = self_test ? synthetic_weights(static_cast<int>(routed->experts.size())) :
            read_weights(model, routed->layer, routed->experts);
        Backend captured_gpu(nullptr, ggml_backend_free);
        if (!self_test) {
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(gpu_device, &props);
            const size_t down_bytes = captured_weights.tensors.at("down.weight").size() +
                captured_weights.tensors.at("down.bias").size();
            require(props.memory_free >= 3 * down_bytes + 512 * 1024 * 1024,
                    "Insufficient GPU headroom for bounded routed replay and observer controls.");
            captured_gpu.reset(ggml_backend_dev_init(gpu_device, nullptr));
            require(captured_gpu != nullptr, "Routed replay GPU initialization failed.");
        }
        auto report = replay_routed_input(cpu.get(), self_test ? cpu.get() : captured_gpu.get(),
                                         captured_weights, *routed, self_test);
        report["capture_source_file"] = activation_capture.filename().string();
        report["capture_record"] = capture_record;
        report["cpu_threads"] = threads;
        report["model"] = model.filename().string();
        report["routing_experts_modified"] = false;
        report["cuda_disable_fusion"] = std::getenv("GGML_CUDA_DISABLE_FUSION")
            ? json(std::getenv("GGML_CUDA_DISABLE_FUSION")) : json(nullptr);
        report["mxfp4_cpu_q8_parity"] = std::getenv("VRAMBRIDGE_HYBRID_MXFP4_Q8_PARITY")
            ? json(std::getenv("VRAMBRIDGE_HYBRID_MXFP4_Q8_PARITY")) : json(nullptr);
        write_report(report, destination);
        return report.at("all_checks_passed").get<bool>() ? 0 : 1;
    }
    Weights weights = self_test ? synthetic_weights() : read_weights(model, layer, expert_ids);
    const std::vector<float> routing{0.05f, 0.15f, 0.30f, 0.50f};
    Branch reference(cpu.get(), weights, 0, 4, routing, true, rows);
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
        {"activation_pattern", activation_pattern},
        {"routing_weights", routing}, {"cpu_threads", threads}, {"repetitions", repetitions}, {"samples", samples},
        {"rows_per_expert", rows}, {"activation_bytes", weights.hidden * rows * sizeof(float)},
        {"gpu_row_limit", gpu_row_limit},
        {"cpu_row_limit", cpu_row_limit},
        {"execution_split", cpu_activation ? "cpu_activation_gpu_down" : "whole_experts"},
        {"cpu_activation_quantization", cpu_activation && weights.type == GGML_TYPE_MXFP4
            ? "native_q8_0_roundtrip_to_f32" : "none"},
        {"activation_stage_copy_method", cpu_activation ? "synchronous_tensor_copy" : "none"},
        {"activation_stage_copy_destination", self_test ? "cpu_self_test" : "gpu"},
        {"operator_check_enabled", operator_check}, {"timings_qualified", !operator_check},
        {"weight_uploads_in_timing", upload_weights},
        {"weight_upload_method", upload_weights ? "synchronous_pageable_source" : "none"},
        {"weight_upload_destination", self_test ? "cpu_self_test" : "gpu"},
        {"cuda_mmq_precision_override", std::getenv("GGML_CUDA_MMQ_PREC")
            ? json(std::getenv("GGML_CUDA_MMQ_PREC")) : json(nullptr)},
        {"cuda_disable_fusion", std::getenv("GGML_CUDA_DISABLE_FUSION")
            ? json(std::getenv("GGML_CUDA_DISABLE_FUSION")) : json(nullptr)},
        {"cuda_disable_graphs", std::getenv("GGML_CUDA_DISABLE_GRAPHS")
            ? json(std::getenv("GGML_CUDA_DISABLE_GRAPHS")) : json(nullptr)},
        {"mxfp4_cpu_q8_parity", std::getenv("VRAMBRIDGE_HYBRID_MXFP4_Q8_PARITY")
            ? json(std::getenv("VRAMBRIDGE_HYBRID_MXFP4_Q8_PARITY")) : json(nullptr)},
        {"full_model_acceleration_claim", false}, {"warm_cache_microbenchmark", !upload_weights},
        {"separate_gpu_reconciliation_stream", !self_test}, {"hardware_gpu_partition", false},
        {"reduction_strategy", gpu_reduce ? "per_expert_outputs_weighted_and_reduced_on_merge_backend"
                                         : "weighted_partial_vectors_then_merge_add"},
        {"timing_order", "serial/concurrent interleaved; first schedule alternates each repetition"},
        {"split_order", reverse_splits ? "descending_gpu_experts" : "ascending_gpu_experts"},
        {"partitions", json::array()},
        {"limitations", {"Fixed selected weights may fit CPU/GPU caches; this is not a full-model bandwidth roofline.",
                          "Synthetic activations and chosen experts are not actual model hidden states or router decisions.",
                          "Every row selects the same four experts: concentrated reuse, not measured prefill routing.",
                          "Optional GPU row chunks share weights but add graph/concatenation overhead; CPU reference remains full-row.",
                          "Optional CPU row chunks affect the candidate host branch only; the unsplit CPU reference remains full-row.",
                          "CPU-activation split keeps gate/up/SwiGLU on CPU for every expert; only selected down projections move to GPU.",
                          "That split times CPU Q8 roundtrip, F32 effective-input copy and GPU requantization; it is not a packed-Q8 transfer.",
                          "Timed weight uploads recopy the same expert bytes; CPU source-cache misses are not enforced.",
                          "Pageable synchronous uploads do not represent pinned asynchronous DMA or a pipelined executor.",
                          "No attention, router, shared expert, residual, prefill, disk tier, or dynamic cache is executed.",
                          "Correctness tolerance is a microkernel gate, not full-model token/logit equivalence.",
                          "Cross-backend async copies may internally fall back to synchronization.",
                          "Timing includes input upload, dispatch, CPU execution, output staging, dependency copy, merge and final synchronization.",
                          "File loading, graph setup, and verification readback are outside timing; weight recopy is included only when requested."}},
    };
    bool passed = true;
    for (int split = 0; split <= 4; ++split) {
        const int resident_count = reverse_splits ? 4 - split : split;
        std::unique_ptr<Branch> host, resident, activation;
        if (resident_count < 4) host = std::make_unique<Branch>(
            cpu.get(), weights, resident_count, 4, routing, !gpu_reduce, rows, cpu_row_limit);
        if (resident_count > 0) {
            if (cpu_activation) activation = std::make_unique<Branch>(
                cpu.get(), weights, 0, resident_count, routing, false, rows, cpu_row_limit,
                false, BranchStage::CpuActivation);
            resident = std::make_unique<Branch>(
                self_test ? cpu.get() : gpu.get(), weights, 0, resident_count, routing, !gpu_reduce, rows,
                gpu_row_limit, false, cpu_activation ? BranchStage::Down : BranchStage::Full);
        }
        auto merge = std::make_unique<Merge>(self_test ? cpu.get() : reconciliation.get(),
            self_test ? cpu_device : gpu_device, weights.hidden, resident_count, routing, gpu_reduce, self_test, rows);
        json checks = json::array();
        std::vector<std::vector<float>> inputs, references;
        for (int sample = 0; sample < samples; ++sample) {
            auto input = activation_input(weights, rows, sample, self_test, activation_pattern);
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
        std::map<std::string, std::vector<double>> upload_times;
        std::map<std::string, std::vector<double>> activation_times, activation_copy_times;
        std::map<std::string, double> worst_abs, worst_relative;
        std::map<std::string, bool> mode_passed;
        for (int index = -1; index < repetitions; ++index) {
            std::vector<std::string> schedules{"serial"};
            if (!self_test) {
                schedules.push_back("concurrent");
                if (index >= 0 && index % 2) std::reverse(schedules.begin(), schedules.end());
            }
            for (const auto & mode : schedules) {
                const int sample = index < 0 ? 0 : index % samples;
                const auto started = Clock::now();
                if (upload_weights && resident) resident->upload_weights(weights, 0);
                const auto uploaded = Clock::now();
                if (activation) {
                    activation->set_input(inputs[sample]);
                    activation->compute(false);
                }
                const auto prepared = Clock::now();
                if (resident) {
                    if (activation) resident->copy_input(activation->output);
                    else resident->set_input(inputs[sample]);
                }
                const auto copied = Clock::now();
                if (resident) {
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
                    actual.assign(weights.hidden * rows, 0);
                    for (Branch * branch : {host.get(), resident.get()}) {
                        if (!branch) continue;
                        const auto values = branch->result();
                        for (size_t i = 0; i < actual.size(); ++i) actual[i] += values[i];
                    }
                } else {
                    merge->compute(host.get(), resident.get());
                }
                const auto completed = Clock::now();
                if (index >= 0) {
                    timings[mode].push_back(std::chrono::duration<double, std::milli>(completed - started).count());
                    dispatch_times[mode].push_back(std::chrono::duration<double, std::milli>(submitted - prepared).count());
                    activation_times[mode].push_back(std::chrono::duration<double, std::milli>(prepared - uploaded).count());
                    activation_copy_times[mode].push_back(activation
                        ? std::chrono::duration<double, std::milli>(copied - prepared).count() : 0);
                    upload_times[mode].push_back(std::chrono::duration<double, std::milli>(uploaded - started).count());
                    host_times[mode].push_back(std::chrono::duration<double, std::milli>(host_done - submitted).count());
                    merge_times[mode].push_back(std::chrono::duration<double, std::milli>(completed - host_done).count());
                }
                if (!self_test || gpu_reduce) actual = merge->result();
                if (self_test && !gpu_reduce) {
                    merge->compute(host.get(), resident.get());
                    require(compare(actual, merge->result()).at("passed").get<bool>(),
                            "CPU reconciliation graph differs from manual partial sum.");
                }
                auto check = compare(references[sample], actual);
                check["mode"] = mode;
                check["sample"] = sample;
                passed = passed && check.at("passed").get<bool>();
                auto position = mode_passed.try_emplace(mode, true).first;
                position->second = position->second && check.at("passed").get<bool>();
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
                {"all_checks_passed", mode_passed.at(mode)},
                {"median_gpu_input_dispatch_ms", median(dispatch_times[mode])},
                {"median_weight_upload_ms", median(upload_times[mode])},
                {"median_cpu_activation_prepare_ms", median(activation_times[mode])},
                {"median_activation_stage_copy_ms", median(activation_copy_times[mode])},
                {"median_cpu_input_compute_ms", median(host_times[mode])},
                {"median_reconciliation_and_dependency_wait_ms", median(merge_times[mode])},
            };
        }
        report["partitions"].push_back({
            {"gpu_experts", self_test ? 0 : resident_count}, {"cpu_experts", self_test ? 4 : 4 - resident_count},
            {"expert_partition_scope", cpu_activation ? "down_projection_only" : "whole_ffn"},
            {"cpu_gate_up_experts", self_test || cpu_activation ? 4 : 4 - resident_count},
            {"gpu_gate_up_experts", self_test || cpu_activation ? 0 : resident_count},
            {"cpu_down_experts", self_test ? 4 : 4 - resident_count},
            {"gpu_down_experts", self_test ? 0 : resident_count},
            {"split_index", resident_count}, {"modes", modes}, {"correctness", checks},
            {"all_checks_passed", std::all_of(mode_passed.begin(), mode_passed.end(),
                [](const auto & entry) { return entry.second; })},
            {"weight_upload_bytes_per_iteration", upload_weights && resident ? resident->weight_bytes() : 0},
            {"activation_stage_copy_bytes_per_iteration", activation ? ggml_nbytes(activation->output) : 0},
            {"cpu_partial_output_bytes", !self_test && host ? ggml_nbytes(host->output) : 0},
            {"pinned_staging_bytes", !self_test ? ggml_nbytes(merge->cpu) : 0},
        });
    }
    if (operator_check) {
        report["operator_diagnostics"] = cpu_activation
            ? check_cpu_activation_split(cpu.get(), self_test ? cpu.get() : gpu.get(), weights, routing,
                reference, rows, gpu_row_limit, cpu_row_limit, samples, self_test, activation_pattern)
            : check_operators(cpu.get(), self_test ? cpu.get() : gpu.get(), weights, routing, reference,
                rows, gpu_row_limit, samples, self_test, activation_pattern);
        passed = passed && report["operator_diagnostics"].at("all_checks_passed").get<bool>();
    }
    report["all_checks_passed"] = passed;
    report["timings_qualified"] = !operator_check && passed;
    write_report(report, destination);
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
