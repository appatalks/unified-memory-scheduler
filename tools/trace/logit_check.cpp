#include "arg.h"
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-context.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;
struct CloseFile { void operator()(FILE * value) const noexcept { std::fclose(value); } };
using File = std::unique_ptr<FILE, CloseFile>;

struct PrefillLocalization;
static bool localization_callback(ggml_tensor * tensor, bool ask, PrefillLocalization & localization);

struct Diagnostic {
    bool armed = false;
    bool sync_hybrid_gpu = false;
    int last_layer = -1;
    json values = json::array();
    std::string error;
    bool all_layers = false;
    bool attention_details = false;
    bool expert_details = false;
    std::set<std::string> layers;
    std::map<std::string, size_t> occurrences;
    size_t captured_bytes = 0;
    bool prefill_armed = false;
    std::map<int, json> prefill_frames;
    std::map<int, int> empty_prefill_batches;
    json prefill_records = json::array();
    PrefillLocalization * localization = nullptr;
};

static bool prefill_capture_callback(ggml_tensor * tensor, bool ask, Diagnostic & diagnostic) {
    const std::string name = tensor->name;
    const auto dash = name.rfind('-');
    if (dash == std::string::npos) return !ask;
    const std::string prefix = name.substr(0, dash);
    const std::string suffix = name.substr(dash + 1);
    if ((suffix != "0" && suffix != "17" && suffix != "35") ||
        (prefix != "ffn_moe_up" && prefix != "ffn_moe_weighted" && prefix != "ffn_moe_out")) return !ask;
    const int layer = std::stoi(suffix);
    auto & frame = diagnostic.prefill_frames[layer];
    if (frame.is_object() && frame.contains("output")) return !ask;
    if (ask) return true;
    const std::string key = prefix == "ffn_moe_up" ? "input" :
        prefix == "ffn_moe_weighted" ? "probabilities" : "output";
    const auto * source = key == "output" ? tensor : tensor->src[1];
    if (!source) throw std::runtime_error("Missing original FFN probability input");
    const bool probabilities = key == "probabilities";
    const int width = probabilities ? 4 : 2880;
    const bool three_dimensional = key != "output";
    const int64_t native_rows = source->ne[three_dimensional ? 2 : 1];
    if (native_rows == 0) {
        if (key == "input") ++diagnostic.empty_prefill_batches[layer];
        return true;
    }
    if (source->type != GGML_TYPE_F32 ||
        source->ne[0] != (probabilities ? 1 : width) ||
        (three_dimensional ? source->ne[1] != (probabilities ? 4 : 1) : source->ne[2] != 1) ||
        source->ne[3] != 1 || source->nb[0] != 4 || (probabilities && source->nb[1] != 4) ||
        native_rows < 1 || native_rows > 2048) {
        throw std::runtime_error("Unsupported actual-prefill capture layout: " + name);
    }
    if (key == "input") {
        if (!frame.is_null()) throw std::runtime_error("Incomplete FFN frame before next input");
        if (!tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer)) {
            throw std::runtime_error("Prefill capture requires CPU expert projection outputs");
        }
        frame = {{"layer", layer}, {"native_rows", native_rows},
                 {"rows", std::min<int64_t>(native_rows, 64)},
                 {"row_offset", std::max<int64_t>(0, native_rows - 64)}, {"hidden", 2880},
                 {"input_source", "ffn_moe_up.src1"}, {"routing_source", "ffn_moe_up.src2"}};
    } else if (frame.is_null() || frame.at("native_rows") != native_rows || frame.contains(key)) {
        throw std::runtime_error("Unaligned or duplicate actual-prefill capture component: " + name);
    }
    const int rows = frame.at("rows");
    const size_t stride = source->nb[three_dimensional ? 2 : 1];
    json values = json::array();
    for (int row = 0; row < rows; ++row) {
        const size_t offset = (row + frame.at("row_offset").get<size_t>()) * stride;
        if (offset > ggml_nbytes(source) || width * sizeof(float) > ggml_nbytes(source) - offset) {
            throw std::runtime_error("Actual-prefill capture stride exceeds storage");
        }
        std::vector<float> selected(width);
        ggml_backend_tensor_get(source, selected.data(), offset, selected.size() * sizeof(float));
        for (float value : selected) {
            if (!std::isfinite(value) || (probabilities && value < 0)) {
                throw std::runtime_error("Nonfinite or invalid actual-prefill values");
            }
            values.push_back(value);
        }
    }
    frame[key] = std::move(values);
    if (key == "input") {
        const auto * ids = tensor->src[2];
        if (!ids || ids->type != GGML_TYPE_I32 || ids->ne[0] != 4 ||
            ids->ne[1] != native_rows || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != 4) {
            throw std::runtime_error("Unsupported original per-token routing input");
        }
        json selected_ids = json::array();
        for (int row = 0; row < rows; ++row) {
            int32_t selected[4];
            const size_t offset = (row + frame.at("row_offset").get<size_t>()) * ids->nb[1];
            if (offset > ggml_nbytes(ids) || sizeof(selected) > ggml_nbytes(ids) - offset) {
                throw std::runtime_error("Original routing stride exceeds storage");
            }
            ggml_backend_tensor_get(ids, selected, offset, sizeof(selected));
            std::set<int32_t> unique;
            for (int32_t value : selected) {
                if (value < 0 || value >= 128 || !unique.insert(value).second) {
                    throw std::runtime_error("Invalid actual-prefill expert IDs");
                }
                selected_ids.push_back(value);
            }
        }
        frame["expert_ids"] = std::move(selected_ids);
    }
    return true;
}

static bool diagnostic_callback(ggml_tensor * tensor, bool ask, void * data) noexcept {
    auto & diagnostic = *static_cast<Diagnostic *>(data);
    if (diagnostic.localization) {
        try { return localization_callback(tensor, ask, *diagnostic.localization); }
        catch (const std::exception & exc) { diagnostic.error = exc.what(); return false; }
    }
    if (diagnostic.prefill_armed) {
        try { return prefill_capture_callback(tensor, ask, diagnostic); }
        catch (const std::exception & exc) { diagnostic.error = exc.what(); return false; }
    }
    const std::string name = tensor->name;
    if (diagnostic.sync_hybrid_gpu && name.rfind("hybrid_gpu_weighted-", 0) == 0) {
        return true;
    }
    const auto dash = name.rfind('-');
    const std::string prefix = name.substr(0, dash);
    const bool wanted = diagnostic.armed && ((name == "result_norm" || name == "result_output" ||
        name == "model.input_embed" || name == "hc_init" || name == "ple_embd") ||
        (dash != std::string::npos &&
        (diagnostic.all_layers || diagnostic.layers.count(name.substr(dash + 1))) &&
        (prefix == "attn_post_norm" || prefix == "ffn_norm" || prefix == "ffn_moe_topk" ||
         prefix == "ffn_moe_weights_softmax" || prefix == "ffn_moe_down_biased" ||
         prefix == "ffn_moe_weighted" || prefix == "hybrid_cpu_down_biased" ||
         prefix == "hybrid_cpu_weighted" || prefix == "hybrid_gpu_weighted" ||
         prefix == "ffn_moe_out" || prefix == "hc_norm" || prefix == "hc_gate" ||
         prefix == "hc_mixed" || prefix == "hc_inject" || prefix == "hc_combine" ||
         prefix == "linear_attn_qkv_mixed" || prefix == "conv_output_raw" ||
         prefix == "q_conv_predelta" || prefix == "k_conv_predelta" || prefix == "v_conv_predelta" ||
         prefix == "beta_sigmoid" || prefix == "gate" || prefix == "final_output" ||
         prefix == "linear_attn_out" || prefix == "attn_output" || prefix == "ffn_out" ||
         prefix == "ffn_shexp_gated" || prefix == "l_last" || prefix == "ple_gated_value" ||
         prefix == "ple_conv_out" ||
         (diagnostic.expert_details &&
          (prefix == "ffn_moe_gate_up" || prefix == "ffn_moe_gate" || prefix == "ffn_moe_up" ||
           prefix == "ffn_moe_swiglu" || prefix == "ffn_moe_down" ||
           prefix == "hybrid_cpu_gate_up" || prefix == "hybrid_cpu_gate" || prefix == "hybrid_cpu_up" ||
           prefix == "hybrid_cpu_swiglu" || prefix == "hybrid_cpu_down" ||
           prefix == "hybrid_gpu_gate_up" || prefix == "hybrid_gpu_gate" || prefix == "hybrid_gpu_up" ||
           prefix == "hybrid_gpu_swiglu" || prefix == "hybrid_gpu_down")) ||
         (diagnostic.attention_details &&
          (prefix == "Qcur_full" || prefix == "Qcur_normed" || prefix == "Kcur_normed" ||
           prefix == "Qcur" || prefix == "Kcur" || prefix == "Vcur" || prefix == "indexer_q" ||
           prefix == "indexer_top_k" || prefix == "attn_pregate" || prefix == "attn_gated" ||
           prefix == "kqv_out" || prefix == "kq_soft_max" || prefix == "kqv" ||
           prefix == "gate_sigmoid")))));
    if (ask) return wanted;
    if (!wanted) return true;
    try {
        const size_t count = ggml_nelements(tensor);
        if (count > 1000000 || ggml_nbytes(tensor) > 16*1024*1024 ||
            ggml_nbytes(tensor) > 64*1024*1024 - diagnostic.captured_bytes) {
            throw std::runtime_error("Diagnostic tensor exceeds the bounded capture size: " + name);
        }
        json entry = {{"name", name}, {"shape", {tensor->ne[0], tensor->ne[1], tensor->ne[2]}},
                      {"shape4", {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}},
                      {"occurrence", diagnostic.occurrences[name]++},
                      {"strides", {tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]}}};
        std::vector<uint8_t> raw(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
        const auto offset = [&](size_t index) {
            size_t result = 0;
            for (int dimension = 0; dimension < 4; ++dimension) {
                result += (index % tensor->ne[dimension])*tensor->nb[dimension];
                index /= tensor->ne[dimension];
            }
            return result;
        };
        if (tensor->type == GGML_TYPE_F32) {
            std::vector<float> values(count);
            for (size_t i = 0; i < count; ++i) std::memcpy(&values[i], raw.data() + offset(i), sizeof(float));
            entry["values"] = values;
        } else if (tensor->type == GGML_TYPE_I32) {
            std::vector<int32_t> values(count);
            for (size_t i = 0; i < count; ++i) std::memcpy(&values[i], raw.data() + offset(i), sizeof(int32_t));
            entry["values"] = values;
        } else throw std::runtime_error("Unexpected diagnostic tensor type");
        diagnostic.values.push_back(entry);
        diagnostic.captured_bytes += raw.size();
        return true;
    } catch (const std::exception & exc) {
        diagnostic.error = exc.what();
        return false;
    }
}

static void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

static void write_bytes(FILE * output, const void * data, size_t bytes) {
    require(std::fwrite(data, 1, bytes, output) == bytes, "Reference write failed");
}

static void read_bytes(FILE * input, void * data, size_t bytes) {
    require(std::fread(data, 1, bytes, input) == bytes, "Reference truncated");
}

static File open_reference(const fs::path & path, json & header, size_t requests, int steps) {
    File file(std::fopen(path.c_str(), "rb"));
    require(file != nullptr, "Cannot open reference");
    char magic[8];
    read_bytes(file.get(), magic, sizeof(magic));
    require(std::memcmp(magic, "VRLOG001", 8) == 0, "Reference magic mismatch");
    uint64_t bytes;
    read_bytes(file.get(), &bytes, sizeof(bytes));
    require(bytes > 0 && bytes <= 1024 * 1024, "Reference header size invalid");
    std::string serialized(bytes, '\0');
    read_bytes(file.get(), serialized.data(), serialized.size());
    header = json::parse(serialized);
    require(header.is_object() && header.at("schema_version") == 1 &&
            header.at("steps") == steps && header.at("requests") == requests &&
            header.at("vocab").is_number_integer() && header.at("vocab") > 0 &&
            header.at("vocab") <= 1000000 && header.at("context").is_number_integer() &&
            header.at("context") > steps && header.at("context") <= 16777216 &&
            header.at("model").is_string() && header.at("backend_commit").is_string() &&
            header.at("contains_token_ids") == true, "Reference header configuration invalid");
    const long payload = std::ftell(file.get());
    require(payload >= 0, "Reference position unavailable");
    for (size_t request = 0; request < requests; ++request) {
        uint64_t tokens;
        read_bytes(file.get(), &tokens, sizeof(tokens));
        require(tokens > 0 && tokens <= header.at("context").get<uint64_t>() - steps,
                "Reference prompt token count invalid");
        const uint64_t skip = tokens*sizeof(llama_token) +
            uint64_t(steps)*(sizeof(llama_token) + header.at("vocab").get<uint64_t>()*sizeof(float));
        require(skip <= uint64_t(std::numeric_limits<long>::max()) &&
                std::fseek(file.get(), static_cast<long>(skip), SEEK_CUR) == 0, "Reference seek failed");
    }
    read_bytes(file.get(), magic, sizeof(magic));
    require(std::memcmp(magic, "ENDLOG01", 8) == 0 && std::fgetc(file.get()) == EOF,
            "Reference trailer or file length mismatch");
    require(std::fseek(file.get(), payload, SEEK_SET) == 0, "Reference rewind failed");
    return file;
}

static void publish(File & file, const fs::path & partial, const fs::path & final) {
    require(std::fclose(file.release()) == 0, "Result close failed");
    fs::create_hard_link(partial, final);
    fs::remove(partial);
}

#include "prefill_localization.h"

static void decode(llama_context * ctx, const std::vector<llama_token> & tokens, bool all_logits = false) {
    auto batch = common_batch_get_one(ctx, tokens);
    if (all_logits) {
        for (int i = 0; i < batch.size(); ++i) require(batch.set_output(i, true), "Cannot request block logits");
    }
    const int limit = static_cast<int>(llama_n_batch(ctx));
    for (int offset = 0; offset < batch.size(); offset += limit) {
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE,
            batch.get_sub_batch(offset, std::min(limit, batch.size() - offset))) == 0, "Decode failed");
    }
}

static int run(int argc, char ** argv) {
    fs::path corpus, reference_out, reference_in, output, debug_output, prefill_output;
    fs::path localize_output, localize_reference, teacher_forced_output;
    std::string localize_focus = "ffn";
    bool explicit_localize_focus = false;
    int steps = 32;
    int decode_group = 1;
    int64_t template_epoch = 0;
    std::string reasoning_effort = "low";
    bool eval_callback_check = false;
    bool sync_hybrid_gpu = false;
    bool all_layer_diagnostics = false;
    bool cpu_reference = false;
    int diagnostic_layer_limit = -1;
    bool attention_details = false;
    bool expert_details = false;
    bool rollback_check = false;
    std::vector<char *> backend_args{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--localization-self-test") {
            const auto result = PrefillLocalization::routing_difference(
                {2, 31, 78, 127, 2, 31, 78, 127}, {31, 2, 78, 127, 2, 31, 78, 55});
            require(result.at("different_ordered_rows") == 2 && result.at("different_expert_set_rows") == 1 &&
                    result.at("first_different_expert_set_row") == 1, "Routing localization self-test failed");
            std::cout << result.dump() << '\n';
            return 0;
        }
        if (arg == "--debug-all-layers") {
            all_layer_diagnostics = true;
            continue;
        }
        if (arg == "--cpu-reference") {
            cpu_reference = true;
            continue;
        }
        if (arg == "--rollback-check") {
            rollback_check = true;
            continue;
        }
        if (arg == "--debug-attention") {
            attention_details = true;
            continue;
        }
        if (arg == "--debug-experts") {
            expert_details = true;
            continue;
        }
        if (arg == "--help") {
            std::cout << "Usage: vrambridge-logit-check --corpus corpus.jsonl --output report.json "
                         "(--reference-out logits.bin | --reference-in logits.bin) [--steps 32] [--batch-check 1] "
                         "[--template-epoch UNIX_SECONDS] [--reasoning-effort default|none|low|medium|high] "
                         "[--rollback-check] "
                         "[--debug-output tensors.json [--debug-all-layers | --debug-layer-limit N] [--debug-attention] [--debug-experts]] [--cpu-reference] "
                         "[--prefill-capture-output private-directory/ffn.json] "
                         "[--localize-output private.bin | --localize-reference private.bin] "
                         "[--localize-focus ffn|attention1|rounding1] "
                         "[--teacher-forced-reference-out private.bin] "
                         "[pinned backend model arguments]\n"
                         "Compares full teacher-forced logits, never generation throughput. Reference contains token IDs; keep private.\n";
            return 0;
        }
        if (arg == "--eval-callback-check" || arg == "--serialized-check") {
            eval_callback_check = true;
            if (arg == "--serialized-check") {
                std::cerr << "Deprecated --serialized-check: use --eval-callback-check; this does not serialize execution.\n";
            }
            continue;
        }
        if (arg == "--sync-hybrid-gpu") {
            sync_hybrid_gpu = true;
            continue;
        }
        if (arg == "--corpus" || arg == "--output" || arg == "--reference-out" ||
            arg == "--reference-in" || arg == "--steps" || arg == "--debug-output" ||
            arg == "--batch-check" || arg == "--debug-layer-limit" || arg == "--template-epoch" ||
            arg == "--reasoning-effort" || arg == "--prefill-capture-output" ||
            arg == "--localize-output" || arg == "--localize-reference" ||
            arg == "--teacher-forced-reference-out" || arg == "--localize-focus") {
            require(++i < argc, "Missing checker option value");
            const std::string value = argv[i];
            if (arg == "--corpus") corpus = value;
            else if (arg == "--output") output = value;
            else if (arg == "--reference-out") reference_out = value;
            else if (arg == "--reference-in") reference_in = value;
            else if (arg == "--debug-output") debug_output = value;
            else if (arg == "--prefill-capture-output") prefill_output = value;
            else if (arg == "--localize-output") localize_output = value;
            else if (arg == "--localize-reference") localize_reference = value;
            else if (arg == "--teacher-forced-reference-out") teacher_forced_output = value;
            else if (arg == "--localize-focus") { localize_focus = value; explicit_localize_focus = true; }
            else if (arg == "--reasoning-effort") {
                require(value == "default" || value == "none" || value == "low" ||
                        value == "medium" || value == "high",
                        "Reasoning effort must be default, none, low, medium, or high");
                reasoning_effort = value;
            }
            else if (arg == "--template-epoch") {
                size_t consumed = 0;
                template_epoch = std::stoll(value, &consumed);
                const auto maximum = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::duration::max()).count();
                require(consumed == value.size() && template_epoch >= 0 && template_epoch <= maximum,
                        "Template epoch must be a nonnegative representable Unix timestamp");
            }
            else {
                size_t consumed = 0;
                const int parsed = std::stoi(value, &consumed);
                require(consumed == value.size(), "Invalid integer option");
                if (arg == "--debug-layer-limit") {
                    require(parsed >= 1 && parsed <= 1024, "Diagnostic layer limit must be between 1 and 1024");
                    diagnostic_layer_limit = parsed;
                } else {
                    (arg == "--batch-check" ? decode_group : steps) = parsed;
                }
            }
        } else backend_args.push_back(argv[i]);
    }
    require(!corpus.empty() && !output.empty() && reference_in.empty() != reference_out.empty() &&
            steps >= 1 && steps <= 128 && decode_group >= 1 && decode_group <= 32 &&
            (reference_out.empty() || decode_group == 1), "Required checker arguments or step/block range invalid");
    require(!rollback_check || (!reference_in.empty() && decode_group >= 2),
            "Rollback checking requires a reference and a block of at least two");
    bool prefill_down_requested = false;
    if (const char * mode = std::getenv("VRAMBRIDGE_HYBRID_PREFILL_DOWN")) {
        require(std::string(mode) == "0" || std::string(mode) == "1",
                "VRAMBRIDGE_HYBRID_PREFILL_DOWN must be 0 or 1");
        prefill_down_requested = std::string(mode) == "1";
    }
#ifndef LLAMA_VRAMBRIDGE_PREFILL_DOWN_API
    require(!prefill_down_requested, "Resident-down prefill requires the separate research backend phase API");
#endif
    bool prefill_cpu_control = false;
    if (const char * control = std::getenv("VRAMBRIDGE_HYBRID_PREFILL_DOWN_CPU_CONTROL")) {
        require((std::string(control) == "0" || std::string(control) == "1") &&
                (std::string(control) == "0" || prefill_down_requested),
                "Prefill-down CPU control requires enabled research mode and value 0 or 1");
        prefill_cpu_control = std::string(control) == "1";
    }
    bool prefill_audit = false;
    if (const char * audit = std::getenv("VRAMBRIDGE_HYBRID_PREFILL_DOWN_AUDIT")) {
        require((std::string(audit) == "0" || std::string(audit) == "1") &&
                (std::string(audit) == "0" || prefill_down_requested),
                "Prefill-down audit requires enabled research mode and value 0 or 1");
        prefill_audit = std::string(audit) == "1";
    }
    int prefill_layer = -1;
    if (const char * selected = std::getenv("VRAMBRIDGE_HYBRID_PREFILL_DOWN_LAYER")) {
        size_t used = 0;
        prefill_layer = std::stoi(selected, &used);
        require(prefill_down_requested && used == std::string(selected).size() &&
                prefill_layer >= 0 && prefill_layer < 35, "Prefill-down diagnostic layer must be 0-34");
    }
    require(!fs::exists(output) && !fs::exists(output.string() + ".part"), "Output already exists");
    require(reference_out.empty() || (!fs::exists(reference_out) && !fs::exists(reference_out.string() + ".part")),
            "Reference output already exists");
    require(debug_output.empty() || (!fs::exists(debug_output) && !fs::exists(debug_output.string() + ".part")),
            "Diagnostic output already exists");
    require(prefill_output.empty() || (!fs::exists(prefill_output) && !fs::exists(prefill_output.string() + ".part")),
            "Prefill capture output already exists");
    std::set<fs::path> destinations;
    for (const auto & path : {output, reference_out, debug_output, prefill_output, localize_output, teacher_forced_output}) {
        if (path.empty()) continue;
        require(!fs::exists(path) && !fs::exists(path.string() + ".part"), "Output or partial already exists");
        require(destinations.insert(fs::weakly_canonical(path)).second &&
                destinations.insert(fs::weakly_canonical(path.string() + ".part")).second,
                "Checker output paths overlap");
    }
    const bool localizing = !localize_output.empty() || !localize_reference.empty();
    require((localize_focus == "ffn" || localize_focus == "attention1" || localize_focus == "rounding1") &&
            (!explicit_localize_focus || localizing), "Localization focus requires a valid localization run");
    if (const char * permit = std::getenv("VRAMBRIDGE_HYBRID_PREFILL_DOWN_LOCALIZE")) {
        require((std::string(permit) == "0" || std::string(permit) == "1") &&
                (std::string(permit) == "0" || (localizing && prefill_down_requested)),
                "Prefill localization permission requires an enabled research localization run");
    }
    require(localize_output.empty() || localize_reference.empty(), "Choose localization output or reference, not both");
    require(!localizing || (!reference_in.empty() && prefill_output.empty() && debug_output.empty() &&
            !eval_callback_check && !sync_hybrid_gpu && !cpu_reference &&
            std::getenv("GGML_CUDA_DISABLE_FUSION") &&
            std::string(std::getenv("GGML_CUDA_DISABLE_FUSION")) == "1"),
            "Localization requires a comparison reference, no other observer, and CUDA fusion disabled");
    if (localizing && prefill_down_requested) {
#ifndef LLAMA_VRAMBRIDGE_LOCALIZATION_API
        throw std::runtime_error("Localization requires the separate diagnostic backend overlay");
#else
        require(std::getenv("VRAMBRIDGE_HYBRID_PREFILL_DOWN_LOCALIZE") &&
                std::string(std::getenv("VRAMBRIDGE_HYBRID_PREFILL_DOWN_LOCALIZE")) == "1",
                "Localization requires explicit backend observer permission");
#endif
    }
    if (!teacher_forced_output.empty()) {
        require(!reference_in.empty(), "Teacher-forced export requires --reference-in");
        PrefillLocalization::private_parent(teacher_forced_output);
    }
    if (!prefill_output.empty()) {
        require(!reference_in.empty(), "Prefill capture requires --reference-in for observer qualification");
        const auto parent = prefill_output.parent_path().empty() ? fs::path(".") : prefill_output.parent_path();
        require(fs::is_directory(parent) &&
                (fs::status(parent).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none,
                "Prefill capture requires an existing private parent directory (mode 0700)");
        require(std::getenv("VRAMBRIDGE_HYBRID_PLAN") == nullptr,
                "Prefill capture requires native CPU experts without a hybrid plan");
    }
    require(!all_layer_diagnostics || !debug_output.empty(), "--debug-all-layers requires --debug-output");
    require(diagnostic_layer_limit < 0 || !debug_output.empty(), "--debug-layer-limit requires --debug-output");
    require(!all_layer_diagnostics || diagnostic_layer_limit < 0, "Choose all layers or a layer limit, not both");
    require(!attention_details || !debug_output.empty(), "--debug-attention requires --debug-output");
    require(!expert_details || !debug_output.empty(), "--debug-experts requires --debug-output");
    std::ifstream prompts(corpus);
    require(prompts.good(), "Cannot open corpus");
    std::vector<json> requests;
    std::set<std::string> ids;
    std::string line;
    while (std::getline(prompts, line)) {
        if (line.empty()) continue;
        auto record = json::parse(line);
        require(record.is_object() && record.at("id").is_string() && !record.at("id").get<std::string>().empty() &&
                record.at("prompt").is_string() && !record.at("prompt").get<std::string>().empty() &&
                (record.at("split") == "train" || record.at("split") == "test") &&
                (record.at("category") == "coding" || record.at("category") == "reasoning" ||
                 record.at("category") == "prose") &&
                ids.insert(record.at("id").get<std::string>()).second, "Invalid or duplicate corpus record");
        if (record.at("split") == "test") {
            require(record.at("prompt").is_string() && record.at("id").is_string(), "Invalid corpus record");
            requests.push_back(record);
        }
    }
    require(prompts.eof() && !requests.empty() && requests.size() <= 20, "Invalid held-out corpus");
    require(prefill_output.empty() || requests.size() <= 3, "Prefill capture is bounded to three held-out requests");
    require(!localizing || requests.size() == 1, "Localization is bounded to one held-out request");
    require(!localizing || requests.front().at("id").get<std::string>().size() <= 128,
            "Localization request ID exceeds 128 bytes");
    std::unique_ptr<PrefillLocalization> localization;
    if (localizing) localization = std::make_unique<PrefillLocalization>(
        localize_output, localize_reference, localize_focus);
    json reference_header;
    File binary = reference_in.empty() ? File(nullptr) : open_reference(reference_in, reference_header, requests.size(), steps);
    if (!reference_in.empty()) {
        require(reference_header.value("reasoning_effort", json("low")) == reasoning_effort,
                "Reference reasoning effort differs");
    }
    const std::string identity = llama_commit();
    require(identity.size() >= 7 &&
            std::string("620d2225cc436ab0b001e3200e590e3b286ef9ab").rfind(identity, 0) == 0,
            "Pinned backend mismatch");
    ggml_backend_load_all_from_path(VRAMBRIDGE_LLAMA_LIBRARY_DIR);
    ggml_backend_load_all();
    common_init();
    common_params params;
    Diagnostic diagnostic;
    diagnostic.sync_hybrid_gpu = sync_hybrid_gpu;
    diagnostic.all_layers = all_layer_diagnostics;
    diagnostic.attention_details = attention_details;
    diagnostic.expert_details = expert_details;
    diagnostic.localization = localization.get();
    params.sampling.temp = 0;
    require(common_params_parse(static_cast<int>(backend_args.size()), backend_args.data(), params,
                                LLAMA_EXAMPLE_COMMON), "Invalid backend options");
    require(!params.model.path.empty(), "Supply a local model path");
    bool usable_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        usable_gpu = usable_gpu || type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    require(params.n_gpu_layers == 0 || usable_gpu,
            "GPU placement requested but no usable GPU device initialized; no model loaded");
    params.warmup = false;
    params.n_outputs_max = decode_group;
    params.n_outputs_max_per_seq = decode_group;
    if (!debug_output.empty() || eval_callback_check || sync_hybrid_gpu || !prefill_output.empty() || localizing) {
        params.cb_eval = diagnostic_callback;
        params.cb_eval_user_data = &diagnostic;
    }
    auto initialized = common_init_from_params(params);
    require(initialized && initialized->context() && initialized->model(), "Model initialization failed");
    auto * ctx = initialized->context();
    if (cpu_reference) {
        auto * sched = ctx->get_sched();
        int configured = 0;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            auto * backend = ggml_backend_sched_get_backend(sched, i);
            auto * device = ggml_backend_get_device(backend);
            if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) continue;
            using set_reference_fn = void (*)(ggml_backend_t, bool);
            auto set_reference = reinterpret_cast<set_reference_fn>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device), "ggml_backend_cpu_set_use_ref"));
            require(set_reference != nullptr, "Selected CPU backend lacks its reference-mode hook");
            set_reference(backend, true);
            ++configured;
        }
        require(configured > 0, "No CPU backend available for reference mode");
        std::cerr << "Diagnostic CPU reference path enabled; not an optimized serving mode.\n";
    }
    require(decode_group <= static_cast<int>(llama_n_batch(ctx)) &&
            decode_group <= static_cast<int>(llama_n_ubatch(ctx)), "Checker block exceeds actual batch capacity");
    const auto * model = initialized->model();
    if (!prefill_output.empty() || localizing) {
        char architecture[64]{};
        require(llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture)) > 0 &&
                std::string(architecture) == "gpt-oss" && llama_model_n_layer(model) == 36 &&
                llama_model_n_embd(model) == 2880, "Prefill capture supports the GPT-OSS-120B layout only");
    }
    require(!localizing || llama_n_ubatch(ctx) <= 512, "Localization supports at most 512 physical rows");
    diagnostic.last_layer = llama_model_n_layer(model) - 1;
    if (diagnostic_layer_limit > 0) {
        for (int layer = 0; layer <= diagnostic.last_layer && layer < diagnostic_layer_limit; ++layer) {
            diagnostic.layers.insert(std::to_string(layer));
        }
    } else {
        diagnostic.layers = {"0", std::to_string(diagnostic.last_layer)};
    }
    const auto * vocab = llama_model_get_vocab(model);
    const int count = llama_vocab_n_tokens(vocab);
    require(count > 0 && count <= 1000000, "Invalid vocabulary");
    auto templates = common_chat_templates_init(model, params.chat_template);
    require(templates != nullptr, "Template initialization failed");
    json header = {{"schema_version", 1}, {"vocab", count}, {"steps", steps}, {"requests", requests.size()},
                   {"context", llama_n_ctx(ctx)}, {"model", fs::path(params.model.path).filename().string()},
                   {"contains_token_ids", true}, {"backend_commit", llama_commit()}};
    if (cpu_reference) header["cpu_reference"] = true;
    if (reasoning_effort != "low") header["reasoning_effort"] = reasoning_effort;
    if (!reference_out.empty()) {
        if (!reference_out.parent_path().empty()) fs::create_directories(reference_out.parent_path());
        binary.reset(std::fopen((reference_out.string() + ".part").c_str(), "wx"));
        require(binary != nullptr, "Cannot exclusively create reference");
        const auto serialized = header.dump();
        const uint64_t bytes = serialized.size();
        write_bytes(binary.get(), "VRLOG001", 8);
        write_bytes(binary.get(), &bytes, sizeof(bytes));
        write_bytes(binary.get(), serialized.data(), serialized.size());
    } else {
        require(reference_header == header, "Reference model/context/vocabulary configuration differs");
    }
    File teacher_file(nullptr);
    if (!teacher_forced_output.empty()) {
        teacher_file.reset(std::fopen((teacher_forced_output.string() + ".part").c_str(), "wx"));
        require(teacher_file != nullptr, "Cannot create private teacher-forced reference");
        fs::permissions(teacher_forced_output.string() + ".part", fs::perms::owner_read | fs::perms::owner_write);
        const auto serialized = header.dump();
        const uint64_t bytes = serialized.size();
        write_bytes(teacher_file.get(), "VRLOG001", 8);
        write_bytes(teacher_file.get(), &bytes, sizeof(bytes));
        write_bytes(teacher_file.get(), serialized.data(), serialized.size());
    }
    if (localization) {
        json descriptor = {{"schema_version", 1}, {"records", localization->expected_records}, {"contains_sensitive_activations", true},
            {"model", header["model"]}, {"backend_commit", header["backend_commit"]},
            {"context", llama_n_ctx(ctx)}, {"logical_batch", llama_n_batch(ctx)},
            {"physical_batch", llama_n_ubatch(ctx)}, {"request", requests.front().at("id")}};
        if (localize_focus != "ffn") descriptor["focus"] = localize_focus;
        localization->configure(descriptor);
    }
    json report = {{"schema_version", 1}, {"backend_commit", llama_commit()},
                   {"source_sha256", VRAMBRIDGE_LOGIT_SOURCE_SHA256},
                   {"teacher_forced", true}, {"steps_per_request", steps}, {"vocab", count},
                   {"decode_block_size", decode_group},
                   {"logical_batch_size", llama_n_batch(ctx)},
                   {"physical_batch_size", llama_n_ubatch(ctx)},
                   {"template_epoch_seconds", template_epoch},
                   {"reasoning_effort", reasoning_effort},
                   {"cpu_reference", cpu_reference},
                   {"inactive_eval_callback", eval_callback_check},
                   {"mode", reference_out.empty() ? "reference_comparison" : "reference_creation"},
                   {"comparison_reference_file", reference_in.filename().string()},
                   {"general_model_quality_established", false},
                   {"synchronize_hybrid_gpu_output", sync_hybrid_gpu},
                   {"prefill_capture_enabled", !prefill_output.empty()},
                   {"prefill_localization_enabled", localizing},
                   {"teacher_forced_reference_exported", !teacher_forced_output.empty()},
                   {"localization_source_sha256", VRAMBRIDGE_LOCALIZATION_SOURCE_SHA256},
                   {"resident_down_prefill_requested", prefill_down_requested},
                   {"resident_down_prefill_cpu_control", prefill_cpu_control},
                   {"resident_down_prefill_audit", prefill_audit},
                   {"resident_down_prefill_diagnostic_layer", prefill_layer < 0 ? json(nullptr) : json(prefill_layer)},
                   {"cuda_disable_fusion", std::getenv("GGML_CUDA_DISABLE_FUSION")
                        ? json(std::getenv("GGML_CUDA_DISABLE_FUSION")) : json(nullptr)},
                   {"model", header["model"]}, {"context", header["context"]},
                   {"full_model_acceleration_claim", false}, {"requests", json::array()},
                   {"gate", "maximum absolute logit error <= 0.01; relative L2 <= 0.001; matched nonfinite entries skipped"}};
    bool passed = true;
    for (const auto & request : requests) {
        common_chat_templates_inputs chat;
        common_chat_msg system; system.role = "system"; system.content = "You are a helpful assistant.";
        common_chat_msg user; user.role = "user"; user.content = request.at("prompt").get<std::string>();
        chat.messages = {system, user};
        if (reasoning_effort != "default") {
            chat.chat_template_kwargs["reasoning_effort"] = json(reasoning_effort).dump();
        }
        chat.now = std::chrono::system_clock::time_point{std::chrono::seconds{template_epoch}};
        const auto rendered = common_chat_templates_apply(templates.get(), chat);
        const auto tokens = common_tokenize(ctx, rendered.prompt.str(), false, true);
        require(!tokens.empty() && tokens.size() + steps <= llama_n_ctx(ctx), "Prompt exceeds actual context");
        const uint64_t size = tokens.size();
        if (teacher_file) {
            write_bytes(teacher_file.get(), &size, sizeof(size));
            write_bytes(teacher_file.get(), tokens.data(), tokens.size() * sizeof(llama_token));
        }
        if (!reference_out.empty()) {
            write_bytes(binary.get(), &size, sizeof(size));
            write_bytes(binary.get(), tokens.data(), tokens.size() * sizeof(llama_token));
        } else {
            uint64_t expected_size;
            read_bytes(binary.get(), &expected_size, sizeof(expected_size));
            require(size == expected_size, "Prompt token count differs");
            std::vector<llama_token> expected(tokens.size());
            read_bytes(binary.get(), expected.data(), expected.size() * sizeof(llama_token));
            require(expected == tokens, "Prompt tokens differ; refusing an unaligned logit comparison");
        }
        std::vector<llama_token> expected_tokens;
        std::vector<float> expected_logits;
        if (reference_out.empty()) {
            expected_tokens.resize(steps);
            expected_logits.resize(size_t(steps)*count);
            for (int step = 0; step < steps; ++step) {
                read_bytes(binary.get(), &expected_tokens[step], sizeof(llama_token));
                require(expected_tokens[step] >= 0 && expected_tokens[step] < count,
                        "Reference token outside vocabulary");
                read_bytes(binary.get(), expected_logits.data() + size_t(step)*count, count*sizeof(float));
            }
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        diagnostic.armed = false;
        diagnostic.prefill_frames.clear();
        diagnostic.empty_prefill_batches.clear();
        diagnostic.prefill_armed = !prefill_output.empty();
        if (localization) { localization->request = request.at("id"); localization->armed = true; }
#ifdef LLAMA_VRAMBRIDGE_PREFILL_DOWN_API
        llama_set_prefill_phase(ctx, true);
#endif
        decode(ctx, tokens);
        if (localization) localization->armed = false;
#ifdef LLAMA_VRAMBRIDGE_PREFILL_DOWN_API
        llama_set_prefill_phase(ctx, false);
#endif
        diagnostic.prefill_armed = false;
        diagnostic.armed = false;
        require(diagnostic.error.empty(), "Diagnostic tensor collection failed: " + diagnostic.error);
        if (!prefill_output.empty()) {
            for (int layer : {0, 17, 35}) {
                auto & frame = diagnostic.prefill_frames[layer];
                require(frame.is_object() && frame.contains("input") && frame.contains("expert_ids") &&
                        frame.contains("probabilities") && frame.contains("output"),
                        "Incomplete actual-prefill FFN frame at layer " + std::to_string(layer));
                frame["request"] = request.at("id");
                frame["category"] = request.at("category");
                frame["prompt_tokens"] = tokens.size();
                frame["observed_empty_batches"] = diagnostic.empty_prefill_batches[layer];
                diagnostic.prefill_records.push_back(std::move(frame));
            }
        }
        double maximum = 0, maximum_l2 = 0;
        int active_block_start = 0, next_decode_step = 0, decode_windows = 0;
        int rollback_windows = 0, discarded_tokens = 0;
        int top1_matches = 0;
        json position_errors = json::array();
        for (int step = 0; step < steps; ++step) {
            const int output_index = step == 0 ? -1 : step - active_block_start - 1;
            auto * logits = llama_get_logits_ith(ctx, output_index);
            require(logits != nullptr, "Missing output logits");
            require(std::any_of(logits, logits + count, [](float value) { return std::isfinite(value); }) &&
                    std::none_of(logits, logits + count, [](float value) { return std::isnan(value); }),
                    "Logit position has no finite values or contains NaN");
            llama_token token = static_cast<llama_token>(
                std::max_element(logits, logits + count) - logits);
            if (!reference_out.empty()) {
                write_bytes(binary.get(), &token, sizeof(token));
                write_bytes(binary.get(), logits, count * sizeof(float));
                ++top1_matches;
            } else {
                const llama_token expected = expected_tokens[step];
                const float * reference = expected_logits.data() + size_t(step)*count;
                double error2 = 0, norm2 = 0;
                top1_matches += token == static_cast<llama_token>(
                    std::max_element(reference, reference + count) - reference);
                for (int i = 0; i < count; ++i) {
                    if (!std::isfinite(reference[i]) || !std::isfinite(logits[i])) {
                        require(reference[i] == logits[i], "Nonfinite logits differ");
                        continue;
                    }
                    const double delta = double(logits[i]) - reference[i];
                    maximum = std::max(maximum, std::abs(delta));
                    error2 += delta * delta;
                    norm2 += double(reference[i]) * reference[i];
                }
                maximum_l2 = std::max(maximum_l2, std::sqrt(error2 / std::max(norm2, 1e-30)));
                position_errors.push_back({{"position", step}, {"cumulative_max_abs", maximum},
                                            {"cumulative_relative_l2", maximum_l2}});
                token = expected; // identical continuation, even if the candidate argmax differs
            }
            if (teacher_file) {
                write_bytes(teacher_file.get(), &token, sizeof(token));
                write_bytes(teacher_file.get(), logits, count * sizeof(float));
            }
            diagnostic.armed = !debug_output.empty() && step == 0 &&
                request.at("id") == requests.front().at("id");
            if (step == next_decode_step) {
                if (reference_out.empty()) {
                    const auto end = expected_tokens.begin() + std::min(step + decode_group, steps);
                    std::vector<llama_token> proposed(expected_tokens.begin() + step, end);
                    const int evaluated = static_cast<int>(proposed.size());
                    const int accepted = rollback_check ? std::min(evaluated, 1 + decode_windows % decode_group) : evaluated;
                    const auto previous = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
                    if (rollback_check) {
                        for (int i = accepted; i < evaluated; ++i) proposed[i] = (proposed[i] + 1) % count;
                    }
                    decode(ctx, proposed, decode_group > 1);
                    if (rollback_check && accepted < evaluated) {
                        llama_synchronize(ctx);
                        require(llama_memory_seq_rm(llama_get_memory(ctx), 0, previous + accepted + 1, -1),
                                "Model memory cannot discard an unaccepted suffix; rollback is unsupported");
                        require(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == previous + accepted,
                                "Rollback retained the wrong target positions");
                        ++rollback_windows;
                        discarded_tokens += evaluated - accepted;
                    }
                    active_block_start = step;
                    next_decode_step = step + accepted;
                } else {
                    decode(ctx, std::vector<llama_token>{token});
                    active_block_start = step;
                    next_decode_step = step + 1;
                }
                ++decode_windows;
            }
            diagnostic.armed = false;
            require(diagnostic.error.empty(), "Diagnostic tensor collection failed: " + diagnostic.error);
        }
        const bool good = maximum <= .01 && maximum_l2 <= .001;
        passed = passed && good;
        report["requests"].push_back({{"id", request.at("id")}, {"category", request.at("category")},
                                      {"prompt_tokens", tokens.size()},
                                      {"max_abs_logit_error", maximum}, {"max_relative_l2", maximum_l2},
                                      {"top1_matches", top1_matches}, {"positions", steps},
                                      {"passed", reference_out.empty() ? json(good) : json(nullptr)},
                                      {"position_errors", position_errors}});
        if (rollback_check) {
            report["requests"].back()["rollback"] = {{"rejected_suffix_windows", rollback_windows},
                {"discarded_input_tokens", discarded_tokens}, {"decode_windows", decode_windows}};
        }
        std::cerr << request.at("id") << ": max logit error " << maximum
                  << ", relative L2 " << maximum_l2 << ", top1 " << top1_matches << '/' << steps << '\n';
    }
    if (!reference_out.empty()) {
        write_bytes(binary.get(), "ENDLOG01", 8);
        publish(binary, reference_out.string() + ".part", reference_out);
    } else {
        char trailer[8];
        read_bytes(binary.get(), trailer, sizeof(trailer));
        require(std::memcmp(trailer, "ENDLOG01", 8) == 0 && std::fgetc(binary.get()) == EOF,
                "Reference trailer or file length mismatch");
    }
    report["all_checks_passed"] = reference_out.empty() ? json(passed) : json(nullptr);
    if (rollback_check) report["rollback_check"] = true;
    if (teacher_file) {
        write_bytes(teacher_file.get(), "ENDLOG01", 8);
        publish(teacher_file, teacher_forced_output.string() + ".part", teacher_forced_output);
    }
    if (localization) {
        localization->finish();
        report["prefill_localization"] = {{"raw_activations_recorded_in_report", false},
            {"performance_measurement", false}, {"records", localization->records},
            {"verified_native_fp16_cache_stores", localization->verified_cache_stores},
            {"focus", localize_focus},
            {"comparison_scope", "Tensor differences are diagnostic; full-logit checks use the supplied reference, which may be an observer-preservation reference."},
            {"comparisons", localization->metrics}};
    }
    if (!prefill_output.empty()) {
        json capture = {{"schema_version", 1}, {"kind", "actual_prefill_ffn_capture"},
            {"contains_sensitive_activations", true}, {"backend_commit", llama_commit()},
            {"observer_full_logit_checks_passed", passed},
            {"execution_controls", {{"cpu_reference", cpu_reference},
                {"cuda_disable_fusion", std::getenv("GGML_CUDA_DISABLE_FUSION")
                    ? json(std::getenv("GGML_CUDA_DISABLE_FUSION")) : json(nullptr)}}},
            {"sampling", "last_64_rows_of_first_nonempty_ffn_batch_per_selected_layer"},
            {"source_sha256", VRAMBRIDGE_LOGIT_SOURCE_SHA256}, {"model", header["model"]},
            {"context", llama_n_ctx(ctx)}, {"logical_batch_size", llama_n_batch(ctx)},
            {"physical_batch_size", llama_n_ubatch(ctx)}, {"records", diagnostic.prefill_records}};
        File file(std::fopen((prefill_output.string() + ".part").c_str(), "wx"));
        require(file != nullptr, "Cannot exclusively create private prefill capture");
        fs::permissions(prefill_output.string() + ".part", fs::perms::owner_read | fs::perms::owner_write);
        const auto serialized = capture.dump() + "\n";
        require(serialized.size() <= 128 * 1024 * 1024, "Private prefill capture exceeds 128 MiB");
        write_bytes(file.get(), serialized.data(), serialized.size());
        publish(file, prefill_output.string() + ".part", prefill_output);
        report["prefill_capture_records"] = diagnostic.prefill_records.size();
    }
    if (!debug_output.empty()) {
        File debug(std::fopen((debug_output.string() + ".part").c_str(), "wx"));
        require(debug != nullptr, "Cannot exclusively create diagnostic output");
        const auto serialized = diagnostic.values.dump() + "\n";
        write_bytes(debug.get(), serialized.data(), serialized.size());
        publish(debug, debug_output.string() + ".part", debug_output);
    }
    if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
    File file(std::fopen((output.string() + ".part").c_str(), "wx"));
    require(file != nullptr, "Cannot exclusively create checker report");
    const auto serialized = report.dump(2) + "\n";
    write_bytes(file.get(), serialized.data(), serialized.size());
    publish(file, output.string() + ".part", output);
    localization.reset();
    initialized.reset();
    llama_backend_free();
    return passed ? 0 : 1;
}

int main(int argc, char ** argv) {
    try { return run(argc, argv); }
    catch (const std::exception & exc) {
        std::cerr << "vrambridge-logit-check: " << exc.what() << '\n';
        return 2;
    }
}
