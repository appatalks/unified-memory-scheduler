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

struct Diagnostic {
    bool armed = false;
    bool sync_hybrid_gpu = false;
    int last_layer = -1;
    json values = json::array();
    std::string error;
    bool all_layers = false;
    bool attention_details = false;
    std::set<std::string> layers;
    std::map<std::string, size_t> occurrences;
    size_t captured_bytes = 0;
};

static bool diagnostic_callback(ggml_tensor * tensor, bool ask, void * data) noexcept {
    auto & diagnostic = *static_cast<Diagnostic *>(data);
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
    fs::path corpus, reference_out, reference_in, output, debug_output;
    int steps = 32;
    int decode_group = 1;
    int64_t template_epoch = 0;
    bool eval_callback_check = false;
    bool sync_hybrid_gpu = false;
    bool all_layer_diagnostics = false;
    bool cpu_reference = false;
    int diagnostic_layer_limit = -1;
    bool attention_details = false;
    std::vector<char *> backend_args{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--debug-all-layers") {
            all_layer_diagnostics = true;
            continue;
        }
        if (arg == "--cpu-reference") {
            cpu_reference = true;
            continue;
        }
        if (arg == "--debug-attention") {
            attention_details = true;
            continue;
        }
        if (arg == "--help") {
            std::cout << "Usage: vrambridge-logit-check --corpus corpus.jsonl --output report.json "
                         "(--reference-out logits.bin | --reference-in logits.bin) [--steps 32] [--batch-check 1] "
                         "[--template-epoch UNIX_SECONDS] "
                         "[--debug-output tensors.json [--debug-all-layers | --debug-layer-limit N] [--debug-attention]] [--cpu-reference] "
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
            arg == "--batch-check" || arg == "--debug-layer-limit" || arg == "--template-epoch") {
            require(++i < argc, "Missing checker option value");
            const std::string value = argv[i];
            if (arg == "--corpus") corpus = value;
            else if (arg == "--output") output = value;
            else if (arg == "--reference-out") reference_out = value;
            else if (arg == "--reference-in") reference_in = value;
            else if (arg == "--debug-output") debug_output = value;
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
    require(!fs::exists(output) && !fs::exists(output.string() + ".part"), "Output already exists");
    require(reference_out.empty() || (!fs::exists(reference_out) && !fs::exists(reference_out.string() + ".part")),
            "Reference output already exists");
    require(debug_output.empty() || (!fs::exists(debug_output) && !fs::exists(debug_output.string() + ".part")),
            "Diagnostic output already exists");
    std::set<fs::path> destinations;
    for (const auto & path : {output, reference_out, debug_output}) {
        if (path.empty()) continue;
        require(destinations.insert(fs::weakly_canonical(path)).second &&
                destinations.insert(fs::weakly_canonical(path.string() + ".part")).second,
                "Checker output paths overlap");
    }
    require(!all_layer_diagnostics || !debug_output.empty(), "--debug-all-layers requires --debug-output");
    require(diagnostic_layer_limit < 0 || !debug_output.empty(), "--debug-layer-limit requires --debug-output");
    require(!all_layer_diagnostics || diagnostic_layer_limit < 0, "Choose all layers or a layer limit, not both");
    require(!attention_details || !debug_output.empty(), "--debug-attention requires --debug-output");
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
    json reference_header;
    File binary = reference_in.empty() ? File(nullptr) : open_reference(reference_in, reference_header, requests.size(), steps);
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
    params.n_outputs_max_per_seq = decode_group;
    if (!debug_output.empty() || eval_callback_check || sync_hybrid_gpu) {
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
    json report = {{"schema_version", 1}, {"backend_commit", llama_commit()},
                   {"source_sha256", VRAMBRIDGE_LOGIT_SOURCE_SHA256},
                   {"teacher_forced", true}, {"steps_per_request", steps}, {"vocab", count},
                   {"decode_block_size", decode_group},
                   {"template_epoch_seconds", template_epoch},
                   {"cpu_reference", cpu_reference},
                   {"inactive_eval_callback", eval_callback_check},
                   {"mode", reference_out.empty() ? "reference_comparison" : "reference_creation"},
                   {"general_model_quality_established", false},
                   {"synchronize_hybrid_gpu_output", sync_hybrid_gpu},
                   {"model", header["model"]}, {"context", header["context"]},
                   {"full_model_acceleration_claim", false}, {"requests", json::array()},
                   {"gate", "maximum absolute logit error <= 0.01; relative L2 <= 0.001; matched nonfinite entries skipped"}};
    bool passed = true;
    for (const auto & request : requests) {
        common_chat_templates_inputs chat;
        common_chat_msg system; system.role = "system"; system.content = "You are a helpful assistant.";
        common_chat_msg user; user.role = "user"; user.content = request.at("prompt").get<std::string>();
        chat.messages = {system, user};
        chat.chat_template_kwargs["reasoning_effort"] = "\"low\"";
        chat.now = std::chrono::system_clock::time_point{std::chrono::seconds{template_epoch}};
        const auto rendered = common_chat_templates_apply(templates.get(), chat);
        const auto tokens = common_tokenize(ctx, rendered.prompt.str(), false, true);
        require(!tokens.empty() && tokens.size() + steps <= llama_n_ctx(ctx), "Prompt exceeds actual context");
        const uint64_t size = tokens.size();
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
        decode(ctx, tokens);
        diagnostic.armed = false;
        require(diagnostic.error.empty(), "Diagnostic tensor collection failed: " + diagnostic.error);
        double maximum = 0, maximum_l2 = 0;
        int top1_matches = 0;
        json position_errors = json::array();
        for (int step = 0; step < steps; ++step) {
            const int output_index = step == 0 ? -1 : (step - 1) % decode_group;
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
                top1_matches += expected == token;
                double error2 = 0, norm2 = 0;
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
            diagnostic.armed = !debug_output.empty() && step == 0 &&
                request.at("id") == requests.front().at("id");
            if (step % decode_group == 0) {
                if (reference_out.empty()) {
                    const auto end = expected_tokens.begin() + std::min(step + decode_group, steps);
                    decode(ctx, std::vector<llama_token>(expected_tokens.begin() + step, end), decode_group > 1);
                } else {
                    decode(ctx, std::vector<llama_token>{token});
                }
            }
            diagnostic.armed = false;
            require(diagnostic.error.empty(), "Diagnostic tensor collection failed: " + diagnostic.error);
        }
        const bool good = maximum <= .01 && maximum_l2 <= .001;
        passed = passed && good;
        report["requests"].push_back({{"id", request.at("id")}, {"category", request.at("category")},
                                      {"max_abs_logit_error", maximum}, {"max_relative_l2", maximum_l2},
                                      {"top1_matches", top1_matches}, {"positions", steps},
                                      {"passed", reference_out.empty() ? json(good) : json(nullptr)},
                                      {"position_errors", position_errors}});
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
