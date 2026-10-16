#include "arg.h"
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-model.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

struct Layer {
    int experts = 0;
    int active = 0;
    uint64_t bytes = 0;
    uint64_t host_bytes = 0;
    std::set<std::string> weights;
    std::set<std::string> types;
};

struct Request {
    std::string id;
    std::string split;
    std::string category;
    std::string prompt;
    int tokens;
};

struct CloseFile {
    void operator()(FILE * file) const noexcept {
        std::fclose(file);
    }
};

struct Trace {
    std::unique_ptr<FILE, CloseFile> output;
    std::map<int, Layer> layers;
    std::map<int, int> counts;
    std::string request;
    std::string phase;
    std::string error;
    int token = 0;

    void emit(const json & value) {
        const std::string line = value.dump() + '\n';
        if (std::fwrite(line.data(), 1, line.size(), output.get()) != line.size()) {
            throw std::runtime_error("Failed to write routing trace.");
        }
    }
};

static bool routing_callback(ggml_tensor * tensor, bool ask, void * data) noexcept {
    auto & trace = *static_cast<Trace *>(data);
    if (trace.phase != "decode") {
        return !ask;
    }
    const std::string prefix = "ffn_moe_topk-";
    try {
        const std::string name = tensor->name;
        const bool selected = name.rfind(prefix, 0) == 0;
        if (ask) {
            return selected;
        }
        if (!selected) {
            return true;
        }
        const std::string suffix = name.substr(prefix.size());
        size_t consumed = 0;
        const int layer_id = std::stoi(suffix, &consumed);
        if (consumed != suffix.size() || !trace.layers.count(layer_id)) {
            throw std::runtime_error("Unsupported routing tensor name: " + name);
        }
        const auto & layer = trace.layers.at(layer_id);
        if (tensor->type != GGML_TYPE_I32 || tensor->ne[0] != layer.active ||
            tensor->ne[1] != 1 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
            throw std::runtime_error("Unsupported routing tensor shape/type: " + name);
        }
        std::vector<uint8_t> values(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size());
        for (int64_t column = 0; column < tensor->ne[1]; ++column) {
            std::vector<int32_t> experts;
            std::set<int32_t> unique;
            for (int64_t rank = 0; rank < tensor->ne[0]; ++rank) {
                const size_t offset = rank * tensor->nb[0] + column * tensor->nb[1];
                if (offset + sizeof(int32_t) > values.size()) {
                    throw std::runtime_error("Routing tensor stride exceeds its storage.");
                }
                int32_t expert;
                std::memcpy(&expert, values.data() + offset, sizeof(expert));
                if (expert < 0 || expert >= layer.experts || !unique.insert(expert).second) {
                    throw std::runtime_error("Out-of-range or duplicate routed expert.");
                }
                experts.push_back(expert);
            }
            trace.emit({{"type", "route"}, {"request", trace.request}, {"phase", trace.phase},
                        {"token", trace.token}, {"layer", layer_id}, {"experts", experts}});
            ++trace.counts[layer_id];
        }
        return true;
    } catch (const std::exception & exc) {
        trace.error = exc.what();
        return false;
    }
}

static std::vector<Request> read_corpus(const fs::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open routing corpus.");
    }
    std::vector<Request> requests;
    std::set<std::string> ids;
    std::set<std::string> prompts;
    bool testing = false;
    bool training = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto value = json::parse(line);
        const std::set<std::string> allowed = {"id", "split", "category", "prompt", "max_tokens"};
        if (!value.is_object()) {
            throw std::runtime_error("Corpus records must be JSON objects.");
        }
        for (const auto & item : value.items()) {
            if (!allowed.count(item.key())) {
                throw std::runtime_error("Unknown corpus field: " + item.key());
            }
        }
        if (!value.at("max_tokens").is_number_integer() || value.at("max_tokens").is_boolean() ||
            value.at("max_tokens") < 1 || value.at("max_tokens") > 8192) {
            throw std::runtime_error("Corpus max_tokens must be an integer.");
        }
        Request request{value.at("id").get<std::string>(), value.at("split").get<std::string>(),
                        value.at("category").get<std::string>(), value.at("prompt").get<std::string>(),
                        value.at("max_tokens").get<int>()};
        if (!std::regex_match(request.id, std::regex("[A-Za-z0-9_-]{1,100}")) ||
            request.category.empty() || request.category.size() > 100 ||
            request.prompt.empty() || request.prompt.size() > 1024 * 1024 ||
            request.tokens < 1 || request.tokens > 8192 ||
            !ids.insert(request.id).second || !prompts.insert(request.prompt).second) {
            throw std::runtime_error("Invalid or duplicate corpus request.");
        }
        if (request.split == "test") {
            testing = true;
        } else if (request.split == "train" && !testing) {
            training = true;
        } else {
            throw std::runtime_error("Corpus requires train records before held-out test records.");
        }
        requests.push_back(std::move(request));
    }
    if (!input.eof() || !training || !testing) {
        throw std::runtime_error("Corpus must contain readable training and test requests.");
    }
    return requests;
}

static std::map<int, Layer> inspect_experts(const llama_model * model) {
    std::map<int, Layer> layers;
    const std::regex pattern(R"(^blk\.([0-9]+)\.ffn_(gate_up|gate|up|down)_exps\.(weight|bias)$)");
    const auto & tensors = llama_internal_get_tensor_map(model);
    for (const auto & [name, tensor] : tensors) {
        std::smatch match;
        if (!std::regex_match(name, match, pattern)) {
            continue;
        }
        const int id = std::stoi(match[1]);
        auto & layer = layers[id];
        const bool weight = match[3] == "weight";
        if (weight) {
            if (tensor->ne[2] < 1 || tensor->ne[3] != 1 ||
                (layer.experts && layer.experts != tensor->ne[2])) {
                throw std::runtime_error("Unsupported expert axis in " + name);
            }
            layer.experts = static_cast<int>(tensor->ne[2]);
            layer.weights.insert(match[2]);
        }
    }
    if (layers.empty()) {
        throw std::runtime_error("No supported routed-expert tensors found; dense models are not traceable.");
    }
    for (auto & [id, layer] : layers) {
        if (!layer.weights.count("down") ||
            !(layer.weights.count("gate_up") ||
              (layer.weights.count("gate") && layer.weights.count("up"))) ||
            (layer.weights.count("gate_up") && (layer.weights.count("gate") || layer.weights.count("up")))) {
            throw std::runtime_error("Incomplete routed-expert tensor group in layer " + std::to_string(id));
        }
        layer.active = static_cast<int>(model->hparams.n_expert_used(id));
        if (layer.experts < 1 || layer.active < 1 || layer.active > layer.experts) {
            throw std::runtime_error("Invalid per-layer expert counts.");
        }
    }
    for (const auto & [name, tensor] : tensors) {
        std::smatch match;
        if (!std::regex_match(name, match, pattern)) {
            continue;
        }
        auto & layer = layers.at(std::stoi(match[1]));
        const bool shape = (tensor->ne[2] == layer.experts && tensor->ne[3] == 1) ||
            (match[3] == "bias" && tensor->ne[1] == layer.experts && tensor->ne[2] == 1 && tensor->ne[3] == 1);
        const uint64_t total = ggml_nbytes(tensor);
        if (!shape || total % layer.experts != 0 || !tensor->buffer) {
            throw std::runtime_error("Unsupported expert storage layout: " + name);
        }
        const uint64_t per_expert = total / layer.experts;
        layer.bytes += per_expert;
        if (ggml_backend_buffer_is_host(tensor->buffer)) {
            layer.host_bytes += per_expert;
        }
        layer.types.insert(ggml_type_name(tensor->type));
    }
    return layers;
}

static void check_counts(const Trace & trace, int expected) {
    if (!trace.error.empty()) {
        throw std::runtime_error(trace.error);
    }
    for (const auto & [id, layer] : trace.layers) {
        const auto found = trace.counts.find(id);
        if (found == trace.counts.end() || found->second != expected) {
            throw std::runtime_error("Missing or duplicate routing observations in layer " + std::to_string(id));
        }
    }
}

static void evaluate_prompt(llama_context * ctx, const std::vector<llama_token> & tokens) {
    auto batch = common_batch_get_one(ctx, tokens);
    const int limit = static_cast<int>(llama_n_batch(ctx));
    if (limit < 1) {
        throw std::runtime_error("Invalid backend batch capacity.");
    }
    for (int offset = 0; offset < batch.size(); offset += limit) {
        const int count = std::min(limit, batch.size() - offset);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(offset, count)) != 0) {
            throw std::runtime_error("Prompt evaluation failed.");
        }
    }
}

static void run_request(Trace & trace, const Request & request, llama_context * ctx,
                        const common_chat_templates * templates) {
    llama_memory_clear(llama_get_memory(ctx), true);
    common_chat_templates_inputs chat;
    common_chat_msg system;
    system.role = "system";
    system.content = "You are a helpful assistant.";
    common_chat_msg user;
    user.role = "user";
    user.content = request.prompt;
    chat.messages = {system, user};
    chat.chat_template_kwargs["reasoning_effort"] = "\"low\"";
    chat.now = std::chrono::system_clock::time_point{};
    const auto rendered = common_chat_templates_apply(templates, chat);
    const auto * model = llama_get_model(ctx);
    const auto * vocab = llama_model_get_vocab(model);
    const auto tokens = common_tokenize(ctx, rendered.prompt.str(), false, true);
    if (tokens.empty() || tokens.size() + request.tokens > llama_n_ctx(ctx)) {
        throw std::runtime_error("Prompt plus requested decode exceeds the actual context.");
    }
    trace.request = request.id;
    trace.counts.clear();
    trace.emit({{"type", "request_start"}, {"id", request.id}, {"split", request.split},
                {"category", request.category}, {"prompt_tokens", tokens.size()},
                {"max_tokens", request.tokens}});
    // Prefill can prune the final layer to output positions; it is not a complete routing trace.
    trace.phase.clear();
    evaluate_prompt(ctx, tokens);
    trace.counts.clear();
    trace.phase = "decode";
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_init_greedy(), llama_sampler_free);
    int decoded = 0;
    std::string stop = "limit";
    std::vector<llama_token> generated;
    for (; decoded < request.tokens; ++decoded) {
        const llama_token token = llama_sampler_sample(sampler.get(), ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            stop = "eos";
            break;
        }
        generated.push_back(token);
        trace.token = decoded;
        const std::vector<llama_token> next{token};
        auto next_batch = common_batch_get_one(ctx, next);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, next_batch.get()) != 0) {
            throw std::runtime_error("Decode failed: " + trace.error);
        }
        check_counts(trace, decoded + 1);
    }
    trace.phase.clear();
    llama_memory_clear(llama_get_memory(ctx), true);
    evaluate_prompt(ctx, tokens);
    sampler.reset(llama_sampler_init_greedy());
    for (const auto expected : generated) {
        const llama_token actual = llama_sampler_sample(sampler.get(), ctx, -1);
        if (actual != expected) {
            throw std::runtime_error("Routing observation changed the greedy token sequence.");
        }
        const std::vector<llama_token> next{actual};
        auto reference_batch = common_batch_get_one(ctx, next);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, reference_batch.get()) != 0) {
            throw std::runtime_error("Reference decode failed.");
        }
    }
    if (stop == "eos" && !llama_vocab_is_eog(vocab, llama_sampler_sample(sampler.get(), ctx, -1))) {
        throw std::runtime_error("Routing observation changed the generation stop.");
    }
    trace.emit({{"type", "request_end"}, {"id", request.id}, {"decode_tokens", decoded},
                {"stop_reason", stop}, {"token_sequence_equivalent", true}});
    if (std::fflush(trace.output.get()) != 0) {
        throw std::runtime_error("Failed to flush routing trace.");
    }
    std::cerr << request.id << ": " << decoded << " routed generated tokens; callback-inactive reference matched "
                 "(not a speed benchmark)\n";
}

static int run(int argc, char ** argv) {
    fs::path corpus;
    fs::path destination;
    std::vector<char *> backend_args{argv[0]};
    bool cpu_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "Usage: vrambridge-trace --trace-corpus corpus.jsonl --trace-output trace.jsonl "
                         "[matching llama.cpp arguments, including --model]\n"
                         "Only the pinned official backend is supported. Collection synchronizes routing tensors; "
                         "its timings are not inference benchmarks. Prompt/output text is not stored.\n";
            return 0;
        }
        if (arg == "--trace-corpus" || arg == "--trace-output") {
            if (++i >= argc) {
                throw std::runtime_error("Missing trace option value.");
            }
            (arg == "--trace-corpus" ? corpus : destination) = argv[i];
        } else {
            if ((arg == "--n-gpu-layers" || arg == "--gpu-layers" || arg == "-ngl") && i + 1 < argc) {
                cpu_only = std::string(argv[i + 1]) == "0";
            }
            backend_args.push_back(argv[i]);
        }
    }
    if (corpus.empty() || destination.empty()) {
        throw std::runtime_error("--trace-corpus and --trace-output are required.");
    }
    const auto requests = read_corpus(corpus);
    const fs::path partial = destination.string() + ".part";
    if (fs::exists(destination) || fs::exists(partial)) {
        throw std::runtime_error("Refusing to overwrite an existing trace or partial trace.");
    }
    const std::string identity = llama_commit();
    if (identity.size() < 7 ||
        std::string("620d2225cc436ab0b001e3200e590e3b286ef9ab").rfind(identity, 0) != 0) {
        throw std::runtime_error("Linked llama.cpp build does not match the supported commit.");
    }
    Trace trace;
    ggml_backend_load_all_from_path(VRAMBRIDGE_LLAMA_LIBRARY_DIR);
    if (ggml_backend_dev_count() == 0) {
        throw std::runtime_error("No backends loaded from the matching library directory.");
    }
    bool usable_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        usable_gpu = usable_gpu || type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                     type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    if (!cpu_only && !usable_gpu) {
        throw std::runtime_error("GPU tracing requested, but no usable GPU device initialized. "
                                 "CPU-only tracing requires explicit --n-gpu-layers 0; no model was loaded.");
    }
    common_init();
    common_params params;
    params.sampling.temp = 0;
    if (!common_params_parse(static_cast<int>(backend_args.size()), backend_args.data(),
                             params, LLAMA_EXAMPLE_COMMON)) {
        throw std::runtime_error("Invalid backend arguments.");
    }
    if (params.model.path.empty() || params.moe_cache_size != 0 ||
        !params.speculative.draft.mparams.path.empty() || !params.lora_adapters.empty() || params.sampling.temp != 0 ||
        params.speculative.types != std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE}) {
        throw std::runtime_error("Supply a local model and greedy sampling; expert caching and speculation must be disabled.");
    }
    if (params.n_gpu_layers != 0 && !llama_supports_gpu_offload()) {
        throw std::runtime_error("GPU placement requested, but no usable GPU backend was loaded.");
    }
    params.warmup = false;
    params.cb_eval = routing_callback;
    params.cb_eval_user_data = &trace;
    llama_backend_init();
    auto initialized = common_init_from_params(params);
    if (!initialized->model() || !initialized->context()) {
        throw std::runtime_error("Failed to initialize tracing model.");
    }
    trace.layers = inspect_experts(initialized->model());
    if (!destination.parent_path().empty()) {
        fs::create_directories(destination.parent_path());
    }
    trace.output.reset(std::fopen(partial.c_str(), "wx"));
    if (!trace.output) {
        throw std::runtime_error("Cannot create partial trace.");
    }
    json layers = json::array();
    uint64_t device_bytes = 0;
    for (const auto & [id, layer] : trace.layers) {
        layers.push_back({{"id", id}, {"experts", layer.experts}, {"active", layer.active},
                          {"bytes_per_expert", layer.bytes}, {"static_host_bytes_per_expert", layer.host_bytes},
                          {"tensor_types", layer.types}});
        device_bytes += (layer.bytes - layer.host_bytes) * layer.experts;
    }
    char arch[128]{};
    if (llama_model_meta_val_str(initialized->model(), "general.architecture", arch, sizeof(arch)) < 0) {
        throw std::runtime_error("Model architecture metadata is missing.");
    }
    trace.emit({{"type", "metadata"}, {"schema_version", 1}, {"instrumented", true},
                {"capture_scope", "decode_only"},
                {"collector_source_sha256", VRAMBRIDGE_TRACE_SOURCE_SHA256},
                {"template_policy", "model Jinja, fixed system message, epoch date, low reasoning effort"},
                {"corpus_unique_prompts", true},
                {"generated_text_recorded", false},
                {"token_equivalence_check", "callback active vs inactive, same prompt and greedy sampling"},
                {"initialization", {{"cpu_threads", params.cpuparams.n_threads},
                                    {"batch_threads", params.cpuparams_batch.n_threads},
                                    {"batch_size", params.n_batch}, {"microbatch_size", params.n_ubatch},
                                    {"cache_type_k", ggml_type_name(params.cache_type_k)},
                                    {"cache_type_v", ggml_type_name(params.cache_type_v)},
                                    {"load_mode_enum", static_cast<int>(params.load_mode)},
                                    {"op_offload", !params.no_op_offload}}},
                {"backend_commit", llama_commit()}, {"architecture", arch},
                {"model", fs::path(params.model.path).filename().string()},
                {"context_tokens", llama_n_ctx(initialized->context())}, {"sampling", "greedy"},
                {"static_device_expert_bytes", device_bytes}, {"layers", layers}});
    auto templates = common_chat_templates_init(initialized->model(), params.chat_template);
    if (!templates) {
        throw std::runtime_error("Cannot initialize model chat template.");
    }
    for (const auto & request : requests) {
        run_request(trace, request, initialized->context(), templates.get());
    }
    trace.emit({{"type", "trace_end"}, {"requests", requests.size()}, {"complete", true}});
    if (std::fclose(trace.output.release()) != 0) {
        throw std::runtime_error("Failed to close routing trace.");
    }
    fs::create_hard_link(partial, destination);
    fs::remove(partial);
    initialized.reset();
    llama_backend_free();
    std::cout << destination.string() << '\n';
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & exc) {
        std::cerr << "vrambridge-trace: " << exc.what() << '\n';
        return 2;
    }
}
