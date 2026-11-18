#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string cpu_backend;
    std::string gpu_backend;
    std::string prompt;
    int context = 2048;
    int batch = 512;
    int threads = 4;
    int gpu_layers = 0;
    int max_tokens = 32;
    bool token_ids = false;
    bool trace_logits = false;
};

int number(const std::string & value, int min, int max) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < min || result > max) {
        throw std::runtime_error("invalid numeric option: " + value);
    }
    return result;
}

Options parse(int argc, char ** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--token-ids") { result.token_ids = true; continue; }
        if (key == "--trace-logits") { result.trace_logits = true; continue; }
        if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--model") result.model = value;
        else if (key == "--cpu-backend") result.cpu_backend = value;
        else if (key == "--gpu-backend") result.gpu_backend = value;
        else if (key == "--prompt") result.prompt = value;
        else if (key == "--context") result.context = number(value, 32, 131072);
        else if (key == "--batch") result.batch = number(value, 1, 2048);
        else if (key == "--threads") result.threads = number(value, 1, 256);
        else if (key == "--gpu-layers") result.gpu_layers = number(value, 0, 1000);
        else if (key == "--max-tokens") result.max_tokens = number(value, 1, 4096);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (result.model.empty() || result.cpu_backend.empty() || result.prompt.empty())
        throw std::runtime_error("--model, --cpu-backend and --prompt are required");
    if (result.prompt.size() > 1048576) throw std::runtime_error("prompt exceeds 1 MiB");
    if (result.gpu_layers > 0 && result.gpu_backend.empty())
        throw std::runtime_error("GPU layers require an explicit --gpu-backend");
    if (result.gpu_layers == 0 && !result.gpu_backend.empty())
        throw std::runtime_error("--gpu-backend requires --gpu-layers > 0");
    if (result.token_ids && result.trace_logits)
        throw std::runtime_error("choose --token-ids or --trace-logits, not both");
    return result;
}

struct Batch {
    llama_batch value;
    explicit Batch(int count) : value(llama_batch_init(count, 0, 1)) {
        if (!value.token || !value.pos || !value.n_seq_id || !value.seq_id || !value.logits) {
            llama_batch_free(value);
            throw std::runtime_error("batch allocation failed");
        }
        value.n_tokens = 0;
    }
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
    void add(llama_token token, int position, bool logits) {
        const int index = value.n_tokens++;
        value.token[index] = token;
        value.pos[index] = position;
        value.n_seq_id[index] = 1;
        value.seq_id[index][0] = 0;
        value.logits[index] = logits;
    }
};

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

void decode(llama_context * context, const Batch & batch) {
    const int status = llama_decode(context, batch.value);
    if (status != 0) throw std::runtime_error("decode failed: " + std::to_string(status));
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & prompt) {
    const int needed = llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), nullptr, 0, true, false);
    if (needed >= 0 || needed == std::numeric_limits<int>::min()) throw std::runtime_error("tokenization failed");
    std::vector<llama_token> tokens(static_cast<size_t>(-needed));
    const int count = llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), tokens.data(),
                                     static_cast<int>(tokens.size()), true, false);
    if (count != static_cast<int>(tokens.size())) throw std::runtime_error("tokenization returned an unexpected count");
    return tokens;
}

void write_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> bytes(64);
    int count = llama_token_to_piece(vocab, token, bytes.data(), static_cast<int>(bytes.size()), 0, false);
    if (count < 0) {
        if (count == std::numeric_limits<int>::min()) throw std::runtime_error("token piece overflow");
        bytes.resize(static_cast<size_t>(-count));
        count = llama_token_to_piece(vocab, token, bytes.data(), static_cast<int>(bytes.size()), 0, false);
    }
    if (count < 0 || static_cast<size_t>(count) > bytes.size() ||
        std::fwrite(bytes.data(), 1, static_cast<size_t>(count), stdout) != static_cast<size_t>(count))
        throw std::runtime_error("cannot write generated token");
}

int run(const Options & options) {
    if (!ggml_backend_load(options.cpu_backend.c_str())) throw std::runtime_error("CPU backend could not be loaded");
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))
        throw std::runtime_error("CPU backend did not register a CPU device");
    if (options.gpu_layers && !ggml_backend_load(options.gpu_backend.c_str()))
        throw std::runtime_error("GPU backend could not be loaded");
    Backend backend;
    ggml_backend_dev_t cpu_devices[] = {nullptr};
    ggml_backend_dev_t gpu_devices[] = {nullptr, nullptr};
    auto model_params = llama_model_default_params();
    if (options.gpu_layers) {
        gpu_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!gpu_devices[0] || !llama_supports_gpu_offload()) throw std::runtime_error("GPU backend unavailable");
        model_params.devices = gpu_devices;
    } else {
        model_params.devices = cpu_devices;
    }
    model_params.n_gpu_layers = options.gpu_layers;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(options.model.c_str(), model_params), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    if (!vocab) throw std::runtime_error("model has no vocabulary");
    const auto tokens = tokenize(vocab, options.prompt);
    if (tokens.size() + static_cast<size_t>(options.max_tokens) > static_cast<size_t>(options.context))
        throw std::runtime_error("prompt and output budget exceed the requested context");
    auto params = llama_context_default_params();
    params.n_ctx = options.context;
    params.n_batch = options.batch;
    params.n_ubatch = options.batch;
    params.n_seq_max = 1;
    params.n_threads = options.threads;
    params.n_threads_batch = options.threads;
    params.offload_kqv = options.gpu_layers > 0;
    params.op_offload = options.gpu_layers > 0;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), params), llama_free);
    if (!context || llama_n_ctx_seq(context.get()) < static_cast<uint32_t>(options.context))
        throw std::runtime_error("context creation failed or yielded a smaller context");
    const int chunk = std::min(options.batch, static_cast<int>(llama_n_batch(context.get())));
    if (chunk < 1) throw std::runtime_error("backend returned an invalid batch size");
    for (size_t start = 0; start < tokens.size(); start += static_cast<size_t>(chunk)) {
        const int count = std::min(chunk, static_cast<int>(tokens.size() - start));
        Batch batch(count);
        for (int i = 0; i < count; ++i) {
            const size_t index = start + static_cast<size_t>(i);
            batch.add(tokens[index], static_cast<int>(index), index + 1 == tokens.size());
        }
        decode(context.get(), batch);
    }
    const int vocab_size = llama_vocab_n_tokens(vocab);
    if (vocab_size <= 0) throw std::runtime_error("invalid vocabulary size");
    for (int step = 0; step < options.max_tokens; ++step) {
        const float * logits = llama_get_logits_ith(context.get(), -1);
        if (!logits) throw std::runtime_error("backend returned no logits");
        for (int i = 0; i < vocab_size; ++i)
            if (!std::isfinite(logits[i])) throw std::runtime_error("backend returned non-finite logits");
        const llama_token token = static_cast<llama_token>(std::max_element(logits, logits + vocab_size) - logits);
        if (options.trace_logits) {
            std::printf("{\"token\":%d,\"logits\":[", token);
            for (int i = 0; i < vocab_size; ++i) std::printf(i ? ",%.9g" : "%.9g", static_cast<double>(logits[i]));
            std::puts("]}");
        } else if (options.token_ids) {
            std::printf("{\"token\":%d}\n", token);
        } else if (!llama_vocab_is_eog(vocab, token)) {
            write_piece(vocab, token);
        }
        if (std::fflush(stdout) != 0) throw std::runtime_error("cannot flush output");
        if (llama_vocab_is_eog(vocab, token) || step + 1 == options.max_tokens) break;
        Batch batch(1);
        batch.add(token, static_cast<int>(tokens.size()) + step, true);
        decode(context.get(), batch);
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::puts("Experimental single-sequence inference control; requires pinned llama.cpp libraries.\n"
                  "--model PATH --cpu-backend PATH --prompt TEXT [--max-tokens N] [--context N] [--batch N]\n"
                  "[--threads N] [--gpu-backend PATH --gpu-layers N] [--token-ids | --trace-logits]");
        return 0;
    }
    try { return run(parse(argc, argv)); }
    catch (const std::exception & error) {
        std::fprintf(stderr, "vrambridge-engine: %s\n", error.what());
        return 2;
    }
}
