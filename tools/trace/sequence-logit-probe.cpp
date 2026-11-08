#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct Batch {
    llama_batch value;
    explicit Batch(int size) : value(llama_batch_init(size, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
};

struct CloseFile {
    void operator()(FILE * file) const noexcept { std::fclose(file); }
};
using File = std::unique_ptr<FILE, CloseFile>;

void add(llama_batch & batch, llama_token token, int position, int sequence, bool output) {
    const int i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = position;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = sequence;
    batch.logits[i] = output;
}

std::vector<std::vector<float>> evaluate(
        llama_context * context, const std::vector<llama_token> & tokens, int sequences, bool interleaved, int vocab,
        bool all_outputs = false) {
    llama_memory_clear(llama_get_memory(context), true);
    Batch batch(static_cast<int>(tokens.size()) * sequences);
    std::vector<int> outputs(sequences);
    const auto append = [&](int seq, int pos) {
        const bool output = pos == static_cast<int>(tokens.size()) - 1;
        if (output) outputs[seq] = batch.value.n_tokens;
        add(batch.value, tokens[pos], pos, seq, output || all_outputs);
    };
    if (interleaved) {
        for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos)
            for (int seq = 0; seq < sequences; ++seq) append(seq, pos);
    } else {
        for (int seq = 0; seq < sequences; ++seq)
            for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) append(seq, pos);
    }
    require(llama_decode(context, batch.value) == 0, "llama_decode failed");
    std::vector<std::vector<float>> result;
    for (int seq = 0; seq < sequences; ++seq) {
        const auto * logits = llama_get_logits_ith(context, outputs[seq]);
        require(logits != nullptr, "missing output logits");
        result.emplace_back(logits, logits + vocab);
    }
    return result;
}

std::vector<float> evaluate_chunked(
        llama_context * context, const std::vector<llama_token> & tokens, int split, int vocab,
        bool all_outputs = false) {
    require(split > 0 && split < static_cast<int>(tokens.size()), "invalid split");
    llama_memory_clear(llama_get_memory(context), true);
    std::vector<float> result;
    for (int part = 0; part < 2; ++part) {
        const int begin = part == 0 ? 0 : split;
        const int end = part == 0 ? split : static_cast<int>(tokens.size());
        Batch batch(end - begin);
        for (int pos = begin; pos < end; ++pos) {
            add(batch.value, tokens[pos], pos, 0, all_outputs || (part == 1 && pos == end - 1));
        }
        require(llama_decode(context, batch.value) == 0, "chunked llama_decode failed");
        if (part == 1) {
            const auto * logits = llama_get_logits_ith(context, end - begin - 1);
            require(logits != nullptr, "missing chunked logits");
            result.assign(logits, logits + vocab);
        }
    }
    return result;
}

std::vector<float> decode_one(llama_context * context, llama_token token, int position, int vocab) {
    Batch batch(1);
    add(batch.value, token, position, 0, true);
    require(llama_decode(context, batch.value) == 0, "continuation decode failed");
    const auto * logits = llama_get_logits_ith(context, 0);
    require(logits != nullptr, "missing continuation logits");
    return {logits, logits + vocab};
}

bool compare(const char * name, int sequence, const std::vector<float> & reference, const std::vector<float> & candidate) {
    require(reference.size() == candidate.size(), "vocabulary differs");
    double maximum = 0.0, sum = 0.0, norm = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(candidate[i]), "nonfinite logits");
        const double delta = double(reference[i]) - candidate[i];
        maximum = std::max(maximum, std::abs(delta));
        sum += delta * delta;
        norm += double(reference[i]) * reference[i];
    }
    const double relative = std::sqrt(sum / std::max(norm, 1e-30));
    const bool pass = maximum <= 0.01 && relative <= 0.001;
    const bool top_equal =
        (std::max_element(reference.begin(), reference.end()) - reference.begin()) ==
        (std::max_element(candidate.begin(), candidate.end()) - candidate.begin());
    std::printf("{\"case\":\"%s\",\"sequence\":%d,\"max_absolute\":%.9g,"
                "\"relative_l2\":%.9g,\"passes\":%s,\"top_equal\":%s}\n",
                name, sequence, maximum, relative, pass ? "true" : "false", top_equal ? "true" : "false");
    return pass;
}

std::vector<float> reference_file(const char * path, const char * mode,
                                  const std::vector<llama_token> & tokens, const std::vector<float> & logits) {
    require(std::strcmp(mode, "write") == 0 || std::strcmp(mode, "read") == 0, "expected reference mode write or read");
    File file(std::fopen(path, std::strcmp(mode, "write") == 0 ? "wbx" : "rb"));
    require(file != nullptr, "cannot open reference file");
    const uint32_t header[2] = {static_cast<uint32_t>(tokens.size()), static_cast<uint32_t>(logits.size())};
    if (std::strcmp(mode, "write") == 0) {
        require(std::fwrite(header, sizeof(uint32_t), 2, file.get()) == 2, "cannot write reference header");
        require(std::fwrite(tokens.data(), sizeof(llama_token), tokens.size(), file.get()) == tokens.size(),
                "cannot write reference tokens");
        require(std::fwrite(logits.data(), sizeof(float), logits.size(), file.get()) == logits.size(),
                "cannot write reference logits");
        require(std::fflush(file.get()) == 0, "cannot flush reference file");
        return logits;
    }
    uint32_t stored[2];
    require(std::fread(stored, sizeof(uint32_t), 2, file.get()) == 2 &&
            stored[0] == header[0] && stored[1] == header[1], "reference dimensions differ");
    std::vector<llama_token> stored_tokens(tokens.size());
    require(std::fread(stored_tokens.data(), sizeof(llama_token), stored_tokens.size(), file.get()) == stored_tokens.size() &&
            stored_tokens == tokens, "reference tokens differ");
    std::vector<float> result(logits.size());
    require(std::fread(result.data(), sizeof(float), result.size(), file.get()) == result.size() &&
            std::fgetc(file.get()) == EOF, "reference logits missing or malformed");
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        const std::vector<float> reference{1.0f, -2.0f};
        const std::vector<float> changed{1.02f, -2.0f};
        return compare("self_equal", 0, reference, reference) &&
               !compare("self_changed", 0, reference, changed) ? 0 : 1;
    }
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "Usage: %s MODEL write|read PRIVATE_REFERENCE [gpu-cpu-experts] | --self-test\n", argv[0]);
        return 2;
    }
    try {
#if !defined(_WIN32)
        umask(0077);
#endif
        llama_backend_init();
        const bool gpu = argc == 5 && std::strcmp(argv[4], "gpu-cpu-experts") == 0;
        require(argc == 4 || gpu, "invalid placement mode");
        ggml_backend_dev_t cpu_devices[] = {nullptr};
        ggml_backend_dev_t gpu_devices[] = {nullptr, nullptr};
        llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr}
        };
        auto model_params = llama_model_default_params();
        if (gpu) {
            gpu_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            require(gpu_devices[0] && llama_supports_gpu_offload() &&
                    std::strncmp(ggml_backend_dev_name(gpu_devices[0]), "CUDA", 4) == 0,
                    "CUDA GPU backend unavailable");
            model_params.devices = gpu_devices;
            model_params.n_gpu_layers = -1;
            model_params.tensor_buft_overrides = overrides;
        } else {
            model_params.devices = cpu_devices;
            model_params.n_gpu_layers = 0;
        }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        require(model != nullptr, "model load failed");
        auto params = llama_context_default_params();
        params.n_ctx = gpu ? 16384 : 131072;
        params.n_seq_max = gpu ? 1 : 8;
        params.n_batch = 2048;
        params.n_ubatch = 512;
        if (const char * size = std::getenv("VRAMBRIDGE_PROBE_UBATCH")) {
            require(std::strcmp(size, "1") == 0 || std::strcmp(size, "512") == 0,
                    "VRAMBRIDGE_PROBE_UBATCH must be 1 or 512");
            params.n_ubatch = std::strcmp(size, "1") == 0 ? 1 : 512;
        }
        params.n_threads = 8;
        params.n_threads_batch = 16;
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
        params.offload_kqv = gpu;
        params.op_offload = gpu;
        params.kv_unified = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), params), llama_free);
        require(context != nullptr, "context creation failed");
        require(llama_n_ctx_seq(context.get()) == 16384, "unexpected sequence context");
        auto * vocab = llama_model_get_vocab(model.get());
        int repetitions = 1;
        if (const char * requested = std::getenv("VRAMBRIDGE_PROBE_PROMPT_REPEAT")) {
            char * end = nullptr;
            const long count = std::strtol(requested, &end, 10);
            require(end != requested && *end == '\0' && count >= 1 && count <= 64,
                    "VRAMBRIDGE_PROBE_PROMPT_REPEAT must be between 1 and 64");
            repetitions = static_cast<int>(count);
        }
        std::string prompt;
        for (int i = 0; i < repetitions; ++i) {
            if (i) prompt += '\n';
            prompt += "Agent: Design a safe, bounded queue for independent model requests.";
        }
        int size = -llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()), nullptr, 0, true, true);
        require(size >= 3 && size <= 2048, "unexpected prompt length");
        std::vector<llama_token> tokens(size);
        require(llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()), tokens.data(), size, true, true) == size,
                "tokenization failed");
        const int vocabulary = llama_vocab_n_tokens(vocab);
        const char * outputs_env = std::getenv("VRAMBRIDGE_PROBE_ALL_OUTPUTS");
        require(!outputs_env || std::strcmp(outputs_env, "0") == 0 || std::strcmp(outputs_env, "1") == 0,
                "VRAMBRIDGE_PROBE_ALL_OUTPUTS must be 0 or 1");
        const bool all_outputs = outputs_env && std::strcmp(outputs_env, "1") == 0;
        const auto current = evaluate(context.get(), tokens, 1, false, vocabulary, all_outputs)[0];
        const auto reference = reference_file(argv[3], argv[2], tokens, current);
        std::printf("{\"prompt_tokens\":%d,\"vocabulary\":%d,\"context_per_sequence\":%u}\n",
                    size, vocabulary, llama_n_ctx_seq(context.get()));
        bool passed = compare("single", 0, reference, current);
        passed = compare("single_repeat", 0, reference,
                         evaluate(context.get(), tokens, 1, false, vocabulary, all_outputs)[0]) && passed;
        for (int split : {1, size / 2, size - 1}) {
            passed = compare("chunked", split, reference,
                             evaluate_chunked(context.get(), tokens, split, vocabulary, all_outputs)) && passed;
        }
        int continuation_steps = 0;
        if (const char * requested = std::getenv("VRAMBRIDGE_PROBE_CONTINUATION_STEPS")) {
            char * end = nullptr;
            const long count = std::strtol(requested, &end, 10);
            require(end != requested && *end == '\0' && count >= 0 && count <= 128,
                    "VRAMBRIDGE_PROBE_CONTINUATION_STEPS must be between 0 and 128");
            continuation_steps = static_cast<int>(count);
        }
        if (continuation_steps) {
            auto last = evaluate(context.get(), tokens, 1, false, vocabulary, all_outputs)[0];
            std::vector<llama_token> next;
            std::vector<std::vector<float>> expected;
            for (int step = 0; step < continuation_steps; ++step) {
                next.push_back(static_cast<llama_token>(std::max_element(last.begin(), last.end()) - last.begin()));
                last = decode_one(context.get(), next.back(), size + step, vocabulary);
                expected.push_back(last);
            }
            for (int split : {1, size / 2, size - 1}) {
                evaluate_chunked(context.get(), tokens, split, vocabulary, all_outputs);
                for (int step = 0; step < continuation_steps; ++step) {
                    const auto actual = decode_one(context.get(), next[step], size + step, vocabulary);
                    passed = compare("chunked_continuation", split * 1000 + step, expected[step], actual) && passed;
                }
            }
        }
        if (gpu) {
            context.reset();
            model.reset();
            llama_backend_free();
            return passed ? 0 : 2;
        }
        for (int sequences : {2, 4, 8}) {
            const auto outputs = evaluate(context.get(), tokens, sequences, false, vocabulary);
            for (int seq = 0; seq < sequences; ++seq) {
                passed = compare("sequence_major", sequences * 10 + seq, reference, outputs[seq]) && passed;
            }
        }
        const auto outputs = evaluate(context.get(), tokens, 8, true, vocabulary);
        for (int seq = 0; seq < 8; ++seq) passed = compare("interleaved_8", seq, reference, outputs[seq]) && passed;
        context.reset();
        model.reset();
        llama_backend_free();
        return passed ? 0 : 2;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "probe failed: %s\n", exc.what());
        return 1;
    }
}
