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

struct Reference {
    std::vector<llama_token> prompt;
    std::vector<llama_token> next;
    std::vector<float> prefill;
    std::vector<std::vector<float>> decode;
};

constexpr int kSteps = 32;

void add(llama_batch & batch, llama_token token, int position, int sequence, bool output) {
    const int i = batch.n_tokens++;
    batch.token[i] = token;
    batch.pos[i] = position;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = sequence;
    batch.logits[i] = output;
}

std::vector<float> prefill(llama_context * context, const std::vector<llama_token> & tokens, int sequence, int vocab) {
    Batch batch(static_cast<int>(tokens.size()));
    for (int pos = 0; pos < static_cast<int>(tokens.size()); ++pos) {
        add(batch.value, tokens[pos], pos, sequence, pos == static_cast<int>(tokens.size()) - 1);
    }
    require(llama_decode(context, batch.value) == 0, "prefill failed");
    const auto * logits = llama_get_logits_ith(context, static_cast<int>(tokens.size()) - 1);
    require(logits != nullptr, "missing prefill logits");
    return {logits, logits + vocab};
}

std::vector<std::vector<float>> decode(
        llama_context * context, const std::vector<Reference> & references, int sequences, int step, int vocab) {
    Batch batch(sequences);
    for (int seq = 0; seq < sequences; ++seq) {
        add(batch.value, references[seq].next.at(step), static_cast<int>(references[seq].prompt.size()) + step, seq, true);
    }
    require(llama_decode(context, batch.value) == 0, "decode failed");
    std::vector<std::vector<float>> result;
    for (int seq = 0; seq < sequences; ++seq) {
        const auto * logits = llama_get_logits_ith(context, seq);
        require(logits != nullptr, "missing decode logits");
        result.emplace_back(logits, logits + vocab);
    }
    return result;
}

std::vector<float> decode_single(llama_context * context, const Reference & reference, int step, int vocab) {
    Batch batch(1);
    add(batch.value, reference.next.at(step), static_cast<int>(reference.prompt.size()) + step, 0, true);
    require(llama_decode(context, batch.value) == 0, "single decode failed");
    const auto * logits = llama_get_logits_ith(context, 0);
    require(logits != nullptr, "missing single decode logits");
    return {logits, logits + vocab};
}

bool compare(const char * name, int sequences, int sequence, int step,
             const std::vector<float> & reference, const std::vector<float> & candidate) {
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
    std::printf("{\"case\":\"%s\",\"sequences\":%d,\"sequence\":%d,\"step\":%d,\"max_absolute\":%.9g,"
                "\"relative_l2\":%.9g,\"passes\":%s,\"top_equal\":%s}\n",
                name, sequences, sequence, step, maximum, relative, pass ? "true" : "false", top_equal ? "true" : "false");
    return pass;
}

void save_reference(const char * path, const std::vector<Reference> & refs, int vocab, bool gpu) {
    File file(std::fopen(path, "wbx"));
    require(file != nullptr, "cannot create reference file");
    const uint32_t header[4] = {static_cast<uint32_t>(refs.size()), static_cast<uint32_t>(vocab), kSteps, gpu ? 1u : 0u};
    require(std::fwrite(header, sizeof(uint32_t), 4, file.get()) == 4, "cannot write reference header");
    for (const auto & ref : refs) {
        const uint32_t count = static_cast<uint32_t>(ref.prompt.size());
        require(std::fwrite(&count, sizeof(count), 1, file.get()) == 1, "cannot write token count");
        require(std::fwrite(ref.prompt.data(), sizeof(llama_token), count, file.get()) == count, "cannot write prompt");
        require(std::fwrite(ref.prefill.data(), sizeof(float), vocab, file.get()) == static_cast<size_t>(vocab),
                "cannot write prefill logits");
        require(ref.next.size() == kSteps && ref.decode.size() == kSteps, "reference steps incomplete");
        for (int step = 0; step < kSteps; ++step) {
            require(std::fwrite(&ref.next[step], sizeof(llama_token), 1, file.get()) == 1, "cannot write next token");
            require(std::fwrite(ref.decode[step].data(), sizeof(float), vocab, file.get()) == static_cast<size_t>(vocab),
                    "cannot write decode logits");
        }
    }
    require(std::fflush(file.get()) == 0, "cannot flush reference file");
}

void load_reference(const char * path, std::vector<Reference> & refs, int vocab, bool gpu) {
    File file(std::fopen(path, "rb"));
    require(file != nullptr, "cannot open reference file");
    uint32_t header[4];
    require(std::fread(header, sizeof(uint32_t), 4, file.get()) == 4 &&
            header[0] == refs.size() && header[1] == static_cast<uint32_t>(vocab) &&
            header[2] == kSteps && header[3] == (gpu ? 1u : 0u),
            "reference dimensions differ");
    for (auto & ref : refs) {
        uint32_t count;
        require(std::fread(&count, sizeof(count), 1, file.get()) == 1 && count == ref.prompt.size(),
                "reference prompt length differs");
        std::vector<llama_token> tokens(count);
        require(std::fread(tokens.data(), sizeof(llama_token), count, file.get()) == count && tokens == ref.prompt,
                "reference tokens differ");
        ref.prefill.resize(vocab);
        require(std::fread(ref.prefill.data(), sizeof(float), vocab, file.get()) == static_cast<size_t>(vocab),
                "reference prefill logits missing");
        for (int step = 0; step < kSteps; ++step) {
            llama_token next;
            require(std::fread(&next, sizeof(next), 1, file.get()) == 1 && next >= 0 && next < vocab,
                    "reference next token is invalid");
            ref.next.push_back(next);
            ref.decode.emplace_back(vocab);
            require(std::fread(ref.decode.back().data(), sizeof(float), vocab, file.get()) == static_cast<size_t>(vocab),
                    "reference decode logits missing");
        }
    }
    require(std::fgetc(file.get()) == EOF, "reference has trailing data");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        const std::vector<float> reference{1.0f, -2.0f};
        const std::vector<float> changed{1.02f, -2.0f};
        return compare("self_equal", 1, 0, 0, reference, reference) &&
               !compare("self_changed", 1, 0, 0, reference, changed) ? 0 : 1;
    }
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "Usage: %s MODEL write|read PRIVATE_REFERENCE [gpu-cpu-experts] | --self-test\n", argv[0]);
        return 2;
    }
    try {
#if !defined(_WIN32)
        umask(0077);
#endif
        require(std::strcmp(argv[2], "write") == 0 || std::strcmp(argv[2], "read") == 0, "invalid reference mode");
        const bool gpu = argc == 5 && std::strcmp(argv[4], "gpu-cpu-experts") == 0;
        require(argc == 4 || gpu, "invalid placement mode");
        llama_backend_init();
        ggml_backend_dev_t cpu_devices[] = {nullptr};
        ggml_backend_dev_t gpu_devices[] = {nullptr, nullptr};
        llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()},
            {nullptr, nullptr}
        };
        auto model_params = llama_model_default_params();
        if (gpu) {
            gpu_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            require(gpu_devices[0] != nullptr && llama_supports_gpu_offload() &&
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
        params.n_ctx = 131072;
        params.n_seq_max = 8;
        params.n_batch = 2048;
        params.n_ubatch = 512;
        params.n_threads = 8;
        params.n_threads_batch = 16;
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
        params.offload_kqv = gpu;
        params.op_offload = gpu;
        params.kv_unified = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), params), llama_free);
        require(context != nullptr && llama_n_ctx_seq(context.get()) == 16384, "unexpected sequence context");
        const auto * vocab = llama_model_get_vocab(model.get());
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const std::vector<std::string> prompts = {
            "Agent: Review a bounded work queue.",
            "Agent: Find a cancellation race.",
            "Agent: Explain how a slot keeps state.",
            "Agent: Propose tests for worker errors.",
            "Agent: Describe GPU memory pressure.",
            "Agent: Compare two queue designs.",
            "Agent: Explain why prompts should not leak.",
            "Agent: Summarize a safe shutdown strategy."
        };
        std::vector<Reference> refs;
        for (const auto & prompt : prompts) {
            const int count = -llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()), nullptr, 0, true, true);
            require(count >= 3 && count <= 128, "unexpected prompt length");
            Reference ref;
            ref.prompt.resize(count);
            require(llama_tokenize(vocab, prompt.c_str(), static_cast<int>(prompt.size()), ref.prompt.data(), count, true, true) == count,
                    "tokenization failed");
            refs.push_back(std::move(ref));
        }
        if (std::strcmp(argv[2], "write") == 0) {
            for (auto & ref : refs) {
                llama_memory_clear(llama_get_memory(context.get()), true);
                ref.prefill = prefill(context.get(), ref.prompt, 0, n_vocab);
                const std::vector<float> * previous = &ref.prefill;
                for (int step = 0; step < kSteps; ++step) {
                    ref.next.push_back(static_cast<llama_token>(
                        std::max_element(previous->begin(), previous->end()) - previous->begin()));
                    ref.decode.push_back(decode_single(context.get(), ref, step, n_vocab));
                    previous = &ref.decode.back();
                }
            }
            save_reference(argv[3], refs, n_vocab, gpu);
        } else {
            load_reference(argv[3], refs, n_vocab, gpu);
        }
        std::printf("{\"prompts\":%zu,\"steps\":%d,\"vocabulary\":%d,\"context_per_sequence\":%u,\"placement\":\"%s\"}\n",
                    refs.size(), kSteps, n_vocab, llama_n_ctx_seq(context.get()), gpu ? "gpu-cpu-experts" : "cpu");
        bool passed = true;
        const char * all_widths = std::getenv("VRAMBRIDGE_PROBE_ALL_WIDTHS");
        require(!all_widths || std::strcmp(all_widths, "0") == 0 || std::strcmp(all_widths, "1") == 0,
                "VRAMBRIDGE_PROBE_ALL_WIDTHS must be 0 or 1");
        const std::vector<int> widths = all_widths && std::strcmp(all_widths, "1") == 0
            ? std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8} : std::vector<int>{2, 4, 8};
        for (int sequences : widths) {
            llama_memory_clear(llama_get_memory(context.get()), true);
            for (int seq = 0; seq < sequences; ++seq) {
                const auto actual = prefill(context.get(), refs[seq].prompt, seq, n_vocab);
                passed = compare("serial_prefill", sequences, seq, -1, refs[seq].prefill, actual) && passed;
            }
            for (int step = 0; step < kSteps; ++step) {
                const auto outputs = decode(context.get(), refs, sequences, step, n_vocab);
                for (int seq = 0; seq < sequences; ++seq) {
                    passed = compare("batched_decode", sequences, seq, step, refs[seq].decode[step], outputs[seq]) && passed;
                }
            }
        }
        context.reset();
        model.reset();
        llama_backend_free();
        return passed ? 0 : 2;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "probe failed: %s\n", exc.what());
        return 1;
    }
}
