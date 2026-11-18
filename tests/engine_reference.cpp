#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6 || !ggml_backend_load(argv[2])) return 2;
    llama_backend_init();
    int result = 2;
    {
        ggml_backend_dev_t cpu_devices[] = {nullptr};
        auto mp = llama_model_default_params();
        mp.devices = cpu_devices;
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) return 2;
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048;
        const int chunk = std::atoi(argv[5]);
        if (chunk <= 0) return 2;
        cp.n_batch = chunk;
        cp.n_ubatch = chunk;
        cp.n_seq_max = 1;
        cp.n_threads = 2;
        cp.n_threads_batch = 2;
        cp.offload_kqv = false;
        cp.op_offload = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) return 2;
        const auto * vocab = llama_model_get_vocab(model.get());
        const std::string prompt = argv[3];
        const int n = -llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), nullptr, 0, true, false);
        if (n <= 0) return 2;
        std::vector<llama_token> prompt_tokens(static_cast<size_t>(n));
        if (llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), prompt_tokens.data(), n, true, false) != n) return 2;
        for (int start = 0; start < n; start += chunk) {
            const int count = std::min(chunk, n - start);
            if (llama_decode(ctx.get(), llama_batch_get_one(prompt_tokens.data() + start, count)) != 0) return 2;
        }
        const int size = llama_vocab_n_tokens(vocab);
        for (int step = 0; step < std::atoi(argv[4]); ++step) {
            const auto * logits = llama_get_logits_ith(ctx.get(), -1);
            if (!logits || size <= 0) return 2;
            llama_token token = static_cast<llama_token>(std::max_element(logits, logits + size) - logits);
            std::printf("{\"token\":%d,\"logits\":[", token);
            for (int i = 0; i < size; ++i) std::printf(i ? ",%.9g" : "%.9g", static_cast<double>(logits[i]));
            std::puts("]}");
            if (llama_vocab_is_eog(vocab, token) || step + 1 == std::atoi(argv[4])) break;
            if (llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) != 0) return 2;
        }
        result = 0;
    }
    llama_backend_free();
    return result;
}
