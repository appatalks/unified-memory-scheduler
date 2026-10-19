#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;

static void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        fs::path model_path, corpus, output;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::puts("Usage: vrambridge-render-corpus --model first.gguf --corpus corpus.jsonl --output /private/tokens.json");
                return 0;
            }
            require(i + 1 < argc, "Missing argument value");
            const std::string value = argv[++i];
            if (option == "--model") model_path = value;
            else if (option == "--corpus") corpus = value;
            else if (option == "--output") output = value;
            else throw std::runtime_error("Unknown option: " + option);
        }
        require(!model_path.empty() && !corpus.empty() && !output.empty(), "All paths are required");
        require(fs::is_regular_file(corpus) && fs::file_size(corpus) <= 2 * 1024 * 1024, "Corpus missing or over2MiB");
        struct stat info{};
        require(!output.parent_path().empty() && stat(output.parent_path().c_str(), &info) == 0 &&
                S_ISDIR(info.st_mode) && info.st_uid == getuid() && !(info.st_mode & 077),
                "Output parent must be owned and private");
        require(!fs::exists(output), "Refusing to overwrite token corpus");
        llama_backend_init();
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path.c_str(), params), llama_model_free);
        require(bool(model), "Cannot load original GGUF vocabulary");
        const auto * vocab = llama_model_get_vocab(model.get());
        auto templates = common_chat_templates_init(model.get(), "");
        json records = json::array();
        std::ifstream input(corpus);
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            require(records.size() < 32 && line.size() <= 512 * 1024, "Corpus record bound exceeded");
            auto request = json::parse(line);
            require(request.at("prompt").is_string(), "Prompt must be text");
            common_chat_templates_inputs chat;
            common_chat_msg system, user;
            system.role = "system";
            system.content = "You are a helpful assistant.";
            user.role = "user";
            user.content = request.at("prompt").get<std::string>();
            chat.messages = {system, user};
            chat.now = std::chrono::system_clock::time_point{std::chrono::seconds{0}};
            chat.chat_template_kwargs["reasoning_effort"] = "\"low\"";
            chat.chat_template_kwargs["enable_thinking"] = "true";
            chat.chat_template_kwargs["preserve_thinking"] = "true";
            const auto rendered = common_chat_templates_apply(templates.get(), chat);
            const auto tokens = common_tokenize(vocab, rendered.prompt.str(), false, true);
            require(!tokens.empty() && tokens.size() <= 12000, "Prompt token count outside study bound");
            records.push_back({{"id", request.at("id")}, {"split", request.at("split")},
                {"category", request.at("category")}, {"rendered_prompt", rendered.prompt.str()}, {"tokens", tokens}});
        }
        require(input.eof() && !records.empty(), "Cannot read complete nonempty corpus");
        const json result = {{"schema_version", 1}, {"kind", "private_shared_token_corpus"},
            {"contains_token_ids_and_prompts", true}, {"model", model_path.filename().string()},
            {"backend_commit", llama_commit()}, {"template_epoch_seconds", 0}, {"reasoning_effort", "low"},
            {"records", records}};
        auto * file = std::fopen(output.c_str(), "wx");
        require(file != nullptr && fchmod(fileno(file), 0600) == 0, "Cannot create private token corpus");
        const auto text = result.dump() + "\n";
        require(std::fwrite(text.data(), 1, text.size(), file) == text.size() && std::fclose(file) == 0,
                "Cannot finish token corpus");
        templates.reset();
        model.reset();
        llama_backend_free();
        std::cout << output.string() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "vrambridge-render-corpus: " << error.what() << '\n';
        return 2;
    }
}
