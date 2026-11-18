#include "gguf_experts.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>

namespace vrambridge::experimental {
namespace {

struct Component {
    std::array<std::int64_t, 4> shape;
    ggml_type type;
    std::vector<float> values;
    std::vector<std::uint8_t> packed;
};

std::vector<std::filesystem::path> shards(const std::filesystem::path & first) {
    std::vector<std::filesystem::path> result{first};
    std::smatch match;
    const auto name = first.filename().string();
    if (std::regex_match(name, match, std::regex(R"((.*)-00001-of-([0-9]{5})\.gguf)"))) {
        const int count = std::stoi(match[2]);
        if (count < 1 || count > 100) throw std::invalid_argument("invalid GGUF shard count");
        for (int i = 2; i <= count; ++i) {
            char suffix[64];
            std::snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, count);
            result.push_back(first.parent_path() / (match[1].str() + suffix));
        }
    }
    return result;
}

} // namespace

LoadedExperts load_gguf_experts(const std::filesystem::path & first_shard, int layer,
                                const std::vector<std::uint32_t> & ids, F32Activation activation) {
    if (layer < 0 || layer >= 4096 || ids.empty() || ids.size() > 16 ||
        std::set<std::uint32_t>(ids.begin(), ids.end()).size() != ids.size())
        throw std::invalid_argument("invalid layer or selected expert IDs");
    std::map<std::string, Component> components;
    const auto prefix = "blk." + std::to_string(layer) + ".ffn_";
    for (const auto & shard : shards(first_shard)) {
        if (!std::filesystem::is_regular_file(shard)) throw std::runtime_error("GGUF shard missing or not a file");
        ggml_context * raw = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_from_file(shard.c_str(), {true, &raw}), gguf_free);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> metadata(raw, ggml_free);
        if (!file || !metadata) throw std::runtime_error("GGUF metadata could not be read");
        for (const std::string part : {"gate", "up", "down"}) {
            for (const std::string kind : {"weight", "bias"}) {
                const std::string key = part + "." + kind;
                const std::string name = prefix + part + "_exps." + kind;
                const auto tensor_id = gguf_find_tensor(file.get(), name.c_str());
                if (tensor_id < 0) continue;
                if (components.count(key)) throw std::runtime_error("duplicate GGUF expert tensor across shards");
                const auto * tensor = ggml_get_tensor(metadata.get(), name.c_str());
                if (!tensor || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[3] != 1)
                    throw std::runtime_error("invalid GGUF expert tensor shape");
                const auto count = static_cast<std::size_t>(kind == "weight" ? tensor->ne[2] : tensor->ne[1]);
                if (count == 0 || count > 65536 || (kind == "bias" && tensor->ne[2] != 1) ||
                    *std::max_element(ids.begin(), ids.end()) >= count)
                    throw std::runtime_error("GGUF expert ID outside tensor");
                const auto elements = ggml_nelements(tensor) / static_cast<std::int64_t>(count);
                if (elements <= 0 || static_cast<std::uint64_t>(elements) > (32ULL * 1024 * 1024) / ids.size())
                    throw std::runtime_error("selected expert tensor exceeds bounded size");
                const auto bytes = ggml_nbytes(tensor);
                if (bytes % count) throw std::runtime_error("nonuniform GGUF expert tensor layout");
                const auto stride = bytes / count;
                const auto file_size = std::filesystem::file_size(shard);
                const auto data_offset = gguf_get_data_offset(file.get());
                const auto tensor_offset = gguf_get_tensor_offset(file.get(), tensor_id);
                if (data_offset > file_size || tensor_offset > file_size - data_offset)
                    throw std::runtime_error("GGUF tensor offset exceeds shard");
                const auto offset = data_offset + tensor_offset;
                if (bytes > file_size - offset) throw std::runtime_error("GGUF tensor exceeds shard");
                const auto * traits = ggml_get_type_traits(tensor->type);
                if (!traits || (tensor->type != GGML_TYPE_F32 && !traits->to_float))
                    throw std::runtime_error("GGUF expert quantization has no dequantizer");
                Component result{{tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}, tensor->type,
                                 std::vector<float>(static_cast<std::size_t>(elements) * ids.size()), {}};
                if (kind == "weight") result.packed.resize(stride * ids.size());
                std::vector<std::uint8_t> packed(stride);
                std::ifstream input(shard, std::ios::binary);
                if (!input) throw std::runtime_error("GGUF shard could not be opened");
                for (std::size_t local = 0; local < ids.size(); ++local) {
                    input.seekg(static_cast<std::streamoff>(offset + stride * ids[local]));
                    input.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(stride));
                    if (!input) throw std::runtime_error("GGUF expert slice could not be read");
                    if (kind == "weight") std::memcpy(result.packed.data() + local * stride, packed.data(), stride);
                    auto * output = result.values.data() + local * static_cast<std::size_t>(elements);
                    if (tensor->type == GGML_TYPE_F32) {
                        if (stride != static_cast<std::size_t>(elements) * sizeof(float))
                            throw std::runtime_error("invalid F32 expert stride");
                        std::memcpy(output, packed.data(), stride);
                    } else {
                        traits->to_float(packed.data(), output, elements);
                    }
                }
                components.emplace(key, std::move(result));
            }
        }
    }
    if (!components.count("gate.weight") || !components.count("up.weight") || !components.count("down.weight"))
        throw std::runtime_error("required separate expert tensors are missing");
    const auto & gate = components.at("gate.weight");
    const auto & up = components.at("up.weight");
    const auto & down = components.at("down.weight");
    if (gate.shape != up.shape || down.shape[0] != gate.shape[1] || down.shape[1] != gate.shape[0] ||
        down.shape[2] != gate.shape[2]) throw std::runtime_error("incompatible expert projection shapes");
    const auto hidden = static_cast<std::size_t>(gate.shape[0]);
    const auto intermediate = static_cast<std::size_t>(gate.shape[1]);
    for (const std::string part : {"gate", "up", "down"}) {
        const auto key = part + ".bias";
        if (!components.count(key)) continue;
        const auto & shape = components.at(key).shape;
        if (shape[0] != static_cast<std::int64_t>(part == "down" ? hidden : intermediate) || shape[1] != gate.shape[2])
            throw std::runtime_error("incompatible expert bias shape");
    }
    LoadedExperts loaded{{hidden, intermediate, ids.size(), activation,
                          std::move(components.at("gate.weight").values), std::move(components.at("up.weight").values),
                          std::move(components.at("down.weight").values), {}, {}, {}}, ids,
                         {ggml_type_name(gate.type), ggml_type_name(up.type), ggml_type_name(down.type)},
                         {gate.type, up.type, down.type},
                         {std::move(components.at("gate.weight").packed), std::move(components.at("up.weight").packed),
                          std::move(components.at("down.weight").packed)}};
    if (components.count("gate.bias")) loaded.weights.gate_bias = std::move(components.at("gate.bias").values);
    if (components.count("up.bias")) loaded.weights.up_bias = std::move(components.at("up.bias").values);
    if (components.count("down.bias")) loaded.weights.down_bias = std::move(components.at("down.bias").values);
    loaded.weights.validate();
    return loaded;
}

} // namespace vrambridge::experimental
