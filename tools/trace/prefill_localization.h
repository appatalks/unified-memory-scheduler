#pragma once

struct PrefillLocalization {
    struct Payload {
        json meta;
        std::vector<float> floats;
        std::vector<int32_t> ids;
        const void * data() const { return floats.empty() ? static_cast<const void *>(ids.data()) : floats.data(); }
        size_t bytes() const { return 4 * (floats.size() + ids.size()); }
    };

    fs::path output_path;
    File file{nullptr};
    bool writing;
    bool attention_focus;
    bool rounding_focus;
    size_t expected_records;
    bool armed = false;
    std::string request;
    std::set<std::pair<int, std::string>> seen;
    json metrics = json::array();
    json reference_header;
    size_t records = 0;
    size_t verified_cache_stores = 0;
    std::map<std::string, std::vector<ggml_fp16_t>> expected_cache_values;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu{nullptr, ggml_backend_free};

    static int width_of(const std::string & kind) {
        if (kind == "input" || kind == "output" || kind == "residual0" || kind == "attn_norm1" ||
            kind == "attn_output1" || kind == "ffn_residual1" || kind == "postnorm1") return 2880;
        if (kind == "q_projection1" || kind == "q_biased1" || kind == "attn_context1") return 4096;
        if (kind == "k_projection1" || kind == "k_biased1" || kind == "v_projection1" || kind == "v_biased1") return 512;
        if (kind == "q_rope1") return 4096;
        if (kind == "k_rope1" || kind == "stored_k1" || kind == "stored_v1") return 512;
        if (kind == "router") return 128;
        if (kind == "ids" || kind == "probabilities") return 4;
        return 0;
    }

    static json routing_difference(const std::vector<int32_t> & expected, const std::vector<int32_t> & actual) {
        require(!actual.empty() && actual.size() == expected.size() && actual.size() % 4 == 0,
                "Routing comparison shape invalid");
        size_t ordered_rows = 0, set_rows = 0, first_set_row = 0;
        for (size_t offset = 0; offset < actual.size(); offset += 4) {
            std::vector<int32_t> a(actual.begin() + offset, actual.begin() + offset + 4);
            std::vector<int32_t> b(expected.begin() + offset, expected.begin() + offset + 4);
            ordered_rows += a != b;
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            if (a != b) { if (!set_rows) first_set_row = offset / 4; ++set_rows; }
        }
        return {{"different_ordered_rows", ordered_rows}, {"different_expert_set_rows", set_rows},
            {"first_different_expert_set_row", set_rows ? json(first_set_row) : json(nullptr)}};
    }

    static void private_parent(const fs::path & path) {
        const auto parent = path.parent_path().empty() ? fs::path(".") : path.parent_path();
        require(fs::is_directory(parent) &&
                (fs::status(parent).permissions() & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none,
                "Sensitive output requires an existing private parent directory (mode 0700)");
    }

    static json read_meta(FILE * file) {
        uint64_t bytes;
        read_bytes(file, &bytes, sizeof(bytes));
        require(bytes > 0 && bytes <= 16384, "Localization record metadata size invalid");
        std::string serialized(bytes, '\0');
        read_bytes(file, serialized.data(), serialized.size());
        auto meta = json::parse(serialized);
        require(meta.at("request").is_string() && !meta.at("request").get<std::string>().empty() &&
                meta.at("request").get<std::string>().size() <= 128 &&
                meta.at("layer").is_number_integer() && meta.at("layer") >= 0 && meta.at("layer") < 36 &&
                meta.at("rows").is_number_integer() && meta.at("rows") >= 1 && meta.at("rows") <= 512,
                "Localization record identity/rows invalid");
        const std::string kind = meta.at("component");
        const int width = width_of(kind);
        require(width && meta.at("width") == width && meta.at("type") == (kind == "ids" ? "i32" : "f32"),
                "Localization component layout invalid");
        return meta;
    }

    PrefillLocalization(const fs::path & output, const fs::path & input, const std::string & focus = "ffn") :
            output_path(output), writing(!output.empty()), attention_focus(focus == "attention1"),
            rounding_focus(focus == "rounding1"), expected_records(rounding_focus ? 6 : attention_focus ? 12 : 180) {
        require(output.empty() != input.empty(), "Choose localization output or reference input, not both");
        if (writing) {
            private_parent(output);
            return;
        }
        require(fs::is_regular_file(input) && fs::file_size(input) <= 512 * 1024 * 1024,
                "Localization reference must be a bounded regular file");
        file.reset(std::fopen(input.c_str(), "rb"));
        require(file != nullptr, "Cannot open localization reference");
        char magic[8];
        read_bytes(file.get(), magic, 8);
        require(std::memcmp(magic, "VRLOC001", 8) == 0, "Localization reference magic mismatch");
        uint64_t size;
        read_bytes(file.get(), &size, sizeof(size));
        require(size > 0 && size <= 16384, "Localization header size invalid");
        std::string serialized(size, '\0');
        read_bytes(file.get(), serialized.data(), serialized.size());
        reference_header = json::parse(serialized);
        require(reference_header.at("schema_version") == 1 && reference_header.at("records") == expected_records &&
                reference_header.at("contains_sensitive_activations") == true,
                "Localization reference header invalid");
        const long start = std::ftell(file.get());
        require(start >= 0, "Localization reference offset unavailable");
        std::set<std::pair<int, std::string>> keys;
        for (size_t i = 0; i < expected_records; ++i) {
            const auto meta = read_meta(file.get());
            require(keys.emplace(meta.at("layer").get<int>(), meta.at("component").get<std::string>()).second,
                    "Duplicate localization record");
            const uint64_t bytes = uint64_t(meta.at("rows").get<int>()) * meta.at("width").get<int>() * 4;
            require(std::fseek(file.get(), static_cast<long>(bytes), SEEK_CUR) == 0,
                    "Localization reference seek failed");
        }
        read_bytes(file.get(), magic, 8);
        require(std::memcmp(magic, "ENDLOC01", 8) == 0 && std::fgetc(file.get()) == EOF,
                "Localization reference trailer/length mismatch");
        require(std::fseek(file.get(), start, SEEK_SET) == 0, "Localization reference rewind failed");
    }

    void configure(const json & descriptor) {
        if (!writing) {
            require(reference_header == descriptor, "Localization reference model/context/request differs");
            if (attention_focus || rounding_focus) return;
            auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            require(device != nullptr, "Localization requires a native CPU quantizer");
            cpu.reset(ggml_backend_dev_init(device, nullptr));
            require(cpu != nullptr, "Cannot initialize localization CPU backend");
            auto threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device), "ggml_backend_set_n_threads"));
            require(threads != nullptr, "Localization CPU thread control unavailable");
            threads(cpu.get(), 1);
            return;
        }
        file.reset(std::fopen((output_path.string() + ".part").c_str(), "wx"));
        require(file != nullptr, "Cannot exclusively create private localization trace");
        fs::permissions(output_path.string() + ".part", fs::perms::owner_read | fs::perms::owner_write);
        write_bytes(file.get(), "VRLOC001", 8);
        const auto serialized = descriptor.dump();
        const uint64_t bytes = serialized.size();
        write_bytes(file.get(), &bytes, sizeof(bytes));
        write_bytes(file.get(), serialized.data(), serialized.size());
    }

    std::vector<uint8_t> quantize(const std::vector<float> & values, int rows) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
        require(ctx != nullptr, "Cannot allocate localization quantization metadata");
        auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2880, rows);
        auto * packed = ggml_cast(ctx.get(), input, GGML_TYPE_Q8_0);
        require(ggml_backend_supports_op(cpu.get(), packed), "Native localization Q8 cast is unsupported");
        auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, packed);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()), ggml_backend_buffer_free);
        require(buffer != nullptr && values.size() * sizeof(float) == ggml_nbytes(input),
                "Localization quantizer input allocation/layout failed");
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS,
                "Localization native Q8 quantization failed");
        std::vector<uint8_t> result(ggml_nbytes(packed));
        ggml_backend_tensor_get(packed, result.data(), 0, result.size());
        return result;
    }

    json compare(const Payload & expected, const Payload & actual) {
        json result = actual.meta;
        size_t different = 0, first = 0;
        double maximum = 0, scale = 0, error2 = 0, norm2 = 0;
        const size_t count = actual.bytes() / 4;
        for (size_t i = 0; i < count; ++i) {
            const double a = actual.floats.empty() ? actual.ids[i] : actual.floats[i];
            const double b = expected.floats.empty() ? expected.ids[i] : expected.floats[i];
            require(std::isfinite(a) && std::isfinite(b), "Nonfinite localization payload");
            if (a != b) { if (!different) first = i; ++different; }
            const double delta = a - b;
            maximum = std::max(maximum, std::abs(delta));
            scale = std::max(scale, std::abs(b));
            error2 += delta * delta;
            norm2 += b * b;
        }
        result["different_values"] = different;
        result["first_different_row"] = different ? json(first / actual.meta.at("width").get<int>()) : json(nullptr);
        result["max_abs"] = maximum;
        result["reference_max_abs"] = scale;
        result["relative_l2"] = std::sqrt(error2 / std::max(norm2, 1e-30));
        if (actual.meta.at("component") == "ids") {
            result.update(routing_difference(expected.ids, actual.ids));
        }
        if (rounding_focus && actual.meta.at("component") != "attn_context1") {
            size_t changed = 0, first_half = 0;
            double half_abs = 0, half_error2 = 0, half_norm2 = 0;
            for (size_t i = 0; i < actual.floats.size(); ++i) {
                const auto a = ggml_fp32_to_fp16(actual.floats[i]);
                const auto b = ggml_fp32_to_fp16(expected.floats[i]);
                if (a != b) { if (!changed) first_half = i; ++changed; }
                const double av = ggml_fp16_to_fp32(a), bv = ggml_fp16_to_fp32(b);
                half_abs = std::max(half_abs, std::abs(av - bv));
                half_error2 += (av - bv) * (av - bv);
                half_norm2 += bv * bv;
            }
            result["fp16_rounding"] = {{"different_codes", changed}, {"effective_max_abs", half_abs},
                {"effective_relative_l2", std::sqrt(half_error2 / std::max(half_norm2, 1e-30))},
                {"first_different_row", changed ? json(first_half / actual.meta.at("width").get<int>()) : json(nullptr)},
                {"actual_cache_storage", actual.meta.at("component") == "stored_k1" ||
                                         actual.meta.at("component") == "stored_v1"}};
        }
        if (actual.meta.at("component") == "input") {
            const auto before = quantize(expected.floats, actual.meta.at("rows"));
            const auto after = quantize(actual.floats, actual.meta.at("rows"));
            require(before.size() == after.size() && before.size() % 34 == 0, "Unexpected native Q8 block size");
            size_t scales = 0, codes = 0, first_block = 0;
            double quant_abs = 0, quant_error2 = 0, quant_norm2 = 0;
            for (size_t offset = 0; offset < before.size(); offset += 34) {
                ggml_fp16_t db, da;
                std::memcpy(&db, before.data() + offset, 2);
                std::memcpy(&da, after.data() + offset, 2);
                if (db != da) { if (!scales && !codes) first_block = offset / 34; ++scales; }
                for (int j = 2; j < 34; ++j) {
                    int8_t qb, qa;
                    std::memcpy(&qb, before.data() + offset + j, 1);
                    std::memcpy(&qa, after.data() + offset + j, 1);
                    if (qb != qa) { if (!scales && !codes) first_block = offset / 34; ++codes; }
                    const double b = double(ggml_fp16_to_fp32(db)) * qb;
                    const double a = double(ggml_fp16_to_fp32(da)) * qa;
                    quant_abs = std::max(quant_abs, std::abs(a - b));
                    quant_error2 += (a - b) * (a - b);
                    quant_norm2 += b * b;
                }
            }
            result["native_q8"] = {{"blocks", before.size() / 34}, {"different_scales", scales},
                {"different_codes", codes}, {"effective_max_abs", quant_abs},
                {"effective_relative_l2", std::sqrt(quant_error2 / std::max(quant_norm2, 1e-30))},
                {"first_different_row", scales || codes ? json(first_block / 90) : json(nullptr)}};
        }
        return result;
    }

    bool callback(ggml_tensor * tensor, bool ask) {
        if (!armed) return !ask;
        const std::string name = tensor->name;
        const auto dash = name.rfind('-');
        std::string prefix;
        int layer = -1;
        std::string stored_kind;
        ggml_tensor sampled{};
        if (rounding_focus && tensor->op == GGML_OP_SET_ROWS && tensor->src[2]) {
            const std::string target = tensor->src[2]->name;
            if (target.rfind("cache_", 0) == 0 && (target.ends_with("k_l1") || target.ends_with("v_l1"))) {
                stored_kind = target.ends_with("k_l1") ? "stored_k1" : "stored_v1";
                layer = 1;
            }
        }
        if (stored_kind.empty()) {
            if (dash == std::string::npos) return !ask;
            const std::string suffix = name.substr(dash + 1);
            prefix = name.substr(0, dash);
            if (suffix.empty() || suffix.size() > 2 ||
                !std::all_of(suffix.begin(), suffix.end(), [](char c) { return c >= '0' && c <= '9'; })) return !ask;
            layer = std::stoi(suffix);
        }
        if (layer < 0 || layer >= 36) return !ask;
        std::string kind = prefix == "hybrid_cpu_up" || prefix == "ffn_moe_up" ? "input" :
            prefix == "ffn_moe_probs" ? "router" : prefix == "ffn_moe_topk" ? "ids" :
            prefix == "ffn_moe_weights_softmax" ? "probabilities" : prefix == "ffn_moe_out" ? "output" : "";
        if (attention_focus) {
            kind = layer == 0 && prefix == "l_out" ? "residual0" : "";
            if (layer == 1) {
                if (prefix == "attn_norm") kind = "attn_norm1";
                else if (prefix == "Qcur" && tensor->op == GGML_OP_MUL_MAT) kind = "q_projection1";
                else if (prefix == "Qcur" && tensor->op == GGML_OP_ADD) kind = "q_biased1";
                else if (prefix == "Kcur" && tensor->op == GGML_OP_MUL_MAT) kind = "k_projection1";
                else if (prefix == "Kcur" && tensor->op == GGML_OP_ADD) kind = "k_biased1";
                else if (prefix == "Vcur" && tensor->op == GGML_OP_MUL_MAT) kind = "v_projection1";
                else if (prefix == "Vcur" && tensor->op == GGML_OP_ADD) kind = "v_biased1";
                else if (prefix == "kqv_out") kind = "attn_context1";
                else if (prefix == "attn_out") kind = "attn_output1";
                else if (prefix == "ffn_inp") kind = "ffn_residual1";
                else if (prefix == "attn_post_norm") kind = "postnorm1";
            }
        }
        if (rounding_focus) {
            kind = stored_kind;
            if (layer == 1 && kind.empty()) {
                if (prefix == "Qcur" && tensor->op == GGML_OP_ROPE) kind = "q_rope1";
                else if (prefix == "Kcur" && tensor->op == GGML_OP_ROPE) kind = "k_rope1";
                else if (prefix == "Vcur" && tensor->op == GGML_OP_ADD) kind = "v_biased1";
                else if (prefix == "kqv_out") kind = "attn_context1";
            }
        }
        if (kind.empty() || seen.count({layer, kind})) return !ask;
        auto * source = kind == "input" ? tensor->src[1] : tensor;
        const bool stored = !stored_kind.empty();
        if (stored) {
            require(tensor->src[0] && tensor->src[1] && tensor->type == GGML_TYPE_F16 &&
                    tensor->ne[0] == 512 && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
                    tensor->src[0]->ne[1] > 0 && tensor->src[0]->ne[1] <= 512,
                    "Unsupported native KV storage layout");
            sampled = *tensor;
            sampled.ne[1] = tensor->src[0]->ne[1];
            source = &sampled;
        }
        require(source != nullptr, "Missing localization source");
        const int width = width_of(kind);
        const auto count = ggml_nelements(source);
        if (!count) return !ask;
        const bool input_layout = kind == "input" && source->ne[0] == 2880 &&
            source->ne[1] == 1 && source->ne[3] == 1;
        const bool probability_layout = kind == "probabilities" && source->ne[0] == 1 &&
            source->ne[1] == 4 && source->ne[3] == 1;
        const bool matrix_layout = kind != "input" && kind != "probabilities" &&
            source->ne[0] == width && source->ne[2] == 1 && source->ne[3] == 1;
        const bool head_layout = (kind == "q_rope1" || kind == "k_rope1") &&
            source->ne[0] * source->ne[1] == width && source->ne[3] == 1;
        require(input_layout || probability_layout || matrix_layout || head_layout,
                "Unsupported localization semantic layout");
        require(source->type == (stored ? GGML_TYPE_F16 : kind == "ids" ? GGML_TYPE_I32 : GGML_TYPE_F32) &&
                count % width == 0 && count / width <= 512 && ggml_nbytes(source) <= 32 * 1024 * 1024,
                "Unsupported localization tensor layout");
        if (ask) return true;
        if (stored) {
            const auto * indices = tensor->src[1];
            require((indices->type == GGML_TYPE_I64 || indices->type == GGML_TYPE_I32) &&
                    indices->ne[0] == source->ne[1], "Unsupported native KV indices");
            for (int64_t row = 0; row < source->ne[1]; ++row) {
                int64_t index = 0;
                ggml_backend_tensor_get(indices, &index, row * indices->nb[0],
                    indices->type == GGML_TYPE_I64 ? sizeof(int64_t) : sizeof(int32_t));
                require(index == row, "KV localization requires aligned first-batch cache positions");
            }
        }
        Payload actual;
        actual.meta = {{"request", request}, {"layer", layer}, {"component", kind},
            {"width", width}, {"rows", count / width}, {"type", kind == "ids" ? "i32" : "f32"}};
        if (stored) actual.meta["native_storage_type"] = "f16";
        std::vector<uint8_t> raw(ggml_nbytes(source));
        ggml_backend_tensor_get(source, raw.data(), 0, raw.size());
        if (kind == "ids") actual.ids.resize(count);
        else actual.floats.resize(count);
        for (int64_t i = 0; i < count; ++i) {
            size_t index = i, offset = 0;
            for (int d = 0; d < 4; ++d) { offset += (index % source->ne[d]) * source->nb[d]; index /= source->ne[d]; }
            require(offset <= raw.size() && (stored ? 2 : 4) <= raw.size() - offset,
                    "Localization stride exceeds storage");
            if (kind == "ids") {
                std::memcpy(&actual.ids[i], raw.data() + offset, 4);
                require(actual.ids[i] >= 0 && actual.ids[i] < 128, "Invalid localization router ID");
            } else {
                if (stored) {
                    ggml_fp16_t value;
                    std::memcpy(&value, raw.data() + offset, 2);
                    actual.floats[i] = ggml_fp16_to_fp32(value);
                } else {
                    std::memcpy(&actual.floats[i], raw.data() + offset, 4);
                }
                require(std::isfinite(actual.floats[i]), "Nonfinite localization activation");
            }
        }
        if (rounding_focus && (kind == "k_rope1" || kind == "v_biased1")) {
            auto & expected = expected_cache_values[kind == "k_rope1" ? "stored_k1" : "stored_v1"];
            for (float value : actual.floats) expected.push_back(ggml_fp32_to_fp16(value));
        }
        if (stored) {
            const auto & expected = expected_cache_values.at(kind);
            require(expected.size() == actual.floats.size(), "Native KV prediction size differs");
            for (size_t i = 0; i < expected.size(); ++i) {
                require(expected[i] == ggml_fp32_to_fp16(actual.floats[i]),
                        "Native KV storage differs from predicted IEEE FP16 rounding");
            }
            ++verified_cache_stores;
        }
        if (writing) {
            const auto serialized = actual.meta.dump();
            const uint64_t bytes = serialized.size();
            write_bytes(file.get(), &bytes, sizeof(bytes));
            write_bytes(file.get(), serialized.data(), serialized.size());
            write_bytes(file.get(), actual.data(), actual.bytes());
        } else {
            Payload expected;
            expected.meta = read_meta(file.get());
            require(expected.meta == actual.meta, "Localization reference tensor alignment differs");
            if (kind == "ids") expected.ids.resize(count);
            else expected.floats.resize(count);
            read_bytes(file.get(), expected.floats.empty() ? static_cast<void *>(expected.ids.data()) :
                       expected.floats.data(), actual.bytes());
            metrics.push_back(compare(expected, actual));
        }
        seen.emplace(layer, kind);
        ++records;
        return true;
    }

    void finish() {
        require(seen.size() == expected_records && records == expected_records,
                "Incomplete first-batch localization coverage");
        if (writing) {
            write_bytes(file.get(), "ENDLOC01", 8);
            publish(file, output_path.string() + ".part", output_path);
        } else {
            char magic[8];
            read_bytes(file.get(), magic, 8);
            require(std::memcmp(magic, "ENDLOC01", 8) == 0 && std::fgetc(file.get()) == EOF,
                    "Localization reference not consumed exactly");
        }
    }
};

static bool localization_callback(ggml_tensor * tensor, bool ask, PrefillLocalization & localization) {
    return localization.callback(tensor, ask);
}
