#include "expert_dispatch.hpp"
#include "f32_cuda.hpp"
#include "gguf_experts.hpp"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef VRAMBRIDGE_NATIVE_EXPERTS
#include "native_cpu.hpp"
#include "quantized_cuda.hpp"
#endif

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

using namespace vrambridge::experimental;

namespace {

struct Capture {
    int layer = 0;
    bool stages = false;
    bool inject = false;
    bool skip_prefill = false;
    bool armed = false;
    bool injected = false;
    bool iq4_exact = false;
    bool q6_exact = false;
    bool mxfp4_exact = false;
    bool allow_nonidentical_layer = false;
    bool use_captured_activation = false;
    bool injected_layer_bitwise = true;
    bool output_host = false;
    std::string injection_mode = "mixed";
    std::string model_path, cpu_plugin;
    F32Activation activation = F32Activation::silu;
    double injected_max_abs = 0, injected_relative_l2 = 0;
    std::size_t native_down_changed = 0;
    double native_down_max_abs = 0;
    bool native_down_checked = false;
    std::vector<float> input, probabilities, output;
    std::vector<std::uint32_t> ids;
    std::map<std::string, std::vector<float>> stage_values;
    std::string error;
};

void capture_stage(Capture & capture, const std::string & stage, const ggml_tensor * tensor) {
    if (!capture.stages || capture.stage_values.count(stage)) return;
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[0] > 65536 ||
        tensor->ne[1] <= 0 || tensor->ne[1] > 16 || tensor->ne[2] <= 0 ||
        tensor->ne[3] != 1 || tensor->nb[0] != sizeof(float))
        throw std::runtime_error("unsupported stage tensor shape");
    const auto width = static_cast<std::size_t>(tensor->ne[0]);
    const auto selected = static_cast<std::size_t>(tensor->ne[1]);
    std::vector<float> values(width * selected);
    for (std::size_t rank = 0; rank < selected; ++rank) {
        const auto offset = static_cast<std::size_t>(tensor->ne[2] - 1) * tensor->nb[2] + rank * tensor->nb[1];
        if (offset > ggml_nbytes(tensor) || width * sizeof(float) > ggml_nbytes(tensor) - offset)
            throw std::runtime_error("stage tensor offset out of bounds");
        ggml_backend_tensor_get(tensor, values.data() + rank * width, offset, width * sizeof(float));
    }
    capture.stage_values.emplace(stage, std::move(values));
}

#if defined(__unix__) || defined(__APPLE__)
class NativeRoundtrip {
public:
    explicit NativeRoundtrip(const char * plugin) {
        handle_ = dlopen(plugin, RTLD_NOW | RTLD_LOCAL);
        if (!handle_) throw std::runtime_error("cannot open CPU plugin for native roundtrip");
        traits_ = reinterpret_cast<Traits>(dlsym(handle_, "ggml_get_type_traits_cpu"));
        q8k_ = reinterpret_cast<ggml_to_float_t>(dlsym(RTLD_DEFAULT, "dequantize_row_q8_K"));
        if (!traits_ || !q8k_) {
            dlclose(handle_);
            handle_ = nullptr;
            throw std::runtime_error("native roundtrip symbols unavailable");
        }
    }
    ~NativeRoundtrip() { if (handle_) dlclose(handle_); }
    NativeRoundtrip(const NativeRoundtrip &) = delete;
    NativeRoundtrip & operator=(const NativeRoundtrip &) = delete;

    ggml_type dot_type(ggml_type weight) const {
        const auto * type = traits_(weight);
        if (!type) throw std::runtime_error("native weight type is unsupported");
        return type->vec_dot_type;
    }
    void convert(std::vector<float> & values, ggml_type dot) const {
        if (dot == GGML_TYPE_F32) return;
        const auto * native = traits_(dot);
        const auto * generic = ggml_get_type_traits(dot);
        if (!native || !native->from_float || !generic || generic->blck_size <= 0 ||
            values.size() % static_cast<std::size_t>(generic->blck_size))
            throw std::runtime_error("unsupported native activation quantization");
        const auto dequantize = dot == GGML_TYPE_Q8_K ? q8k_ : generic->to_float;
        if (!dequantize) throw std::runtime_error("native activation dequantizer unavailable");
        std::vector<std::uint8_t> packed(ggml_row_size(dot, static_cast<std::int64_t>(values.size())));
        native->from_float(values.data(), packed.data(), static_cast<std::int64_t>(values.size()));
        dequantize(packed.data(), values.data(), static_cast<std::int64_t>(values.size()));
    }
private:
    using Traits = const ggml_type_traits_cpu * (*)(ggml_type);
    void * handle_ = nullptr;
    Traits traits_ = nullptr;
    ggml_to_float_t q8k_ = nullptr;
};
#endif

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void inject_candidate(Capture & capture, ggml_tensor * output);

bool observe(ggml_tensor * tensor, bool ask, void * context) {
    auto & capture = *static_cast<Capture *>(context);
    if (capture.skip_prefill && !capture.armed) return !ask;
    const std::string name = tensor->name;
    const auto suffix = "-" + std::to_string(capture.layer);
    const bool input = name == "ffn_moe_up" + suffix && capture.input.empty();
    const bool weights = name == "ffn_moe_weighted" + suffix && capture.probabilities.empty();
    const bool output = name == "ffn_moe_out" + suffix && capture.output.empty();
    std::string stage;
    if (capture.stages) {
        if (name == "ffn_moe_up" + suffix) stage = "up";
        else if (name == "ffn_moe_gate" + suffix) stage = "gate";
        else if (name == "ffn_moe_swiglu" + suffix) stage = "activation";
        else if (name == "ffn_moe_down" + suffix) stage = "down";
    }
    const bool stage_needed = !stage.empty() && !capture.stage_values.count(stage);
    if (!input && !weights && !output && !stage_needed) return !ask;
    if (ask) return true;
    try {
        if (!input && !weights && !output) {
            capture_stage(capture, stage, tensor);
            return true;
        }
        const auto * source = output ? tensor : tensor->src[1];
        require(source && source->type == GGML_TYPE_F32 && source->ne[3] == 1 && source->nb[0] == sizeof(float),
                "unsupported captured F32 tensor");
        if (input) {
            const auto * ids = tensor->src[2];
            require(source->ne[0] > 0 && source->ne[0] <= 65536 && source->ne[1] == 1 && source->ne[2] > 0 &&
                    ids && ids->type == GGML_TYPE_I32 && ids->ne[0] > 0 && ids->ne[0] <= 16 &&
                    ids->ne[1] == source->ne[2] && ids->nb[0] == sizeof(std::int32_t), "unsupported captured route shape");
            const auto row = static_cast<std::size_t>(source->ne[2] - 1);
            capture.input.resize(static_cast<std::size_t>(source->ne[0]));
            std::vector<std::int32_t> selected(static_cast<std::size_t>(ids->ne[0]));
            const auto source_offset = row * source->nb[2];
            const auto ids_offset = row * ids->nb[1];
            require(source_offset <= ggml_nbytes(source) && capture.input.size() * sizeof(float) <= ggml_nbytes(source) - source_offset &&
                    ids_offset <= ggml_nbytes(ids) && selected.size() * sizeof(std::int32_t) <= ggml_nbytes(ids) - ids_offset,
                    "captured route or input exceeds tensor storage");
            ggml_backend_tensor_get(source, capture.input.data(), source_offset, capture.input.size() * sizeof(float));
            ggml_backend_tensor_get(ids, selected.data(), ids_offset, selected.size() * sizeof(std::int32_t));
            std::set<std::int32_t> unique;
            for (const auto id : selected) {
                require(id >= 0 && unique.insert(id).second, "invalid captured expert ID");
                capture.ids.push_back(static_cast<std::uint32_t>(id));
            }
            if (stage_needed) capture_stage(capture, stage, tensor);
        } else if (weights) {
            require(!capture.ids.empty() && source->ne[0] == 1 && source->ne[1] == static_cast<std::int64_t>(capture.ids.size()) &&
                    source->ne[2] > 0 && source->nb[1] == sizeof(float), "unsupported captured probability shape");
            capture.probabilities.resize(capture.ids.size());
            const auto offset = static_cast<std::size_t>(source->ne[2] - 1) * source->nb[2];
            require(offset <= ggml_nbytes(source) && capture.probabilities.size() * sizeof(float) <= ggml_nbytes(source) - offset,
                    "captured probabilities exceed tensor storage");
            ggml_backend_tensor_get(source, capture.probabilities.data(), offset,
                                    capture.probabilities.size() * sizeof(float));
        } else {
            require(!capture.input.empty() && source->ne[0] == static_cast<std::int64_t>(capture.input.size()) &&
                    source->ne[1] > 0 && source->ne[2] == 1, "unsupported captured output shape");
            capture.output.resize(capture.input.size());
            capture.output_host = source->buffer && ggml_backend_buffer_is_host(source->buffer);
            const auto offset = static_cast<std::size_t>(source->ne[1] - 1) * source->nb[1];
            require(offset <= ggml_nbytes(source) && capture.output.size() * sizeof(float) <= ggml_nbytes(source) - offset,
                    "captured expert output exceeds tensor storage");
            ggml_backend_tensor_get(source, capture.output.data(), offset,
                                    capture.output.size() * sizeof(float));
            if (capture.inject) inject_candidate(capture, tensor);
        }
        return true;
    } catch (const std::exception & error) {
        capture.error = error.what();
        return false;
    }
}

struct Error {
    double max_abs = 0, relative_l2 = 0, max_reference = 0;
    bool passed = false;
};

struct Timing {
    std::vector<double> cpu_ms, cpu2_ms, mixed_ms, iq4_ms;
    bool all_correct = true;
    bool iq4_bitwise = true;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values.size() % 2 ? values[values.size() / 2] :
        (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2.0;
}

void print_samples(const char * name, const std::vector<double> & values) {
    std::printf(",\"%s\":[", name);
    for (std::size_t i = 0; i < values.size(); ++i) std::printf(i ? ",%.9g" : "%.9g", values[i]);
    std::putchar(']');
}

Error compare(const std::vector<float> & reference, const std::vector<float> & actual,
              double absolute, double relative) {
    require(reference.size() == actual.size() && !reference.empty(), "comparison shape mismatch");
    Error result;
    double error2 = 0, reference2 = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(actual[i]), "non-finite comparison");
        const double difference = static_cast<double>(actual[i]) - reference[i];
        result.max_abs = std::max(result.max_abs, std::abs(difference));
        result.max_reference = std::max(result.max_reference, std::abs(static_cast<double>(reference[i])));
        error2 += difference * difference;
        reference2 += static_cast<double>(reference[i]) * reference[i];
    }
    result.relative_l2 = std::sqrt(error2 / std::max(reference2, 1e-30));
    result.passed = result.max_abs <= absolute && result.relative_l2 <= relative;
    return result;
}

void inject_candidate(Capture & capture, ggml_tensor * output) {
#if defined(VRAMBRIDGE_NATIVE_EXPERTS) && (defined(__unix__) || defined(__APPLE__))
    require(!capture.injected && !capture.input.empty() && capture.ids.size() == capture.probabilities.size(),
            "incomplete or duplicate expert injection");
    if (capture.iq4_exact || capture.q6_exact || capture.mxfp4_exact)
        require(capture.output_host, "exact quantized injection requires a CPU-resident native layer output");
    const auto loaded = load_gguf_experts(capture.model_path, capture.layer, capture.ids, capture.activation);
    require(loaded.weights.hidden == capture.input.size(), "injected expert hidden width differs");
    NativeCpuExperts native(loaded, capture.cpu_plugin);
    NativeRoundtrip roundtrip(capture.cpu_plugin.c_str());
    const auto down_type = roundtrip.dot_type(loaded.quant_types[2]);
    F32Transforms transforms;
    transforms.down_input = [&](std::vector<float> & values) { roundtrip.convert(values, down_type); };
    std::vector<Route> row;
    std::vector<std::uint8_t> resident(capture.ids.size());
    for (std::size_t i = 0; i < capture.ids.size(); ++i) {
        row.push_back({static_cast<std::uint32_t>(i), capture.probabilities[i]});
        resident[i] = static_cast<std::uint8_t>(i % 2 == 0);
    }
    std::unique_ptr<F32CudaExperts> cuda;
    std::unique_ptr<Iq4NativeCudaDown> iq4;
    std::unique_ptr<Q6NativeCudaDown> q6;
    std::unique_ptr<Mxfp4NativeCudaDown> mxfp4;
    if (capture.iq4_exact) iq4 = std::make_unique<Iq4NativeCudaDown>(loaded, resident);
    else if (capture.q6_exact) q6 = std::make_unique<Q6NativeCudaDown>(loaded, resident);
    else if (capture.mxfp4_exact) mxfp4 = std::make_unique<Mxfp4NativeCudaDown>(loaded, resident);
    else cuda = std::make_unique<F32CudaExperts>(loaded.weights, resident);
    const auto selected_activations = [&](const std::vector<ExpertJob> & jobs) {
        if (!capture.use_captured_activation) return native.activations(capture.input, jobs);
        require(capture.stage_values.count("activation") != 0, "native activation capture is missing");
        const auto & all = capture.stage_values.at("activation");
        require(all.size() == capture.ids.size() * loaded.weights.intermediate, "native activation shape differs");
        std::vector<float> selected;
        selected.reserve(jobs.size() * loaded.weights.intermediate);
        for (const auto & job : jobs) {
            require(job.row == 0 && job.rank < capture.ids.size() && job.expert == job.rank,
                    "native activation route differs");
            selected.insert(selected.end(), all.begin() + job.rank * loaded.weights.intermediate,
                            all.begin() + (job.rank + 1) * loaded.weights.intermediate);
        }
        return selected;
    };
    if (capture.use_captured_activation) {
        require(capture.stage_values.count("down") != 0 && loaded.weights.down_bias.empty(),
                "unbiased native down capture is required");
        std::vector<ExpertJob> all_jobs;
        for (std::size_t i = 0; i < capture.ids.size(); ++i)
            all_jobs.push_back({0, i, static_cast<std::uint32_t>(i)});
        const auto computed = native.execute_down(selected_activations(all_jobs), all_jobs);
        const auto & observed = capture.stage_values.at("down");
        require(computed.size() == observed.size(), "native down comparison shape differs");
        for (std::size_t i = 0; i < computed.size(); ++i) {
            capture.native_down_changed += std::memcmp(&computed[i], &observed[i], sizeof(float)) != 0;
            capture.native_down_max_abs = std::max(capture.native_down_max_abs,
                std::abs(static_cast<double>(computed[i]) - observed[i]));
        }
        capture.native_down_checked = true;
    }
    const auto cpu = [&](const std::vector<ExpertJob> & jobs) {
        return capture.use_captured_activation ? native.execute_down(selected_activations(jobs), jobs)
                                               : native.execute(capture.input, jobs);
    };
    const auto gpu = [&](const std::vector<ExpertJob> & jobs) {
        const auto activations = selected_activations(jobs);
        if (!capture.iq4_exact && !capture.q6_exact && !capture.mxfp4_exact) return cuda->execute_down(activations, jobs, transforms);
        const auto packed = native.pack_down(activations);
        const auto actual = capture.iq4_exact ? iq4->execute(packed, jobs) :
                            capture.q6_exact ? q6->execute(packed, jobs) : mxfp4->execute(packed, jobs);
        const auto expected = native.execute_down(activations, jobs);
        require(actual.size() == expected.size() &&
                std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                "quantized GPU output did not match native CPU bits");
        return actual;
    };
    const auto candidate = ExpertPlan(loaded.weights.hidden, {row}, resident).execute(cpu, gpu);
    const auto native_output = ExpertPlan(loaded.weights.hidden, {row}, std::vector<std::uint8_t>(resident.size(), 0)).execute(cpu, {});
    const auto & replacement = capture.injection_mode == "mixed" ? candidate :
                               capture.injection_mode == "native-cpu" ? native_output : capture.output;
    const auto scale = std::abs(*std::max_element(capture.output.begin(), capture.output.end(),
        [](float a, float b) { return std::abs(a) < std::abs(b); }));
    const auto delta = compare(capture.output, replacement, 1e-5 + 1e-4 * scale, 1e-4);
    capture.injected_max_abs = delta.max_abs;
    capture.injected_relative_l2 = delta.relative_l2;
    require(delta.passed, "expert replacement failed the native layer correctness gate");
    capture.injected_layer_bitwise =
        std::memcmp(capture.output.data(), replacement.data(), replacement.size() * sizeof(float)) == 0;
    if ((capture.iq4_exact || capture.q6_exact || capture.mxfp4_exact) && capture.injection_mode == "mixed" &&
        !capture.allow_nonidentical_layer && !capture.injected_layer_bitwise) {
        std::size_t changed = 0;
        for (std::size_t i = 0; i < replacement.size(); ++i)
            changed += std::memcmp(&capture.output[i], &replacement[i], sizeof(float)) != 0;
        char message[192];
        std::snprintf(message, sizeof(message), "quantized layer bits differ: changed=%zu max_abs=%.9g relative_l2=%.9g native_down_checked=%s native_down_changed=%zu native_down_max_abs=%.9g",
                      changed, delta.max_abs, delta.relative_l2, capture.native_down_checked ? "true" : "false",
                      capture.native_down_changed, capture.native_down_max_abs);
        throw std::runtime_error(message);
    }
    const auto offset = static_cast<std::size_t>(output->ne[1] - 1) * output->nb[1];
    require(offset <= ggml_nbytes(output) && replacement.size() * sizeof(float) <= ggml_nbytes(output) - offset,
            "expert replacement exceeds output storage");
    if (capture.injection_mode != "compute-only")
        ggml_backend_tensor_set(output, replacement.data(), offset, replacement.size() * sizeof(float));
    std::vector<float> checked(replacement.size());
    ggml_backend_tensor_get(output, checked.data(), offset, checked.size() * sizeof(float));
    require(checked == replacement, "expert replacement readback failed");
    capture.injected = true;
#else
    (void) capture;
    (void) output;
    throw std::runtime_error("expert injection is unsupported on this platform");
#endif
}

struct Inference {
    std::vector<std::vector<float>> logits;
    std::vector<llama_token> forced;
    std::vector<float> first_layer_output;
    int injections = 0;
    double max_layer_abs = 0, max_layer_relative_l2 = 0;
    bool all_layers_bitwise = true;
};

Error compare_inference(const Inference & reference, const Inference & actual, bool & same_top) {
    require(reference.logits.size() == actual.logits.size() && !reference.logits.empty(), "logit position count differs");
    Error combined;
    combined.passed = true;
    same_top = true;
    for (std::size_t i = 0; i < reference.logits.size(); ++i) {
        const auto error = compare(reference.logits[i], actual.logits[i], 0.01, 0.001);
        combined.max_abs = std::max(combined.max_abs, error.max_abs);
        combined.relative_l2 = std::max(combined.relative_l2, error.relative_l2);
        combined.passed = combined.passed && error.passed;
        same_top = same_top && (std::max_element(reference.logits[i].begin(), reference.logits[i].end()) - reference.logits[i].begin() ==
                                std::max_element(actual.logits[i].begin(), actual.logits[i].end()) - actual.logits[i].begin());
    }
    return combined;
}

Inference infer(llama_model * model, Capture * capture, const std::string & prompt, int gpu_layers,
                int steps, const std::vector<llama_token> * forced = nullptr) {
    auto params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_seq_max = 1;
    params.n_threads = 8;
    params.n_threads_batch = 8;
    params.offload_kqv = gpu_layers > 0;
    params.op_offload = gpu_layers > 0;
    params.cb_eval = capture ? observe : nullptr;
    params.cb_eval_user_data = capture;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model, params), llama_free);
    require(context != nullptr, "context creation failed");
    const auto * vocab = llama_model_get_vocab(model);
    const int count = -llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), nullptr, 0, true, false);
    require(count > 0 && count + steps < 512, "invalid prompt tokenization or continuation length");
    require(!forced || forced->size() == static_cast<std::size_t>(steps), "invalid forced continuation");
    std::vector<llama_token> tokens(static_cast<std::size_t>(count));
    require(llama_tokenize(vocab, prompt.data(), static_cast<int>(prompt.size()), tokens.data(), count, true, false) == count,
            "prompt tokenization failed");
    Inference result;
    const auto process = [&](llama_token * data, int length) {
        const int status = llama_decode(context.get(), llama_batch_get_one(data, length));
        if (capture) require(capture->error.empty(), capture->error.c_str());
        require(status == 0, "prompt or continuation inference failed");
        if (capture && result.first_layer_output.empty() && !capture->output.empty())
            result.first_layer_output = capture->output;
        if (capture && capture->inject && (!capture->skip_prefill || capture->armed)) {
            require(capture->injected, "expert injection was not executed");
            ++result.injections;
            result.max_layer_abs = std::max(result.max_layer_abs, capture->injected_max_abs);
            result.max_layer_relative_l2 = std::max(result.max_layer_relative_l2, capture->injected_relative_l2);
            result.all_layers_bitwise = result.all_layers_bitwise && capture->injected_layer_bitwise;
        }
        const auto * logits = llama_get_logits_ith(context.get(), -1);
        require(logits != nullptr, "missing inference logits");
        result.logits.emplace_back(logits, logits + llama_vocab_n_tokens(vocab));
    };
    process(tokens.data(), count);
    for (int step = 0; step < steps; ++step) {
        llama_token token = forced ? forced->at(step) : static_cast<llama_token>(
            std::max_element(result.logits.back().begin(), result.logits.back().end()) - result.logits.back().begin());
        result.forced.push_back(token);
        if (capture) capture->armed = true;
        if (capture && capture->inject) {
            capture->input.clear(); capture->probabilities.clear(); capture->output.clear(); capture->ids.clear();
            capture->stage_values.clear(); capture->error.clear(); capture->injected = false;
            capture->injected_max_abs = 0; capture->injected_relative_l2 = 0; capture->injected_layer_bitwise = true;
            capture->output_host = false;
        }
        process(&token, 1);
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "Usage: %s FIRST_GGUF CPU_PLUGIN GPU_PLUGIN silu|openai [--native-activation] [--native-kernel] [--inject] [--decode-only] [--iq4-exact|--q6-exact|--mxfp4-exact] [--native-captured-activation] [--allow-nonidentical-layer] [--injection-mode mixed|original|compute-only|native-cpu] [--steps N] [--stage-diagnostics] [--benchmark N] [--gpu-layers N] [--layer N] [--case fact|coding|reasoning|prose]\n", argv[0]);
        return 2;
    }
    try {
        bool native_activation = false;
        bool native_kernel = false;
        bool inject = false;
        bool decode_only = false;
        bool iq4_exact = false;
        bool q6_exact = false;
        bool mxfp4_exact = false;
        bool allow_nonidentical_layer = false;
        bool captured_activation = false;
        bool stages = false;
        int layer = 0;
        int gpu_layers = 4;
        int repetitions = 0;
        int steps = 0;
        std::string case_name = "fact";
        std::string injection_mode = "mixed";
        for (int i = 5; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--native-activation") { native_activation = true; continue; }
            if (option == "--native-kernel") { native_kernel = true; continue; }
            if (option == "--inject") { inject = true; continue; }
            if (option == "--decode-only") { decode_only = true; continue; }
            if (option == "--iq4-exact") { iq4_exact = true; continue; }
            if (option == "--q6-exact") { q6_exact = true; continue; }
            if (option == "--mxfp4-exact") { mxfp4_exact = true; continue; }
            if (option == "--allow-nonidentical-layer") { allow_nonidentical_layer = true; continue; }
            if (option == "--native-captured-activation") { captured_activation = true; continue; }
            if (option == "--stage-diagnostics") { stages = true; continue; }
            require(i + 1 < argc, "missing diagnostic option value");
            const std::string value = argv[++i];
            if (option == "--case") case_name = value;
            else if (option == "--injection-mode") injection_mode = value;
            else if (option == "--layer") {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), layer);
                require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && layer >= 0 && layer < 4096,
                        "invalid layer");
            } else if (option == "--gpu-layers") {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), gpu_layers);
                require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && gpu_layers >= 0 && gpu_layers <= 1000,
                        "invalid GPU layer count");
            } else if (option == "--benchmark") {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), repetitions);
                require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && repetitions >= 1 && repetitions <= 10,
                        "invalid benchmark repetition count");
            } else if (option == "--steps") {
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), steps);
                require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && steps >= 0 && steps <= 4,
                        "invalid teacher-forced continuation count");
            } else throw std::runtime_error("unknown diagnostic option");
        }
        std::string prompt;
        if (case_name == "fact") prompt = "The capital of France is";
        else if (case_name == "coding") prompt = "Write a thread-safe bounded queue with cancellation handling.";
        else if (case_name == "reasoning") prompt = "Calculate 17 times 23 and explain the steps.";
        else if (case_name == "prose") prompt = "Explain why cache residency affects expert execution latency.";
        else throw std::runtime_error("unknown diagnostic case");
        const std::string activation = argv[4];
        require(activation == "silu" || activation == "openai", "unsupported activation");
        require(!stages || (native_activation && activation == "silu"), "stage diagnostics require native-activation and silu");
        require(!native_kernel || native_activation, "native kernel diagnostic requires native activation mode");
        require(!inject || native_kernel, "expert injection requires native kernel diagnostic");
        require(!decode_only || (inject && steps > 0), "decode-only diagnostic requires injection and continuation steps");
        require(static_cast<int>(iq4_exact) + static_cast<int>(q6_exact) + static_cast<int>(mxfp4_exact) <= 1,
                "only one exact quantization format may be selected");
        require(!(iq4_exact || q6_exact || mxfp4_exact) || inject, "exact quantization diagnostic requires expert injection");
        require(!allow_nonidentical_layer || (inject && (iq4_exact || q6_exact || mxfp4_exact)),
                "nonidentical layer diagnostic requires exact quantization injection");
        require(!captured_activation || (inject && (iq4_exact || q6_exact || mxfp4_exact) && activation == "silu"),
                "captured activation diagnostic requires a quantized SILU injection");
        require(injection_mode == "mixed" || injection_mode == "original" || injection_mode == "compute-only" ||
                injection_mode == "native-cpu", "unknown injection mode");
        require(inject || injection_mode == "mixed", "injection mode requires --inject");
        require(repetitions == 0 || native_kernel, "benchmark requires native kernel diagnostic");
        require(ggml_backend_load(argv[2]) && ggml_backend_load(argv[3]), "backend plugin load failed");
        llama_backend_init();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU), nullptr};
        require(devices[0] && llama_supports_gpu_offload(), "GPU backend unavailable");
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = gpu_layers;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "model load failed");
        const auto baseline = infer(model.get(), nullptr, prompt, gpu_layers, steps);
        const auto repeated = infer(model.get(), nullptr, prompt, gpu_layers, steps, &baseline.forced);
        bool repeat_top = false;
        const auto baseline_repeat = compare_inference(baseline, repeated, repeat_top);
        if (!baseline_repeat.passed || !repeat_top) {
            std::printf("{\"baseline_repeat\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"top_equal\":%s,\"pass\":false}}\n",
                        baseline_repeat.max_abs, baseline_repeat.relative_l2, repeat_top ? "true" : "false");
            return 1;
        }
        Capture capture;
        capture.layer = layer;
        capture.stages = stages || captured_activation;
        capture.skip_prefill = decode_only;
        const auto observed = infer(model.get(), &capture, prompt, gpu_layers, steps, &baseline.forced);
        require(!capture.input.empty() && !capture.ids.empty() && !capture.probabilities.empty() && !capture.output.empty(),
                "incomplete actual expert observation");
        bool same_top = false;
        const auto observer = compare_inference(baseline, observed, same_top);
        if (!observer.passed || !same_top) {
            std::printf("{\"observer\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"top_equal\":%s,\"pass\":false}}\n",
                        observer.max_abs, observer.relative_l2, same_top ? "true" : "false");
            return 1;
        }
        Capture injected_capture;
        Error injected_logits, injection_control;
        bool injected_top = false;
        Inference injected;
        if (inject) {
            injected_capture.layer = layer;
            injected_capture.inject = true;
            injected_capture.skip_prefill = decode_only;
            injected_capture.iq4_exact = iq4_exact;
            injected_capture.q6_exact = q6_exact;
            injected_capture.mxfp4_exact = mxfp4_exact;
            injected_capture.allow_nonidentical_layer = allow_nonidentical_layer;
            injected_capture.use_captured_activation = captured_activation;
            injected_capture.stages = captured_activation;
            injected_capture.injection_mode = injection_mode;
            injected_capture.model_path = argv[1];
            injected_capture.cpu_plugin = argv[2];
            injected_capture.activation = activation == "silu" ? F32Activation::silu : F32Activation::openai_swiglu;
            injected = infer(model.get(), &injected_capture, prompt, gpu_layers, steps, &baseline.forced);
            require(injected.injections == steps + (decode_only ? 0 : 1), "not every expert injection was executed");
            injected_logits = compare_inference(baseline, injected, injected_top);
            const auto scale = std::abs(*std::max_element(capture.output.begin(), capture.output.end(),
                [](float a, float b) { return std::abs(a) < std::abs(b); }));
            injection_control = compare(capture.output, injected.first_layer_output, 1e-5 + 1e-4 * scale, 1e-4);
        }
        model.reset();
        const auto loaded = load_gguf_experts(argv[1], layer, capture.ids,
            activation == "silu" ? F32Activation::silu : F32Activation::openai_swiglu);
        require(loaded.weights.hidden == capture.input.size(), "model and captured hidden widths differ");
        std::vector<Route> row;
        std::vector<std::uint8_t> none(capture.ids.size(), 0), mixed(capture.ids.size());
        for (std::size_t i = 0; i < capture.ids.size(); ++i) {
            row.push_back({static_cast<std::uint32_t>(i), capture.probabilities[i]});
            mixed[i] = static_cast<std::uint8_t>(i % 2 == 0);
        }
        const std::vector<std::vector<Route>> routes{row};
        const auto cpu = [&](const std::vector<ExpertJob> & jobs) {
            return execute_f32_cpu(loaded.weights, capture.input, jobs);
        };
        const auto cpu_output = ExpertPlan(loaded.weights.hidden, routes, none).execute(cpu, {});
        F32CudaExperts cuda(loaded.weights, mixed);
        const auto gpu = [&](const std::vector<ExpertJob> & jobs) { return cuda.execute(capture.input, jobs); };
        const auto mixed_output = ExpertPlan(loaded.weights.hidden, routes, mixed).execute(cpu, gpu);
        const auto scale = std::abs(*std::max_element(
            capture.output.begin(), capture.output.end(), [](float a, float b) { return std::abs(a) < std::abs(b); }));
        const auto cpu_error = compare(capture.output, cpu_output, 1e-5 + 1e-4 * scale, 1e-4);
        const auto mixed_error = compare(capture.output, mixed_output, 1e-5 + 1e-4 * cpu_error.max_reference, 1e-4);
        const auto split_error = compare(cpu_output, mixed_output, 1e-5 + 1e-4 * cpu_error.max_reference, 1e-4);
        Error native_error, native_mixed_error, native_split_error, native_kernel_error, kernel_mixed_error, kernel_split_error;
        Timing timing;
        std::string gate_dot, down_dot;
        std::map<std::string, Error> stage_errors;
        std::map<std::string, Error> kernel_stage_errors;
        if (native_activation) {
#if defined(__unix__) || defined(__APPLE__)
            NativeRoundtrip native(argv[2]);
            const auto gate_type = native.dot_type(loaded.quant_types[0]);
            const auto up_type = native.dot_type(loaded.quant_types[1]);
            const auto down_type = native.dot_type(loaded.quant_types[2]);
            require(gate_type == up_type, "gate and up require different activation formats");
            gate_dot = ggml_type_name(gate_type);
            down_dot = ggml_type_name(down_type);
            F32Transforms transforms;
            transforms.gate_up_input = [&](std::vector<float> & values) { native.convert(values, gate_type); };
            transforms.down_input = [&](std::vector<float> & values) { native.convert(values, down_type); };
            std::map<std::string, std::vector<float>> cpu_stages;
            if (stages) {
                require(loaded.weights.gate_bias.empty() && loaded.weights.up_bias.empty() && loaded.weights.down_bias.empty(),
                        "stage diagnostics currently require unbiased experts");
                transforms.observe = [&](const ExpertJob &, F32Stage stage, const std::vector<float> & values) {
                    const char * key = stage == F32Stage::gate ? "gate" : stage == F32Stage::up ? "up" :
                                       stage == F32Stage::activation ? "activation" : "down";
                    auto & all = cpu_stages[key];
                    all.insert(all.end(), values.begin(), values.end());
                };
            }
            const auto native_cpu = [&](const std::vector<ExpertJob> & jobs) {
                return execute_f32_cpu(loaded.weights, capture.input, jobs, transforms);
            };
            const auto native_cpu_output = ExpertPlan(loaded.weights.hidden, routes, none).execute(native_cpu, {});
            native_error = compare(capture.output, native_cpu_output, 1e-5 + 1e-4 * cpu_error.max_reference, 1e-4);
            if (stages) {
                for (const std::string key : {"gate", "up", "activation", "down"}) {
                    require(capture.stage_values.count(key) && cpu_stages.count(key), "missing observed stage");
                    const auto & reference = capture.stage_values.at(key);
                    const auto max_ref = std::abs(*std::max_element(reference.begin(), reference.end(),
                        [](float a, float b) { return std::abs(a) < std::abs(b); }));
                    stage_errors.emplace(key, compare(reference, cpu_stages.at(key), 1e-5 + 1e-4 * max_ref, 1e-4));
                }
                transforms.observe = {};
            }
            const auto native_gpu = [&](const std::vector<ExpertJob> & jobs) {
                return cuda.execute(capture.input, jobs, transforms);
            };
            const auto native_mixed = ExpertPlan(loaded.weights.hidden, routes, mixed).execute(native_cpu, native_gpu);
            native_mixed_error = compare(capture.output, native_mixed, 1e-5 + 1e-4 * cpu_error.max_reference, 1e-4);
            native_split_error = compare(native_cpu_output, native_mixed, 1e-5 + 1e-4 * cpu_error.max_reference, 1e-4);
#ifdef VRAMBRIDGE_NATIVE_EXPERTS
            if (native_kernel) {
                NativeCpuExperts exact(loaded, argv[2]);
                F32Transforms exact_observer;
                std::map<std::string, std::vector<float>> exact_stages;
                if (stages) {
                    exact_observer.observe = [&](const ExpertJob &, F32Stage stage, const std::vector<float> & values) {
                        const char * key = stage == F32Stage::gate ? "gate" : stage == F32Stage::up ? "up" :
                                           stage == F32Stage::activation ? "activation" : "down";
                        auto & all = exact_stages[key];
                        all.insert(all.end(), values.begin(), values.end());
                    };
                }
                const auto exact_cpu = [&](const std::vector<ExpertJob> & jobs) {
                    return exact.execute(capture.input, jobs, exact_observer);
                };
                const auto exact_cpu_output = ExpertPlan(loaded.weights.hidden, routes, none).execute(exact_cpu, {});
                native_kernel_error = compare(capture.output, exact_cpu_output, 1e-5 + 1e-4 * cpu_error.max_reference, 1e-4);
                if (stages) {
                    for (const std::string key : {"gate", "up", "activation", "down"}) {
                        require(capture.stage_values.count(key) && exact_stages.count(key), "missing native observed stage");
                        const auto & reference = capture.stage_values.at(key);
                        const auto max_ref = std::abs(*std::max_element(reference.begin(), reference.end(),
                            [](float a, float b) { return std::abs(a) < std::abs(b); }));
                        kernel_stage_errors.emplace(key, compare(reference, exact_stages.at(key), 1e-5 + 1e-4 * max_ref, 1e-4));
                    }
                    exact_observer.observe = {};
                }
                const auto exact_gpu_down = [&](const std::vector<ExpertJob> & jobs) {
                    return cuda.execute_down(exact.activations(capture.input, jobs), jobs, transforms);
                };
                const auto exact_mixed_output = ExpertPlan(loaded.weights.hidden, routes, mixed).execute(exact_cpu, exact_gpu_down);
                kernel_mixed_error = compare(capture.output, exact_mixed_output, 1e-5 + 1e-4 * cpu_error.max_reference, 1e-4);
                kernel_split_error = compare(exact_cpu_output, exact_mixed_output, 1e-5 + 1e-4 * cpu_error.max_reference, 1e-4);
                if (repetitions) {
                    const ExpertPlan cpu_plan(loaded.weights.hidden, routes, none);
                    const ExpertPlan mixed_plan(loaded.weights.hidden, routes, mixed);
                    std::unique_ptr<Iq4NativeCudaDown> timed_iq4;
                    std::unique_ptr<Q6NativeCudaDown> timed_q6;
                    std::unique_ptr<Mxfp4NativeCudaDown> timed_mxfp4;
                    if (iq4_exact) timed_iq4 = std::make_unique<Iq4NativeCudaDown>(loaded, mixed);
                    if (q6_exact) timed_q6 = std::make_unique<Q6NativeCudaDown>(loaded, mixed);
                    if (mxfp4_exact) timed_mxfp4 = std::make_unique<Mxfp4NativeCudaDown>(loaded, mixed);
                    const auto exact_iq4_down = [&](const std::vector<ExpertJob> & jobs) {
                        const auto packed = exact.pack_down(exact.activations(capture.input, jobs));
                        return iq4_exact ? timed_iq4->execute(packed, jobs) :
                               q6_exact ? timed_q6->execute(packed, jobs) : timed_mxfp4->execute(packed, jobs);
                    };
                    const auto run = [&](int mode) {
                        const auto start = std::chrono::steady_clock::now();
                        const auto output = mode == 0 ? cpu_plan.execute(exact_cpu, {}) :
                                            mode == 1 ? mixed_plan.execute(exact_cpu, exact_cpu) :
                                            mode == 2 ? mixed_plan.execute(exact_cpu, exact_gpu_down) :
                                                        mixed_plan.execute(exact_cpu, exact_iq4_down);
                        const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                        const bool correct = compare(capture.output, output, 1e-5 + 1e-4 * cpu_error.max_reference, 1e-4).passed;
                        timing.all_correct = timing.all_correct && correct;
                        if (mode == 3) {
                            const bool bitwise = output.size() == exact_cpu_output.size() &&
                                std::memcmp(output.data(), exact_cpu_output.data(), output.size() * sizeof(float)) == 0;
                            timing.iq4_bitwise = timing.iq4_bitwise && bitwise;
                        }
                        return milliseconds;
                    };
                    run(0);
                    run(1);
                    run(2);
                    if (iq4_exact || q6_exact || mxfp4_exact) run(3);
                    const int modes = (iq4_exact || q6_exact || mxfp4_exact) ? 4 : 3;
                    for (int repeat = 0; repeat < repetitions; ++repeat) {
                        for (int offset = 0; offset < modes; ++offset) {
                            const int mode = (repeat + offset) % modes;
                            const double elapsed = run(mode);
                            (mode == 0 ? timing.cpu_ms : mode == 1 ? timing.cpu2_ms :
                             mode == 2 ? timing.mixed_ms : timing.iq4_ms).push_back(elapsed);
                        }
                    }
                }
            }
#else
            if (native_kernel) throw std::runtime_error("native CPU kernel diagnostic is unavailable on this platform");
#endif
#else
            throw std::runtime_error("native activation diagnostic is unavailable on this platform");
#endif
        }
        std::printf("{\"layer\":%d,\"case\":\"%s\",\"baseline_repeat\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"top_equal\":true,\"pass\":true},"
                    "\"observer\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"top_equal\":true,\"pass\":true},"
                    "\"selected_experts\":%zu,\"hidden\":%zu,\"input\":\"actual_prompt_last_row\","
                    "\"cpu_vs_native\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s},"
                    "\"mixed_vs_native\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s},"
                    "\"mixed_vs_f32_cpu\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s}",
                    layer, case_name.c_str(), baseline_repeat.max_abs, baseline_repeat.relative_l2,
                    observer.max_abs, observer.relative_l2, capture.ids.size(), loaded.weights.hidden,
                    cpu_error.max_abs, cpu_error.relative_l2, cpu_error.passed ? "true" : "false",
                    mixed_error.max_abs, mixed_error.relative_l2, mixed_error.passed ? "true" : "false",
                    split_error.max_abs, split_error.relative_l2, split_error.passed ? "true" : "false");
        if (native_activation)
            std::printf(",\"cpu_native_roundtrip\":{\"gate_up_type\":\"%s\",\"down_type\":\"%s\","
                        "\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s},"
                        "\"mixed_native_roundtrip\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s},"
                        "\"mixed_roundtrip_vs_cpu_roundtrip\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s}",
                        gate_dot.c_str(), down_dot.c_str(), native_error.max_abs, native_error.relative_l2,
                        native_error.passed ? "true" : "false", native_mixed_error.max_abs, native_mixed_error.relative_l2,
                        native_mixed_error.passed ? "true" : "false", native_split_error.max_abs, native_split_error.relative_l2,
                        native_split_error.passed ? "true" : "false");
        if (stages) {
            std::printf(",\"stage_errors\":{");
            bool first = true;
            for (const auto & [name, error] : stage_errors) {
                std::printf(first ? "\"%s\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s}"
                                  : ",\"%s\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s}",
                            name.c_str(), error.max_abs, error.relative_l2, error.passed ? "true" : "false");
                first = false;
            }
            std::putchar('}');
            if (native_kernel) {
                std::printf(",\"native_kernel_stage_errors\":{");
                bool first_kernel = true;
                for (const auto & [name, error] : kernel_stage_errors) {
                    std::printf(first_kernel ? "\"%s\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s}"
                                             : ",\"%s\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s}",
                                name.c_str(), error.max_abs, error.relative_l2, error.passed ? "true" : "false");
                    first_kernel = false;
                }
                std::putchar('}');
            }
        }
        if (native_kernel)
            std::printf(",\"cpu_native_kernel\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s},"
                        "\"native_cpu_gpu_down\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s},"
                        "\"native_cpu_gpu_down_vs_cpu\":{\"max_abs\":%.9g,\"relative_l2\":%.9g,\"pass\":%s}",
                        native_kernel_error.max_abs, native_kernel_error.relative_l2, native_kernel_error.passed ? "true" : "false",
                        kernel_mixed_error.max_abs, kernel_mixed_error.relative_l2, kernel_mixed_error.passed ? "true" : "false",
                        kernel_split_error.max_abs, kernel_split_error.relative_l2, kernel_split_error.passed ? "true" : "false");
        if (inject)
            std::printf(",\"injection\":{\"mode\":\"%s\",\"decode_only\":%s,\"iq4_exact\":%s,\"q6_exact\":%s,\"mxfp4_exact\":%s,\"captured_activation\":%s,\"allow_nonidentical_layer\":%s,\"layer_bitwise\":%s,\"positions\":%zu,\"interventions\":%d,\"original_layer_control_pass\":%s,\"layer_max_abs\":%.9g,"
                        "\"layer_relative_l2\":%.9g,\"logit_max_abs\":%.9g,\"logit_relative_l2\":%.9g,"
                        "\"top_equal\":%s,\"pass\":%s}", injection_mode.c_str(), decode_only ? "true" : "false", iq4_exact ? "true" : "false", q6_exact ? "true" : "false", mxfp4_exact ? "true" : "false", captured_activation ? "true" : "false",
                        allow_nonidentical_layer ? "true" : "false", injected.all_layers_bitwise ? "true" : "false",
                        injected.logits.size(), injected.injections, injection_control.passed ? "true" : "false",
                        injected.max_layer_abs, injected.max_layer_relative_l2,
                        injected_logits.max_abs, injected_logits.relative_l2, injected_top ? "true" : "false",
                        injected_logits.passed && injected_top && injection_control.passed ? "true" : "false");
        if (repetitions) {
            std::printf(",\"benchmark\":{\"warmups_per_path\":1,\"repetitions\":%d,\"cpu_median_ms\":%.9g,"
                        "\"cpu_two_workers_median_ms\":%.9g,\"mixed_median_ms\":%.9g,\"all_correct\":%s", repetitions,
                        median(timing.cpu_ms), median(timing.cpu2_ms), median(timing.mixed_ms),
                        timing.all_correct ? "true" : "false");
            print_samples("cpu_samples_ms", timing.cpu_ms);
            print_samples("cpu_two_workers_samples_ms", timing.cpu2_ms);
            print_samples("mixed_samples_ms", timing.mixed_ms);
            if (iq4_exact) {
                std::printf(",\"iq4_exact_median_ms\":%.9g,\"iq4_bitwise\":%s", median(timing.iq4_ms),
                            timing.iq4_bitwise ? "true" : "false");
                print_samples("iq4_exact_samples_ms", timing.iq4_ms);
            }
            if (q6_exact) {
                std::printf(",\"q6_exact_median_ms\":%.9g,\"q6_bitwise\":%s", median(timing.iq4_ms),
                            timing.iq4_bitwise ? "true" : "false");
                print_samples("q6_exact_samples_ms", timing.iq4_ms);
            }
            if (mxfp4_exact) {
                std::printf(",\"mxfp4_exact_median_ms\":%.9g,\"mxfp4_bitwise\":%s", median(timing.iq4_ms),
                            timing.iq4_bitwise ? "true" : "false");
                print_samples("mxfp4_exact_samples_ms", timing.iq4_ms);
            }
            std::putchar('}');
        }
        std::puts("}");
        llama_backend_free();
        const bool passed = native_kernel ? native_kernel_error.passed && kernel_mixed_error.passed && kernel_split_error.passed
                            : native_activation ? native_error.passed && native_mixed_error.passed && native_split_error.passed
                                                : cpu_error.passed && mixed_error.passed && split_error.passed;
        return passed && (!inject || (injection_control.passed && injected_logits.passed && injected_top)) &&
                       (!repetitions || (timing.all_correct && (!(iq4_exact || q6_exact || mxfp4_exact) || timing.iq4_bitwise))) ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "actual expert probe: %s\n", error.what());
        return 2;
    }
}
