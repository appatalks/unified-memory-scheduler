#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

struct Fixture {
    char path[64] = "/tmp/vrambridge-expert-kernel-XXXXXX";
    int fd = mkstemp(path);
    void * mapping = nullptr;
    size_t size = 0;
    ~Fixture() {
        if (mapping) munmap(mapping, size);
        if (fd >= 0) close(fd);
        unlink(path);
    }
};

int main() {
    try {
        json cases = json::array();
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_M}) {
            Fixture fixture;
            if (fixture.fd < 0) throw std::runtime_error("Cannot create kernel fixture.");
            constexpr int width = 256, outputs = 16, experts = 16;
            std::vector<float> original(width * outputs * experts);
            for (size_t i = 0; i < original.size(); ++i) original[i] = std::sin(float(i % 997) * .031f);
            const size_t bytes = ggml_row_size(type, width) * outputs * experts;
            std::vector<uint8_t> data(bytes);
            const std::vector<float> importance(width, 1.0f);
            if (type == GGML_TYPE_F32) {
                std::memcpy(data.data(), original.data(), bytes);
            } else if (ggml_quantize_chunk(type, original.data(), data.data(), 0, outputs * experts, width, importance.data()) != bytes) {
                throw std::runtime_error("Cannot quantize kernel fixture.");
            }
            const size_t page = size_t(sysconf(_SC_PAGESIZE));
            fixture.size = (page + bytes + page - 1) / page * page;
            std::vector<uint8_t> file(fixture.size);
            std::memcpy(file.data() + page, data.data(), bytes);
            if (write(fixture.fd, file.data(), file.size()) != ssize_t(file.size()) || fsync(fixture.fd) != 0) {
                throw std::runtime_error("Cannot write kernel fixture.");
            }
            fixture.mapping = mmap(nullptr, fixture.size, PROT_READ, MAP_PRIVATE, fixture.fd, 0);
            if (fixture.mapping == MAP_FAILED) {
                fixture.mapping = nullptr;
                throw std::runtime_error("Cannot map kernel fixture.");
            }
            auto free_ctx = [](ggml_context * ctx) { ggml_free(ctx); };
            std::unique_ptr<ggml_context, decltype(free_ctx)> ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), free_ctx);
            auto * weights = ggml_new_tensor_3d(ctx.get(), type, width, outputs, experts);
            auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, 2, 1);
            auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, 1);
            weights->data = static_cast<char *>(fixture.mapping) + page;
            weights->buffer = ggml_backend_cpu_buffer_from_ptr(fixture.mapping, fixture.size);
            if (!weights->buffer) throw std::runtime_error("Cannot wrap mapped source.");
            auto free_buffer = [](ggml_backend_buffer_t buffer) { ggml_backend_buffer_free(buffer); };
            std::unique_ptr<ggml_backend_buffer, decltype(free_buffer)> mapped(weights->buffer, free_buffer);
            auto * output = ggml_mul_mat_id(ctx.get(), weights, input, ids);
            auto * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, output);
            std::unique_ptr<ggml_backend_buffer, decltype(free_buffer)> storage(
                ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()), free_buffer);
            std::vector<float> activations(width * 2);
            for (size_t i = 0; i < activations.size(); ++i) activations[i] = float(int(i % 17) - 8) * .1f;
            ggml_backend_tensor_set(input, activations.data(), 0, activations.size() * sizeof(float));
            auto free_backend = [](ggml_backend_t backend) { ggml_backend_free(backend); };
            std::unique_ptr<ggml_backend, decltype(free_backend)> backend(ggml_backend_cpu_init(), free_backend);
            ggml_backend_cpu_set_n_threads(backend.get(), 2);
            json results = json::array();
            void * source_pointer = weights->data;
            for (const auto & selected : {std::vector<int32_t>{0, 7}, {3, 12}, {0, 7}, {1, 2}}) {
                ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int32_t));
                if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("Native expert computation failed.");
                }
                if (weights->data != source_pointer) throw std::runtime_error("Source pointer was not restored.");
                std::vector<uint32_t> bits(ggml_nelements(output));
                ggml_backend_tensor_get(output, bits.data(), 0, bits.size() * sizeof(uint32_t));
                results.push_back(bits);
            }
            cases.push_back({{"type", ggml_type_name(type)}, {"output_bits", results},
                             {"source_pointer_restored", true}});
        }
        std::cout << json({{"cases", cases}}).dump() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Expert disk native-kernel test: " << error.what() << '\n';
        return 1;
    }
}
