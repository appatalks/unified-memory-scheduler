#include "gguf_experts.hpp"

#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace vrambridge::experimental;

namespace {

void print_component(const char * name, const std::vector<float> & values, std::size_t count, bool first) {
    std::printf(first ? "\"%s\":[" : ",\"%s\":[", name);
    if (!values.empty()) {
        const auto size = values.size() / count;
        const std::size_t samples[] = {0, size / 3, size / 2, size - 1};
        bool initial = true;
        for (std::size_t expert = 0; expert < count; ++expert) {
            for (const auto offset : samples) {
                std::printf(initial ? "%.9g" : ",%.9g", static_cast<double>(values[expert * size + offset]));
                initial = false;
            }
        }
    }
    std::putchar(']');
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try {
        const auto loaded = load_gguf_experts(argv[1], 0, {2, 0, 1}, F32Activation::silu);
        const auto & w = loaded.weights;
        std::putchar('{');
        print_component("gate", w.gate, w.experts, true);
        print_component("up", w.up, w.experts, false);
        print_component("down", w.down, w.experts, false);
        print_component("gate_bias", w.gate_bias, w.experts, false);
        print_component("up_bias", w.up_bias, w.experts, false);
        print_component("down_bias", w.down_bias, w.experts, false);
        std::puts("}");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "GGUF expert inspect: %s\n", error.what());
        return 2;
    }
}
