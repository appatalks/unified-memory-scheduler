#include "llama.h"
#include "llama-graph.h"

#include <iostream>
#include <stdexcept>

#ifndef LLAMA_VRAMBRIDGE_PREFILL_DOWN_API
#error "This test requires the separate resident-down prefill research backend"
#endif

int main() {
    try {
        llm_graph_params prompt{};
        llm_graph_params decode{};
        if (prompt.prefill_down_phase || decode.prefill_down_phase) {
            throw std::runtime_error("Research prefill phase must default off");
        }
        prompt.prefill_down_phase = true;
        if (prompt.allow_reuse(decode) || decode.allow_reuse(prompt)) {
            throw std::runtime_error("Graphs must not be reused across explicit prefill/decode phases");
        }
        std::cout << "PASS: default-off phase and bidirectional graph-cache separation\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Prefill phase: " << error.what() << '\n';
        return 1;
    }
}
