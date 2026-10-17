#include "llama-moe-cache.h"

#include <iostream>
#include <stdexcept>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        require(llama_moe_colder_admission({0, 100, 99}, {1, 0, 1}),
                "actual recent frequency must dominate the seed prior");
        require(!llama_moe_colder_admission({2, 0, 1}, {1, 100, 99}),
                "a more frequent expert must not be evicted for seed order");
        require(llama_moe_colder_admission({0, 0, 99}, {0, 100, 1}),
                "unseen unseeded experts should precede strong cold-start seeds");
        require(!llama_moe_colder_admission({0, 100, 1}, {0, 0, 99}),
                "the strongest seed must not be the first cold-start victim");
        require(llama_moe_colder_admission({2, 100, 1}, {2, 0, 99}),
                "active-frequency ties must use recency, not training priority");
        require(!llama_moe_colder_admission({2, 0, 99}, {2, 100, 1}),
                "recent evidence must override seed preference when frequency ties");
        require(llama_moe_colder_admission({0, 0, 1}, {0, 0, 99}),
                "untrained cold-start ties should fall back to LRU");
        require(!llama_moe_colder_admission({4, 10, 12}, {4, 10, 12}),
                "equal ranks must preserve a stable victim");
        require(llama_moe_probation_margin(false, 0) == 4, "unobserved experts retain baseline admission");
        require(llama_moe_probation_margin(true, 0) == 7, "unreused lifetime requires stronger readmission evidence");
        require(llama_moe_probation_margin(true, 1) == 6, "one future hit reduces probation");
        require(llama_moe_probation_margin(true, 2) == 5, "two future hits reduce probation");
        require(llama_moe_probation_margin(true, 3) == 4, "productive lifetimes keep baseline margin");
        require(llama_moe_probation_margin(true, UINT64_MAX) == 4, "large reuse counts must not overflow");
        require(llama_moe_anchor_slots(100, 4, 80) == 24, "anchor quota must use available tensor-derived slots");
        require(llama_moe_anchor_slots(100, 4, 10) == 10, "anchor quota cannot exceed provided training seeds");
        require(llama_moe_anchor_slots(100, 4, 0) == 0, "causal cold-start plans cannot invent training seeds");
        require(llama_moe_anchor_slots(4, 4, 0) == 0, "reserved zero IDs cannot become anchors");
        require(llama_moe_cold_gpu_budget(0) == 0, "resident work must not cause a transient upload");
        require(llama_moe_cold_gpu_budget(1) == 0, "one miss must retain the CPU path");
        require(llama_moe_cold_gpu_budget(2) == 1, "two misses can split one expert to each device");
        require(llama_moe_cold_gpu_budget(10) == 1, "offload must remain within the single scratch slot");
        require(llama_moe_cold_gpu_budget(128) == 1, "budget must not assume a model's top-k");
        require(llama_moe_gpu_slot_limit(100, false) == 100, "disabled mode must retain its original ID bounds");
        require(100 < llama_moe_gpu_slot_limit(100, true), "transient scratch ID must be valid for GPU upload");
        require(101 == llama_moe_gpu_slot_limit(100, true), "transient mode must expose exactly one extra slot");
        std::cout << "PASS: frequency, cold-start seed priority, LRU and stable ties\n";
        return 0;
    } catch (const std::exception & exc) {
        std::cerr << "MoE admission: " << exc.what() << '\n';
        return 1;
    }
}
