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
        std::cout << "PASS: frequency, cold-start seed priority, LRU and stable ties\n";
        return 0;
    } catch (const std::exception & exc) {
        std::cerr << "MoE admission: " << exc.what() << '\n';
        return 1;
    }
}
