#include "expert_dispatch.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace vrambridge::experimental;
using namespace std::chrono_literals;

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void rejects(Function function, const char * message) {
    bool failed = false;
    try { function(); } catch (const std::exception &) { failed = true; }
    require(failed, message);
}

float value(const ExpertJob & job, std::size_t column) {
    return static_cast<float>(100 * job.expert + 10 * job.row + column);
}

void check_routing() {
    const std::vector<std::vector<Route>> routes{
        {{3, 0.25f}, {0, 0.75f}, {2, -0.5f}},
        {{1, 0.2f}, {3, 0.8f}}
    };
    const ExpertPlan plan(3, routes, {0, 1, 0, 1});
    require(plan.rows() == 2 && plan.width() == 3 && plan.cpu_jobs().size() == 2 &&
            plan.gpu_jobs().size() == 3, "incorrect partition sizes");
    const auto runner = [](const std::vector<ExpertJob> & jobs) {
        std::vector<float> output;
        for (const auto & job : jobs)
            for (std::size_t col = 0; col < 3; ++col) output.push_back(value(job, col));
        return output;
    };
    const auto actual = plan.execute(runner, runner);
    for (std::size_t row = 0; row < routes.size(); ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            float expected = 0;
            for (std::size_t rank = 0; rank < routes[row].size(); ++rank) {
                const auto & route = routes[row][rank];
                expected += route.weight * value({row, rank, route.expert}, col);
            }
            require(actual[row * 3 + col] == expected, "routing weight or expert output changed");
        }
    }
    const auto one = [](const std::vector<ExpertJob> & jobs) { return std::vector<float>(jobs.size(), 1.0f); };
    require(ExpertPlan(1, {{{0, 1.0f}}}, {0}).execute(one, {}) == std::vector<float>{1.0f},
            "CPU-only execution failed");
    require(ExpertPlan(1, {{{0, 1.0f}}}, {1}).execute({}, one) == std::vector<float>{1.0f},
            "GPU-only execution failed");
}

void check_reduction_order() {
    const ExpertPlan plan(1, {{{0, 1}, {1, 1}, {2, 1}}}, {0, 1, 0});
    const auto runner = [](const std::vector<ExpertJob> & jobs) {
        std::vector<float> result;
        for (const auto & job : jobs) result.push_back(job.expert == 0 ? 1e20f : job.expert == 1 ? -1e20f : 1.0f);
        return result;
    };
    require(plan.execute(runner, runner) == std::vector<float>{1.0f}, "expert-rank reduction order changed");
}

void check_overlap_and_failure_cleanup() {
    const ExpertPlan plan(1, {{{0, 1}, {1, 1}}}, {0, 1});
    std::promise<void> gpu_entered, cpu_entered;
    auto gpu_signal = gpu_entered.get_future();
    auto cpu_signal = cpu_entered.get_future();
    const auto gpu = [&](const std::vector<ExpertJob> &) {
        gpu_entered.set_value();
        if (cpu_signal.wait_for(2s) != std::future_status::ready) throw std::runtime_error("CPU did not overlap GPU");
        return std::vector<float>{2.0f};
    };
    const auto cpu = [&](const std::vector<ExpertJob> &) {
        if (gpu_signal.wait_for(2s) != std::future_status::ready) throw std::runtime_error("GPU was not started");
        cpu_entered.set_value();
        return std::vector<float>{1.0f};
    };
    require(plan.execute(cpu, gpu) == std::vector<float>{3.0f}, "concurrent execution failed");

    std::atomic<bool> gpu_completed{false};
    const auto slow_gpu = [&](const std::vector<ExpertJob> &) {
        std::this_thread::sleep_for(20ms);
        gpu_completed = true;
        return std::vector<float>{2.0f};
    };
    const ExpertRunner failed_cpu = [](const std::vector<ExpertJob> &) -> std::vector<float> {
        throw std::runtime_error("CPU failure");
    };
    rejects([&] { plan.execute(failed_cpu, slow_gpu); }, "CPU failure was hidden");
    require(gpu_completed, "GPU work outlived a failed execution");
}

void check_invalid_inputs() {
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    rejects([] { ExpertPlan(0, {{{0, 1}}}, {0}); }, "zero width accepted");
    rejects([] { ExpertPlan(1, {}, {0}); }, "zero rows accepted");
    rejects([] { ExpertPlan(1, {{{0, 1}}}, {2}); }, "invalid residency accepted");
    rejects([] { ExpertPlan(1, {{{1, 1}}}, {0}); }, "out-of-range expert accepted");
    rejects([] { ExpertPlan(1, {{{0, 1}, {0, 1}}}, {0, 0}); }, "duplicate route accepted");
    rejects([&] { ExpertPlan(1, {{{0, nan}}}, {0}); }, "non-finite weight accepted");
    const ExpertPlan plan(1, {{{0, 1}, {1, 1}}}, {0, 1});
    bool cpu_called = false;
    const auto cpu = [&](const std::vector<ExpertJob> & jobs) {
        cpu_called = true;
        return std::vector<float>(jobs.size(), 1.0f);
    };
    rejects([&] { plan.execute(cpu, {}); }, "missing GPU backend accepted");
    require(!cpu_called, "work started before missing-backend validation");
    const auto empty = [](const std::vector<ExpertJob> &) { return std::vector<float>{}; };
    const auto nan_output = [&](const std::vector<ExpertJob> & jobs) { return std::vector<float>(jobs.size(), nan); };
    rejects([&] { plan.execute(cpu, empty); }, "wrong backend output shape accepted");
    rejects([&] { plan.execute(cpu, nan_output); }, "non-finite backend output accepted");
}

} // namespace

int main() {
    try {
        check_routing();
        check_reduction_order();
        check_overlap_and_failure_cleanup();
        check_invalid_inputs();
        std::puts("expert dispatch routing, ordering, overlap and failure gates passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "expert dispatch: %s\n", error.what());
        return 1;
    }
}
