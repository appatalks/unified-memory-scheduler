#include "vrambridge/info.hpp"

#include <chrono>
#include <charconv>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc {};
    gmtime_r(&time, &utc);
    char value[32] {};
    std::strftime(value, sizeof(value), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return value;
}

std::size_t parse_size_mib(const std::string & value) {
    const auto mib = std::stoull(value);
    if (mib == 0) {
        throw std::invalid_argument("benchmark size must be greater than zero");
    }
    return mib * 1024ULL * 1024ULL;
}

void print_usage(std::ostream & output) {
    output << "Usage: vrambridge-info [--json] [--output FILE] [--no-bench] "
              "[--size-mib N] [--iterations N] [--device N] [--ram-threads N]\n";
}

} // namespace

int main(int argc, char ** argv) {
    bool json = false;
    bool run_benchmarks = true;
    std::filesystem::path output_path;
    vrambridge::BenchmarkOptions options;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string argument(argv[index]);
            if (argument == "--json") {
                json = true;
            } else if (argument == "--output" && index + 1 < argc) {
                output_path = argv[++index];
            } else if (argument == "--no-bench") {
                run_benchmarks = false;
            } else if (argument == "--size-mib" && index + 1 < argc) {
                options.bytes = parse_size_mib(argv[++index]);
            } else if (argument == "--iterations" && index + 1 < argc) {
                options.iterations = std::stoi(argv[++index]);
            } else if (argument == "--device" && index + 1 < argc) {
                options.cuda_device = std::stoi(argv[++index]);
            } else if (argument == "--ram-threads" && index + 1 < argc) {
                const std::string value = argv[++index];
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), options.ram_threads);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || options.ram_threads < 1) {
                    throw std::invalid_argument("RAM threads must be a positive integer");
                }
            } else if (argument == "--help" || argument == "-h") {
                print_usage(std::cout);
                return 0;
            } else {
                throw std::invalid_argument("unknown or incomplete argument: " + argument);
            }
        }

        vrambridge::HardwareReport report;
        report.generated_at_utc = utc_timestamp();
        report.system = vrambridge::collect_system_info();
        report.cuda = vrambridge::collect_cuda_info();
        if (run_benchmarks) {
            report.benchmarks = vrambridge::run_ram_benchmarks(options);
            auto cuda_results = vrambridge::run_cuda_benchmarks(options);
            report.benchmarks.insert(report.benchmarks.end(), cuda_results.begin(), cuda_results.end());
        }

        std::ofstream output_file;
        std::ostream * output = &std::cout;
        if (!output_path.empty()) {
            if (!output_path.parent_path().empty()) {
                std::filesystem::create_directories(output_path.parent_path());
            }
            output_file.open(output_path);
            if (!output_file) {
                throw std::runtime_error("cannot open output file: " + output_path.string());
            }
            output = &output_file;
        }

        if (json) {
            vrambridge::print_json_report(report, *output);
        } else {
            vrambridge::print_human_report(report, *output);
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "vrambridge-info: " << error.what() << '\n';
        print_usage(std::cerr);
        return 2;
    }
}