#include "vrambridge/info.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <set>
#include <string_view>
#include <sys/utsname.h>
#include <thread>
#include <vector>

namespace vrambridge {
namespace {

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n\"");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n\"");
    return value.substr(first, last - first + 1);
}

std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::map<std::string, std::uint64_t> read_meminfo() {
    std::ifstream input("/proc/meminfo");
    std::map<std::string, std::uint64_t> values;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        std::uint64_t value = 0;
        std::string unit;
        if (!(fields >> key >> value)) {
            continue;
        }
        fields >> unit;
        if (!key.empty() && key.back() == ':') {
            key.pop_back();
        }
        values[key] = unit == "kB" ? value * 1024ULL : value;
    }
    return values;
}

std::string read_cpu_model() {
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        const auto key = trim(line.substr(0, separator));
        if (key == "model name" || key == "Hardware" || key == "Processor") {
            return trim(line.substr(separator + 1));
        }
    }
    return "unknown";
}

unsigned int read_physical_core_count() {
    std::ifstream input("/proc/cpuinfo");
    std::set<std::pair<int, int>> cores;
    std::string line;
    int physical_id = 0;
    int core_id = -1;
    auto record_core = [&] {
        if (core_id >= 0) {
            cores.emplace(physical_id, core_id);
        }
        physical_id = 0;
        core_id = -1;
    };
    while (std::getline(input, line)) {
        if (line.empty()) {
            record_core();
            continue;
        }
        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        const auto key = trim(line.substr(0, separator));
        const auto value = trim(line.substr(separator + 1));
        if (key == "physical id") {
            physical_id = std::stoi(value);
        } else if (key == "core id") {
            core_id = std::stoi(value);
        }
    }
    record_core();
    return static_cast<unsigned int>(cores.size());
}

std::string read_numa_topology() {
    const std::filesystem::path root("/sys/devices/system/node");
    std::error_code error;
    std::vector<std::string> nodes;
    for (const auto & entry : std::filesystem::directory_iterator(root, error)) {
        if (error || !entry.is_directory()) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (!name.starts_with("node") || name.size() <= 4 ||
            !std::all_of(name.begin() + 4, name.end(), [](unsigned char value) { return std::isdigit(value); })) {
            continue;
        }
        const auto cpus = trim(read_text_file(entry.path() / "cpulist"));
        nodes.push_back(name + " CPUs " + (cpus.empty() ? "unknown" : cpus));
    }
    std::sort(nodes.begin(), nodes.end());
    std::ostringstream output;
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        if (index != 0) {
            output << "; ";
        }
        output << nodes[index];
    }
    return output.str().empty() ? "unknown" : output.str();
}

std::string read_os_name() {
    std::ifstream input("/etc/os-release");
    std::string line;
    while (std::getline(input, line)) {
        if (line.starts_with("PRETTY_NAME=")) {
            return trim(line.substr(std::string_view("PRETTY_NAME=").size()));
        }
    }
    return "Linux";
}

bool has_iommu_groups() {
    const std::filesystem::path groups("/sys/kernel/iommu_groups");
    std::error_code error;
    if (!std::filesystem::is_directory(groups, error)) {
        return false;
    }
    return std::filesystem::directory_iterator(groups, error) != std::filesystem::directory_iterator();
}

std::string iommu_parameters() {
    const auto command_line = read_text_file("/proc/cmdline");
    std::istringstream input(command_line);
    std::string argument;
    std::vector<std::string> matches;
    while (input >> argument) {
        if (argument.find("iommu") != std::string::npos) {
            matches.push_back(argument);
        }
    }
    std::ostringstream output;
    for (std::size_t index = 0; index < matches.size(); ++index) {
        if (index != 0) {
            output << ' ';
        }
        output << matches[index];
    }
    return output.str();
}

std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character);
            } else {
                output << character;
            }
        }
    }
    return output.str();
}

double gib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1ULL << 30);
}

std::string cuda_version_string(int version) {
    if (version <= 0) {
        return "unknown";
    }
    std::ostringstream output;
    output << version / 1000 << '.' << (version % 1000) / 10;
    return output.str();
}

BandwidthResult measure_ram_write(std::vector<std::uint64_t> & buffer, int iterations) {
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::fill(buffer.begin(), buffer.end(), static_cast<std::uint64_t>(iteration + 1));
    }
    const auto stopped = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stopped - started).count();
    const std::size_t bytes = buffer.size() * sizeof(buffer.front());
    return {
        .name = "ram_sequential_write",
        .direction = "cpu_to_ram",
        .memory_kind = "pageable",
        .bytes = bytes,
        .iterations = iterations,
        .seconds = seconds,
        .gb_per_second = static_cast<double>(bytes) * iterations / seconds / 1.0e9,
        .available = true,
    };
}

BandwidthResult measure_ram_read(const std::vector<std::uint64_t> & buffer, int iterations) {
    std::uint64_t checksum = 0;
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        checksum = std::accumulate(buffer.begin(), buffer.end(), checksum);
    }
    const auto stopped = std::chrono::steady_clock::now();
    if (checksum == 0) {
        throw std::runtime_error("RAM benchmark produced an invalid checksum");
    }
    const double seconds = std::chrono::duration<double>(stopped - started).count();
    const std::size_t bytes = buffer.size() * sizeof(buffer.front());
    return {
        .name = "ram_sequential_read",
        .direction = "ram_to_cpu",
        .memory_kind = "pageable",
        .bytes = bytes,
        .iterations = iterations,
        .seconds = seconds,
        .gb_per_second = static_cast<double>(bytes) * iterations / seconds / 1.0e9,
        .available = true,
    };
}

} // namespace

SystemInfo collect_system_info() {
    const auto memory = read_meminfo();
    struct utsname uts {};
    const bool have_uname = uname(&uts) == 0;

    SystemInfo info;
    info.cpu_model = read_cpu_model();
    info.physical_core_count = read_physical_core_count();
    info.logical_cpu_count = std::thread::hardware_concurrency();
    info.ram_total_bytes = memory.contains("MemTotal") ? memory.at("MemTotal") : 0;
    info.ram_available_bytes = memory.contains("MemAvailable") ? memory.at("MemAvailable") : 0;
    info.numa_nodes = trim(read_text_file("/sys/devices/system/node/online"));
    if (info.numa_nodes.empty()) {
        info.numa_nodes = "unknown";
    }
    info.numa_topology = read_numa_topology();
    info.os = read_os_name();
    info.kernel = have_uname ? std::string(uts.release) : "unknown";
    info.hugepages_total = memory.contains("HugePages_Total") ? memory.at("HugePages_Total") : 0;
    info.hugepages_free = memory.contains("HugePages_Free") ? memory.at("HugePages_Free") : 0;
    info.hugepage_size_bytes = memory.contains("Hugepagesize") ? memory.at("Hugepagesize") : 0;
    info.iommu_groups_detected = has_iommu_groups();
    info.iommu_command_line = iommu_parameters();
    return info;
}

std::vector<BandwidthResult> run_ram_benchmarks(const BenchmarkOptions & options) {
    if (options.iterations <= 0) {
        throw std::invalid_argument("iterations must be greater than zero");
    }
    const std::size_t elements = std::max<std::size_t>(1, options.bytes / sizeof(std::uint64_t));
    std::vector<std::uint64_t> buffer(elements, 1);
    std::vector<BandwidthResult> results;
    results.push_back(measure_ram_write(buffer, options.iterations));
    results.push_back(measure_ram_read(buffer, options.iterations));
    return results;
}

void print_human_report(const HardwareReport & report, std::ostream & output) {
    std::uint64_t free_vram = 0;
    for (const auto & gpu : report.cuda.devices) {
        free_vram += gpu.vram_free_bytes;
    }

    output << "VRAMBridge Hardware Report\n\n";
    output << "System:\n";
    output << "  CPU: " << report.system.cpu_model << '\n';
        output << "  CPU cores: " << report.system.physical_core_count << " physical, "
            << report.system.logical_cpu_count << " logical\n";
    output << "  RAM: " << std::fixed << std::setprecision(2) << gib(report.system.ram_total_bytes)
           << " GiB total, " << gib(report.system.ram_available_bytes) << " GiB available\n";
    output << "  NUMA nodes: " << report.system.numa_nodes << '\n';
    output << "  NUMA topology: " << report.system.numa_topology << '\n';
    output << "  OS/kernel: " << report.system.os << " / " << report.system.kernel << '\n';
    output << "  Huge pages: " << report.system.hugepages_free << '/' << report.system.hugepages_total
           << " free, " << report.system.hugepage_size_bytes / 1024 << " KiB each\n";
    output << "  IOMMU groups: " << (report.system.iommu_groups_detected ? "detected" : "not detected") << '\n';

    output << "\nCUDA:\n";
    if (!report.cuda.available) {
        output << "  unavailable: " << report.cuda.error << '\n';
    } else {
         output << "  NVIDIA driver: " << report.cuda.nvidia_driver_version << '\n';
         output << "  CUDA driver/runtime: " << cuda_version_string(report.cuda.driver_version) << " / "
             << cuda_version_string(report.cuda.runtime_version) << '\n';
        for (const auto & gpu : report.cuda.devices) {
            output << "  GPU " << gpu.index << ": " << gpu.name << " (compute " << gpu.compute_major << '.' << gpu.compute_minor << ")\n";
            output << "    VRAM: " << gib(gpu.vram_free_bytes) << " GiB free / " << gib(gpu.vram_total_bytes) << " GiB total\n";
                 output << "    PCI: " << gpu.pci_bus_id << ", Gen " << gpu.pcie_generation_current << " x"
                     << gpu.pcie_width_current << " (max Gen " << gpu.pcie_generation_max << " x" << gpu.pcie_width_max << ")\n";
                 output << "    PCIe theoretical: " << gpu.pcie_theoretical_gb_per_second << " GB/s per direction\n";
                 output << "    VRAM theoretical: " << gpu.vram_theoretical_gb_per_second << " GB/s\n";
                 output << "    Resizable BAR: " << gpu.resizable_bar_status << '\n';
        }
    }

    output << "\nMeasured bandwidth:\n";
    for (const auto & result : report.benchmarks) {
        output << "  " << std::left << std::setw(28) << result.name << std::right;
        if (result.available) {
            output << std::fixed << std::setprecision(2) << result.gb_per_second << " GB/s\n";
        } else {
            output << "unavailable: " << result.error << '\n';
        }
    }

    output << "\nCurrent free memory envelope:\n";
    output << "  Fast tier (VRAM): " << gib(free_vram) << " GiB\n";
    output << "  Slow tier (RAM):  " << gib(report.system.ram_available_bytes) << " GiB\n";
    output << "  Tiers are not performance-equivalent and are not reported as a single VRAM capacity.\n";
}

void print_json_report(const HardwareReport & report, std::ostream & output) {
    const auto quoted = [](std::string_view value) { return '"' + json_escape(value) + '"'; };
    output << std::setprecision(10);
    output << "{\n";
    output << "  \"schema_version\": 1,\n";
    output << "  \"generated_at_utc\": " << quoted(report.generated_at_utc) << ",\n";
    output << "  \"system\": {\n";
    output << "    \"cpu_model\": " << quoted(report.system.cpu_model) << ",\n";
    output << "    \"physical_core_count\": " << report.system.physical_core_count << ",\n";
    output << "    \"logical_cpu_count\": " << report.system.logical_cpu_count << ",\n";
    output << "    \"ram_total_bytes\": " << report.system.ram_total_bytes << ",\n";
    output << "    \"ram_available_bytes\": " << report.system.ram_available_bytes << ",\n";
    output << "    \"numa_nodes\": " << quoted(report.system.numa_nodes) << ",\n";
    output << "    \"numa_topology\": " << quoted(report.system.numa_topology) << ",\n";
    output << "    \"os\": " << quoted(report.system.os) << ",\n";
    output << "    \"kernel\": " << quoted(report.system.kernel) << ",\n";
    output << "    \"hugepages_total\": " << report.system.hugepages_total << ",\n";
    output << "    \"hugepages_free\": " << report.system.hugepages_free << ",\n";
    output << "    \"hugepage_size_bytes\": " << report.system.hugepage_size_bytes << ",\n";
    output << "    \"iommu_groups_detected\": " << (report.system.iommu_groups_detected ? "true" : "false") << ",\n";
    output << "    \"iommu_command_line\": " << quoted(report.system.iommu_command_line) << "\n";
    output << "  },\n";
    output << "  \"cuda\": {\n";
    output << "    \"available\": " << (report.cuda.available ? "true" : "false") << ",\n";
    output << "    \"error\": " << quoted(report.cuda.error) << ",\n";
    output << "    \"runtime_version\": " << report.cuda.runtime_version << ",\n";
    output << "    \"driver_version\": " << report.cuda.driver_version << ",\n";
    output << "    \"nvidia_driver_version\": " << quoted(report.cuda.nvidia_driver_version) << ",\n";
    output << "    \"devices\": [";
    for (std::size_t index = 0; index < report.cuda.devices.size(); ++index) {
        const auto & gpu = report.cuda.devices[index];
        output << (index == 0 ? "\n" : ",\n");
        output << "      {\"index\": " << gpu.index << ", \"name\": " << quoted(gpu.name)
               << ", \"compute_major\": " << gpu.compute_major << ", \"compute_minor\": " << gpu.compute_minor
               << ", \"vram_total_bytes\": " << gpu.vram_total_bytes << ", \"vram_free_bytes\": " << gpu.vram_free_bytes
               << ", \"pci_bus_id\": " << quoted(gpu.pci_bus_id) << ", \"pcie_generation_current\": " << gpu.pcie_generation_current
               << ", \"pcie_generation_max\": " << gpu.pcie_generation_max << ", \"pcie_width_current\": " << gpu.pcie_width_current
               << ", \"pcie_width_max\": " << gpu.pcie_width_max << ", \"pcie_theoretical_gb_per_second\": "
               << gpu.pcie_theoretical_gb_per_second << ", \"vram_theoretical_gb_per_second\": "
               << gpu.vram_theoretical_gb_per_second << ", \"resizable_bar_status\": "
               << quoted(gpu.resizable_bar_status) << '}';
    }
    output << (report.cuda.devices.empty() ? "]\n" : "\n    ]\n");
    output << "  },\n";
    output << "  \"benchmarks\": [";
    for (std::size_t index = 0; index < report.benchmarks.size(); ++index) {
        const auto & result = report.benchmarks[index];
        output << (index == 0 ? "\n" : ",\n");
        output << "    {\"name\": " << quoted(result.name) << ", \"direction\": " << quoted(result.direction)
               << ", \"memory_kind\": " << quoted(result.memory_kind) << ", \"bytes\": " << result.bytes
               << ", \"iterations\": " << result.iterations << ", \"seconds\": " << result.seconds
               << ", \"gb_per_second\": " << result.gb_per_second << ", \"available\": "
               << (result.available ? "true" : "false") << ", \"error\": " << quoted(result.error) << '}';
    }
    output << (report.benchmarks.empty() ? "]\n" : "\n  ]\n");
    output << "}\n";
}

} // namespace vrambridge