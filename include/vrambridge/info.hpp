#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace vrambridge {

struct BandwidthResult {
    std::string name;
    std::string direction;
    std::string memory_kind;
    std::size_t bytes = 0;
    int iterations = 0;
    double seconds = 0.0;
    double gb_per_second = 0.0;
    bool available = false;
    std::string error;
};

struct GpuInfo {
    int index = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    std::uint64_t vram_total_bytes = 0;
    std::uint64_t vram_free_bytes = 0;
    std::string pci_bus_id;
    int pcie_generation_current = 0;
    int pcie_generation_max = 0;
    int pcie_width_current = 0;
    int pcie_width_max = 0;
    double pcie_theoretical_gb_per_second = 0.0;
    double vram_theoretical_gb_per_second = 0.0;
    std::string resizable_bar_status = "unknown";
};

struct CudaInfo {
    bool available = false;
    std::string error;
    int runtime_version = 0;
    int driver_version = 0;
    std::string nvidia_driver_version;
    std::vector<GpuInfo> devices;
};

struct SystemInfo {
    std::string cpu_model;
    unsigned int physical_core_count = 0;
    unsigned int logical_cpu_count = 0;
    std::uint64_t ram_total_bytes = 0;
    std::uint64_t ram_available_bytes = 0;
    std::string numa_nodes;
    std::string numa_topology;
    std::string os;
    std::string kernel;
    std::uint64_t hugepages_total = 0;
    std::uint64_t hugepages_free = 0;
    std::uint64_t hugepage_size_bytes = 0;
    bool iommu_groups_detected = false;
    std::string iommu_command_line;
};

struct HardwareReport {
    std::string generated_at_utc;
    SystemInfo system;
    CudaInfo cuda;
    std::vector<BandwidthResult> benchmarks;
};

struct BenchmarkOptions {
    std::size_t bytes = 256ULL * 1024ULL * 1024ULL;
    int iterations = 5;
    int cuda_device = 0;
};

SystemInfo collect_system_info();
CudaInfo collect_cuda_info();
std::vector<BandwidthResult> run_ram_benchmarks(const BenchmarkOptions & options);
std::vector<BandwidthResult> run_cuda_benchmarks(const BenchmarkOptions & options);

void print_human_report(const HardwareReport & report, std::ostream & output);
void print_json_report(const HardwareReport & report, std::ostream & output);

} // namespace vrambridge