#include "vrambridge/info.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <regex>
#include <vector>

namespace vrambridge {
namespace {

void check_cuda(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        return {};
    }
    std::string value;
    std::getline(input, value);
    return value;
}

int parse_leading_integer(const std::string & value) {
    if (value.empty()) {
        return 0;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception &) {
        return 0;
    }
}

double parse_link_speed(const std::string & value) {
    if (value.empty()) {
        return 0.0;
    }
    try {
        return std::stod(value);
    } catch (const std::exception &) {
        return 0.0;
    }
}

int pcie_generation(double giga_transfers_per_second) {
    if (giga_transfers_per_second >= 64.0) return 6;
    if (giga_transfers_per_second >= 32.0) return 5;
    if (giga_transfers_per_second >= 16.0) return 4;
    if (giga_transfers_per_second >= 8.0) return 3;
    if (giga_transfers_per_second >= 5.0) return 2;
    if (giga_transfers_per_second >= 2.5) return 1;
    return 0;
}

double pcie_gb_per_second_per_lane(int generation) {
    switch (generation) {
    case 1: return 0.250;
    case 2: return 0.500;
    case 3: return 0.984615;
    case 4: return 1.969231;
    case 5: return 3.938462;
    case 6: return 7.563077;
    default: return 0.0;
    }
}

std::filesystem::path pci_sysfs_path(const cudaDeviceProp & properties) {
    std::ostringstream bdf;
    bdf << std::hex << std::setfill('0') << std::setw(4) << properties.pciDomainID << ':'
        << std::setw(2) << properties.pciBusID << ':' << std::setw(2) << properties.pciDeviceID << ".0";
    return std::filesystem::path("/sys/bus/pci/devices") / bdf.str();
}

std::string detect_resizable_bar(const std::filesystem::path & pci_path, std::uint64_t vram_total_bytes) {
    std::ifstream resources(pci_path / "resource");
    for (int resource = 0; resource < 6; ++resource) {
        std::uint64_t start = 0;
        std::uint64_t end = 0;
        std::uint64_t flags = 0;
        resources >> std::hex >> start >> end >> flags;
        const auto resize_path = pci_path / ("resource" + std::to_string(resource) + "_resize");
        if (std::filesystem::exists(resize_path)) {
            const auto value = read_text_file(resize_path);
            const std::uint64_t aperture_bytes = end >= start && start != 0 ? end - start + 1 : 0;
            std::ostringstream status;
            if (aperture_bytes >= vram_total_bytes) {
                status << "enabled (" << std::fixed << std::setprecision(2)
                       << static_cast<double>(aperture_bytes) / static_cast<double>(1ULL << 30) << " GiB aperture)";
            } else if (aperture_bytes > 0) {
                status << "supported, partial aperture (" << std::fixed << std::setprecision(2)
                       << static_cast<double>(aperture_bytes) / static_cast<double>(1ULL << 30) << " GiB)";
            } else {
                status << "supported (current size unavailable)";
            }
            if (!value.empty()) {
                status << ", kernel mask " << value;
            }
            return status.str();
        }
    }
    return "not exposed by kernel";
}

std::string nvidia_driver_version() {
    std::ifstream input("/proc/driver/nvidia/version");
    std::string first_line;
    std::getline(input, first_line);
    const std::regex release_pattern(R"(\s([0-9]+\.[0-9]+(?:\.[0-9]+)?)\s+Release Build)");
    std::smatch match;
    return std::regex_search(first_line, match, release_pattern) ? match[1].str() : "unknown";
}

GpuInfo make_gpu_info(int device) {
    cudaDeviceProp properties {};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    check_cuda(cudaSetDevice(device), "cudaSetDevice");

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    char bus_id[32] {};
    check_cuda(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device), "cudaDeviceGetPCIBusId");
    const auto pci_path = pci_sysfs_path(properties);
    const double current_speed = parse_link_speed(read_text_file(pci_path / "current_link_speed"));
    const double max_speed = parse_link_speed(read_text_file(pci_path / "max_link_speed"));
    int memory_clock_khz = 0;
    int memory_bus_width = 0;
    check_cuda(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device),
        "cudaDeviceGetAttribute(memory clock)");
    check_cuda(cudaDeviceGetAttribute(&memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, device),
        "cudaDeviceGetAttribute(memory bus width)");

    GpuInfo info;
    info.index = device;
    info.name = properties.name;
    info.compute_major = properties.major;
    info.compute_minor = properties.minor;
    info.vram_total_bytes = total_bytes;
    info.vram_free_bytes = free_bytes;
    info.pci_bus_id = bus_id;
    info.pcie_generation_current = pcie_generation(current_speed);
    info.pcie_generation_max = pcie_generation(max_speed);
    info.pcie_width_current = parse_leading_integer(read_text_file(pci_path / "current_link_width"));
    info.pcie_width_max = parse_leading_integer(read_text_file(pci_path / "max_link_width"));
    info.pcie_theoretical_gb_per_second = pcie_gb_per_second_per_lane(info.pcie_generation_current) * info.pcie_width_current;
    info.vram_theoretical_gb_per_second = 2.0 * static_cast<double>(memory_clock_khz) * 1000.0
        * static_cast<double>(memory_bus_width) / 8.0 / 1.0e9;
    info.resizable_bar_status = detect_resizable_bar(pci_path, total_bytes);
    return info;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (pointer_) {
            cudaFree(pointer_);
        }
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;
    void * get() const { return pointer_; }

private:
    void * pointer_ = nullptr;
};

class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t bytes) {
        check_cuda(cudaHostAlloc(&pointer_, bytes, cudaHostAllocDefault), "cudaHostAlloc");
    }
    ~PinnedBuffer() {
        if (pointer_) {
            cudaFreeHost(pointer_);
        }
    }
    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer & operator=(const PinnedBuffer &) = delete;
    void * get() const { return pointer_; }

private:
    void * pointer_ = nullptr;
};

class Stream {
public:
    Stream() {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    ~Stream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    Stream(const Stream &) = delete;
    Stream & operator=(const Stream &) = delete;
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

BandwidthResult measure_copy(
        const std::string & name,
        const std::string & direction,
        const std::string & memory_kind,
        void * destination,
        const void * source,
        std::size_t bytes,
        int iterations,
        cudaMemcpyKind kind,
        cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(destination, source, bytes, kind, stream), "warm-up cudaMemcpyAsync");
    check_cuda(cudaStreamSynchronize(stream), "warm-up cudaStreamSynchronize");

    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        check_cuda(cudaMemcpyAsync(destination, source, bytes, kind, stream), "cudaMemcpyAsync");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    const auto stopped = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stopped - started).count();

    return {
        .name = name,
        .direction = direction,
        .memory_kind = memory_kind,
        .bytes = bytes,
        .iterations = iterations,
        .seconds = seconds,
        .gb_per_second = static_cast<double>(bytes) * iterations / seconds / 1.0e9,
        .available = true,
    };
}

} // namespace

CudaInfo collect_cuda_info() {
    CudaInfo info;
    cudaError_t status = cudaRuntimeGetVersion(&info.runtime_version);
    if (status != cudaSuccess) {
        info.error = std::string("cudaRuntimeGetVersion: ") + cudaGetErrorString(status);
        return info;
    }
    status = cudaDriverGetVersion(&info.driver_version);
    if (status != cudaSuccess) {
        info.error = std::string("cudaDriverGetVersion: ") + cudaGetErrorString(status);
        return info;
    }
    info.nvidia_driver_version = nvidia_driver_version();

    int count = 0;
    status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        info.error = std::string("cudaGetDeviceCount: ") + cudaGetErrorString(status);
        return info;
    }
    if (count == 0) {
        info.error = "no CUDA devices found";
        return info;
    }

    try {
        int original_device = 0;
        cudaGetDevice(&original_device);
        for (int device = 0; device < count; ++device) {
            info.devices.push_back(make_gpu_info(device));
        }
        cudaSetDevice(original_device);
        info.available = true;
    } catch (const std::exception & error) {
        info.devices.clear();
        info.error = error.what();
    }
    return info;
}

std::vector<BandwidthResult> run_cuda_benchmarks(const BenchmarkOptions & options) {
    if (options.iterations <= 0) {
        throw std::invalid_argument("iterations must be greater than zero");
    }

    int count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess || count == 0) {
        return {{
            .name = "cuda_transfers",
            .direction = "host_device",
            .memory_kind = "unavailable",
            .available = false,
            .error = count_status == cudaSuccess ? "no CUDA devices found" : cudaGetErrorString(count_status),
        }};
    }
    if (options.cuda_device < 0 || options.cuda_device >= count) {
        throw std::invalid_argument("CUDA device index is out of range");
    }

    check_cuda(cudaSetDevice(options.cuda_device), "cudaSetDevice");
    std::vector<std::byte> pageable(options.bytes, std::byte{0x2a});
    PinnedBuffer pinned(options.bytes);
    DeviceBuffer device_source(options.bytes);
    DeviceBuffer device_destination(options.bytes);
    Stream stream;
    std::fill_n(static_cast<std::byte *>(pinned.get()), options.bytes, std::byte{0x2a});

    std::vector<BandwidthResult> results;
    results.push_back(measure_copy("pageable_ram_to_gpu", "host_to_device", "pageable", device_source.get(),
        pageable.data(), options.bytes, options.iterations, cudaMemcpyHostToDevice, stream.get()));
    results.push_back(measure_copy("pinned_ram_to_gpu", "host_to_device", "pinned", device_source.get(),
        pinned.get(), options.bytes, options.iterations, cudaMemcpyHostToDevice, stream.get()));
    results.push_back(measure_copy("gpu_to_pageable_ram", "device_to_host", "pageable", pageable.data(),
        device_source.get(), options.bytes, options.iterations, cudaMemcpyDeviceToHost, stream.get()));
    results.push_back(measure_copy("gpu_to_pinned_ram", "device_to_host", "pinned", pinned.get(),
        device_source.get(), options.bytes, options.iterations, cudaMemcpyDeviceToHost, stream.get()));
    results.push_back(measure_copy("gpu_vram_copy", "device_to_device", "device", device_destination.get(),
        device_source.get(), options.bytes, options.iterations, cudaMemcpyDeviceToDevice, stream.get()));
    return results;
}

} // namespace vrambridge