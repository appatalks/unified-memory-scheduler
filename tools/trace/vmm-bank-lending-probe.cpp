#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(CUresult result, const char * operation) {
    if (result == CUDA_SUCCESS) return;
    const char * name = nullptr;
    cuGetErrorName(result, &name);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "unknown CUDA error"));
}

struct Resources {
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    CUmemGenericAllocationHandle handle = 0;
    CUdeviceptr bank = 0, scratch = 0, sink = 0;
    size_t bytes = 0;
    bool bank_mapped = false, scratch_mapped = false, handle_created = false;
    ~Resources() {
        if (stream) cuStreamSynchronize(stream);
        if (executable) cuGraphExecDestroy(executable);
        if (graph) cuGraphDestroy(graph);
        if (sink) cuMemFree(sink);
        if (bank_mapped) cuMemUnmap(bank, bytes);
        if (scratch_mapped) cuMemUnmap(scratch, bytes);
        if (bank) cuMemAddressFree(bank, bytes);
        if (scratch) cuMemAddressFree(scratch, bytes);
        if (handle_created) cuMemRelease(handle);
        if (stream) cuStreamDestroy(stream);
        if (context) cuCtxDestroy(context);
    }
};

double median(std::vector<double> samples) {
    if (samples.empty()) throw std::runtime_error("empty timing sample");
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    return samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) / 2;
}

void require_bytes(const std::vector<uint8_t> & data, uint8_t expected) {
    if (!std::all_of(data.begin(), data.end(), [expected](uint8_t v) { return v == expected; })) {
        throw std::runtime_error("VMM alias contents differ");
    }
}

using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        return median({3, 1, 2}) == 2 && median({1, 4, 2, 3}) == 2.5 ? 0 : 1;
    }
    if (argc > 2) {
        std::fprintf(stderr, "Usage: %s [16-1024 MiB] | --self-test\n", argv[0]);
        return 2;
    }
    try {
        char * end = nullptr;
        const long mib = argc == 2 ? std::strtol(argv[1], &end, 10) : 64;
        if ((argc == 2 && (end == argv[1] || *end != '\0')) || mib < 16 || mib > 1024) {
            throw std::runtime_error("size must be 16-1024 MiB");
        }
        check(cuInit(0), "cuInit");
        CUdevice device;
        check(cuDeviceGet(&device, 0), "cuDeviceGet");
        int supported = 0;
        check(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device), "VMM support");
        if (!supported) throw std::runtime_error("device does not support virtual memory management");
        Resources r;
#if CUDA_VERSION >= 13000
        check(cuCtxCreate(&r.context, nullptr, 0, device), "cuCtxCreate");
#else
        check(cuCtxCreate(&r.context, 0, device), "cuCtxCreate");
#endif
        check(cuStreamCreate(&r.stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        CUmemAllocationProp properties{};
        properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties.location.id = device;
        size_t granularity = 0;
        check(cuMemGetAllocationGranularity(&granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "allocation granularity");
        const size_t target = size_t(mib) * 1024 * 1024;
        if (!granularity || granularity > target) throw std::runtime_error("unsupported allocation granularity");
        r.bytes = ((target + granularity - 1) / granularity) * granularity;
        if (r.bytes > size_t(1024) * 1024 * 1024) throw std::runtime_error("allocation exceeds bound");
        check(cuMemAddressReserve(&r.bank, r.bytes, granularity, 0, 0), "reserve bank address");
        check(cuMemAddressReserve(&r.scratch, r.bytes, granularity, 0, 0), "reserve scratch address");
        check(cuMemCreate(&r.handle, r.bytes, &properties, 0), "allocate backing");
        r.handle_created = true;
        CUmemAccessDesc access{};
        access.location = properties.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        const auto map = [&](CUdeviceptr address) {
            check(cuMemMap(address, r.bytes, 0, r.handle, 0), "map backing");
            if (address == r.bank) r.bank_mapped = true; else r.scratch_mapped = true;
            check(cuMemSetAccess(address, r.bytes, &access, 1), "set access");
        };
        map(r.bank);
        check(cuMemAlloc(&r.sink, r.bytes), "allocate graph read sink");
        check(cuGraphCreate(&r.graph, 0), "create graph");
        CUDA_MEMCPY3D copy_params{};
        copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy_params.srcDevice = r.bank;
        copy_params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copy_params.dstDevice = r.sink;
        copy_params.WidthInBytes = r.bytes;
        copy_params.Height = 1;
        copy_params.Depth = 1;
        CUgraphNode node;
        check(cuGraphAddMemcpyNode(&node, r.graph, nullptr, 0, &copy_params, r.context), "add graph read");
        check(cuGraphInstantiateWithFlags(&r.executable, r.graph, 0), "instantiate graph");
        std::vector<uint8_t> copy(r.bytes), refill(r.bytes, 0x36);
        std::vector<double> lend, restore, upload, replay;
        constexpr int warmup = 1, measured = 12;
        for (int iteration = 0; iteration < warmup + measured; ++iteration) {
            check(cuMemsetD8(r.bank, 0x5a, r.bytes), "initialize bank");
            check(cuCtxSynchronize(), "initialization synchronization");
            check(cuGraphLaunch(r.executable, r.stream), "initial graph launch");
            check(cuStreamSynchronize(r.stream), "initial graph synchronization");
            check(cuMemcpyDtoH(copy.data(), r.sink, r.bytes), "verify initial graph read");
            require_bytes(copy, 0x5a);
            auto start = Clock::now();
            check(cuMemUnmap(r.bank, r.bytes), "unmap bank");
            r.bank_mapped = false;
            map(r.scratch);
            const double lend_ms = milliseconds(start);
            check(cuMemcpyDtoH(copy.data(), r.scratch, r.bytes), "read scratch alias");
            require_bytes(copy, 0x5a);
            check(cuMemsetD8(r.scratch, 0xc3, r.bytes), "write scratch alias");
            check(cuCtxSynchronize(), "scratch synchronization");
            start = Clock::now();
            check(cuMemUnmap(r.scratch, r.bytes), "unmap scratch");
            r.scratch_mapped = false;
            map(r.bank);
            const double restore_ms = milliseconds(start);
            check(cuMemcpyDtoH(copy.data(), r.bank, r.bytes), "read returned bank");
            require_bytes(copy, 0xc3);
            start = Clock::now();
            check(cuMemcpyHtoD(r.bank, refill.data(), r.bytes), "refill bank");
            check(cuCtxSynchronize(), "refill synchronization");
            const double upload_ms = milliseconds(start);
            start = Clock::now();
            check(cuGraphLaunch(r.executable, r.stream), "replay original graph");
            check(cuStreamSynchronize(r.stream), "replay synchronization");
            const double replay_ms = milliseconds(start);
            check(cuMemcpyDtoH(copy.data(), r.sink, r.bytes), "verify replayed graph read");
            require_bytes(copy, 0x36);
            if (iteration >= warmup) {
                lend.push_back(lend_ms);
                restore.push_back(restore_ms);
                upload.push_back(upload_ms);
                replay.push_back(replay_ms);
            }
        }
        std::printf("{\"status\":\"complete\",\"bytes\":%zu,\"granularity\":%zu,\"warmup\":%d,\"cycles\":%d,"
                    "\"stable_graph_replays\":%d,\"lend_map_median_ms\":%.6f,\"restore_map_median_ms\":%.6f,"
                    "\"pageable_refill_median_ms\":%.6f,\"graph_replay_median_ms\":%.6f}\n",
                    r.bytes, granularity, warmup, measured, measured, median(lend), median(restore), median(upload), median(replay));
        return 0;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "VMM bank-lending probe failed: %s\n", exc.what());
        return 1;
    }
}
