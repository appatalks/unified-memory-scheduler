#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

namespace {

uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

struct Origin {
    std::string path;
    size_t offset = 0;
    uint64_t inode = 0;
    unsigned device_major = 0;
    unsigned device_minor = 0;
};

Origin mapped_origin(const void * data, size_t size) {
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(data);
    require(size && pointer <= UINTPTR_MAX - size, "Invalid source range.");
    std::ifstream maps("/proc/self/maps");
    require(bool(maps), "Cannot inspect source mappings.");
    for (std::string line; std::getline(maps, line);) {
        std::istringstream input(line);
        std::string range, permissions, offset, device, inode, path;
        input >> range >> permissions >> offset >> device >> inode;
        std::getline(input >> std::ws, path);
        const auto separator = range.find('-');
        if (separator == std::string::npos) continue;
        const uintptr_t begin = std::stoull(range.substr(0, separator), nullptr, 16);
        const uintptr_t end = std::stoull(range.substr(separator + 1), nullptr, 16);
        if (begin <= pointer && pointer + size <= end) {
            require(!path.empty() && path.front() == '/' && path.find(" (deleted)") == std::string::npos,
                    "Expert source is not a live file mapping.");
            const auto colon = device.find(':');
            require(colon != std::string::npos, "Missing source mapping device identity.");
            return {path, std::stoull(offset, nullptr, 16) + pointer - begin, std::stoull(inode),
                    unsigned(std::stoul(device.substr(0, colon), nullptr, 16)),
                    unsigned(std::stoul(device.substr(colon + 1), nullptr, 16))};
        }
    }
    throw std::runtime_error("Expert source does not fit one file mapping.");
}

struct Window {
    int fd = -1;
    size_t page, file_offset, prefix, length, stride, experts;
    struct stat identity{};
    char * address = nullptr;
    const char * original;
    std::vector<uint32_t> references;

    Window(const Origin & origin, size_t bytes, size_t expert_stride, size_t count, size_t page_size, const void * source) :
        page(page_size), file_offset(origin.offset / page * page), prefix(origin.offset % page),
        length((prefix + bytes + page - 1) / page * page), stride(expert_stride), experts(count),
        original(static_cast<const char *>(source) - prefix) {
        fd = open(origin.path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        require(fd >= 0, "Cannot open expert source for direct reads.");
        try {
            require(fstat(fd, &identity) == 0 && S_ISREG(identity.st_mode) &&
                    identity.st_ino == origin.inode && major(identity.st_dev) == origin.device_major &&
                    minor(identity.st_dev) == origin.device_minor &&
                    origin.offset <= size_t(identity.st_size) && bytes <= size_t(identity.st_size) - origin.offset,
                    "Invalid expert source file or range.");
            address = static_cast<char *>(mmap(nullptr, length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            if (address == MAP_FAILED) {
                address = nullptr;
                throw std::runtime_error("Cannot reserve sparse expert address range.");
            }
            references.resize(length / page);
        } catch (...) {
            if (address) munmap(address, length);
            close(fd);
            throw;
        }
    }
    ~Window() {
        if (address) munmap(address, length);
        if (fd >= 0) close(fd);
    }
    void unchanged() const {
        struct stat current{};
        require(fstat(fd, &current) == 0 && current.st_size == identity.st_size &&
                current.st_mtim.tv_sec == identity.st_mtim.tv_sec &&
                current.st_mtim.tv_nsec == identity.st_mtim.tv_nsec, "Expert source changed during execution.");
    }
    std::pair<size_t, size_t> pages(size_t expert) const {
        require(expert < experts, "Expert ID outside source domain.");
        const size_t begin = prefix + expert * stride;
        return {begin / page, (begin + stride + page - 1) / page};
    }
};

using Key = std::pair<Window *, size_t>;
using Page = std::pair<Window *, size_t>;

struct ReadJob {
    Window * window;
    size_t first;
    size_t count;
};

class ExpertCache {
    size_t page_;
    size_t capacity_;
    unsigned workers_;
    bool source_aware_;
    std::map<std::tuple<const void *, size_t, size_t>, std::unique_ptr<Window>> windows_;
    std::list<Key> lru_;
    std::map<Key, std::list<Key>::iterator> entries_;

    void evict(const Key & key) {
        auto * window = key.first;
        const auto [begin, end] = window->pages(key.second);
        for (size_t index = begin; index < end; ++index) {
            require(window->references[index] > 0, "Invalid page ownership during eviction.");
            if (--window->references[index] == 0) {
                require(mprotect(window->address + index * page_, page_, PROT_NONE) == 0 &&
                        madvise(window->address + index * page_, page_, MADV_DONTNEED) == 0,
                        "Cannot release an evicted expert page.");
                resident_bytes -= page_;
            }
        }
        lru_.erase(entries_.at(key));
        entries_.erase(key);
        ++evictions;
    }

public:
    uint64_t resident_bytes = 0, peak_resident_bytes = 0, evictions = 0;
    uint64_t logical_bytes = 0, hit_bytes = 0, disk_read_bytes = 0, source_ram_bytes = 0, jobs = 0;

    ExpertCache(size_t capacity, unsigned workers, bool source_aware = false) :
        page_(size_t(sysconf(_SC_PAGESIZE))), capacity_(capacity), workers_(workers), source_aware_(source_aware) {
        require(page_ && capacity_ >= page_ && workers_ > 0 && workers_ <= 4, "Invalid cache bounds.");
    }

    Window * window(const void * source, size_t bytes, size_t stride, size_t count) {
        require(count && count <= 65536 && stride && count <= SIZE_MAX / stride && bytes == stride * count,
                "Unsupported expert tensor layout.");
        auto key = std::make_tuple(source, bytes, stride);
        auto found = windows_.find(key);
        if (found != windows_.end()) return found->second.get();
        require(windows_.size() < 512, "Expert-source descriptor bound exceeded.");
        auto value = std::make_unique<Window>(mapped_origin(source, bytes), bytes, stride, count, page_, source);
        auto * result = value.get();
        windows_.emplace(key, std::move(value));
        return result;
    }

    void prepare(const std::set<Key> & required) {
        require(required.size() <= 65536, "Active expert set exceeds descriptor bound.");
        std::set<Window *> sources;
        for (const auto & key : required) sources.insert(key.first);
        for (auto * source : sources) source->unchanged();
        auto missing_pages = [&] {
            std::set<Page> pages;
            for (const auto & key : required) {
                if (entries_.count(key)) continue;
                const auto [begin, end] = key.first->pages(key.second);
                for (size_t index = begin; index < end; ++index) {
                    if (!key.first->references[index]) pages.emplace(key.first, index);
                }
            }
            return pages;
        };
        auto pages = missing_pages();
        while (pages.size() * page_ > capacity_ - resident_bytes) {
            auto victim = std::find_if(lru_.begin(), lru_.end(), [&](const Key & key) { return !required.count(key); });
            require(victim != lru_.end(), "Current active expert working set exceeds the RAM cache budget.");
            const Key key = *victim;
            evict(key);
            pages = missing_pages();
        }
        std::vector<ReadJob> reads;
        for (const auto & [window, index] : pages) {
            if (!reads.empty() && reads.back().window == window &&
                reads.back().first + reads.back().count == index) {
                ++reads.back().count;
            } else {
                reads.push_back({window, index, 1});
            }
        }
        for (const auto & job : reads) {
            require(mprotect(job.window->address + job.first * page_, job.count * page_, PROT_READ | PROT_WRITE) == 0,
                    "Cannot enable staging range.");
        }
        std::atomic<size_t> next{0};
        std::atomic<uint64_t> disk_bytes{0}, ram_bytes{0}, read_calls{0};
        auto reader = [&] {
            for (size_t index; (index = next.fetch_add(1)) < reads.size();) {
                const auto & job = reads[index];
                std::vector<unsigned char> resident(job.count, 0);
                if (source_aware_) {
                    require(mincore(const_cast<char *>(job.window->original) + job.first * page_,
                                    job.count * page_, resident.data()) == 0,
                            "Cannot observe original source page residency.");
                }
                for (size_t position = 0; position < job.count;) {
                    const auto page_index = job.first + position;
                    const auto offset = job.window->file_offset + page_index * page_;
                    if (resident[position] & 1) {
                        const auto size = std::min(page_, size_t(job.window->identity.st_size) - offset);
                        std::memcpy(job.window->address + page_index * page_,
                                    job.window->original + page_index * page_, size);
                        ram_bytes.fetch_add(size);
                        ++position;
                        continue;
                    }
                    size_t end = position + 1;
                    while (end < job.count && !(resident[end] & 1)) ++end;
                    const auto size = (end - position) * page_;
                    ssize_t count;
                    do {
                        count = pread(job.window->fd, job.window->address + page_index * page_, size, off_t(offset));
                    } while (count < 0 && errno == EINTR);
                    const size_t expected = std::min(size, size_t(job.window->identity.st_size) - offset);
                    require(count >= 0 && size_t(count) == expected, "Failed or short direct expert read.");
                    disk_bytes.fetch_add(size);
                    read_calls.fetch_add(1);
                    position = end;
                }
            }
        };
        std::vector<std::future<void>> work;
        try {
            for (unsigned i = 0; i < std::min<size_t>(workers_, reads.size()); ++i) {
                work.push_back(std::async(std::launch::async, reader));
            }
            for (auto & worker : work) worker.get();
            for (auto * source : sources) source->unchanged();
        } catch (...) {
            for (auto & worker : work) if (worker.valid()) worker.wait();
            for (const auto & [window, index] : pages) {
                mprotect(window->address + index * page_, page_, PROT_NONE);
                madvise(window->address + index * page_, page_, MADV_DONTNEED);
            }
            throw;
        }
        disk_read_bytes += disk_bytes.load();
        source_ram_bytes += ram_bytes.load();
        jobs += read_calls.load();
        resident_bytes += pages.size() * page_;
        peak_resident_bytes = std::max(peak_resident_bytes, resident_bytes);
        for (const auto & key : required) {
            logical_bytes += key.first->stride;
            auto found = entries_.find(key);
            if (found != entries_.end()) {
                hit_bytes += key.first->stride;
                lru_.splice(lru_.end(), lru_, found->second);
            } else {
                const auto [begin, end] = key.first->pages(key.second);
                for (size_t index = begin; index < end; ++index) ++key.first->references[index];
                lru_.push_back(key);
                entries_[key] = std::prev(lru_.end());
            }
        }
        require(resident_bytes <= capacity_, "RAM cache budget exceeded.");
    }
};

bool external_ids(ggml_cgraph * graph, const ggml_tensor * ids) {
    for (auto * tensor = ids; tensor; tensor = tensor->view_src) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (ggml_graph_node(graph, i) == tensor) return false;
        }
    }
    return true;
}

#ifndef VRAMBRIDGE_EXPERT_DISK_SELF_TEST
using Compute = ggml_status (*)(ggml_cgraph *, ggml_cplan *);

struct State {
    Compute original = nullptr;
    std::unique_ptr<ExpertCache> cache;
    FILE * output = nullptr;
    std::mutex mutex;
    uint64_t graphs = 0, prepare_ns = 0, compute_ns = 0, baseline_graphs = 0;
    bool failed = false;
    ~State() { if (output) std::fclose(output); }

    void emit() {
        const auto line = json({{"graphs", graphs}, {"baseline_graphs", baseline_graphs},
            {"resident_bytes", cache->resident_bytes}, {"peak_resident_bytes", cache->peak_resident_bytes},
            {"logical_expert_bytes", cache->logical_bytes}, {"hit_expert_bytes", cache->hit_bytes},
            {"direct_read_requested_bytes", cache->disk_read_bytes}, {"read_jobs", cache->jobs},
            {"source_ram_copy_bytes", cache->source_ram_bytes},
            {"evictions", cache->evictions}, {"prepare_ns", prepare_ns}, {"compute_ns", compute_ns}}).dump() + '\n';
        require(std::fwrite(line.data(), 1, line.size(), output) == line.size() && std::fflush(output) == 0,
                "Cannot write cache observation.");
    }
};

State & state() {
    static State value;
    static const bool initialized = [&] {
        const char * library = std::getenv("VRAMBRIDGE_EXPERT_DISK_LIBRARY");
        const char * output = std::getenv("VRAMBRIDGE_EXPERT_DISK_OUT");
        const char * budget = std::getenv("VRAMBRIDGE_EXPERT_DISK_MIB");
        require(library && output && budget, "Explicit pinned library, output and RAM budget are required.");
        size_t used = 0;
        const std::string text = budget;
        const auto mib = std::stoull(text, &used);
        require(used == text.size() && mib >= 64 && mib <= 65536, "Cache budget must be 64-65536 MiB.");
        const auto * threads = std::getenv("VRAMBRIDGE_EXPERT_DISK_WORKERS");
        size_t parsed = 0;
        const auto workers_value = threads ? std::stoul(threads, &parsed) : 2;
        require(workers_value > 0 && workers_value <= 4 && (!threads || parsed == std::strlen(threads)),
                "Read worker count must be an integer from 1 to 4.");
        const unsigned workers = unsigned(workers_value);
        const char * aware = std::getenv("VRAMBRIDGE_EXPERT_DISK_SOURCE_AWARE");
        require(!aware || std::strcmp(aware, "0") == 0 || std::strcmp(aware, "1") == 0,
                "Source-aware control must be 0 or 1.");
        const bool source_aware = aware && std::strcmp(aware, "1") == 0;
        value.cache = std::make_unique<ExpertCache>(mib * 1024 * 1024, workers, source_aware);
        void * handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
        require(handle, "Matching CPU library is not already loaded.");
        value.original = reinterpret_cast<Compute>(dlsym(handle, "ggml_graph_compute"));
        require(value.original && value.original != &ggml_graph_compute, "Cannot resolve original native CPU compute.");
        value.output = std::fopen(output, "wx");
        require(value.output && fchmod(fileno(value.output), 0600) == 0, "Cannot create exclusive private cache output.");
        std::fprintf(stderr, "VRAMBridge exact expert RAM cache active: %llu MiB; raw file-backed weights, direct reads, %u workers\n",
                     static_cast<unsigned long long>(mib), workers);
        if (source_aware) std::fprintf(stderr, "VRAMBridge source-residency-aware cache fills active\n");
        return true;
    }();
    (void) initialized;
    return value;
}
#endif
} // namespace

#ifndef VRAMBRIDGE_EXPERT_DISK_SELF_TEST
extern "C" ggml_status ggml_graph_compute(ggml_cgraph * graph, ggml_cplan * plan) {
    State * current = nullptr;
    std::unique_lock<std::mutex> lock;
    std::vector<std::pair<ggml_tensor *, void *>> originals;
    try {
        current = &state();
        lock = std::unique_lock<std::mutex>(current->mutex);
        if (current->failed) return GGML_STATUS_FAILED;
        std::set<Key> required;
        std::map<ggml_tensor *, Window *> tensors;
        bool multi_position = false;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT_ID && node->src[2]->ne[1] != 1) multi_position = true;
        }
        if (multi_position) {
            ++current->baseline_graphs;
            return current->original(graph, plan);
        }
        const auto begin = now_ns();
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_MUL_MAT_ID) continue;
            auto * weights = node->src[0];
            auto * ids = node->src[2];
            require(weights && ids && weights->data && ids->data && !weights->extra &&
                    weights->buffer && ids->buffer && ggml_backend_buffer_is_host(weights->buffer) &&
                    ggml_backend_buffer_is_host(ids->buffer) && ggml_is_contiguous(weights) &&
                    weights->ne[2] > 0 && weights->ne[3] == 1 && ids->type == GGML_TYPE_I32 &&
                    ids->ne[0] > 0 && ids->ne[0] <= weights->ne[2] &&
                    ids->ne[2] == 1 && ids->ne[3] == 1 && external_ids(graph, ids),
                    "Unsupported expert source/IDs; no fallback accepted. Use raw --no-repack CPU-MoE decoding.");
            auto * window = current->cache->window(weights->data, ggml_nbytes(weights), weights->nb[2], weights->ne[2]);
            tensors[weights] = window;
            std::set<int32_t> unique;
            for (int rank = 0; rank < ids->ne[0]; ++rank) {
                int32_t expert;
                std::memcpy(&expert, static_cast<const char *>(ids->data) + rank * ids->nb[0], sizeof(expert));
                require(expert >= 0 && expert < weights->ne[2] && unique.insert(expert).second,
                        "Invalid or duplicate selected expert.");
                required.emplace(window, size_t(expert));
            }
        }
        if (required.empty()) return current->original(graph, plan);
        current->cache->prepare(required);
        for (const auto & [tensor, window] : tensors) {
            originals.emplace_back(tensor, tensor->data);
            tensor->data = window->address + window->prefix;
        }
        const auto prepared = now_ns();
        const auto result = current->original(graph, plan);
        const auto computed = now_ns();
        for (auto & [tensor, data] : originals) tensor->data = data;
        originals.clear();
        ++current->graphs;
        current->prepare_ns += prepared - begin;
        current->compute_ns += computed - prepared;
        current->emit();
        return result;
    } catch (const std::exception & error) {
        for (auto & [tensor, data] : originals) tensor->data = data;
        if (current) current->failed = true;
        std::fprintf(stderr, "Exact expert disk cache failed: %s\n", error.what());
        return GGML_STATUS_FAILED;
    }
}
#else
int main() {
    try {
        char path[] = "/tmp/vrambridge-expert-cache-XXXXXX";
        const int fd = mkstemp(path);
        require(fd >= 0, "Cannot create test fixture.");
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        const size_t prefix = 416, stride = page + 144, count = 8;
        const size_t bytes = (prefix + stride * count + page - 1) / page * page;
        std::vector<char> values(bytes);
        for (size_t i = 0; i < bytes; ++i) values[i] = char(i % 251);
        require(write(fd, values.data(), values.size()) == ssize_t(values.size()) && fsync(fd) == 0,
                "Cannot populate test fixture.");
        auto * source = static_cast<char *>(mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0));
        require(source != MAP_FAILED, "Cannot map test fixture.");
        {
            ExpertCache cache(5 * page, 2);
            auto * window = cache.window(source + prefix, stride * count, stride, count);
            cache.prepare({{window, 0}, {window, 1}});
            require(std::memcmp(source + prefix, window->address + prefix, 2 * stride) == 0,
                    "Initial cached expert bytes changed.");
            const auto read_bytes = cache.disk_read_bytes;
            cache.prepare({{window, 0}});
            require(cache.disk_read_bytes == read_bytes && cache.hit_bytes == stride,
                    "Cached expert caused more direct reads.");
            cache.prepare({{window, 6}, {window, 7}});
            require(cache.evictions > 0 && cache.resident_bytes <= 5 * page &&
                    std::memcmp(source + prefix + 6 * stride, window->address + prefix + 6 * stride, 2 * stride) == 0,
                    "Eviction, page overlap or source byte parity failed.");
            cache.prepare({{window, 0}, {window, 1}});
            require(std::memcmp(source + prefix, window->address + prefix, 2 * stride) == 0,
                    "Readmission changed cached expert bytes.");
            bool rejected = false;
            try { cache.prepare({{window, 0}, {window, 2}, {window, 4}, {window, 6}}); }
            catch (const std::runtime_error &) { rejected = true; }
            require(rejected, "Oversized active working set was not rejected.");
        }
        munmap(source, bytes);
        close(fd);
        unlink(path);
        std::cout << "{\"byte_parity\":true,\"hits_avoid_reads\":true,\"overlapping_pages\":true,"
                     "\"eviction_readmission\":true,\"budget_rejection\":true}\n";
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Expert cache self-test failed: %s\n", error.what());
        return 1;
    }
}
#endif
