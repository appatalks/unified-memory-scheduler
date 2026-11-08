#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <unistd.h>

namespace {
struct Capture {
    std::mutex lock;
    std::FILE* file = nullptr;
    uint64_t vocab = 0;
    uint64_t written = 0;
    bool failed = false;

    Capture() {
        const char* path = std::getenv("VRAMBRIDGE_STRATA_LOGITS");
        const char* size = std::getenv("VRAMBRIDGE_STRATA_VOCAB");
        if (!path || !size) return;
        char* end = nullptr;
        vocab = std::strtoull(size, &end, 10);
        if (!end || *end || vocab < 2 || vocab > 1048576) return;
        int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) return;
        file = ::fdopen(fd, "wb");
        if (!file) { ::close(fd); return; }
        const uint64_t magic = 0x315041434c525453ULL;
        if (std::fwrite(&magic, sizeof(magic), 1, file) != 1 ||
            std::fwrite(&vocab, sizeof(vocab), 1, file) != 1) failed = true;
    }

    ~Capture() { if (file) std::fclose(file); }

    void record(const void* data, size_t bytes) {
        if (!file || bytes == 0 || bytes % (vocab * sizeof(float))) return;
        const uint64_t rows = bytes / (vocab * sizeof(float));
        if (rows > 8) return;
        std::lock_guard<std::mutex> guard(lock);
        if (failed) return;
        if (written + bytes > 1024ULL * 1024 * 1024 ||
            std::fwrite(&rows, sizeof(rows), 1, file) != 1 ||
            std::fwrite(data, 1, bytes, file) != bytes || std::fflush(file) != 0) {
            failed = true;
            std::fprintf(stderr, "VRAMBridge Strata logits capture failed\n");
            return;
        }
        written += bytes;
    }
};
}

// Observe existing diagnostic D2H copies, never add a transfer or change arithmetic.
extern "C" int cudaMemcpy(void* dst, const void* src, size_t bytes, int kind) {
    using Copy = int (*)(void*, const void*, size_t, int);
    static Copy next = reinterpret_cast<Copy>(::dlsym(RTLD_NEXT, "cudaMemcpy"));
    if (!next) return 999;
    const int result = next(dst, src, bytes, kind);
    if (result == 0 && kind == 2) {
        static Capture capture;
        capture.record(dst, bytes);
    }
    return result;
}
