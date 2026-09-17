// CPU-only orchestration test double: this is NOT a CPU search backend.
#include <cuda_runtime_api.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

#include "sieve.hpp"

namespace {
thread_local int selected_device = -1;
std::mutex mutex;
std::condition_variable ready;
std::set<int> active;
bool parallel_seen = false;

int setting(const char *name, int fallback) {
    const char *text = std::getenv(name);
    return text ? std::stoi(text) : fallback;
}

void log_event(const char *event, ratpoints_gpu::DenominatorRange range) {
    const char *path = std::getenv("RATPOINTS_TEST_LOG");
    if (path) {
        std::ofstream stream(path, std::ios::app);
        stream << event << ' ' << selected_device << ' '
               << range.first << ' ' << range.last << '\n';
    }
}

struct ActiveJob {
    ratpoints_gpu::DenominatorRange range;
    explicit ActiveJob(ratpoints_gpu::DenominatorRange value) : range(value) {
        std::lock_guard<std::mutex> lock(mutex);
        if (selected_device < 0 || !active.insert(selected_device).second) {
            throw std::runtime_error("worker not bound or device used concurrently");
        }
        parallel_seen = parallel_seen || active.size() > 1;
        log_event("start", range);
        ready.notify_all();
    }
    ~ActiveJob() {
        std::lock_guard<std::mutex> lock(mutex);
        log_event("end", range);
        active.erase(selected_device);
        ready.notify_all();
    }
};
}  // namespace

cudaError_t cudaGetDeviceCount(int *count) {
    if (setting("RATPOINTS_TEST_DRIVER_ERROR", 0)) {
        return cudaErrorUnknown;
    }
    const char *mask = std::getenv("CUDA_VISIBLE_DEVICES");
    *count = mask && !*mask ? 0 : setting("RATPOINTS_TEST_DEVICE_COUNT", 3);
    return *count ? cudaSuccess : cudaErrorNoDevice;
}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *properties, int device) {
    std::snprintf(properties->name, sizeof(properties->name), "Mock GPU %d", device);
    properties->major = 8;
    properties->minor = 9;
    properties->multiProcessorCount = 128;
    properties->totalGlobalMem = 24ULL << 30;
    return cudaSuccess;
}
cudaError_t cudaDeviceGetPCIBusId(char *text, int length, int device) {
    std::snprintf(text, static_cast<size_t>(length), "0000:%02d:00.0", device + 1);
    return cudaSuccess;
}
cudaError_t cudaSetDevice(int device) {
    if (device == setting("RATPOINTS_TEST_ACTIVATE_ERROR", -1)) {
        return cudaErrorUnknown;
    }
    selected_device = device;
    return cudaSuccess;
}
const char *cudaGetErrorString(cudaError_t) { return "mock CUDA failure"; }

namespace ratpoints_gpu {
SieveResult run_modular_sieve(const Coefficients &, long long numerator_bound,
                               DenominatorRange range,
                               const CandidateBatchCallback &callback) {
    ActiveJob job(range);
    if (range.first < 1 || range.last < range.first || range.count() > 65536) {
        throw std::runtime_error("invalid worker range");
    }
    if (setting("RATPOINTS_TEST_REQUIRE_PARALLEL", 0)) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!ready.wait_for(lock, std::chrono::seconds(5), [] { return parallel_seen; })) {
            throw std::runtime_error("workers did not execute concurrently");
        }
    }
    if (selected_device == setting("RATPOINTS_TEST_FAIL_DEVICE", -1)) {
        throw std::runtime_error("injected sieve failure");
    }
    bool sparse = setting("RATPOINTS_TEST_BOUNDARIES_ONLY", 0) != 0;
    long long bound = sparse ? 1 : numerator_bound;
    if ((2.0L * bound + 1) * range.count() > 2000000) {
        throw std::runtime_error("mock search too large; not a production backend");
    }
    CandidateBatch candidates;
    unsigned long long survivors = 0;
    if (selected_device != setting("RATPOINTS_TEST_DROP_DEVICE", -1)) {
        // Deliberately unordered, multi-flush output exercises real sorting,
        // exact GMP verification, deduplication by primitive coordinates, and
        // main-thread formatting independently of any CUDA kernel.
        for (long long b = range.last; b >= range.first; --b) {
            for (long long a = bound; a >= -bound; --a) {
                candidates.numerators.push_back(a);
                candidates.denominators.push_back(static_cast<int>(b));
                ++survivors;
                if (candidates.size() == 97) {
                    callback(candidates);
                    candidates.numerators.clear();
                    candidates.denominators.clear();
                }
            }
        }
        if (candidates.size()) {
            callback(candidates);
        }
    }
    SieveMetrics metrics;
    metrics.word_count = static_cast<unsigned long long>((2 * numerator_bound + 32) / 32);
    return {survivors, metrics};
}
}  // namespace ratpoints_gpu
