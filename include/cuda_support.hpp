#ifndef RATPOINTS_GPU_CUDA_SUPPORT_HPP
#define RATPOINTS_GPU_CUDA_SUPPORT_HPP

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace ratpoints_gpu {

inline void check_cuda(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) : size_(size) {
        if (size > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::overflow_error("device buffer size exceeds host address space");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&data_),
                              size * sizeof(T)),
                   "cudaMalloc");
    }

    ~DeviceBuffer() { cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *data() { return data_; }
    const T *data() const { return data_; }

    void upload(const std::vector<T> &source) {
        if (source.size() > size_) {
            throw std::logic_error("device upload exceeds buffer capacity");
        }
        check_cuda(cudaMemcpy(data_, source.data(), source.size() * sizeof(T),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device");
    }

    void download(std::vector<T> &destination) const {
        if (destination.size() > size_) {
            throw std::logic_error("device download exceeds buffer capacity");
        }
        check_cuda(cudaMemcpy(destination.data(), data_,
                              destination.size() * sizeof(T),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host");
    }

    T download_scalar() const {
        if (size_ != 1) {
            throw std::logic_error("scalar download requires a one-item buffer");
        }
        T value;
        check_cuda(cudaMemcpy(&value, data_, sizeof(T), cudaMemcpyDeviceToHost),
                   "cudaMemcpy scalar device to host");
        return value;
    }

    void clear() {
        check_cuda(cudaMemset(data_, 0, size_ * sizeof(T)), "cudaMemset");
    }

private:
    T *data_ = nullptr;
    size_t size_;
};

class EventInterval {
public:
    EventInterval() {
        check_cuda(cudaEventCreate(&start_), "cudaEventCreate");
        cudaError_t error = cudaEventCreate(&stop_);
        if (error != cudaSuccess) {
            cudaEventDestroy(start_);
            check_cuda(error, "cudaEventCreate");
        }
    }

    ~EventInterval() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    EventInterval(const EventInterval &) = delete;
    EventInterval &operator=(const EventInterval &) = delete;

    void start() { check_cuda(cudaEventRecord(start_), "cudaEventRecord"); }
    void stop() { check_cuda(cudaEventRecord(stop_), "cudaEventRecord"); }

    float milliseconds() const {
        float result;
        check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize");
        check_cuda(cudaEventElapsedTime(&result, start_, stop_),
                   "cudaEventElapsedTime");
        return result;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

}  // namespace ratpoints_gpu

#endif
