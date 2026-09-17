#include "gpu_devices.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <cuda_runtime_api.h>

namespace ratpoints_gpu {
namespace {

void require_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(status));
    }
}

}  // namespace

std::vector<int> parse_device_selection(const std::string &text) {
    if (text == "all") {
        return {};
    }
    std::vector<int> ids;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find(',', start);
        if (end == std::string::npos) {
            end = text.size();
        }
        if (end == start) {
            throw std::invalid_argument("empty GPU ID in --devices");
        }
        int id = 0;
        for (size_t i = start; i < end; ++i) {
            if (text[i] < '0' || text[i] > '9'
                || id > (std::numeric_limits<int>::max() - (text[i] - '0')) / 10) {
                throw std::invalid_argument("--devices requires all or comma-separated nonnegative GPU IDs");
            }
            id = id * 10 + (text[i] - '0');
        }
        if (std::find(ids.begin(), ids.end(), id) != ids.end()) {
            throw std::invalid_argument("duplicate GPU ID in --devices");
        }
        ids.push_back(id);
        start = end + 1;
    }
    if (ids.empty() || text.back() == ',') {
        throw std::invalid_argument("--devices requires all or a nonempty GPU list");
    }
    return ids;
}

std::vector<GpuDevice> available_gpu_devices() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) {
        throw std::runtime_error("no visible CUDA devices; check the driver and CUDA_VISIBLE_DEVICES");
    }
    require_cuda(status, "cudaGetDeviceCount");
    std::vector<GpuDevice> devices;
    for (int id = 0; id < count; ++id) {
        cudaDeviceProp properties{};
        require_cuda(cudaGetDeviceProperties(&properties, id),
                     "cudaGetDeviceProperties");
        char pci_bus_id[32] = {};
        require_cuda(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), id),
                     "cudaDeviceGetPCIBusId");
        devices.push_back({id, properties.name, properties.major,
                           properties.minor, properties.multiProcessorCount,
                           static_cast<unsigned long long>(properties.totalGlobalMem),
                           pci_bus_id});
    }
    return devices;
}

std::vector<GpuDevice> select_gpu_devices(const std::vector<int> &ids) {
    std::vector<GpuDevice> available = available_gpu_devices();
    if (ids.empty()) {
        return available;
    }
    std::vector<GpuDevice> selected;
    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= available.size()) {
            throw std::invalid_argument("GPU ID " + std::to_string(id)
                + " is not visible; run --list-devices (IDs are CUDA-visible ordinals)");
        }
        for (const auto &device : selected) {
            if (device.id == id) {
                throw std::invalid_argument("duplicate GPU ID in selection");
            }
        }
        selected.push_back(available[static_cast<size_t>(id)]);
    }
    return selected;
}

void activate_gpu(int id) {
    require_cuda(cudaSetDevice(id), "cudaSetDevice");
}

}  // namespace ratpoints_gpu
