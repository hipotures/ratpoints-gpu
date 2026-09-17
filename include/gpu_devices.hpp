#ifndef RATPOINTS_GPU_GPU_DEVICES_HPP
#define RATPOINTS_GPU_GPU_DEVICES_HPP

#include <string>
#include <vector>

namespace ratpoints_gpu {

struct GpuDevice {
    int id;
    std::string name;
    int major;
    int minor;
    int multiprocessors;
    unsigned long long memory_bytes;
    std::string pci_bus_id;
};

// An empty selection means all devices visible to the CUDA runtime.
std::vector<int> parse_device_selection(const std::string &text);
std::vector<GpuDevice> available_gpu_devices();
std::vector<GpuDevice> select_gpu_devices(const std::vector<int> &ids);
void activate_gpu(int id);

}  // namespace ratpoints_gpu

#endif
