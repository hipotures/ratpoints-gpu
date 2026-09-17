#ifndef RATPOINTS_TEST_MOCK_CUDA_RUNTIME_API_H
#define RATPOINTS_TEST_MOCK_CUDA_RUNTIME_API_H

// Deliberately minimal test double. Never included by the production build.
#include <cstddef>

enum cudaError_t { cudaSuccess, cudaErrorNoDevice, cudaErrorUnknown };
struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
    int multiProcessorCount;
    std::size_t totalGlobalMem;
};
cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *properties, int device);
cudaError_t cudaDeviceGetPCIBusId(char *text, int length, int device);
cudaError_t cudaSetDevice(int device);
const char *cudaGetErrorString(cudaError_t status);

#endif
