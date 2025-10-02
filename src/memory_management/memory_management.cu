// memory_management.cu
#include "memory_management.hpp"
#include "addition.h"
#include "utility.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <cuda_runtime.h>
#include <stdexcept>
#include <string.h>
#endif

#ifdef __cplusplus
// =============================================================================
// RAII Wrapper Implementation Functions
// =============================================================================

// CudaDevice Implementation
CudaDevice::CudaDevice(int dev) : deviceId(dev), initialized(false) {
    struct cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));
    printf("Setting up Device %d: %s\n", deviceId, deviceProp.name);
    CHECK(cudaSetDevice(deviceId));
    initialized = true;
}

CudaDevice::~CudaDevice() {
    if (initialized) {
        cudaError_t error = cudaDeviceReset();
        if (error != cudaSuccess) {
            printf("Error resetting device %d: %s\n", deviceId, cudaGetErrorString(error));
        } else {
            printf("Device %d reset successfully\n", deviceId);
        }
        initialized = false;
    }
}

int CudaDevice::getId() const {
    return deviceId;
}

void CudaDevice::printProperties() const {
    if (initialized) {
        struct cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));
        
        printf("\n=== CUDA Device Properties ===\n");
        printf("Device %d: %s\n", deviceId, deviceProp.name);
        printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Total Global Memory: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024*1024*1024));
        printf("Shared Memory per Block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max Grid Size: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Max Block Dim: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Warp Size: %d\n", deviceProp.warpSize);
        // printf("Memory Clock Rate: %.2f MHz\n", deviceProp.memoryClockRate / 1000.0f);
        printf("Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
        printf("L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
        printf("Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("==============================\n\n");
    }
}

// Template instantiations for common types
template class HostMemory<float>;
template class HostMemory<double>;
template class HostMemory<int>;
template class HostMemory<char>;

template class DeviceMemory<float>;
template class DeviceMemory<double>;
template class DeviceMemory<int>;
template class DeviceMemory<char>;

// Utility functions for RAII wrappers
namespace RAIIUtils {
    
    // Helper function to create formatted memory allocation messages
    void logMemoryAllocation(const char* type, size_t bytes, bool success) {
        char buffer[256];
        formatBytes(bytes, buffer, sizeof(buffer));
        
        if (success) {
            printf("RAII: Allocated %s memory: %s\n", type, buffer);
        } else {
            printf("RAII: Failed to allocate %s memory: %s\n", type, buffer);
        }
    }
    
    void logMemoryDeallocation(const char* type, size_t bytes) {
        char buffer[256];
        formatBytes(bytes, buffer, sizeof(buffer));
        printf("RAII: Freed %s memory: %s\n", type, buffer);
    }
    
    // Enhanced error checking with context
    void checkCudaError(cudaError_t error, const char* operation, const char* file, int line) {
        if (error != cudaSuccess) {
            printf("RAII CUDA Error in %s at %s:%d\n", operation, file, line);
            printf("Error code: %d, reason: %s\n", error, cudaGetErrorString(error));
            throw std::runtime_error("CUDA operation failed");
        }
    }
}

// Enhanced CHECK macro for RAII operations
#define RAII_CHECK(call, operation) \
{ \
    const cudaError_t error = call; \
    RAIIUtils::checkCudaError(error, operation, __FILE__, __LINE__); \
}

// Factory functions for easier RAII object creation
template<typename T>
std::unique_ptr<HostMemory<T>> createHostMemory(size_t count) {
    return std::make_unique<HostMemory<T>>(count);
}

template<typename T>
std::unique_ptr<DeviceMemory<T>> createDeviceMemory(size_t count) {
    return std::make_unique<DeviceMemory<T>>(count);
}

// Explicit template instantiations for factory functions
template std::unique_ptr<HostMemory<float>> createHostMemory<float>(size_t);
template std::unique_ptr<HostMemory<double>> createHostMemory<double>(size_t);
template std::unique_ptr<HostMemory<int>> createHostMemory<int>(size_t);

template std::unique_ptr<DeviceMemory<float>> createDeviceMemory<float>(size_t);
template std::unique_ptr<DeviceMemory<double>> createDeviceMemory<double>(size_t);
template std::unique_ptr<DeviceMemory<int>> createDeviceMemory<int>(size_t);

#endif // __cplusplus