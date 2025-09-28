// helpfunctions.cu
#include "helpfunctions.hpp"
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

// Timer function - returns current time in seconds
double cpuSecond(void) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Initialize matrix with random data
void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned int) time(&t));
    
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// Check results between CPU and GPU
void checkResult(float *hostRef, float *gpuRef, int N) {
    double epsilon = 1.0E-8;
    bool match = true;
    int errorCount = 0;
    const int len = N < 10 ? N : 10;
    
    for (int i = 0; i < N; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = false;
            errorCount++;
            
            // Show first 10 errors only
            if (errorCount <= len) {
                printf("Mismatch at index %d: host=%5.2f gpu=%5.2f diff=%f\n", 
                       i, hostRef[i], gpuRef[i], fabs(hostRef[i] - gpuRef[i]));
            }
        }
    }
    // Show first 10 values if not error
    if(errorCount == 0){
        for(int i=0; i < len; i++){
            printf("index: %d \t Host Element: %5.2f \t Device Element: %5.2f\n", i, hostRef[i], gpuRef[i]);
        }
    }
    if (match) {
        printf("Arrays match\n\n");
    } else {
        printf("Arrays do not match! Found %d errors out of %d elements.\n\n", 
               errorCount, N);
    }
}

// Print matrix (useful for debugging small matrices)
void printMatrix(float *matrix, int nx, int ny, const char* name) {
    printf("\n%s Matrix (%dx%d):\n", name, nx, ny);
    
    int maxRows = (ny > 8) ? 8 : ny;
    int maxCols = (nx > 8) ? 8 : nx;
    
    for (int i = 0; i < maxRows; i++) {
        for (int j = 0; j < maxCols; j++) {
            printf("%6.2f ", matrix[i * nx + j]);
        }
        if (nx > 8) printf("...");
        printf("\n");
    }
    if (ny > 8) printf("...\n");
    printf("\n");
}

// Format bytes into human-readable format
void formatBytes(size_t bytes, char* buffer, size_t buffer_size) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = (double)bytes;
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    if (unit_index == 0) {
        snprintf(buffer, buffer_size, "%zu %s", bytes, units[unit_index]);
    } else {
        snprintf(buffer, buffer_size, "%.2f %s", size, units[unit_index]);
    }
}

// Print basic device information (requires CUDA headers when used)
void printDeviceInfo(void) {
    printf("=== System Information ===\n");
    
    // Get current time
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Timestamp: %s", asctime(timeinfo));
    
    // Print some basic system info
    printf("Helper functions library loaded successfully.\n");
    printf("Timer precision: microseconds\n");
    printf("Random seed: time-based\n");
    printf("Floating point epsilon: 1.0E-8\n");
    printf("===========================\n\n");
}

// Sum Matrix on Host
void sumMatrixOnHost (float *A, float *B, float *C, const int nx, const int ny) {
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for (int iy=0; iy<ny; iy++) 
    {
        for (int ix=0; ix<nx; ix++) 
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

// Sum Matrix on Device
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < nx && iy < ny)
    {
        unsigned int idx = iy*nx + ix;
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

#ifdef __cplusplus
// =============================================================================
// RAII Wrapper Implementation Functions
// =============================================================================

// CudaDevice Implementation
CudaDevice::CudaDevice(int dev) : deviceId(dev), initialized(false) {
    struct cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));
    printf("RAII: Setting up Device %d: %s\n", deviceId, deviceProp.name);
    CHECK(cudaSetDevice(deviceId));
    initialized = true;
}

CudaDevice::~CudaDevice() {
    if (initialized) {
        cudaError_t error = cudaDeviceReset();
        if (error != cudaSuccess) {
            printf("RAII: Error resetting device %d: %s\n", deviceId, cudaGetErrorString(error));
        } else {
            printf("RAII: Device %d reset successfully\n", deviceId);
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
        printf("Memory Clock Rate: %.2f MHz\n", deviceProp.memoryClockRate / 1000.0f);
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