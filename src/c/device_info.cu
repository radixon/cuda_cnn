#include "device_info.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Error Handling
void check_cuda_error(cudaError_t error, const char* operation){
    if(error != cudaSuccess){
        printf("CUDA Error in %s: %s\n", operation, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Version Information
void print_cuda_version(void){
    int driver_version = 0, runtime_version = 0;

    check_cuda_error(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    check_cuda_error(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");

    printf("CUDA Driver Version: %d.%d\n", driver_version / 1000, (driver_version %100) / 10);
    printf("CUDA Runtime Version: %d.%d\n", runtime_version/1000, (runtime_version % 100) / 10);
}

// Device Count
int get_device_count(void){
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    check_cuda_error(error, "cudaGetDeviceCount");
    return device_count;
}

// Utility Functions
void format_bytes(size_t bytes, char* buffer, size_t buffer_size){
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = (double)bytes;

    while(size >= 1024 && unit < 4){
        size /= 1024.0;
        unit++;
    }

    snprintf(buffer, buffer_size, "%.2f %s", size, units[unit]);
}

void format_frequency(int clock_rate_khz, char* buffer, size_t buffer_size){
    if(clock_rate_khz >= 1000000){
        snprintf(buffer, buffer_size, "%.2f GHZ", clock_rate_khz / 1000000.0);
    }else{
        snprintf(buffer, buffer_size, "%.0f MHz", clock_rate_khz / 1000.0);
    }
}

int is_device_compatible(int device_id){
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
    return (prop.major >= 3);
}

double get_device_memory_gb(int device_id){
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
    return prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
}

const char* get_device_name(int device_id){
    static cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
    return prop.name;
}

// Device Information Functions
void print_device_basic_info(int device_id){
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");

    char memory_str[64];
    format_bytes(prop.totalGlobalMem, memory_str, sizeof(memory_str));

    printf("Device %d: \"%s\"\n", device_id, prop.name);
    printf("Compute Capability: %d.%d", prop.major, prop.minor);

    // Architecture Name
    if(prop.major == 8){printf(" (Ampere) ");}
    else if(prop.major == 7){printf(" (Vota/Turing) ");}
    else if(prop.major == 6){printf(" (Pascal) ");}
    else if(prop.major == 5){printf(" (Maxwell) ");}
    else if(prop.major == 3){printf(" (Kepler) ");}

    printf("\n Global Memory: %s\n", memory_str);
    printf(" Multiprocessors: %d\n", prop.multiProcessorCount);
    printf(" Compatible: %s\n", is_device_compatible(device_id) ? "Yes" : "No");
}

void print_device_memory_info(int device_id){
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");

    char buffer[64];
    printf("\nMemory Information:\n");
    
    format_bytes(prop.totalGlobalMem, buffer, sizeof(buffer));
    printf(" Global Memory: %s\n", buffer);

    format_bytes(prop.totalConstMem, buffer, sizeof(buffer));
    printf(" Constant Memory: %s\n", buffer);

    format_bytes(prop.sharedMemPerBlock, buffer, sizeof(buffer));
    printf(" Shared Memory per Block: %s\n", buffer);

    if(prop.l2CacheSize > 0){
        format_bytes(prop.l2CacheSize, buffer, sizeof(buffer));
        printf(" L2 Cache Size: %s\n", buffer);
    }

    format_frequency(prop.memoryClockRate, buffer, sizeof(buffer));
    printf(" Memory Clock Rate: %s\n", buffer);
    printf(" Memory Bus Width: %d bits\n", prop.memoryBusWidth);

    // Calculate Theoretical Bandwidth
    double bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    printf(" Memory Bandwidth (theoretical): %.1f GB/s\n", bandwidth);
}

void print_device_compute_info(int device_id){
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
    char buffer[64];

    printf("\nCompute Information\n");
    printf(" Multiprocessors: %d\n", prop.multiProcessorCount);

    // Estimate CUDA cores
    int estimated_cores = prop.multiProcessorCount * 128;
    printf(" CUDA Cores: %d\n", estimated_cores);

    format_frequency(prop.clockRate, buffer, sizeof(buffer));
    printf(" GPU Clock Rate: %s\n", buffer);
    printf(" Warp Size: %d\n", prop.warpSize);
    printf(" Registers per Block: %d\n", prop.regsPerBlock);
}

void print_device_thread_info(int device_id){
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");

    printf("\nThread Information:\n");
    printf(" Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf(" Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf(" Max Blcok Dimensions: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf(" Max Grid Dimensions: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

void print_device_detailed_info(int device_id){
    printf("\nCUDA Device Information\n");
    printf("=====================================\n");
    print_device_basic_info(device_id);
    print_device_memory_info(device_id);
    print_device_compute_info(device_id);
    print_device_thread_info(device_id);

    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");

    printf("\nTexture Memory Limits:\n");
    printf(" 1D Texture: %d\n", prop.maxTexture1D);
    printf(" 2D Texture: %d x %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf(" 3D Texture: %d x %d x %d\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
}

void print_system_info(void){
    printf("CUDA System Information\n");
    printf("=============================\n");
    print_cuda_version();

    int device_count = get_device_count();
    printf("Total CUDA Devices: %d\n", device_count);

    if(device_count == 0){
        printf("\nNo CUDA-capable devices found!\n");
        return;
    }

    // Count compatible devices
    int compatible_count = 0;
    for(int i=0; i < device_count; i++){
        if(is_device_compatible(i)){
            compatible_count++;
        }
    }
    printf("Compatible Devices: %d\n", compatible_count);
}

void print_all_devices(void){
    print_system_info();
    int device_count = get_device_count();
    if(device_count == 0){
        return;
    }

    // Print detailed info for each device
    for(int i=0; i < device_count; i++){
        print_device_detailed_info(i);
    }

    // Summary
    printf("======================================================\n");
    printf("Summary:\n");
    for(int i=0; i < device_count; i++){
        print_device_basic_info(i);
        printf("\n");
    }
}