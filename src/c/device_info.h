#ifndef DEVICE_INFO_C_H
#define DEVICE_INFO_C_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error Handling
void check_cuda_error(cudaError_t error, const char* operation);

// Version Information
void print_cuda_version(void);

// Device Count
int get_device_count(void);

// Device Information Functions
void print_device_basic_info(int device_id);
void print_device_memory_info(int device_id);
void print_device_compute_info(int device_id);
void print_device_thread_info(int device_id);
void print_device_detailed_info(int device_id);

// System Information
void print_system_info(void);
void print_all_devices(void);

// Utility Functions
int is_device_compatible(int device_id);
double get_device_memory_gb(int device_id);
const char* get_device_name(int device_id);

// Formatting Helpers
void format_bytes(size_t bytes, char* buffer, size_t buffer_size);
void format_frequency(int clock_rate_khz, char* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif