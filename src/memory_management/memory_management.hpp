// memroy_management.hpp
#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

// CUDA headers needed for RAII wrappers
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
#include <memory>
#include <stdexcept>
#include <string.h>

// RAII wrapper for host memory management
template<typename T>
class HostMemory {
private:
    T* ptr;
    size_t size;

public:
    // Constructor - allocates memory
    explicit HostMemory(size_t count);
    
    // Destructor - automatically frees memory
    ~HostMemory();
    
    // Delete copy constructor and assignment operator to prevent double-free
    HostMemory(const HostMemory&) = delete;
    HostMemory& operator=(const HostMemory&) = delete;
    
    // Move constructor and assignment operator
    HostMemory(HostMemory&& other) noexcept;
    HostMemory& operator=(HostMemory&& other) noexcept;
    
    // Access methods
    T* get();
    const T* get() const;
    T& operator[](size_t index);
    const T& operator[](size_t index) const;
    
    // Utility methods
    void memset(int value);
    size_t getSize() const;
    size_t getCount() const;
};

#ifdef __CUDACC__
// RAII wrapper for device memory management
template<typename T>
class DeviceMemory {
private:
    T* ptr;
    size_t size;

public:
    // Constructor - allocates device memory
    explicit DeviceMemory(size_t count);
    
    // Destructor - automatically frees device memory
    ~DeviceMemory();
    
    // Delete copy constructor and assignment operator
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // Move constructor and assignment operator
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;
    
    // Access methods
    T* get();
    const T* get() const;
    
    // Memory transfer methods
    void copyFromHost(const T* hostPtr);
    void copyToHost(T* hostPtr) const;
    void copyFromHost(const HostMemory<T>& hostMem);
    void copyToHost(HostMemory<T>& hostMem) const;
    
    // Utility methods
    size_t getSize() const;
    size_t getCount() const;
};

// RAII wrapper for CUDA device management
class CudaDevice {
private:
    int deviceId;
    bool initialized;

public:
    // Constructor - sets up device
    explicit CudaDevice(int dev = 0);
    
    // Destructor - resets device
    ~CudaDevice();
    
    // Delete copy constructor and assignment operator
    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(const CudaDevice&) = delete;
    
    // Access methods
    int getId() const;
    void printProperties() const;
};
#endif // __CUDACC__

extern "C" {
#endif // __cplusplus

#define CHECK(call) \
{ \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(-10*error);                                                    \
    }                                                                       \
}

#ifdef __cplusplus
}

// Template method implementations (must be in header for templates)
template<typename T>
inline HostMemory<T>::HostMemory(size_t count) : size(count * sizeof(T)) {
    ptr = static_cast<T*>(malloc(size));
    if (!ptr) {
        throw std::bad_alloc();
    }
}

template<typename T>
inline HostMemory<T>::~HostMemory() {
    if (ptr) {
        free(ptr);
        ptr = nullptr;
    }
}

template<typename T>
inline HostMemory<T>::HostMemory(HostMemory&& other) noexcept 
    : ptr(other.ptr), size(other.size) {
    other.ptr = nullptr;
    other.size = 0;
}

template<typename T>
inline HostMemory<T>& HostMemory<T>::operator=(HostMemory&& other) noexcept {
    if (this != &other) {
        if (ptr) free(ptr);
        ptr = other.ptr;
        size = other.size;
        other.ptr = nullptr;
        other.size = 0;
    }
    return *this;
}

template<typename T>
inline T* HostMemory<T>::get() { return ptr; }

template<typename T>
inline const T* HostMemory<T>::get() const { return ptr; }

template<typename T>
inline T& HostMemory<T>::operator[](size_t index) { return ptr[index]; }

template<typename T>
inline const T& HostMemory<T>::operator[](size_t index) const { return ptr[index]; }

template<typename T>
inline void HostMemory<T>::memset(int value) {
    ::memset(ptr, value, size);
}

template<typename T>
inline size_t HostMemory<T>::getSize() const { return size; }

template<typename T>
inline size_t HostMemory<T>::getCount() const { return size / sizeof(T); }

#ifdef __CUDACC__
template<typename T>
inline DeviceMemory<T>::DeviceMemory(size_t count) : size(count * sizeof(T)) {
    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&ptr), size);
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed");
    }
}

template<typename T>
inline DeviceMemory<T>::~DeviceMemory() {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

template<typename T>
inline DeviceMemory<T>::DeviceMemory(DeviceMemory&& other) noexcept 
    : ptr(other.ptr), size(other.size) {
    other.ptr = nullptr;
    other.size = 0;
}

template<typename T>
inline DeviceMemory<T>& DeviceMemory<T>::operator=(DeviceMemory&& other) noexcept {
    if (this != &other) {
        if (ptr) cudaFree(ptr);
        ptr = other.ptr;
        size = other.size;
        other.ptr = nullptr;
        other.size = 0;
    }
    return *this;
}

template<typename T>
inline T* DeviceMemory<T>::get() { return ptr; }

template<typename T>
inline const T* DeviceMemory<T>::get() const { return ptr; }

template<typename T>
inline void DeviceMemory<T>::copyFromHost(const T* hostPtr) {
    CHECK(cudaMemcpy(ptr, hostPtr, size, cudaMemcpyHostToDevice));
}

template<typename T>
inline void DeviceMemory<T>::copyToHost(T* hostPtr) const {
    CHECK(cudaMemcpy(hostPtr, ptr, size, cudaMemcpyDeviceToHost));
}

template<typename T>
inline void DeviceMemory<T>::copyFromHost(const HostMemory<T>& hostMem) {
    CHECK(cudaMemcpy(ptr, hostMem.get(), size, cudaMemcpyHostToDevice));
}

template<typename T>
inline void DeviceMemory<T>::copyToHost(HostMemory<T>& hostMem) const {
    CHECK(cudaMemcpy(hostMem.get(), ptr, size, cudaMemcpyDeviceToHost));
}

template<typename T>
inline size_t DeviceMemory<T>::getSize() const { return size; }

template<typename T>
inline size_t DeviceMemory<T>::getCount() const { return size / sizeof(T); }
#endif // __CUDACC__

#endif // __cplusplus

#endif // MEMORY_MANAGEMENT_H