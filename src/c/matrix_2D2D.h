#ifndef MATRIX_2D2D_H
#define MATRIX_2D2D_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Error Macro
#define CHECK(call)                                                         \
do{                                                                         \
    const cudaError_t error = call;                                         \
    if(error != cudaSuccess){                                               \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                            \
    }                                                                       \
} while(0)

// Timing Functions
double cpuSecond(void);

// Data Initialization
void initialData(float *ip, int size);

// Host Computation Function
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny);

// Device Computation Function
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny);

// Utility Function
void checkResult(float *hostRef, float *gpuRef, const int N);

// Device Management
void printDeviceInfo(int dev);

#endif