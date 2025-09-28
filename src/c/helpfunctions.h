// helpfunctions.h
#ifndef HELPFUNCTIONS_H
#define HELPFUNCTIONS_H

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

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

// Function declarations
double cpuSecond(void);
void initialData(float *ip, int size);
void checkResult(float *hostRef, float *gpuRef, int N);

// Additional utility functions
void printMatrix(float *matrix, int nx, int ny, const char* name);
void formatBytes(size_t bytes, char* buffer, size_t buffer_size);
void printDeviceInfo(void);

// Sum Matrix on Host
void sumMatrixOnHost (float *A, float *B, float *C, const int nx, const int ny);

// Sum Matrix on Device
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny);

#ifdef __cplusplus
}
#endif

#endif 