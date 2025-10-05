// sobel.h
#ifndef SOBEL_H
#define SOBEL_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C"{
#endif

// Sobel on Host
void sobelVerticalOnHost(float *input, float *output, int width, int height);

// Sobel on Device
__global__ void sobelVerticalOnGPU(float *input, float *output, int width, int height);

// Sobel at a specific location
__device__ float applySobelVertical(float *input, int x, int y, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // SOBEL_H