// addition.h
#ifndef ADDITION_H
#define ADDITION_H

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


// Sum Matrix on Host
void sumMatrixOnHost (float *A, float *B, float *C, const int nx, const int ny);

// Sum Matrix on Device
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny);

#ifdef __cplusplus
}
#endif

#endif // ADDITION_H