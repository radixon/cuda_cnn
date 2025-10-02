// addition.cu
#include "addition.h"
#include <stdlib.h>

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