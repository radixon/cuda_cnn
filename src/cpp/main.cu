// main.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utility.h"
#include "addition.h"
#include "sobel.h"
#include "memory_management.hpp"  

void addition(char *title, char *name);
void sobel(char *title, char *name);

int main(int argc, char **argv) {
    for(int i=1; i < argc; i++){
        if(strcmp(argv[i], "addition") == 0){
            addition(argv[0], argv[i]);
        }
        
        if(strcmp(argv[i], "sobel") == 0){
            sobel(argv[0], argv[i]);
        }
    }
     

    // reset device
    return (0);
}

void addition(char *title, char *name){
    printf("%s Starting %s\n", title, name);

    // set up device
    CudaDevice device(0);
    // device.printProperties();

    // set up date size of matrix
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx*ny;
    printf("Matrix size: nx %d ny %d\n",nx, ny);

    // allocate host memory
    HostMemory<float> h_A(nxy);
    HostMemory<float> h_B(nxy);
    HostMemory<float> hostRef(nxy);
    HostMemory<float> gpuRef(nxy);

    // initialize data at host side
    double iStart = cpuSecond();
    initialData (h_A.get(), nxy);
    initialData (h_B.get(), nxy);
    double iElaps = cpuSecond() - iStart;
    
    hostRef.memset(0);
    gpuRef.memset(0);

    // add matrix at host side for result checks
    iStart = cpuSecond();
    sumMatrixOnHost (h_A.get(), h_B.get(), hostRef.get(), nx,ny);
    iElaps = cpuSecond() - iStart;

    // allocate device global memory
    DeviceMemory<float> d_MatA(nxy);
    DeviceMemory<float> d_MatB(nxy);
    DeviceMemory<float> d_MatC(nxy);

    // transfer data from host to device
    d_MatA.copyFromHost(h_A);
    d_MatB.copyFromHost(h_B);

    // launch kernel at host side
    int dimx = 64;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    iStart = cpuSecond();
    sumMatrixOnGPU2D <<< grid, block >>>(d_MatA.get(), d_MatB.get(), d_MatC.get(), nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
    grid.y, block.x, block.y, iElaps);

    // copy kernel result back to host side
    d_MatC.copyToHost(gpuRef);

    // check device results
    checkResult(hostRef.get(), gpuRef.get(), nxy);
}

void sobel(char *title, char *name){
    printf("%s Starting %s\n", title, name);

    // set up device
    CudaDevice device(0);
    // device.printProperties();

    // set up image dimensions (typically smaller than matrix operations)
    int width = 1024;   // Image width
    int height = 1024;  // Image height
    int nPixels = width * height;
    printf("Image size: width %d height %d\n", width, height);

    // allocate host memory
    HostMemory<float> h_input(nPixels);
    HostMemory<float> hostRef(nPixels);
    HostMemory<float> gpuRef(nPixels);

    // initialize data at host side
    double iStart = cpuSecond();
    generateTestImage(h_input.get(), width, height); // This would typically load an actual image
    double iElaps = cpuSecond() - iStart;
    
    hostRef.memset(0);
    gpuRef.memset(0);

    // add matrix at host side for result checks
    iStart = cpuSecond();
    sobelVerticalOnHost(h_input.get(), hostRef.get(), width, height);
    iElaps = cpuSecond() - iStart;
    printf("sobelVerticalOnHost elapsed %f sec\n", iElaps);

    // allocate device global memory
    DeviceMemory<float> d_input(nPixels);
    DeviceMemory<float> d_output(nPixels);
    // transfer data from host to device
    d_input.copyFromHost(h_input);

    // launch kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    iStart = cpuSecond();
    sobelVerticalOnGPU <<< grid, block >>>(d_input.get(), d_output.get(), width, height);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    printf("sobelVerticalOnGPU <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", 
           grid.x, grid.y, block.x, block.y, iElaps);

    // copy kernel result back to host side
    d_output.copyToHost(gpuRef);

    // check results
    compareSobelResults(hostRef.get(), gpuRef.get(), nPixels, 1e-5f);
}