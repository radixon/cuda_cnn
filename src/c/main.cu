// main.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "addition.h"
#include "sobel.h" 
#include "utility.h" 

void addition(char *title, char *name);
void sobel(char *title, char *name);

int main(int argc, char **argv) {
    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "addition") == 0){
            addition(argv[0], argv[i]);
        }
        
        if(strcmp(argv[i], "sobel") == 0){
            sobel(argv[0], argv[i]);
        }
    }
           
    return (0);
}

void addition(char *title, char *name){
    printf("%s Starting %s\n", title, name);

    // set up device
    int dev = 0;
    struct cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up date size of matrix
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n",nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = cpuSecond();
    initialData (h_A, nxy);
    initialData (h_B, nxy);
    double iElaps = cpuSecond() - iStart;
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = cpuSecond();
    sumMatrixOnHost (h_A, h_B, hostRef, nx,ny);
    iElaps = cpuSecond() - iStart;

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    int dimx = 64;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    iStart = cpuSecond();
    sumMatrixOnGPU2D <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
    grid.y, block.x, block.y, iElaps);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    cudaDeviceReset();
}

void sobel(char *title, char *name){
    printf("%s Starting %s\n", title, name);

    // set up device
    int dev = 0;
    struct cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up image dimensions (typically smaller than matrix operations)
    int width = 1024;   // Image width
    int height = 1024;  // Image height
    int nPixels = width * height;
    int nBytes = nPixels * sizeof(float);
    printf("Image size: width %d height %d\n", width, height);

    // malloc host memory
    float *h_input, *hostRef, *gpuRef;
    h_input = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize input image data
    double iStart = cpuSecond();
    generateTestImage(h_input, width, height); // This would typically load an actual image
    double iElaps = cpuSecond() - iStart;
    
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // apply Sobel on host for result verification
    iStart = cpuSecond();
    sobelVerticalOnHost(h_input, hostRef, width, height);
    iElaps = cpuSecond() - iStart;
    printf("sobelVerticalOnHost elapsed %f sec\n", iElaps);

    // malloc device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, nBytes);
    cudaMalloc((void **)&d_output, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_input, h_input, nBytes, cudaMemcpyHostToDevice);

    // launch Sobel kernel
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    iStart = cpuSecond();
    sobelVerticalOnGPU<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sobelVerticalOnGPU <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", 
           grid.x, grid.y, block.x, block.y, iElaps);

    // copy result back to host
    cudaMemcpy(gpuRef, d_output, nBytes, cudaMemcpyDeviceToHost);

    // check results
    compareSobelResults(hostRef, gpuRef, nPixels, 1e-5f);

    // cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
}