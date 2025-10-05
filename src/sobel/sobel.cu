// sobel.cu
#include "sobel.h"
#include <stdlib.h>
#include <math.h>

// Get pixel value with boundary checking
float getPixelValue(float *input, int x, int y, int width, int height){
    if(x < 0 || x >= width || y < 0 || y >= height){
        return 0.0f;
    }
    return input[y*width + x];
}

// Sobel on Host
void sobelVerticalOnHost(float *input, float *output, const int width, const int height){
    int kernel[3][3] = {
                        {-1, 0, 1},
                        {-2, 0, 2},
                        {-1, 0, 1}
    };

    for(int y=0; y < height; y++){
        for(int x=0; x < width; x++){
            float sum = 0.0f;

            // Apply Sobel kernel
            for(int y_kernel = -1; y_kernel < 2; y_kernel++){
                for(int x_kernel = -1; x_kernel < 2; x_kernel++){
                    float pixel = getPixelValue(input, x + x_kernel, y + y_kernel, width, height);
                    sum += pixel * kernel[y_kernel + 1][x_kernel + 1];
                }
            }

            // Store result
            output[y * width + x] = fabsf(sum);
        }
    }
}

// Sobel on Device
__global__ void sobelVerticalOnGPU(float *input, float *output, int width, int height){
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height){
        unsigned int idx = y * width + x;
        output[idx] = applySobelVertical(input, x, y , width, height);
    }
}

// Sobel operator at a specific location
__device__ float applySobelVertical(float *input, int x, int y, int width, int height){
    // Sobel vertical kernel
    int kernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

    float sum = 0.0f;
    int kernelIndex = 0;

    // Apply kernel
    for(int y_kernel = -1; y_kernel < 2; y_kernel++){
        for(int x_kernel = -1; x_kernel < 2; x_kernel++){
            int idx_x = x + x_kernel;
            int idx_y = y + y_kernel;

            float pixel = 0.0f;
            // Boundary checking
            if(idx_x > -1 && idx_x < width && idx_y > -1 && idx_y < height){
                pixel = input[idx_y * width + idx_x];
            }

            sum += pixel * kernel[kernelIndex];
            kernelIndex++;
        }
    }
    return fabsf(sum);
}