// utility.cu
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "utility.h"

// Timer function - returns current time in seconds
double cpuSecond(void) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Initialize matrix with random data
void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned int) time(&t));
    
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// Generate test image
void generateTestImage(float *image, int width, int height) {
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            // Create a simple pattern with edges
            if(x < width/3) {
                image[y * width + x] = 0.0f;  // Black region
            } else if(x < 2*width/3) {
                image[y * width + x] = 128.0f;  // Gray region
            } else {
                image[y * width + x] = 255.0f;  // White region
            }
            
            // Add some vertical stripes for testing
            if((x % 20) < 5) {
                image[y * width + x] = 255.0f - image[y * width + x];
            }
        }
    }
}

// Check results between CPU and GPU
void checkResult(float *hostRef, float *gpuRef, int N) {
    double epsilon = 1.0E-8;
    bool match = true;
    int errorCount = 0;
    
    for (int i = 0; i < N; i++) {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = false;
            errorCount++;
            
            // Show first 10 errors only
            if (errorCount <= 10) {
                printf("Mismatch at index %d: host=%5.2f gpu=%5.2f diff=%f\n", 
                       i, hostRef[i], gpuRef[i], fabs(hostRef[i] - gpuRef[i]));
            }
        }

        if(i < 10){
            printf("index: %d \t host element: %5.2f \t gpu element: %5.2f\n",i,hostRef[i], gpuRef[i]);
        }
    }
    
    if (match) {
        printf("Arrays match\n\n");
    } else {
        printf("Arrays do not match! Found %d errors out of %d elements.\n\n", 
               errorCount, N);
    }
}

// Function to compare results
bool compareSobelResults(float *host_result, float *gpu_result, int size, float tolerance) {
    int mismatches = 0;
    float max_diff = 0.0f;
    
    for(int i = 0; i < size; i++) {
        float diff = fabsf(host_result[i] - gpu_result[i]);
        if(diff > tolerance) {
            mismatches++;
            if(diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    printf("Comparison Results:\n");
    printf("  Total elements: %d\n", size);
    printf("  Mismatches: %d (%.2f%%)\n", mismatches, (float)mismatches/size * 100.0f);
    printf("  Maximum difference: %f\n", max_diff);
    printf("  Tolerance: %f\n\n", tolerance);
    
    return mismatches == 0;
}

// Print matrix (useful for debugging small matrices)
void printMatrix(float *matrix, int nx, int ny, const char* name) {
    printf("\n%s Matrix (%dx%d):\n", name, nx, ny);
    
    int maxRows = (ny > 8) ? 8 : ny;
    int maxCols = (nx > 8) ? 8 : nx;
    
    for (int i = 0; i < maxRows; i++) {
        for (int j = 0; j < maxCols; j++) {
            printf("%6.2f ", matrix[i * nx + j]);
        }
        if (nx > 8) printf("...");
        printf("\n");
    }
    if (ny > 8) printf("...\n");
    printf("\n");
}

// Format bytes into human-readable format
void formatBytes(size_t bytes, char* buffer, size_t buffer_size) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = (double)bytes;
    
    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }
    
    if (unit_index == 0) {
        snprintf(buffer, buffer_size, "%zu %s", bytes, units[unit_index]);
    } else {
        snprintf(buffer, buffer_size, "%.2f %s", size, units[unit_index]);
    }
}

// Print basic device information (requires CUDA headers when used)
void printDeviceInfo(void) {
    printf("=== System Information ===\n");
    
    // Get current time
    time_t rawtime;
    struct tm *timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    printf("Timestamp: %s", asctime(timeinfo));
    
    // Print some basic system info
    printf("Helper functions library loaded successfully.\n");
    printf("Timer precision: microseconds\n");
    printf("Random seed: time-based\n");
    printf("Floating point epsilon: 1.0E-8\n");
    printf("===========================\n\n");
}
