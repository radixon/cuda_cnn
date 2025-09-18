#include <cuda_runtime.h>
#include <stdio.h>
#include "device_info.h"

int main(int argc, char **argv){
    printf("CUDA Information\n");
    printf("=============================\n\n");

    print_all_devices();
    printf("Device Information retrieval Completed\n");
    return 0;
}