#include <cuda_runtime.h>
#include <iostream>
#include <exception>
#include "device_info.h"

int cudaDeviceInformation(){
    std::cout << "CUDA CNN - Device Information (C++)" << '\n';
    std::cout << std::string(60, '=') << '\n';

    try{
        DeviceInfo::DeviceManager manager;
        manager.printAllDevices();

        std::cout << "\n Device information retrieval completed!" << '\n';
        return 0;
    }catch(const std::exception& e){
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

int main(int argc, char **argv){
    return cudaDeviceInformation();
}