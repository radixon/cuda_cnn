#include "device_info.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

namespace DeviceInfo{
    void checkCudaError(cudaError_t error, const std::string& operation){
        if(error != cudaSuccess){
            throw std::runtime_error(operation + " failed: " + cudaGetErrorString(error));
        }
    }

    std::string formatBytes(size_t bytes){
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit = 0;
        double size = static_cast<double>(bytes);

        while(size >= 1024.0 && unit < 4){
            size /= 1024.0;
            unit++;
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
        return oss.str();
    }

    std::string formatFrequency(int clockRateKHz){
        std::ostringstream oss;
        if(clockRateKHz >= 1000000){
            oss << std::fixed << std::setprecision(2) << clockRateKHz / 1000000.0 << " GHz";
        }
        else{
            oss << std::fixed << std::setprecision(0) << clockRateKHz / 1000.0 << " MHz";
        }
        return oss.str();
    }

    void printCudaVersion(){
        int driverVersion = 0, runtimeVersion = 0;

        checkCudaError(cudaDriverGetVersion(&driverVersion), "cudaDriverGetVersion");
        checkCudaError(cudaRuntimeGetVersion(&runtimeVersion), "cudaRuntimeGetVersion");

        std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << "\n";
        std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << "\n";
    }

    // CudaDevice Implementation
    CudaDevice::CudaDevice(int deviceId) : deviceId_(deviceId){
        checkCudaError(cudaGetDeviceProperties(&properties_, deviceId), "cudaGetDeviceProperties for device " +
                std::to_string(deviceId));
    }

    bool CudaDevice::isCompatible() const{
        return properties_.major >= 3;
    }

    double CudaDevice::getGlobalMemoryGB() const{
        return properties_.totalGlobalMem / pow(1024.0,3);
    }

    void CudaDevice::printBasicInfo() const{
        std::cout << "Device " << deviceId_ << ": \"" << properties_.name << "\"" << "\n";
        std::cout << "Compute Capabiity: " << properties_.major << "." << properties_.minor;

        // Architecture name
        if(properties_.major == 8) {std::cout << " (Ampere)";}
        else if(properties_.major == 7) {std::cout << " (Volta/Turning)";}
        else if(properties_.major == 6) {std::cout << " (Pascal)";}
        else if(properties_.major == 5) {std::cout << " (Maxwell)";}
        else if(properties_.major == 3) {std::cout << " (Kepler)";}

        std::cout << "\n";
        std::cout << " Global Memory: " << formatBytes(properties_.totalGlobalMem) << "\n";
        std::cout << " Multiprocessors: " << properties_.multiProcessorCount << "\n";
        std::cout << " Compatible: " << (isCompatible() ? "Yes" : "No") << "\n";
    }

    void CudaDevice::printMemoryInfo() const{
        std::cout << "\nMemory Information:" << "\n";
        std::cout << " Global Memory: " << formatBytes(properties_.totalGlobalMem) << "\n";
        std::cout << " Constant Memory: " << formatBytes(properties_.totalConstMem) << "\n";
        std::cout << " Shared Memory per Block: " << formatBytes(properties_.sharedMemPerBlock) << "\n";

        if(properties_.l2CacheSize > 0){
            std::cout << " L2 Cache Size: " << formatBytes(properties_.l2CacheSize) << "\n";
        }

        std::cout << " Memory Clock Rate: " << formatFrequency(properties_.memoryClockRate) << "\n";
        std::cout << " Memory Bus Width: " << properties_.memoryBusWidth << " bits" << "\n";

        double bandwidth = 2.0 * properties_.memoryClockRate * (properties_.memoryBusWidth / 8) / 1.0e6;
        std::cout << " Memory Bandwidth (theoretical): " << std::fixed << std::setprecision(1) << bandwidth
                    << " GB/s" << "\n";
    }

    void CudaDevice::printComputeInfo() const{
        std::cout << "\nCompute Information:" << "\n";
        std::cout << " Multiprocessors: " << properties_.multiProcessorCount << "\n";

        // Estimate CUDA cores
        int estimatedCores = properties_.multiProcessorCount * 128;
        std::cout << " CUDA Cores (estimated): " << estimatedCores << "\n";

        std::cout << " GPU Clock Rate: " << formatFrequency(properties_.clockRate) << "\n";
        std::cout << " Warp Size: " << properties_.warpSize << "\n";
        std::cout << " Registers per Block: "  << properties_.regsPerBlock << "\n";
    }

    void CudaDevice::printThreadInfo() const{
        std::cout << "\nThread Information:" << "\n";
        std::cout << " Max Threads per Block: " << properties_.maxThreadsPerBlock << "\n";
        std::cout << " Max Thresds per Multiprocess: " << properties_.maxThreadsPerMultiProcessor << "\n";

        std::cout   << " Max Block Dimensions: "
                    << properties_.maxThreadsDim[0] << " x "
                    << properties_.maxThreadsDim[1] << " x "
                    << properties_.maxThreadsDim[2] << "\n";
        
        std::cout   << " Max Grid Dimensions: "
                    << properties_.maxGridSize[0] << " x "
                    << properties_.maxGridSize[1] << " x "
                    << properties_.maxGridSize[2] << "\n";
    }

    void CudaDevice::printDetailedInfo() const{
        std::cout << "\n" << std::string(60, '=') << "\n";
        printBasicInfo();
        printMemoryInfo();
        printComputeInfo();
        printThreadInfo();

        std::cout << "\nTexture Memory Limits:" << "\n";
        std::cout << " 1D Texture: " << properties_.maxTexture1D << "\n";
        std::cout << " 2D Texture: " 
                    << properties_.maxTexture2D[0] << " x " 
                    << properties_.maxTexture2D[1] << "\n";
        std::cout << " 3D Texture: " 
                    << properties_.maxTexture3D[0] << " x "
                    << properties_.maxTexture3D[1] << " x "
                    << properties_.maxTexture3D[2] << "\n";
    }

    // DeviceManager Implementation
    DeviceManager::DeviceManager(){
        int deviceCount = 0;
        checkCudaError(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");

        for(int i=0; i < deviceCount; i++){
            devices_.emplace_back(i);
        }
    }

    void DeviceManager::printSystemInfo() const{
        std::cout << "CUDA System Information" << "\n";
        std::cout << std::string(40, '=') << "\n";

        printCudaVersion();
        std::cout << " Total CUDA Devices: " << getDeviceCount() << "\n";
        if(getDeviceCount() == 0){
            std::cout << "\nNo CUDA-capable devices found" << "\n";
            return;
        }

        auto compatible = getCompatibleDevices();
        std::cout << "Compatible Devices: " << compatible.size() << "\n";
    }

    void DeviceManager::printAllDevices() const{
        printSystemInfo();
        for(const auto& device : devices_){
            device.printDetailedInfo();
        }

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Summary:" << "\n";
        for(const auto& device : devices_){
            device.printBasicInfo();
            std::cout << "\n";
        }
    }

    std::vector<int> DeviceManager::getCompatibleDevices() const{
        std::vector<int> compatible;
        for(const auto& device: devices_){
            if(device.isCompatible()){
                compatible.push_back(device.getId());
            }
        }

        return compatible;
    }
}