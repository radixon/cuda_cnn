#ifndef DEVICE_INFO_CPP_H
#define DEVICE_INFO_CPP_H

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace DeviceInfo{
    class CudaDevice{
        private:
            int deviceId_;
            cudaDeviceProp properties_;
        
        public:
            explicit CudaDevice(int deviceId);

            // Getters
            int getId() const {return deviceId_; }
            const cudaDeviceProp& getProperties() const { return properties_; }
            std::string getName() const { return std::string(properties_.name); }

            // Device capabilities
            bool isCompatible() const;
            double getGlobalMemoryGB() const;
            int getComputeCapabilityMajor() const { return properties_.major; }
            int getComputeCapabilityMinor() const { return properties_.minor; }

            // Display methods
            void printBasicInfo() const;
            void printMemoryInfo() const;
            void printComputeInfo() const;
            void printThreadInfo() const;
            void printDetailedInfo() const;
    };

    class DeviceManager{
        private:
            std::vector<CudaDevice> devices_;
        
        public:
            DeviceManager();

            int getDeviceCount() const { return devices_.size(); }
            const CudaDevice& getDevice(int index) const { return devices_.at(index); }

            void printAllDevices() const;
            void printSystemInfo() const;
            std::vector<int> getCompatibleDevices() const;
    };

    // Utility functions
    void checkCudaError(cudaError_t error, const std::string& operation);
    std::string formatBytes(size_t bytes);
    std::string formatFrequency(int clockRateKHz);
    void printCudaVersion();
}

#endif