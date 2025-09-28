# CUDA Matrix Addition with RAII Memory Management

A high-performance CUDA CNN implementation.  Currently the repo is of matrix addition demonstrating efficient GPU programming techniques with both traditional C and modern C++ RAII (Resource Acquisition Is Initialization) memory management approaches.  

## Features

* GPU-Accelerated Matrix Addition : Parallel computation using CUDA kernels
* Dual Implementation : Both C and C++ versions available
* RAII Memory Management : Automatic memory cleanup with C++ destructors (C++ version)
* Traditional C Implementation : Manual memory management (C Version)
* Performance Benchmarking : CPU vs GPU timing comparisons
* Error Checking : Comprehensive CUDA error handling and result validation
* Memory Safety : Prevents memory leaks with smart wrapper classes (C++ version)
* Cross-Platform : Compatible with modern CUDA-enabled GPUs

## Requirements

* CUDA Toolkit : Version 10.0 or higher
* GPU : CUDA-capable device (Compute Capability 3.5+)
* Compiler : nvcc with C++11 support
* OS : Linux, Windows, or macOS

## Installation

### Clone The Repository

```bash
git clone https://github.com/radixon/cuda-cnn.git
cd cuda_cnn
```

### Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

## Build Project

```bash
# Both Versions
make all

# C Version
make c

# C++ Version
make cpp

# Debug Versions
make debug      # Both
make debug-c    # C Debug
make debug-cpp  # C++ Debug
```

## Project Structure

```
cuda_cnn/
├── cuda_cnn/
│   ├── src/
│   │   ├── c/                     # Traditional C implementation
│   │   │   ├── main.cu            # Main program with manual memory management
│   │   │   ├── helpfunctions.h    # Traditional C header
│   │   │   └── helpfunctions.cu   # Traditional C implementation
│   │   └── cpp/                   # C++ implementation with RAII
│   │       ├── main.cu            # Main program with RAII wrappers
│   │       ├── helpfunctions.hpp  # Header with RAII declarations
│   │       └── helpfunctions.cu   # Implementation with RAII functions
├── .gitignore                     # 
└── LICENSE                        # Copyright Information
├── README.md                      # Project Description
└── Makefile                       # Build configuration
```

## Usage

```bash
make run    # Direct Execution Both Versions
make run-c  # Direct Execution C Version
make run-cpp    # Direct Execution C++ Version

make benchmark  # Benchmark Both Versions
mkae benchmark-c    # Benchmark C Version
make benchmark-cpp  # Benchmark C++ Version
```

## Implementation Differences

### C 

```C
// Manual memory allocation
float *h_A, *h_B, *hostRef, *gpuRef;
h_A = (float *)malloc(nBytes);
float *d_MatA, *d_MatB, *d_MatC;
cudaMalloc((void **)&d_MatA, nBytes);

// Manual memory cleanup required
free(h_A);
cudaFree(d_MatA);
```

### C++

```C++
// Automatic memory management with RAII wrappers
HostMemory<float> h_A(nxy);
DeviceMemory<float> d_MatA(nxy);

// Memory automatically freed when objects go out of scope
```