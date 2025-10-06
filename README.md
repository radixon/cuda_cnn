# CUDA Matrix Operations with RAII Memory Management

A high-performance CUDA implementation demonstrating efficient GPU programming techniques with both traditional C and modern C++ RAII (Resource Acquisition Is Initialization) memory management approaches. The project currently supports **matrix addition** and **Sobel vertical edge detection**, providing a foundation for CNN and image processing operations.

## Features

### Core Operations
* **GPU-Accelerated Matrix Addition**: Parallel computation using optimized CUDA kernels
* **Sobel Vertical Edge Detection**: GPU-accelerated image processing with convolution operations
* **Dual Implementation**: Both C and C++ versions available for each operation
* **Performance Benchmarking**: CPU vs GPU timing comparisons with detailed metrics

### Memory Management
* **RAII Memory Management**: Automatic memory cleanup with C++ destructors (C++ version)
* **Traditional C Implementation**: Manual memory management with explicit error checking (C version)
* **Memory Safety**: Prevents memory leaks with smart wrapper classes (C++ version)
* **Comprehensive Error Handling**: CUDA error checking and result validation

### Platform Support
* **Cross-Platform**: Compatible with modern CUDA-enabled GPUs
* **Flexible Build System**: Makefile with multiple build targets and configurations

## Requirements

* **CUDA Toolkit**: Version 10.0 or higher
* **GPU**: CUDA-capable device (Compute Capability 3.5+)
* **Compiler**: nvcc with C++14 support
* **OS**: Linux, Windows, or macOS

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

### Check GPU Compatibility

```bash
make check-cuda
make gpu-info
```

### Build Project

```bash
# Build Versions
make     # Both Versions
make c   # C Version
make cpp # C++ Version

# Debug Versions
make debug      # Both
make debug-c    # C Debug
make debug-cpp  # C++ Debug
```

## Project Structure

```
cuda_cnn/
├── src/
│   ├── c/                         # Traditional C implementations
│   │   ├── main.cu               # Matrix addition main (C)
│   │   └── sobel_main.cu         # Sobel edge detection main (C)
│   ├── cpp/                      # C++ implementations with RAII
│   │   ├── main.cu               # Matrix addition main (C++)
│   │   └── sobel_main.cu         # Sobel edge detection main (C++)
│   ├── utility/                  # Common utility functions
│   │   ├── utility.h             # Utility header declarations
│   │   └── utility.cu            # Timer, initialization, validation functions
│   ├── addition/                 # Matrix addition operations
│   │   ├── addition.h            # Matrix addition header
│   │   └── addition.cu           # GPU kernels for matrix addition
│   ├── sobel/                    # Sobel edge detection operations
│   │   ├── sobel.h               # Sobel operation header
│   │   └── sobel.cu              # GPU kernels for Sobel edge detection
│   └── memory_management/        # C++ RAII memory management
│       ├── memory_management.hpp # RAII wrapper class declarations
│       └── memory_management.cu  # RAII implementation
├── build/                        # Compiled object files
├── Makefile                      # Build configuration
├── README.md                     # Project documentation
├── LICENSE                       # Copyright information
└── .gitignore                    # Git ignore rules
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

### Benefits of RAII Approach

* **Exception Safety:** Automatic cleanup even during exceptions
* **Memory Leak Prevention:** Guaranteed resource deallocation
* **Clean Code:** No manual memory management calls
* **CUDA Error Handling:** Integrated error checking in wrapper classes