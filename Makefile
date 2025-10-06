# Matrix Makefile
##########################

# Compiler Settings
NVCC = nvcc
NVCC_FLAGS = -O2 -std=c++14 -Xcompiler -fPIC
CXX = g++
CXX_FLAGS = -O2 -std=c++14 -Wall -Wextra -fPIC

# GPU Architecture
GPU_ARCH = -gencode arch=compute_86,code=sm_86

# CUDA paths
CUDA_PATH ?= /usr/local/cuda
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64
LDLIBS = -lcudart

# Directories
SRC_DIR = src
C_DIR = $(SRC_DIR)/c
CPP_DIR = $(SRC_DIR)/cpp
UTILITY_DIR = $(SRC_DIR)/utility
ADDITION_DIR = $(SRC_DIR)/addition
SOBEL_DIR = $(SRC_DIR)/sobel
MEMORY_MANAGEMENT_DIR = $(SRC_DIR)/memory_management
BUILD_DIR = build

# Source Files
C_SOURCE = $(C_DIR)/main.cu
CPP_SOURCE = $(CPP_DIR)/main.cu
UTILITY_SOURCE = $(UTILITY_DIR)/utility.cu
UTILITY_HEADER = $(UTILITY_DIR)/utility.h
ADDITION_SOURCE = $(ADDITION_DIR)/addition.cu
ADDITION_HEADER = $(ADDITION_DIR)/addition.h
SOBEL_SOURCE = $(SOBEL_DIR)/sobel.cu
SOBEL_HEADER = $(SOBEL_DIR)/sobel.h
MEMORY_MANAGEMENT_SOURCE = $(MEMORY_MANAGEMENT_DIR)/memory_management.cu 
MEMORY_MANAGEMENT_HEADER = $(MEMORY_MANAGEMENT_DIR)/memory_management.hpp 

# Object Files
C_OBJECT = $(BUILD_DIR)/c_main.o 
CPP_OBJECT = $(BUILD_DIR)/cpp_main.o 
UTILITY_OBJECT = $(BUILD_DIR)/utility.o
ADDITION_OBJECT = $(BUILD_DIR)/addition.o
SOBEL_OBJECT = $(BUILD_DIR)/sobel.o
MEMORY_MANAGEMENT_OBJECT = $(BUILD_DIR)/memory_management.o

# Target Executables
TARGET_C_ADDITION = matrix_c
TARGET_CPP_ADDITION = matrix_cpp

# Target Arguments
ARGS ?= addition sobel

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
NC = \033[0m

# Default Target 
.PHONY: all
all: $(BUILD_DIR) $(TARGET_C_ADDITION) $(TARGET_CPP_ADDITION)
	@echo "$(GREEN) Build Complete!$(NC)"
	@echo "$(BLUE)Run C version --> ./$(TARGET_C_ADDITION)$(NC)"
	@echo "$(BLUE)Run C++ version --> ./$(TARGET_CPP_ADDITION)$(NC)"

# Create Build Directory
$(BUILD_DIR):
	@echo "$(YELLOW) Creating build Directory$(NC)"
	@mkdir -p $(BUILD_DIR)

###############################################################################
#	Header Build Rules
###############################################################################
# Compile Utillity Functions
$(BUILD_DIR)/utility.o: $(UTILITY_SOURCE) $(UTILITY_HEADER)
	@echo "$(YELLOW)Compiling utility functions$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $(INCLUDES) -I$(UTILITY_DIR) -c $< -o $@

# Compile Addition Functions
$(BUILD_DIR)/addition.o: $(ADDITION_SOURCE) $(ADDITION_HEADER)
	@echo "$(YELLOW)Compiling addition functions$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $(INCLUDES) -I$(ADDITION_DIR) -c $< -o $@

# Compile Sobel Functions
$(BUILD_DIR)/sobel.o: $(SOBEL_SOURCE) $(SOBEL_HEADER)
	@echo "$(YELLOW)Compiling sobel functions$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $(INCLUDES) -I$(SOBEL_DIR) -c $< -o $@

###############################################################################
#	C (CUDA) Matrix Addition Build Rules
###############################################################################
# Build Executable
$(TARGET_C_ADDITION): $(C_OBJECT) $(UTILITY_OBJECT) $(ADDITION_OBJECT) $(SOBEL_OBJECT)
	@echo "$(BLUE)Linking C Version$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Compile Main
$(BUILD_DIR)/c_main.o: $(C_SOURCE) $(UTILITY_HEADER) $(ADDITION_HEADER) $(SOBEL_HEADER)
	@echo "$(YELLOW)Compiling C main$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $(INCLUDES) -I$(SRC_DIR) -I$(UTILITY_DIR) -I$(ADDITION_DIR) -I$(SOBEL_DIR) -c $< -o $@

###############################################################################
#	C++ (CUDA) Matrix Addition Build Rules
###############################################################################
# Build Executable
$(TARGET_CPP_ADDITION): $(CPP_OBJECT) $(UTILITY_OBJECT) $(ADDITION_OBJECT) $(SOBEL_OBJECT) $(MEMORY_MANAGEMENT_OBJECT)
	@echo "$(BLUE)Linking C++ version$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Compile Main
$(BUILD_DIR)/cpp_main.o: $(CPP_SOURCE) $(UTILITY_HEADER) $(ADDITION_HEADER) $(SOBEL_HEADER) $(MEMORY_MANAGEMENT_HEADER)
	@echo "$(YELLOW)Compiling C++ main$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -I$(SRC_DIR) -I$(UTILITY_DIR) -I$(ADDITION_DIR) -I$(SOBEL_DIR) -I$(MEMORY_MANAGEMENT_DIR) -c $< -o $@

# Compile Memory Management Functions
$(BUILD_DIR)/memory_management.o: $(MEMORY_MANAGEMENT_SOURCE) $(MEMORY_MANAGEMENT_HEADER)
	@echo "$(YELLOW)Compiling memory management functions$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $(INCLUDES) -I$(MEMORY_MANAGEMENT_DIR) -I$(UTILITY_DIR) -I$(ADDITION_DIR) -c $< -o $@

###############################################################################
#	Build Targets
###############################################################################
.PHONY: c cpp
c: $(BUILD_DIR) $(TARGET_C_ADDITION)
	@echo "$(GREEN)C Build Complete --> ./$(TARGET_C_ADDITION)$(NC)"

cpp: $(BUILD_DIR) $(TARGET_CPP_ADDITION)
	@echo "$(GREEN)C++ Build Complete --> ./$(TARGET_CPP_ADDITION)$(NC)"

###############################################################################
#	Debug Build 
###############################################################################
.PHONY: debug debug-c debug-cpp
debug: debug-c debug-cpp
	@echo "$(GREEN)Debug builds complete!$(NC)"

debug-c: NVCC_FLAGS += -g -G -DDEBUG
debug-c: TARGET_C_ADDITION := $(TARGET_C_ADDITION)_debug
debug-c: $(BUILD_DIR) $(TARGET_C_ADDITION)
	@echo "$(GREEN)C debug build complete: ./$(TARGET_C_ADDITION)$(NC)"

debug-cpp: NVCC_FLAGS += -g -G -DDEBUG
debug-cpp: TARGET_CPP_ADDITION := $(TARGET_CPP_ADDITION)_debug
debug-cpp: $(BUILD_DIR) $(TARGET_CPP_ADDITION)
	@echo "$(GREEN)C++ debug build complete: ./$(TARGET_CPP_ADDITION)$(NC)"

###############################################################################
#	Run Program
###############################################################################
.PHONY: run run-c run-cpp
run: run-c run-cpp

run-c: $(TARGET_C_ADDITION_ADDITION)
	@echo "$(BLUE)Running C Matrix Operations$(NC)"
	@./$(TARGET_C_ADDITION) $(ARGS)

run-cpp: $(TARGET_CPP_ADDITION)
	@echo "$(BLUE)Running C++ Matrix Operations$(NC)"
	@./$(TARGET_CPP_ADDITION) $(ARGS)

###############################################################################
#	Clean
###############################################################################
.PHONY: clean clean-all
clean:
	@echo "$(YELLOW)Cleaning build files$(NC)"
	@rm -rf $(BUILD_DIR) $(TARGET_C_ADDITION) $(TARGET_CPP_ADDITION)
	@rm -f $(TARGET_C_ADDITION)_debug $(TARGET_CPP_ADDITION)_debug
	@echo "$(GREEN)Clean complete$(NC)"

clean-all: clean
	@echo "$(YELLOW)Cleaning all generated files$(NC)"
	@rm -f $(C_HEADER)
	@echo "$(GREEN)Deep clean complete$(NC)"

###############################################################################
#	Check CUDA Installation
###############################################################################
.PHONY: check-cuda
check-cuda:
	@echo "$(BLUE)Checking CUDA installation$(NC)"
	@echo "NVCC Version:"
	@nvcc --version 2>/dev/null || echo "$(RED) NVCC not found$(NC)"
	@echo "\nCUDA Devices:"
	@nvidia-smi -L 2>/dev/null || echo "$(RED)No CUDA devices found$(NC)"

###############################################################################
#	GPU Info
###############################################################################
.PHONY: gpu-info
gpu-info:
	@echo "$(BLUE)GPU Information:$(NC)"
	@nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

###############################################################################
#	Performance Testing
###############################################################################
.PHONY: benchmark benchmark-c benchmark-cpp
benchmark: benchmark-c benchmark-cpp

benchmark-c: $(TARGET_C_ADDITION)
	@echo "$(BLUE)Running C benchmark$(NC)"
	@time ./$(TARGET_C_ADDITION)

benchmark-cpp: $(TARGET_CPP_ADDITION)
	@echo "$(BLUE)Running C++ benchmark$(NC)"
	@time ./$(TARGET_CPP_ADDITION)

###############################################################################
#	Show Configuration
###############################################################################
.PHONY: config
config:
	@echo "$(BLUE)Build Configuration:$(NC)"
	@echo "  NVCC: $(NVCC)"
	@echo "  NVCC_FLAGS: $(NVCC_FLAGS)"
	@echo "  GPU_ARCH: $(GPU_ARCH)"
	@echo "  INCLUDES: $(INCLUDES)"
	@echo "  LDFLAGS: $(LDFLAGS)"
	@echo "  LDLIBS: $(LDLIBS)"
	@echo "  C_SOURCE: $(C_SOURCES)"
	@echo "  CPP_SOURCE: $(CPP_SOURCES)"
	@echo "  C_HELPER: $(ADDITION_HEADER)"
	@echo "  CPP_HELPER: $(MEMORY_MANAGEMENT_HEADER)"

###############################################################################
#	Help
###############################################################################
.PHONY: help
help:
	@echo "$(GREEN)CUDA Matrix Addition Makefile$(NC)"
	@echo ""
	@echo "$(YELLOW)Check:$(NC)"
	@echo "  check-files  - Check if source files exist"
	@echo ""
	@echo "$(YELLOW)Build:$(NC)"
	@echo "  all          - Build both versions (default)"
	@echo "  c            - Build C version only"
	@echo "  cpp          - Build C++ version only"
	@echo "  debug        - Build debug versions"
	@echo "  debug-c      - Build C debug version"
	@echo "  debug-cpp    - Build C++ debug version"
	@echo ""
	@echo "$(YELLOW)Run:$(NC)"
	@echo "  run          - Run both versions"
	@echo "  run-c        - Run C version"
	@echo "  run-cpp      - Run C++ version"
	@echo "  benchmark    - Benchmark both versions"
	@echo ""
	@echo "$(YELLOW)Utility:$(NC)"
	@echo "  clean        - Remove build files"
	@echo "  clean-all    - Remove all generated files"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  gpu-info     - Show GPU information"
	@echo "  config       - Show build configuration"
	@echo "  help         - Show this help"
	@echo ""
	@echo "$(YELLOW)Example workflow:$(NC)"
	@echo "  make setup   # Setup project"
	@echo "  # Copy main.cu to src/c/ and src/cpp/"
	@echo "  make run     # Build and run both versions"

.PHONY: all clean help setup debug run benchmark memcheck check-cuda gpu-info \
        config c cpp debug-c debug-cpp run-c run-cpp benchmark-c benchmark-cpp \
        memcheck-c memcheck-cpp create-helper check-files clean-all

