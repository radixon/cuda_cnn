# Matrix Addition
#########################

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
BUILD_DIR = build

# Source Files
C_SOURCES = $(C_DIR)/main.cu
C_HELPER_SOURCE = $(C_DIR)/helpfunctions.cu
C_HELPER_HEADER = $(C_DIR)/helpfunctions.h

CPP_SOURCES = $(CPP_DIR)/main.cu
CPP_HELPER_SOURCE = $(CPP_DIR)/helpfunctions.cu 
CPP_HEADERS = $(CPP_DIR)/helpfunctions.hpp 

# Object Files
C_OBJECTS = $(BUILD_DIR)/c_main.o \
			$(BUILD_DIR)/c_helper.o 

CPP_OBJECTS = $(BUILD_DIR)/cpp_main.o \
              $(BUILD_DIR)/cpp_helpfunctions.o 

# Target Executables
TARGET_C = matrix_add_c
TARGET_CPP = matrix_add_cpp

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
NC = \033[0m

# Default Target 
.PHONY: all
all: $(BUILD_DIR) $(TARGET_C) $(TARGET_CPP)
	@echo "$(GREEN) Build Complete!$(NC)"
	@echo "$(BLUE)Run C version --> ./$(TARGET_C)$(NC)"
	@echo "$(BLUE)Run C++ version --> ./$(TARGET_CPP)$(NC)"

# Create Build Directory
$(BUILD_DIR):
	@echo "$(YELLOW) Creating build Directory$(NC)"
	@mkdir -p $(BUILD_DIR)

###############################################################################
#	C (CUDA) Build Rules
###############################################################################
# Build Executable
$(TARGET_C): $(C_OBJECTS)
	@echo "$(BLUE)Linking C Version$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Compile Main
$(BUILD_DIR)/c_main.o: $(C_SOURCES) $(C_HELPER_HEADER)
	@echo "$(YELLOW)Compiling C main$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $(INCLUDES) -I$(SRC_DIR) -c $< -o $@

# Compile C Helper Functions
$(BUILD_DIR)/c_helper.o: $(C_HELPER_SOURCE) $(C_HELPER_HEADER)
	@echo "$(YELLOW)Compiling C helper functions$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $(INCLUDES) -I$(SRC_DIR) -c $< -o $@

###############################################################################
#	C++ (CUDA) Build Rules
###############################################################################
# Build Executable
$(TARGET_CPP): $(CPP_OBJECTS)
	@echo "$(BLUE)Linking C++ version$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Compile Main
$(BUILD_DIR)/cpp_main.o: $(CPP_DIR)/main.cu $(CPP_HEADERS)
	@echo "$(YELLOW)Compiling C++ main$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -I$(SRC_DIR) -c $< -o $@

# Compile C++ Help Functions
$(BUILD_DIR)/cpp_helpfunctions.o: $(CPP_DIR)/helpfunctions.cu $(CPP_DIR)/helpfunctions.hpp
	@echo "$(YELLOW)Compiling C++ helper functions$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $(INCLUDES) -c $< -o $@

###############################################################################
#	Build Targets
###############################################################################
.PHONY: c cpp
c: $(BUILD_DIR) $(TARGET_C)
	@echo "$(GREEN)C Build Complete --> ./$(TARGET_C)$(NC)"

cpp: $(BUILD_DIR) $(TARGET_CPP)
	@echo "$(GREEN)C++ Build Complete --> ./$(TARGET_CPP)$(NC)"

###############################################################################
#	Debug Build 
###############################################################################
.PHONY: debug debug-c debug-cpp
debug: debug-c debug-cpp
	@echo "$(GREEN)Debug builds complete!$(NC)"

debug-c: NVCC_FLAGS += -g -G -DDEBUG
debug-c: TARGET_C := $(TARGET_C)_debug
debug-c: $(BUILD_DIR) $(TARGET_C)
	@echo "$(GREEN)C debug build complete: ./$(TARGET_C)$(NC)"

debug-cpp: NVCC_FLAGS += -g -G -DDEBUG
debug-cpp: TARGET_CPP := $(TARGET_CPP)_debug
debug-cpp: $(BUILD_DIR) $(TARGET_CPP)
	@echo "$(GREEN)C++ debug build complete: ./$(TARGET_CPP)$(NC)"

###############################################################################
#	Run Program
###############################################################################
.PHONY: run run-c run-cpp
run: run-c run-cpp

run-c: $(TARGET_C)
	@echo "$(BLUE)Running C Matrix Addition$(NC)"
	@./$(TARGET_C)

run-cpp: $(TARGET_CPP)
	@echo "$(BLUE)Running C++ Matrix Addition$(NC)"
	@./$(TARGET_CPP)

###############################################################################
#	Clean
###############################################################################
.PHONY: clean clean-all
clean:
	@echo "$(YELLOW)Cleaning build files$(NC)"
	@rm -rf $(BUILD_DIR) $(TARGET_C) $(TARGET_CPP)
	@rm -f $(TARGET_C)_debug $(TARGET_CPP)_debug
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

benchmark-c: $(TARGET_C)
	@echo "$(BLUE)Running C benchmark$(NC)"
	@time ./$(TARGET_C)

benchmark-cpp: $(TARGET_CPP)
	@echo "$(BLUE)Running C++ benchmark$(NC)"
	@time ./$(TARGET_CPP)

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
	@echo "  C_HELPER: $(C_HELPER_HEADER)"
	@echo "  CPP_HELPER: $(CPP_HEADERS)"

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