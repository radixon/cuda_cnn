# CUDA Device Information
#########################

# Compiler Settings
NVCC = nvcc
CC = gcc
NVCC_FLAGS = -O2 -std=c++11
CC_FLAGS = -O2 -std=c99

# GPU Architecture
GPU_ARCH = -gencode=arch=compute_75,code=sm_75

# Directories
SRC_DIR = src
C_DIR = $(SRC_DIR)/c
CPP_DIR = $(SRC_DIR)/cpp
BUILD_DIR = build

# Source Files
C_SOURCES = $(C_DIR)/main.cu $(C_DIR)/device_info.cu
CPP_SOURCES = $(CPP_DIR)/main.cu $(CPP_DIR)/device_info.cu

# Object Files
C_OBJECTS = $(BUILD_DIR)/c_main.o $(BUILD_DIR)/c_device_info.o
CPP_OBJECTS = $(BUILD_DIR)/cpp_main.o $(BUILD_DIR)/cpp_device_info.o

# Target Executables
TARGET_C = device_info_c
TARGET_CPP = device_info_cpp

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
	@echo "$(BLUE)Run C version: ./$(TARGET_C)$(NC)"
	@echo "$(BLUE)Run C++ version: ./$(TARGET_CPP)$(NC)"

# Create Build Directory
$(BUILD_DIR):
	@echo "$(YELLOW) Creating build Directory$(NC)"
	@mkdir -p $(BUILD_DIR)

###############################################################################
#	C (CUDA) Build Rules
###############################################################################
# Build Executable
$(TARGET_C): $(C_OBJECTS)
	@echo "$(BLUE)Linking C version...$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $^ -o $@

# Compile Main
$(BUILD_DIR)/c_main.o: $(C_DIR)/main.cu
	@echo "$(YELLOW)Compiling C main...$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -I$(C_DIR) -c $< -o $@

# Compile Device Info
$(BUILD_DIR)/c_device_info.o: $(C_DIR)/device_info.cu
	@echo "$(YELLOW)Compiling C device info...$(NC)"
	$(NVCC) $(NVCC_FLAGS) -I$(C_DIR) -c $< -o $@


###############################################################################
#	C++ (CUDA) Build Rules
###############################################################################
# Build Executable
$(TARGET_CPP): $(CPP_OBJECTS)
	@echo "$(BLUE)Linking C++ version...$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) $^ -o $@

# Compile Main
$(BUILD_DIR)/cpp_main.o: $(CPP_DIR)/main.cu
	@echo "$(YELLOW)Compiling C++ main...$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -I$(C_DIR) -c $< -o $@

# Compile Device Info
$(BUILD_DIR)/cpp_device_info.o: $(CPP_DIR)/device_info.cu
	@echo "$(YELLOW)Compiling C++ device info...$(NC)"
	$(NVCC) $(NVCC_FLAGS) -I$(CPP_DIR) -c $< -o $@

###############################################################################
#	Individual Build Targets
###############################################################################
.PHONY: c cpp

c: $(BUILD_DIR) $(TARGET_C)
	@echo "$(GREEN)C version ready: ./$(TARGET_C)$(NC)"

cpp: $(BUILD_DIR) $(TARGET_CPP)
	@echo "$(GREEN)C++ version ready: ./$(TARGET_CPP)$(NC)"

###############################################################################
#	Tests
###############################################################################
.PHONY: test test-c test-cpp
test: test-c test-cpp
	@echo "$(GREEN) All tests passed$(NC)"

test-c: $(TARGET_C)
	@echo "$(BLUE)Testing C version...$(NC)"
	@timeout 30s ./$(TARGET_C) > /dev/null 2>&1 && \
		echo "$(GREEN) C test passed$(NC)" || \
		echo "$(RED) C test failed$(NC)"

test-cpp: $(TARGET_CPP)
	@echo "$(BLUE)Testing C++ version...$(NC)"
	@timeout 30s ./$(TARGET_CPP) > /dev/null 2>&1 && \
		echo "$(GREEN) C++ test passed$(NC)" || \
		echo "$(RED) C++ test failed$(NC)"

###############################################################################
#	Clean
###############################################################################
.PHONY: clean	clean-all
clean:
	@echo "$(YELLOW)Cleaning build files...$(NC)"
	@rm -rf $(BUILD_DIR)
	@rm -f $(TARGET_C) $(TARGET_CPP)
	@echo "$(GREEN) Clean Complete$(NC)"

clean-all:	clean
	@echo "$(YELLOW)Cleaning all temporary files...$(NC)"
	@rm -rf $(BUILD_DIR)
	@rm -f $(TARGET_C) $(TARGET_CPP)
	@rm -f *.tmp *.log
	@echo "$(GREEN) Clean Complete$(NC)"

# Check CUDA Installation
.PHONY:	check-cuda
check-cuda:
	@echo "$(BLUE)Checking CUDA installation...$(NC)"
	@which nvcc > /dev/null || (echo "$(RED) nvcc not found$(NC)" && exit 1)
	@echo "$(GREEN) nvcc found: $$(which nvcc)$(NC)"
	@echo "$(BLUE)CUDA version:$(NC)"
	@nvcc --version | grep "release"
	@echo "$(BLUE)GPU status:$(NC)"
	@nvidia-smi -L 2>/dev/null || echo "$(YELLOW)  nvidia-smi not available$(NC)"

# Setup project (create directories, check dependencies)
.PHONY: setup
setup:
	@echo "$(BLUE)Setting up project...$(NC)"
	@mkdir -p $(SRC_DIR) $(C_DIR) $(BUILD_DIR)
	@echo "$(GREEN) Directories created$(NC)"
	@make check-cuda

# Detect GPU architecture automatically (advanced)
.PHONY: detect-gpu
detect-gpu:
	@echo "$(BLUE)Detecting GPU architecture...$(NC)"
	@nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | \
		head -n1 | awk -F. '{printf "Detected compute capability: %s.%s\n", $$1, $$2}' || \
		echo "$(YELLOW)  Could not detect GPU architecture$(NC)"

# Show disk usage
.PHONY: size
size: all
	@echo "$(BLUE)Build size information:$(NC)"
	@ls -lh $(TARGET_C) 2>/dev/null || echo "Executables not found"
	@du -sh $(BUILD_DIR) 2>/dev/null || echo "Build directory not found"

# Show make version and capabilities
.PHONY: make-info
make-info:
	@echo "$(BLUE)Make information:$(NC)"
	@make --version | head -n1
	@echo "Available features: $(MAKE_VERSION)"

# Memory usage during compilation
.PHONY: memory-usage
memory-usage:
	@echo "$(BLUE)Monitoring memory usage during build...$(NC)"
	@/usr/bin/time -v $(MAKE) clean all 2>&1 | grep -E "(Maximum resident|User time|System time)"

# Performance benchmark
.PHONY: benchmark
benchmark: all
	@echo "$(BLUE)Running performance benchmark...$(NC)"
	@echo "C version timing:"
	@time ./$(TARGET_C) > /dev/null 2>&1 || true

# Security check (basic)
.PHONY: security-check
security-check:
	@echo "$(BLUE)Basic security checks...$(NC)"
	@find $(SRC_DIR) -name "*.c" -o -name "*.cpp" -o -name "*.cu" | \
		xargs grep -n "strcpy\|strcat\|sprintf\|gets" || \
		echo "$(GREEN) No unsafe functions found$(NC)"
	
# Code statistics
.PHONY: stats
stats:
	@echo "$(BLUE)Code statistics:$(NC)"
	@find $(SRC_DIR) -name "*.h" -o -name "*.c" -o -name "*.cpp" -o -name "*.cu" | \
		xargs wc -l | tail -n1
	@echo "File breakdown:"
	@find $(SRC_DIR) -name "*.h" -o -name "*.c" -o -name "*.cpp" -o -name "*.cu" | \
		xargs wc -l | head -n -1

# Dependency check
.PHONY: deps
deps:
	@echo "$(BLUE)Checking dependencies...$(NC)"
	@echo "Make version: $$(make --version | head -n1)"
	@echo "NVCC version: $$(nvcc --version | grep release || echo 'Not found')"
	@echo "GCC version: $$(gcc --version | head -n1 || echo 'Not found')"

# Show GPU memory usage (if nvidia-smi available)
.PHONY: gpu-status
gpu-status:
	@echo "$(BLUE)GPU Status:$(NC)"
	@nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv 2>/dev/null || \
		echo "$(YELLOW)  nvidia-smi not available$(NC)"

# Show help
help:
	@echo "$(GREEN) Available make targets:$(NC)"
	@echo "		all	-	Build everything (default)"
	@echo "		c	-	Build the C version"
	@echo "		test	-	Run all tests"
	@echo "		test-c	-	Run C version test"
	@echo "		size	-	Show build size information"
	@echo "		setup	-	Setup project directories & check CUDA"
	@echo "		stats	-	Show code statistics"
	@echo "		deps	-	Check compiler/dependency versions"
	@echo "		clean	-	Remove build files"
	@echo "		help	-	Show this help"
	@echo " "
	@echo "		clean-all	-	Remove build + temporary files"
	@echo "		check-cuda	-	Check CUDA installation"
	@echo "		detect-gpu	-	Detect GPU architecture"
	@echo "		make-info	-	Show make version and capabilities"
	@echo "		benchmark	-	Run performance benchmark"
	@echo "		gpu-status	-	Show GPU memory and utilization"
	@echo "		memory-usage	-	Monitor memory usage during build"
	@echo "		security-check	-	Scan for unsafe functions"

.PHONY: all test clean help