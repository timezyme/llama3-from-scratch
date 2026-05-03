# This is your make file
# You may change it and we use it to build your code.
# DO NOT CHANGE RECIPE FOR TEST RELATED TARGETS
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -MMD -MP
LDFLAGS :=
NVCC := nvcc
ARCH ?= sm_75
NVCCFLAGS ?= -std=c++17 -O2
NVCCFLAGS += -arch=$(ARCH)
BUILD ?= release

ifeq ($(BUILD),debug)
  CXXFLAGS += -g -O0
else
  CXXFLAGS += -O2
endif

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
BIN_DIR := bin
TARGET := llm
SOURCES := main.cpp $(SRC_DIR)/tokenizer_bpe.cpp $(SRC_DIR)/loader.cpp
OBJECTS := $(BUILD_DIR)/main.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/loader.o
DEPS := $(OBJECTS:.o=.d)
INCLUDES := -I$(INC_DIR) -I.

ifneq ($(shell command -v $(NVCC) 2>/dev/null),)
  CUDA_ENABLED := 1
else
  CUDA_ENABLED := 0
endif

# Shared CUDA kernel and inference objects (used by both main binary and tests_m2m3)
CUDA_KERNEL_OBJECTS :=
MAIN_CUDA_OBJECTS :=

ifeq ($(CUDA_ENABLED),1)
  CUDA_PATH ?= /usr/local/cuda
  LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
  CXXFLAGS += -DCUDA_ENABLED
  NVCCFLAGS += -DCUDA_ENABLED
  CUDA_KERNEL_OBJECTS := $(BUILD_DIR)/matmul.o \
                         $(BUILD_DIR)/rmsnorm.o \
                         $(BUILD_DIR)/rope.o \
                         $(BUILD_DIR)/attention.o \
                         $(BUILD_DIR)/swiglu.o \
                         $(BUILD_DIR)/residual.o
  MAIN_CUDA_OBJECTS := $(BUILD_DIR)/model_weights.o \
                       $(BUILD_DIR)/device_weights.o \
                       $(BUILD_DIR)/inference.o \
                       $(BUILD_DIR)/kv_cache.o \
                       $(CUDA_KERNEL_OBJECTS)
  OBJECTS += $(MAIN_CUDA_OBJECTS)
else
  MATMUL_OBJECT := $(BUILD_DIR)/matmul_cpu.o
  OBJECTS += $(MATMUL_OBJECT)
endif

DEPS := $(OBJECTS:.o=.d)

all: $(BIN_DIR)/$(TARGET)
$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
$(BUILD_DIR)/main.o: main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/tokenizer_bpe.o: $(SRC_DIR)/tokenizer_bpe.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/loader.o: $(SRC_DIR)/loader.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/matmul_cpu.o: kernel/matmul_cpu.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/matmul.o: kernel/matmul.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/model_weights.o: $(SRC_DIR)/model_weights.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/device_weights.o: $(SRC_DIR)/device_weights.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/inference.o: $(SRC_DIR)/inference.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/kv_cache.o: $(SRC_DIR)/kv_cache.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	rm -f a.out

.PHONY: run

run: all
	./$(BIN_DIR)/$(TARGET)
-include $(DEPS)

# ------------------------------------------------------------
# Tests build
# Don't forget to add your required objects as well.

.PHONY: tests

ifeq ($(CUDA_ENABLED),1)
  TEST_MATMUL := $(BUILD_DIR)/matmul.o
else
  TEST_MATMUL := $(BUILD_DIR)/matmul_cpu.o
endif

TEST_OBJECTS := $(BUILD_DIR)/test.o $(BUILD_DIR)/test_api.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/loader.o $(TEST_MATMUL)

tests: $(BIN_DIR)/tests

$(BIN_DIR)/tests: $(TEST_OBJECTS) | $(BIN_DIR)
	$(CXX) $(TEST_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test.o: tests/test.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test_api.o: tests/test_api.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ------------------------------------------------------------
# Milestone 2-3 internal tests (CUDA required)

.PHONY: tests_m2m3

ifeq ($(CUDA_ENABLED),1)

M2M3_KERNEL_OBJECTS := $(CUDA_KERNEL_OBJECTS)

M2M3_TEST_OBJECTS := $(BUILD_DIR)/test_m2m3.o \
                     $(BUILD_DIR)/model_weights.o \
                     $(BUILD_DIR)/device_weights.o \
                     $(BUILD_DIR)/inference.o \
                     $(BUILD_DIR)/kv_cache.o \
                     $(BUILD_DIR)/tokenizer_bpe.o \
                     $(BUILD_DIR)/loader.o \
                     $(M2M3_KERNEL_OBJECTS)

tests_m2m3: $(BIN_DIR)/tests_m2m3

$(BIN_DIR)/tests_m2m3: $(M2M3_TEST_OBJECTS) | $(BIN_DIR)
	$(CXX) $(M2M3_TEST_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_m2m3.o: tests/test_m2m3.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rmsnorm.o: kernel/rmsnorm.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/rope.o: kernel/rope.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/attention.o: kernel/attention.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/swiglu.o: kernel/swiglu.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/residual.o: kernel/residual.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

else

tests_m2m3:
	@echo "ERROR: tests_m2m3 requires CUDA (nvcc not found)"
	@exit 1

endif
