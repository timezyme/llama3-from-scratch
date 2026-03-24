# This is your make file
# You may change it and we use it to build your code.
# DO NOT CHANGE RECIPE FOR TEST RELATED TARGETS 
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -MMD -MP
LDFLAGS :=
NVCC := nvcc
NVCCFLAGS := -std=c++17 -O2
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

ifeq ($(CUDA_ENABLED),1)
  MATMUL_OBJECT := $(BUILD_DIR)/matmul.o
  CUDA_PATH ?= /usr/local/cuda
  LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
else
  MATMUL_OBJECT := $(BUILD_DIR)/matmul_cpu.o
endif

OBJECTS += $(MATMUL_OBJECT)
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

TEST_OBJECTS := $(BUILD_DIR)/test.o $(BUILD_DIR)/test_api.o $(BUILD_DIR)/tokenizer_bpe.o $(BUILD_DIR)/loader.o $(MATMUL_OBJECT)

tests: $(BIN_DIR)/tests

$(BIN_DIR)/tests: $(TEST_OBJECTS) | $(BIN_DIR)
	$(CXX) $(TEST_OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/test.o: tests/test.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/test_api.o: tests/test_api.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
