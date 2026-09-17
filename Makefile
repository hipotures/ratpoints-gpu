NVCC ?= nvcc
PYTHON ?= python3
INSTALL ?= install
CXX ?= g++
HOST_TEST_FLAGS ?= -O2 -std=c++14 -Wall -Wextra -Werror
THREAD_FLAGS := -Xcompiler=-pthread
DEVICES ?= 0,1
HEIGHT ?= 1000000
REPEATS ?= 3

CPPFLAGS ?=
NVCCFLAGS ?= -O3 -std=c++14
WARNING_FLAGS ?= -Xcompiler=-Wall,-Wextra
ARCH_FLAGS ?= -arch=native
LDFLAGS ?=
LDLIBS ?= -lgmpxx -lgmp

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
DESTDIR ?=

TARGET := ratpoints_gpu
BUILD_DIR := build
SIEVE_TEST := $(BUILD_DIR)/test_sieve_streaming
CPP_SOURCES := src/main.cpp src/command_line.cpp src/exact_polynomial.cpp \
	src/output_formatter.cpp src/point_search.cpp src/sieve_plan.cpp src/gpu_devices.cpp
CUDA_SOURCES := src/sieve.cu
CPP_OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
CUDA_OBJECTS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))
OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)
HEADERS := $(wildcard include/*.hpp)
INTERNAL_HEADERS := $(wildcard src/*.hpp)
HOST_TEST := $(BUILD_DIR)/mock_ratpoints_gpu
HOST_PLAN_TEST := $(BUILD_DIR)/test_sieve_plan
HOST_SOURCES := $(filter-out src/sieve_plan.cpp,$(CPP_SOURCES)) tests/mock_backend.cpp

.PHONY: all clean install test test-host test-multi-gpu benchmark record-check print-config

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(WARNING_FLAGS) $(THREAD_FLAGS) $(ARCH_FLAGS) \
		$(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

$(BUILD_DIR)/%.o: src/%.cpp $(HEADERS) $(INTERNAL_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) -Iinclude $(NVCCFLAGS) $(WARNING_FLAGS) $(THREAD_FLAGS) $(ARCH_FLAGS) \
		-c -o $@ $<

$(BUILD_DIR)/%.o: src/%.cu $(HEADERS) $(INTERNAL_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) -Iinclude $(NVCCFLAGS) $(WARNING_FLAGS) $(THREAD_FLAGS) $(ARCH_FLAGS) \
		-c -o $@ $<

$(SIEVE_TEST): tests/test_sieve_streaming.cpp $(HEADERS) $(INTERNAL_HEADERS) \
		$(BUILD_DIR)/sieve.o $(BUILD_DIR)/sieve_plan.o
	$(NVCC) $(CPPFLAGS) -Iinclude $(NVCCFLAGS) $(WARNING_FLAGS) $(THREAD_FLAGS) $(ARCH_FLAGS) \
		-o $@ tests/test_sieve_streaming.cpp $(BUILD_DIR)/sieve.o \
		$(BUILD_DIR)/sieve_plan.o $(LDLIBS)

$(BUILD_DIR):
	mkdir -p $@

install: $(TARGET)
	$(INSTALL) -d "$(DESTDIR)$(BINDIR)"
	$(INSTALL) -m 755 $(TARGET) "$(DESTDIR)$(BINDIR)/$(TARGET)"

test: $(TARGET) $(SIEVE_TEST)
	command -v ratpoints >/dev/null
	$(SIEVE_TEST)
	$(PYTHON) tests/compare_with_ratpoints.py ./$(TARGET)

$(HOST_TEST): $(HOST_SOURCES) $(HEADERS) tests/mock_cuda/cuda_runtime_api.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(HOST_TEST_FLAGS) -pthread -Itests/mock_cuda -Iinclude \
		$(HOST_SOURCES) $(LDFLAGS) -o $@ $(LDLIBS)

$(HOST_PLAN_TEST): tests/test_sieve_plan.cpp src/sieve_plan.cpp $(HEADERS) $(INTERNAL_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(HOST_TEST_FLAGS) -Iinclude -Isrc \
		tests/test_sieve_plan.cpp src/sieve_plan.cpp $(LDFLAGS) -o $@

test-host: $(HOST_TEST) $(HOST_PLAN_TEST)
	$(HOST_PLAN_TEST)
	$(PYTHON) tests/test_multi_gpu.py ./$(HOST_TEST) --devices 0,1 --mock

test-multi-gpu: $(TARGET)
	$(PYTHON) tests/test_multi_gpu.py ./$(TARGET) --devices "$(DEVICES)"

benchmark: $(TARGET)
	$(PYTHON) tests/benchmark_multi_gpu.py --binary ./$(TARGET) \
		--devices "$(DEVICES)" --height "$(HEIGHT)" --repeats "$(REPEATS)"

record-check: $(TARGET)
	$(PYTHON) tests/verify_record_curve.py ./$(TARGET)

print-config:
	@printf '%s\n' \
		'NVCC=$(NVCC)' \
		'NVCCFLAGS=$(NVCCFLAGS)' \
		'ARCH_FLAGS=$(ARCH_FLAGS)' \
		'THREAD_FLAGS=$(THREAD_FLAGS)' \
		'PREFIX=$(PREFIX)' \
		'BINDIR=$(BINDIR)'

clean:
	$(RM) $(TARGET) $(OBJECTS) $(SIEVE_TEST) $(HOST_TEST) $(HOST_PLAN_TEST)
	-rmdir $(BUILD_DIR)
