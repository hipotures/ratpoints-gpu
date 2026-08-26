NVCC ?= nvcc
PYTHON ?= python3
INSTALL ?= install

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
CPP_SOURCES := src/main.cpp src/command_line.cpp src/exact_polynomial.cpp \
	src/output_formatter.cpp src/point_search.cpp src/sieve_plan.cpp
CUDA_SOURCES := src/sieve.cu
CPP_OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
CUDA_OBJECTS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))
OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)
HEADERS := $(wildcard include/*.hpp)
INTERNAL_HEADERS := $(wildcard src/*.hpp)

.PHONY: all clean install test record-check print-config

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(WARNING_FLAGS) $(ARCH_FLAGS) \
		$(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

$(BUILD_DIR)/%.o: src/%.cpp $(HEADERS) $(INTERNAL_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) -Iinclude $(NVCCFLAGS) $(WARNING_FLAGS) $(ARCH_FLAGS) \
		-c -o $@ $<

$(BUILD_DIR)/%.o: src/%.cu $(HEADERS) $(INTERNAL_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) -Iinclude $(NVCCFLAGS) $(WARNING_FLAGS) $(ARCH_FLAGS) \
		-c -o $@ $<

$(BUILD_DIR):
	mkdir -p $@

install: $(TARGET)
	$(INSTALL) -d "$(DESTDIR)$(BINDIR)"
	$(INSTALL) -m 755 $(TARGET) "$(DESTDIR)$(BINDIR)/$(TARGET)"

test: $(TARGET)
	command -v ratpoints >/dev/null
	$(PYTHON) tests/compare_with_ratpoints.py ./$(TARGET)

record-check: $(TARGET)
	$(PYTHON) tests/verify_record_curve.py ./$(TARGET)

print-config:
	@printf '%s\n' \
		'NVCC=$(NVCC)' \
		'NVCCFLAGS=$(NVCCFLAGS)' \
		'ARCH_FLAGS=$(ARCH_FLAGS)' \
		'PREFIX=$(PREFIX)' \
		'BINDIR=$(BINDIR)'

clean:
	$(RM) $(TARGET) $(OBJECTS)
	-rmdir $(BUILD_DIR)
