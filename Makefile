SHELL := /bin/bash
.DEFAULT_GOAL := dvfs_latency_bench

# ---------- Config ----------
BUILD ?= release
SM    ?= 70

# Force using CUDA 12.8 nvcc (avoid picking /usr/bin/nvcc from apt)
CUDA_HOME ?= /usr/local/cuda-12.8
NVCC ?= $(CUDA_HOME)/bin/nvcc

# Host C++ compiler for nvcc (can override to g++-10)
CXX      ?= g++
HOST_CXX ?= $(CXX)

BIN    := dvfs_latency_bench
OBJDIR := build/$(BUILD)

STD := -std=c++17
ARCH := -arch=sm_$(SM)

# Project includes
INCLUDES := -I. -Iworkload

# NVML header path you found
NVML_INC ?= /usr/local/cuda-12.8/targets/x86_64-linux/include
INCLUDES += -I$(NVML_INC)

# Optional NVML library directory (usually not needed)
NVML_LIB ?=
LDFLAGS :=
ifneq ($(NVML_LIB),)
  LDFLAGS += -L$(NVML_LIB)
endif

ifeq ($(BUILD),debug)
  OPT_CXX  := -O0 -g
  OPT_NVCC := -O0 -g -lineinfo
else
  OPT_CXX  := -O3
  OPT_NVCC := -O3 -lineinfo
endif

CXXFLAGS  := $(STD) $(OPT_CXX) $(INCLUDES) -Wall -Wextra -Wno-unused-parameter
NVCCFLAGS := $(STD) $(OPT_NVCC) $(ARCH) $(INCLUDES) -Xcompiler -fPIC -ccbin $(HOST_CXX)

# workload_comm uses std::thread
PTHREAD := -Xcompiler -pthread
LDLIBS  := -lnvidia-ml

# ---------- Sources ----------
SRC_MAIN := main.cc
OBJ_MAIN := $(OBJDIR)/main.o
OBJ_CU   := $(OBJDIR)/workload_compute.o $(OBJDIR)/workload_comm.o
OBJS     := $(OBJ_MAIN) $(OBJ_CU)

# ---------- Rules ----------
.PHONY: all clean distclean list workload_objs

all: dvfs_latency_bench

list:
	@echo "BUILD=$(BUILD)"
	@echo "SM=$(SM)"
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "NVCC=$(NVCC)"
	@echo "HOST_CXX=$(HOST_CXX)"
	@echo "NVML_INC=$(NVML_INC)"
	@echo "NVML_LIB=$(NVML_LIB)"
	@echo "---- tool check ----"
	@which $(NVCC) || true
	@$(NVCC) --version || true
	@$(HOST_CXX) --version | head -n 1 || true

$(OBJDIR):
	@mkdir -p $@

$(OBJ_MAIN): $(SRC_MAIN) workload/workload.h workload/workload_compute.h workload/workload_comm.h | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/workload_compute.o: workload/workload_compute.cu workload/workload_compute.h | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR)/workload_comm.o: workload/workload_comm.cu workload/workload_comm.h | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

workload_objs: $(OBJ_CU)
	@echo "Built workload objects: $(OBJ_CU)"

dvfs_latency_bench: $(OBJS)
	$(NVCC) $(ARCH) $(STD) $(OPT_NVCC) -o $(BIN) $(OBJS) $(PTHREAD) $(LDFLAGS) $(LDLIBS)
	@echo "Built: ./$(BIN)"

clean:
	@rm -rf build

distclean: clean
	@rm -f $(BIN)
