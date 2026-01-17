SHELL := /bin/bash
.DEFAULT_GOAL := dvfs_latency_bench

# ---------- Config ----------
BUILD ?= release
SM    ?= 70
NVCC  ?= nvcc
CXX   ?= g++

BIN    := dvfs_latency_bench
OBJDIR := build/$(BUILD)

STD := -std=c++17
ARCH := -arch=sm_$(SM)

# Project includes
INCLUDES := -I. -Iworkload

# NVML include/lib (can be overridden from command line)
# Your machine:
#   nvml.h at /usr/local/cuda-12.8/targets/x86_64-linux/include/nvml.h
NVML_INC ?= /usr/local/cuda-12.8/targets/x86_64-linux/include
NVML_LIB ?=

INCLUDES += -I$(NVML_INC)

# If NVML_LIB is set, add -L...
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
NVCCFLAGS := $(STD) $(OPT_NVCC) $(ARCH) $(INCLUDES) -Xcompiler -fPIC

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
	@echo "OBJDIR=$(OBJDIR)"
	@echo "BIN=$(BIN)"
	@echo "NVCC=$(NVCC)"
	@echo "CXX=$(CXX)"
	@echo "NVML_INC=$(NVML_INC)"
	@echo "NVML_LIB=$(NVML_LIB)"

$(OBJDIR):
	@mkdir -p $@

# main.o
$(OBJ_MAIN): $(SRC_MAIN) workload/workload.h workload/workload_compute.h workload/workload_comm.h | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# workload_compute.o
$(OBJDIR)/workload_compute.o: workload/workload_compute.cu workload/workload_compute.h | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# workload_comm.o
$(OBJDIR)/workload_comm.o: workload/workload_comm.cu workload/workload_comm.h | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

workload_objs: $(OBJ_CU)
	@echo "Built workload objects: $(OBJ_CU)"

# link
dvfs_latency_bench: $(OBJS)
	$(NVCC) $(ARCH) $(STD) $(OPT_NVCC) -o $(BIN) $(OBJS) $(PTHREAD) $(LDFLAGS) $(LDLIBS)
	@echo "Built: ./$(BIN)"

clean:
	@rm -rf build

distclean: clean
	@rm -f $(BIN)
