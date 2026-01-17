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
INCLUDES := -I. -Iworkload
ARCH := -arch=sm_$(SM)

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
SRC_CU   := workload/workload_compute.cu workload/workload_comm.cu

OBJ_MAIN := $(OBJDIR)/main.o
OBJ_CU   := $(OBJDIR)/workload_compute.o $(OBJDIR)/workload_comm.o
OBJS     := $(OBJ_MAIN) $(OBJ_CU)

# ---------- Rules ----------
.PHONY: dvfs_latency_bench all clean distclean list

all: dvfs_latency_bench

list:
	@echo "BUILD=$(BUILD)"
	@echo "SM=$(SM)"
	@echo "OBJDIR=$(OBJDIR)"
	@echo "BIN=$(BIN)"
	@echo "NVCC=$(NVCC)"
	@echo "CXX=$(CXX)"

# Directory rule: ONLY create directory, no compilation here.
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

# link
dvfs_latency_bench: $(OBJS)
	$(NVCC) $(ARCH) $(STD) $(OPT_NVCC) -o $(BIN) $(OBJS) $(PTHREAD) $(LDLIBS)
	@echo "Built: ./$(BIN)"

clean:
	@rm -rf build

distclean: clean
	@rm -f $(BIN)
