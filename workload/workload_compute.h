// workload_compute.h
#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// Persistent compute workload for DVFS latency experiments.
// Target: NVIDIA V100 (sm_70), should work on sm_70+.

enum class ComputeLoadKind : int {
  ComputeOnly = 0,
  MemoryOnly  = 1,
  Mixed       = 2,
};

struct ComputeWorkloadParams {
  int device_id = 0;

  // Launch shape
  int block_size = 256;

  // If active_blocks > 0, it overrides blocks_per_sm.
  // Otherwise grid = sm_count * blocks_per_sm.
  int blocks_per_sm = 2;
  int active_blocks = 0;

  // Load kind
  ComputeLoadKind kind = ComputeLoadKind::ComputeOnly;

  // Compute intensity (number of FMA ops per "work step")
  int fma_iters = 4096;

  // Memory intensity (number of global memory iterations per "work step")
  int mem_iters = 0;

  // Memory buffer size (bytes). If 0, memory path is disabled unless kind needs it.
  // Recommend using a size well above L2 (e.g., 64MB~512MB) for memory-bound behavior.
  size_t buffer_bytes = 0;

  // Stride in uint4 elements (16B). Larger stride => less locality.
  int mem_stride_u4 = 64;

  // Duty cycle (approx). If duty_off_us > 0, kernel alternates busy/sleep.
  int duty_on_us  = 0;
  int duty_off_us = 0;

  // Stop check interval in outer loop iterations
  int check_interval = 1024;
};

struct ComputeWorkloadHandle {
  int device_id = 0;

  cudaStream_t work_stream = nullptr; // persistent kernel stream
  cudaStream_t ctrl_stream = nullptr; // stop flag update stream

  int* d_stop = nullptr;              // device stop flag
  float* d_sink = nullptr;            // prevent DCE
  uint4* d_buf_u4 = nullptr;          // optional memory buffer
  size_t buf_u4_elems = 0;

  int grid = 0;
  int block = 0;

  bool running = false;
};

void start_compute_workload(const ComputeWorkloadParams& params, ComputeWorkloadHandle* h);
void request_stop_compute_workload(ComputeWorkloadHandle* h);
void stop_compute_workload(ComputeWorkloadHandle* h);
void destroy_compute_workload(ComputeWorkloadHandle* h);
