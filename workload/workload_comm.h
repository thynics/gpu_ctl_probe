// workload_comm.h
#pragma once

#include <atomic>
#include <cstddef>
#include <thread>
#include <cuda_runtime.h>

// Single-GPU memcpy/sync workload for DVFS latency experiments.
// Target: NVIDIA V100 (sm_70), should work on sm_70+.

enum class CommCopyKind : int {
  H2D = 0,      // pinned host -> device
  D2H = 1,      // device -> pinned host
  D2D = 2,      // device -> device
  H2D_D2H = 3,  // ping-pong: H2D then D2H each iteration
};

enum class CommSyncKind : int {
  EventSync = 0,      // cudaEventRecord + cudaEventSynchronize
  StreamSync = 1,     // cudaStreamSynchronize
  EventQuerySpin = 2, // cudaEventRecord + spin on cudaEventQuery
};

struct CommWorkloadParams {
  int device_id = 0;

  CommCopyKind copy_kind = CommCopyKind::H2D_D2H;
  CommSyncKind sync_kind = CommSyncKind::EventSync;

  // Transfer size per memcpy (bytes).
  size_t bytes_per_copy = 8ull * 1024ull * 1024ull;

  // How many memcpy operations to enqueue before synchronizing.
  int burst_copies = 4;

  // Optional: use two streams to create additional waits.
  bool use_two_streams = false;

  // After each burst+sync, sleep on host.
  int duty_off_us = 0;

  // Warmup iterations.
  int warmup_iters = 50;
};

struct CommWorkloadHandle {
  int device_id = 0;

  cudaStream_t stream0 = nullptr;
  cudaStream_t stream1 = nullptr;
  cudaEvent_t  evt0 = nullptr;
  cudaEvent_t  evt1 = nullptr;

  void*  h_buf = nullptr;
  void*  d_buf0 = nullptr;
  void*  d_buf1 = nullptr;
  size_t buf_bytes = 0;

  std::atomic<bool> stop_requested{false};
  std::thread worker;

  bool running = false;
};

void start_comm_workload(const CommWorkloadParams& params, CommWorkloadHandle* h);
void request_stop_comm_workload(CommWorkloadHandle* h);
void stop_comm_workload(CommWorkloadHandle* h);
void destroy_comm_workload(CommWorkloadHandle* h);
