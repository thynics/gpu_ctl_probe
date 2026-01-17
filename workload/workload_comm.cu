// workload_comm.cu
#include "workload_comm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "[CUDA] %s:%d: %s failed: %s\n",                         \
              __FILE__, __LINE__, #call, cudaGetErrorString(_e));              \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#endif

static void comm_worker_loop(CommWorkloadParams params, CommWorkloadHandle* h) {
  CUDA_CHECK(cudaSetDevice(params.device_id));

  auto do_one_iter = [&]() {
    const int burst = std::max(1, params.burst_copies);
    const size_t sz = std::max<size_t>(1, params.bytes_per_copy);

    for (int i = 0; i < burst; ++i) {
      if (params.copy_kind == CommCopyKind::H2D) {
        CUDA_CHECK(cudaMemcpyAsync(h->d_buf0, h->h_buf, sz,
                                  cudaMemcpyHostToDevice, h->stream0));
      } else if (params.copy_kind == CommCopyKind::D2H) {
        CUDA_CHECK(cudaMemcpyAsync(h->h_buf, h->d_buf0, sz,
                                  cudaMemcpyDeviceToHost, h->stream0));
      } else if (params.copy_kind == CommCopyKind::D2D) {
        CUDA_CHECK(cudaMemcpyAsync(h->d_buf1, h->d_buf0, sz,
                                  cudaMemcpyDeviceToDevice, h->stream0));
        std::swap(h->d_buf0, h->d_buf1);
      } else { // H2D_D2H
        CUDA_CHECK(cudaMemcpyAsync(h->d_buf0, h->h_buf, sz,
                                  cudaMemcpyHostToDevice, h->stream0));
        CUDA_CHECK(cudaMemcpyAsync(h->h_buf, h->d_buf0, sz,
                                  cudaMemcpyDeviceToHost, h->stream0));
      }
    }

    if (!params.use_two_streams) {
      if (params.sync_kind == CommSyncKind::StreamSync) {
        CUDA_CHECK(cudaStreamSynchronize(h->stream0));
      } else if (params.sync_kind == CommSyncKind::EventSync) {
        CUDA_CHECK(cudaEventRecord(h->evt0, h->stream0));
        CUDA_CHECK(cudaEventSynchronize(h->evt0));
      } else { // EventQuerySpin
        CUDA_CHECK(cudaEventRecord(h->evt0, h->stream0));
        while (true) {
          cudaError_t q = cudaEventQuery(h->evt0);
          if (q == cudaSuccess) break;
          if (q != cudaErrorNotReady) CUDA_CHECK(q);
        }
      }
    } else {
      CUDA_CHECK(cudaEventRecord(h->evt0, h->stream0));
      CUDA_CHECK(cudaStreamWaitEvent(h->stream1, h->evt0, 0));
      CUDA_CHECK(cudaEventRecord(h->evt1, h->stream1));

      if (params.sync_kind == CommSyncKind::StreamSync) {
        CUDA_CHECK(cudaStreamSynchronize(h->stream1));
      } else if (params.sync_kind == CommSyncKind::EventSync) {
        CUDA_CHECK(cudaEventSynchronize(h->evt1));
      } else { // EventQuerySpin
        while (true) {
          cudaError_t q = cudaEventQuery(h->evt1);
          if (q == cudaSuccess) break;
          if (q != cudaErrorNotReady) CUDA_CHECK(q);
        }
      }
    }

    if (params.duty_off_us > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(params.duty_off_us));
    }
  };

  for (int i = 0; i < std::max(0, params.warmup_iters); ++i) {
    do_one_iter();
  }

  while (!h->stop_requested.load(std::memory_order_relaxed)) {
    do_one_iter();
  }
}

void start_comm_workload(const CommWorkloadParams& params, CommWorkloadHandle* h) {
  if (!h) return;
  if (h->running) {
    destroy_comm_workload(h);
  }

  // Reset only POD-ish fields (avoid memset on std::atomic/std::thread).
  h->device_id = params.device_id;
  h->stream0 = nullptr; h->stream1 = nullptr;
  h->evt0 = nullptr;    h->evt1 = nullptr;
  h->h_buf = nullptr;   h->d_buf0 = nullptr; h->d_buf1 = nullptr;
  h->buf_bytes = 0;
  h->stop_requested.store(false, std::memory_order_relaxed);
  h->running = false;

  CUDA_CHECK(cudaSetDevice(h->device_id));

  CUDA_CHECK(cudaStreamCreateWithFlags(&h->stream0, cudaStreamNonBlocking));
  if (params.use_two_streams) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&h->stream1, cudaStreamNonBlocking));
  }

  CUDA_CHECK(cudaEventCreateWithFlags(&h->evt0, cudaEventDisableTiming));
  if (params.use_two_streams) {
    CUDA_CHECK(cudaEventCreateWithFlags(&h->evt1, cudaEventDisableTiming));
  }

  h->buf_bytes = std::max<size_t>(1, params.bytes_per_copy);

  const bool need_host =
      (params.copy_kind == CommCopyKind::H2D) ||
      (params.copy_kind == CommCopyKind::D2H) ||
      (params.copy_kind == CommCopyKind::H2D_D2H);

  if (need_host) {
    CUDA_CHECK(cudaHostAlloc(&h->h_buf, h->buf_bytes, cudaHostAllocDefault));
    std::memset(h->h_buf, 0xAB, h->buf_bytes);
  }

  CUDA_CHECK(cudaMalloc(&h->d_buf0, h->buf_bytes));
  CUDA_CHECK(cudaMemsetAsync(h->d_buf0, 0, h->buf_bytes, h->stream0));

  if (params.copy_kind == CommCopyKind::D2D) {
    CUDA_CHECK(cudaMalloc(&h->d_buf1, h->buf_bytes));
    CUDA_CHECK(cudaMemsetAsync(h->d_buf1, 0, h->buf_bytes, h->stream0));
  } else {
    h->d_buf1 = nullptr;
  }

  CUDA_CHECK(cudaStreamSynchronize(h->stream0));

  h->running = true;
  h->worker = std::thread(comm_worker_loop, params, h);
}

void request_stop_comm_workload(CommWorkloadHandle* h) {
  if (!h || !h->running) return;
  h->stop_requested.store(true, std::memory_order_relaxed);
}

void stop_comm_workload(CommWorkloadHandle* h) {
  if (!h || !h->running) return;
  request_stop_comm_workload(h);
  if (h->worker.joinable()) {
    h->worker.join();
  }
  h->running = false;
}

void destroy_comm_workload(CommWorkloadHandle* h) {
  if (!h) return;
  if (h->running) {
    stop_comm_workload(h);
  }

  CUDA_CHECK(cudaSetDevice(h->device_id));

  if (h->d_buf1) CUDA_CHECK(cudaFree(h->d_buf1));
  if (h->d_buf0) CUDA_CHECK(cudaFree(h->d_buf0));
  if (h->h_buf)  CUDA_CHECK(cudaFreeHost(h->h_buf));

  if (h->evt1) CUDA_CHECK(cudaEventDestroy(h->evt1));
  if (h->evt0) CUDA_CHECK(cudaEventDestroy(h->evt0));

  if (h->stream1) CUDA_CHECK(cudaStreamDestroy(h->stream1));
  if (h->stream0) CUDA_CHECK(cudaStreamDestroy(h->stream0));

  // Reset
  h->stream0 = nullptr; h->stream1 = nullptr;
  h->evt0 = nullptr;    h->evt1 = nullptr;
  h->h_buf = nullptr;   h->d_buf0 = nullptr; h->d_buf1 = nullptr;
  h->buf_bytes = 0;
  h->stop_requested.store(false, std::memory_order_relaxed);
  h->running = false;
}
