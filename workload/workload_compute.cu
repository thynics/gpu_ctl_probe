// workload_compute.cu
#ifdef __f
#undef __f
#endif


#include "workload_compute.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

// -----------------------------
// Device helpers
// -----------------------------
__device__ __forceinline__ uint32_t xorshift32(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

__device__ __forceinline__ void busy_fma(float &a, float &b, float &c, int iters) {
#pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    a = fmaf(a, b, c);
    b = fmaf(b, c, a);
    c = fmaf(c, a, b);
  }
}

__device__ __forceinline__ uint64_t clock64_relaxed() {
#if __CUDA_ARCH__ >= 300
  return clock64();
#else
  return 0;
#endif
}

__device__ __forceinline__ void sleep_us(int us) {
  if (us <= 0) return;
#if __CUDA_ARCH__ >= 700
  __nanosleep(static_cast<unsigned int>(us) * 1000u);
#else
  uint64_t start = clock64_relaxed();
  uint64_t cycles = static_cast<uint64_t>(us) * 1000ull;
  while (clock64_relaxed() - start < cycles) { /* spin */ }
#endif
}

// -----------------------------
// Persistent kernel
// -----------------------------
__global__ void persistent_compute_kernel(const int* __restrict__ d_stop,
                                          float* __restrict__ d_sink,
                                          uint4* __restrict__ buf_u4,
                                          size_t buf_u4_elems,
                                          int kind,
                                          int fma_iters,
                                          int mem_iters,
                                          int mem_stride_u4,
                                          int duty_on_us,
                                          int duty_off_us,
                                          int check_interval) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float a = 1.0f + 0.001f * (tid & 1023);
  float b = 1.5f + 0.002f * ((tid >> 3) & 1023);
  float c = 2.0f + 0.003f * ((tid >> 6) & 1023);

  uint32_t rng = 0x9e3779b9u ^ static_cast<uint32_t>(tid);
  size_t idx = (static_cast<size_t>(tid) * 17u) % (buf_u4_elems ? buf_u4_elems : 1u);

  uint64_t busy_start = 0;
  if (duty_off_us > 0 && duty_on_us > 0) {
    busy_start = clock64_relaxed();
  }

  int iter = 0;
  for (;;) {
    if ((iter % max(1, check_interval)) == 0) {
      int s = *((volatile const int*)d_stop);
      if (s != 0) break;
    }

    if (duty_off_us > 0 && duty_on_us > 0) {
      uint64_t now = clock64_relaxed();
      if ((now - busy_start) > 50000ull) {
        sleep_us(duty_off_us);
        busy_start = clock64_relaxed();
      }
    }

    if (kind == static_cast<int>(ComputeLoadKind::ComputeOnly) ||
        kind == static_cast<int>(ComputeLoadKind::Mixed)) {
      busy_fma(a, b, c, fma_iters);
    }

    if ((kind == static_cast<int>(ComputeLoadKind::MemoryOnly) ||
         kind == static_cast<int>(ComputeLoadKind::Mixed)) &&
        buf_u4 != nullptr && buf_u4_elems > 0 && mem_iters > 0) {
#pragma unroll 1
      for (int m = 0; m < mem_iters; ++m) {
        rng = xorshift32(rng);
        idx = (idx + static_cast<size_t>(mem_stride_u4) + (rng & 15u)) % buf_u4_elems;

        uint4 v = buf_u4[idx];
        a = fmaf(a, 1.000001f, static_cast<float>(v.x & 255u));
        b = fmaf(b, 1.000002f, static_cast<float>(v.y & 255u));
        c = fmaf(c, 1.000003f, static_cast<float>(v.z & 255u));

        v.x ^= rng;
        buf_u4[idx] = v;
      }
    }

    if ((iter & 0x3FF) == 0) {
      d_sink[tid] = a + b + c;
    }

    ++iter;
  }

  d_sink[tid] = a + b + c;
}

// -----------------------------
// Host utilities
// -----------------------------
static size_t round_up_pow2(size_t x) {
  if (x <= 1) return 1;
  size_t p = 1;
  while (p < x) p <<= 1;
  return p;
}

static int compute_grid_blocks(const ComputeWorkloadParams& p, int sm_count) {
  if (p.active_blocks > 0) return p.active_blocks;
  int bpsm = std::max(1, p.blocks_per_sm);
  return std::max(1, sm_count * bpsm);
}

// -----------------------------
// Public API
// -----------------------------
void start_compute_workload(const ComputeWorkloadParams& params,
                            ComputeWorkloadHandle* h) {
  if (!h) return;
  if (h->running) {
    destroy_compute_workload(h);
  }

  *h = ComputeWorkloadHandle{};
  h->device_id = params.device_id;

  CUDA_CHECK(cudaSetDevice(h->device_id));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, h->device_id));
  const int sm_count = prop.multiProcessorCount;

  h->block = std::max(32, params.block_size);
  h->grid  = compute_grid_blocks(params, sm_count);

  CUDA_CHECK(cudaStreamCreateWithFlags(&h->work_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&h->ctrl_stream, cudaStreamNonBlocking));

  CUDA_CHECK(cudaMalloc(&h->d_stop, sizeof(int)));
  int zero = 0;
  CUDA_CHECK(cudaMemcpyAsync(h->d_stop, &zero, sizeof(int),
                            cudaMemcpyHostToDevice, h->ctrl_stream));
  CUDA_CHECK(cudaStreamSynchronize(h->ctrl_stream));

  const size_t n_threads = static_cast<size_t>(h->grid) * static_cast<size_t>(h->block);
  CUDA_CHECK(cudaMalloc(&h->d_sink, n_threads * sizeof(float)));

  bool need_mem = (params.kind != ComputeLoadKind::ComputeOnly) ||
                  (params.mem_iters > 0) ||
                  (params.buffer_bytes > 0);

  if (need_mem) {
    size_t bytes = params.buffer_bytes;
    if (bytes == 0) bytes = 128ull * 1024ull * 1024ull;
    bytes = std::max<size_t>(bytes, 1 * 1024 * 1024);

    size_t elems = bytes / sizeof(uint4);
    elems = round_up_pow2(std::max<size_t>(elems, 1024));
    h->buf_u4_elems = elems;

    CUDA_CHECK(cudaMalloc(&h->d_buf_u4, elems * sizeof(uint4)));
    CUDA_CHECK(cudaMemsetAsync(h->d_buf_u4, 0, elems * sizeof(uint4), h->work_stream));
  }

  persistent_compute_kernel<<<h->grid, h->block, 0, h->work_stream>>>(
      h->d_stop,
      h->d_sink,
      h->d_buf_u4,
      h->buf_u4_elems,
      static_cast<int>(params.kind),
      std::max(0, params.fma_iters),
      std::max(0, params.mem_iters),
      std::max(1, params.mem_stride_u4),
      params.duty_on_us,
      params.duty_off_us,
      std::max(1, params.check_interval));

  CUDA_CHECK(cudaGetLastError());
  h->running = true;
}

void request_stop_compute_workload(ComputeWorkloadHandle* h) {
  if (!h || !h->running) return;
  CUDA_CHECK(cudaSetDevice(h->device_id));
  int one = 1;
  CUDA_CHECK(cudaMemcpyAsync(h->d_stop, &one, sizeof(int),
                            cudaMemcpyHostToDevice, h->ctrl_stream));
  CUDA_CHECK(cudaStreamSynchronize(h->ctrl_stream));
}

void stop_compute_workload(ComputeWorkloadHandle* h) {
  if (!h || !h->running) return;
  request_stop_compute_workload(h);
  CUDA_CHECK(cudaSetDevice(h->device_id));
  CUDA_CHECK(cudaStreamSynchronize(h->work_stream));
  h->running = false;
}

void destroy_compute_workload(ComputeWorkloadHandle* h) {
  if (!h) return;
  if (h->running) stop_compute_workload(h);

  CUDA_CHECK(cudaSetDevice(h->device_id));

  if (h->d_buf_u4) CUDA_CHECK(cudaFree(h->d_buf_u4));
  if (h->d_sink)   CUDA_CHECK(cudaFree(h->d_sink));
  if (h->d_stop)   CUDA_CHECK(cudaFree(h->d_stop));

  if (h->work_stream) CUDA_CHECK(cudaStreamDestroy(h->work_stream));
  if (h->ctrl_stream) CUDA_CHECK(cudaStreamDestroy(h->ctrl_stream));

  *h = ComputeWorkloadHandle{};
}
