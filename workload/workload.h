// workload.h
#pragma once

#include "workload_compute.h"
#include "workload_comm.h"

enum class WorkloadMode : int { Idle = 0, Compute = 1, Comm = 2 };

struct WorkloadConfig {
  WorkloadMode mode = WorkloadMode::Idle;
  ComputeWorkloadParams compute{};
  CommWorkloadParams comm{};
};

struct WorkloadHandle {
  WorkloadMode mode = WorkloadMode::Idle;
  ComputeWorkloadHandle compute{};
  CommWorkloadHandle comm{};
  bool running = false;
};

inline void start_workload(const WorkloadConfig& cfg, WorkloadHandle* h) {
  if (!h) return;
  h->mode = cfg.mode;
  h->running = false;

  switch (cfg.mode) {
    case WorkloadMode::Idle:
      return;
    case WorkloadMode::Compute:
      start_compute_workload(cfg.compute, &h->compute);
      h->running = true;
      return;
    case WorkloadMode::Comm:
      start_comm_workload(cfg.comm, &h->comm);
      h->running = true;
      return;
  }
}

inline void request_stop_workload(WorkloadHandle* h) {
  if (!h || !h->running) return;
  switch (h->mode) {
    case WorkloadMode::Idle:   return;
    case WorkloadMode::Compute: request_stop_compute_workload(&h->compute); return;
    case WorkloadMode::Comm:    request_stop_comm_workload(&h->comm); return;
  }
}

inline void stop_workload(WorkloadHandle* h) {
  if (!h || !h->running) return;
  switch (h->mode) {
    case WorkloadMode::Idle: break;
    case WorkloadMode::Compute: stop_compute_workload(&h->compute); break;
    case WorkloadMode::Comm: stop_comm_workload(&h->comm); break;
  }
  h->running = false;
}

inline void destroy_workload(WorkloadHandle* h) {
  if (!h) return;
  if (h->running) stop_workload(h);
  switch (h->mode) {
    case WorkloadMode::Idle: break;
    case WorkloadMode::Compute: destroy_compute_workload(&h->compute); break;
    case WorkloadMode::Comm: destroy_comm_workload(&h->comm); break;
  }
  h->mode = WorkloadMode::Idle;
}
