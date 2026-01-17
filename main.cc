// main.cc
// DVFS latency benchmark (GPU graphics clock only; mem locked NOT_SUPPORTED on this platform)
//
// What we control:
//   - GPU "graphics" clock (NVML_CLOCK_GRAPHICS), via:
//       * locked GPU clocks: nvmlDeviceSetGpuLockedClocks(min=max=target)
//       * applications clocks: nvmlDeviceSetApplicationsClocks(mem_fixed, gpu_target)
//     (mem_fixed is kept constant; we do NOT change mem clock)
//
// What we measure:
//   - API latency: time spent in the NVML set-call(s)
//   - Settle latency: time from before set-call to N consecutive polls reaching target
//   - Workload intensity metrics (NVML, averaged over a window):
//       * utilization.gpu (%), utilization.memory (%)
//       * PCIe TX/RX throughput (KB/s)
//       * power (mW), energy (mJ)
//
// Notes:
//   - This version intentionally drops *any* mem DVFS knob.
//   - "comm" still matters as an activity profile (copy engines + PCIe) that can affect DVFS behavior.

#include <nvml.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "workload.h"

// -----------------------------
// Helpers
// -----------------------------
static inline uint64_t now_us() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count();
}

static inline void sleep_us(int us) {
  if (us <= 0) return;
  std::this_thread::sleep_for(std::chrono::microseconds(us));
}

static inline void sleep_ms(int ms) {
  if (ms <= 0) return;
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

static std::string nvml_err(nvmlReturn_t r) { return std::string(nvmlErrorString(r)); }

#define NVML_CHECK(call) do { \
  nvmlReturn_t _r = (call); \
  if (_r != NVML_SUCCESS) { \
    std::fprintf(stderr, "[NVML] %s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, nvmlErrorString(_r)); \
    std::exit(2); \
  } \
} while(0)

static bool nvml_ok(nvmlReturn_t r) { return r == NVML_SUCCESS; }

static bool has_arg(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], key) == 0) return true;
  return false;
}

static std::string get_arg(int argc, char** argv, const char* key, const std::string& def = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  }
  return def;
}

static int get_arg_int(int argc, char** argv, const char* key, int def) {
  std::string s = get_arg(argc, argv, key, "");
  if (s.empty()) return def;
  return std::atoi(s.c_str());
}

static long long get_arg_ll(int argc, char** argv, const char* key, long long def) {
  std::string s = get_arg(argc, argv, key, "");
  if (s.empty()) return def;
  return std::atoll(s.c_str());
}

static bool get_arg_bool(int argc, char** argv, const char* key, bool def) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], key) == 0) {
      if (i + 1 < argc) {
        if (argv[i + 1][0] == '-') return true;
        return std::atoi(argv[i + 1]) != 0;
      }
      return true;
    }
  }
  return def;
}

static std::vector<unsigned int> sort_unique(std::vector<unsigned int> v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

static std::vector<unsigned int> pick_evenly(const std::vector<unsigned int>& sorted_asc, int k) {
  std::vector<unsigned int> out;
  if (sorted_asc.empty() || k <= 0) return out;
  if ((int)sorted_asc.size() <= k) return sorted_asc;
  out.reserve(k);
  for (int i = 0; i < k; ++i) {
    double t = (k == 1) ? 0.0 : (double)i / (double)(k - 1);
    size_t idx = (size_t)(t * (double)(sorted_asc.size() - 1) + 0.5);
    out.push_back(sorted_asc[idx]);
  }
  return sort_unique(out);
}

static std::vector<unsigned int> parse_u32_list_csv(const std::string& s) {
  std::vector<unsigned int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    char* end = nullptr;
    long long x = std::strtoll(tok.c_str(), &end, 10);
    if (end == tok.c_str()) continue;
    if (x < 0) continue;
    v.push_back((unsigned int)x);
  }
  return sort_unique(v);
}

static void print_usage() {
  std::fprintf(stderr,
    "Usage: dvfs_latency_bench [options]\n"
    "  --device N               GPU index (default 0)\n"
    "  --mode idle|compute|comm (default idle)\n"
    "  --out file.csv           output CSV (default dvfs_latency_gpu_only.csv)\n"
    "\n"
    "Clock points (GPU graphics clock only):\n"
    "  --core_list a,b,c        explicit graphics clocks MHz (optional)\n"
    "  --pick_core K            if no core_list, pick K points evenly (default 6)\n"
    "  --transitions adjacent|extreme|all (default adjacent)\n"
    "\n"
    "Clock API selection:\n"
    "  --api auto|locked|app    (default auto)\n"
    "    locked: nvmlDeviceSetGpuLockedClocks\n"
    "    app   : nvmlDeviceSetApplicationsClocks(mem_fixed, gpu_target)  (mem_fixed constant)\n"
    "    auto  : prefer locked; fallback to app if locked not usable\n"
    "\n"
    "Timing (defaults are intentionally longer for credibility):\n"
    "  --iters N                datapoints per transition (default 300)\n"
    "  --poll_us U              settle poll interval (default 2000)\n"
    "  --stable_n N             require N consecutive polls at target (default 10)\n"
    "  --timeout_ms T           settle timeout per transition (default 8000)\n"
    "  --warmup_trans N         warmup transitions (default 100)\n"
    "  --warmup_ms M            warmup sleep before measuring (default 10000)\n"
    "\n"
    "Load intensity metrics (NVML averaged):\n"
    "  --metric_window_ms W     window for averaging util/pcie (default 200)\n"
    "  --metric_interval_ms I   sampling interval inside window (default 50)\n"
    "\n"
    "Compute workload options (mode=compute):\n"
    "  --compute_kind compute|mem|mixed (default compute)\n"
    "  --blocks_per_sm N (default 2)\n"
    "  --fma_iters N      (default 4096)\n"
    "  --mem_iters N      (default 0)\n"
    "  --buffer_mb N      (default 0 -> auto if needed)\n"
    "  --mem_stride_u4 N  (default 64)\n"
    "\n"
    "Comm workload options (mode=comm):\n"
    "  --comm_copy h2d|d2h|d2d|h2d_d2h (default h2d_d2h)\n"
    "  --comm_sync event|stream|spin   (default event)\n"
    "  --comm_bytes N    bytes per copy (default 8MB)\n"
    "  --comm_burst N    copies before sync (default 4)\n"
    "  --comm_two_streams 0/1 (default 0)\n"
    "  --comm_duty_off_us U (default 0)\n"
    "\n"
    "Note: changing clocks usually requires admin/root privileges.\n"
  );
}

// -----------------------------
// NVML wrappers
// -----------------------------
struct NvmlCtx {
  nvmlDevice_t dev{};
  int device_id = 0;
  std::string name;
};

static NvmlCtx nvml_open_device(int device_id) {
  NvmlCtx c;
  c.device_id = device_id;

  NVML_CHECK(nvmlInit_v2());

  nvmlDevice_t dev;
  NVML_CHECK(nvmlDeviceGetHandleByIndex_v2(device_id, &dev));
  c.dev = dev;

  char name[128];
  std::memset(name, 0, sizeof(name));
  NVML_CHECK(nvmlDeviceGetName(dev, name, sizeof(name) - 1));
  c.name = name;

  return c;
}

static void nvml_close() { nvmlShutdown(); }

static std::optional<unsigned int> nvml_get_clock_mhz(nvmlDevice_t dev, nvmlClockType_t t) {
  unsigned int mhz = 0;
  nvmlReturn_t r = nvmlDeviceGetClockInfo(dev, t, &mhz);
  if (!nvml_ok(r)) return std::nullopt;
  return mhz;
}

static std::optional<unsigned int> nvml_get_app_clock_mhz(nvmlDevice_t dev, nvmlClockType_t t) {
  unsigned int mhz = 0;
  nvmlReturn_t r = nvmlDeviceGetApplicationsClock(dev, t, &mhz);
  if (!nvml_ok(r)) return std::nullopt;
  return mhz;
}

static std::optional<unsigned int> nvml_get_power_mw(nvmlDevice_t dev) {
  unsigned int mw = 0;
  nvmlReturn_t r = nvmlDeviceGetPowerUsage(dev, &mw);
  if (!nvml_ok(r)) return std::nullopt;
  return mw;
}

static std::optional<unsigned long long> nvml_get_energy_mj(nvmlDevice_t dev) {
  unsigned long long mj = 0;
  nvmlReturn_t r = nvmlDeviceGetTotalEnergyConsumption(dev, &mj);
  if (r == NVML_ERROR_NOT_SUPPORTED) return std::nullopt;
  if (!nvml_ok(r)) return std::nullopt;
  return mj;
}

static std::vector<unsigned int> nvml_get_supported_mem_clocks(nvmlDevice_t dev) {
  unsigned int n = 256;
  std::vector<unsigned int> mem(n);
  nvmlReturn_t r = nvmlDeviceGetSupportedMemoryClocks(dev, &n, mem.data());
  if (r == NVML_ERROR_INSUFFICIENT_SIZE) {
    mem.resize(n);
    r = nvmlDeviceGetSupportedMemoryClocks(dev, &n, mem.data());
  }
  if (!nvml_ok(r) || n == 0) return {};
  mem.resize(n);
  return sort_unique(mem);
}

static std::vector<unsigned int> nvml_get_supported_graphics_clocks(nvmlDevice_t dev, unsigned int mem_mhz) {
  unsigned int n = 512;
  std::vector<unsigned int> gr(n);
  nvmlReturn_t r = nvmlDeviceGetSupportedGraphicsClocks(dev, mem_mhz, &n, gr.data());
  if (r == NVML_ERROR_INSUFFICIENT_SIZE) {
    gr.resize(n);
    r = nvmlDeviceGetSupportedGraphicsClocks(dev, mem_mhz, &n, gr.data());
  }
  if (!nvml_ok(r) || n == 0) return {};
  gr.resize(n);
  return sort_unique(gr);
}

struct Metrics {
  double gpu_util_pct = 0.0;
  double mem_util_pct = 0.0;
  double pcie_tx_kbps = 0.0;
  double pcie_rx_kbps = 0.0;
};

static std::optional<Metrics> sample_metrics_avg(nvmlDevice_t dev, int window_ms, int interval_ms) {
  window_ms = std::max(0, window_ms);
  interval_ms = std::max(1, interval_ms);
  if (window_ms == 0) {
    Metrics m{};
    nvmlUtilization_t util{};
    unsigned int tx = 0, rx = 0;

    if (nvml_ok(nvmlDeviceGetUtilizationRates(dev, &util))) {
      m.gpu_util_pct = util.gpu;
      m.mem_util_pct = util.memory;
    }
    if (nvml_ok(nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_TX_BYTES, &tx))) m.pcie_tx_kbps = tx;
    if (nvml_ok(nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_RX_BYTES, &rx))) m.pcie_rx_kbps = rx;
    return m;
  }

  int samples = 0;
  double sum_gpu = 0, sum_mem = 0, sum_tx = 0, sum_rx = 0;

  uint64_t t_end = now_us() + (uint64_t)window_ms * 1000ull;
  while (now_us() < t_end) {
    nvmlUtilization_t util{};
    unsigned int tx = 0, rx = 0;

    if (nvml_ok(nvmlDeviceGetUtilizationRates(dev, &util))) {
      sum_gpu += util.gpu;
      sum_mem += util.memory;
    }
    if (nvml_ok(nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_TX_BYTES, &tx))) sum_tx += tx;
    if (nvml_ok(nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_RX_BYTES, &rx))) sum_rx += rx;

    samples++;
    sleep_ms(interval_ms);
  }

  if (samples <= 0) return std::nullopt;

  Metrics m{};
  m.gpu_util_pct = sum_gpu / samples;
  m.mem_util_pct = sum_mem / samples;
  m.pcie_tx_kbps = sum_tx / samples;
  m.pcie_rx_kbps = sum_rx / samples;
  return m;
}

// -----------------------------
// Clock API (GPU-only control)
// -----------------------------
enum class ClockApiKind : int { Auto = 0, Locked = 1, App = 2 };

static ClockApiKind parse_api_kind(const std::string& s) {
  if (s == "locked") return ClockApiKind::Locked;
  if (s == "app") return ClockApiKind::App;
  return ClockApiKind::Auto;
}

static const char* api_kind_str(ClockApiKind k) {
  switch (k) {
    case ClockApiKind::Locked: return "locked";
    case ClockApiKind::App:    return "app";
    default:                   return "auto";
  }
}

struct SetClockApiLatencyUs {
  uint64_t api_us = 0;
};

static nvmlReturn_t set_locked_gpu(nvmlDevice_t dev, unsigned int gpu_mhz, uint64_t* api_us) {
  uint64_t t0 = now_us();
  nvmlReturn_t r = nvmlDeviceSetGpuLockedClocks(dev, gpu_mhz, gpu_mhz);
  uint64_t t1 = now_us();
  if (api_us) *api_us = (t1 - t0);
  return r;
}

static nvmlReturn_t set_app_gpu(nvmlDevice_t dev, unsigned int mem_fixed_mhz, unsigned int gpu_mhz, uint64_t* api_us) {
  uint64_t t0 = now_us();
  nvmlReturn_t r = nvmlDeviceSetApplicationsClocks(dev, mem_fixed_mhz, gpu_mhz);
  uint64_t t1 = now_us();
  if (api_us) *api_us = (t1 - t0);
  return r;
}

static void reset_locked_gpu(nvmlDevice_t dev) { nvmlDeviceResetGpuLockedClocks(dev); }
static void reset_app(nvmlDevice_t dev) { nvmlDeviceResetApplicationsClocks(dev); }

struct SettleResult {
  bool ok = false;
  uint64_t t_total_us = 0;
  uint64_t t_after_call_us = 0;
  int polls = 0;
  unsigned int final_gpu_mhz = 0;
  std::string status; // ok(locked/app), timeout(...), set_xxx:ERR
};

static SettleResult set_and_settle_gpu_only(nvmlDevice_t dev,
                                            unsigned int target_gpu_mhz,
                                            unsigned int mem_fixed_mhz,
                                            int poll_us,
                                            int stable_n,
                                            int timeout_ms,
                                            ClockApiKind api_kind,
                                            bool locked_gpu_ok,
                                            bool app_ok,
                                            SetClockApiLatencyUs* api_lat_out) {
  SettleResult res;
  stable_n = std::max(1, stable_n);
  poll_us  = std::max(0, poll_us);
  timeout_ms = std::max(1, timeout_ms);

  uint64_t t_start = now_us();
  uint64_t api_us = 0;

  // choose API
  ClockApiKind used = api_kind;
  if (api_kind == ClockApiKind::Auto) {
    used = locked_gpu_ok ? ClockApiKind::Locked : ClockApiKind::App;
  }
  if (used == ClockApiKind::Locked && !locked_gpu_ok) used = ClockApiKind::App;
  if (used == ClockApiKind::App && !app_ok) used = ClockApiKind::Locked;

  // set
  nvmlReturn_t rset = NVML_SUCCESS;
  std::string which;

  if (used == ClockApiKind::Locked) {
    which = "set_locked_gpu";
    rset = set_locked_gpu(dev, target_gpu_mhz, &api_us);
  } else {
    which = "set_app_gpu";
    rset = set_app_gpu(dev, mem_fixed_mhz, target_gpu_mhz, &api_us);
  }

  uint64_t t_after = now_us();

  if (api_lat_out) api_lat_out->api_us = api_us;

  if (!nvml_ok(rset)) {
    res.ok = false;
    res.final_gpu_mhz = nvml_get_clock_mhz(dev, NVML_CLOCK_GRAPHICS).value_or(0);
    res.status = which + ":" + nvml_err(rset);
    return res;
  }

  // settle poll
  int consec = 0;
  int polls = 0;
  uint64_t deadline = t_start + (uint64_t)timeout_ms * 1000ull;

  while (true) {
    auto cg_act = nvml_get_clock_mhz(dev, NVML_CLOCK_GRAPHICS);
    if (cg_act.has_value()) res.final_gpu_mhz = *cg_act;

    bool hit = false;
    if (used == ClockApiKind::Locked) {
      if (cg_act.has_value() && *cg_act == target_gpu_mhz) hit = true;
    } else {
      // app: use configured app clock to avoid boost mismatch
      auto cg_app = nvml_get_app_clock_mhz(dev, NVML_CLOCK_GRAPHICS);
      if (cg_app.has_value() && *cg_app == target_gpu_mhz) hit = true;
    }

    polls++;
    consec = hit ? (consec + 1) : 0;

    uint64_t t_now = now_us();
    if (consec >= stable_n) {
      res.ok = true;
      res.polls = polls;
      res.t_total_us = t_now - t_start;
      res.t_after_call_us = t_now - t_after;
      res.status = std::string("ok(") + api_kind_str(used) + ")";
      return res;
    }
    if (t_now >= deadline) {
      res.ok = false;
      res.polls = polls;
      res.t_total_us = t_now - t_start;
      res.t_after_call_us = t_now - t_after;
      res.status = std::string("timeout(") + api_kind_str(used) + ")";
      return res;
    }
    if (poll_us > 0) sleep_us(poll_us);
  }
}

// -----------------------------
// Transitions
// -----------------------------
enum class TransitionPlan : int { Adjacent = 0, Extreme = 1, AllPairs = 2 };

static TransitionPlan parse_plan(const std::string& s) {
  if (s == "adjacent") return TransitionPlan::Adjacent;
  if (s == "extreme") return TransitionPlan::Extreme;
  if (s == "all") return TransitionPlan::AllPairs;
  return TransitionPlan::Adjacent;
}

static std::vector<std::pair<unsigned int, unsigned int>>
build_transitions(const std::vector<unsigned int>& gpu_points, TransitionPlan plan) {
  std::vector<std::pair<unsigned int, unsigned int>> out;
  if (gpu_points.size() < 2) return out;

  std::vector<unsigned int> ps = gpu_points;
  std::sort(ps.begin(), ps.end());
  ps = sort_unique(ps);

  if (plan == TransitionPlan::AllPairs) {
    for (size_t i = 0; i < ps.size(); ++i)
      for (size_t j = 0; j < ps.size(); ++j)
        if (i != j) out.push_back({ps[i], ps[j]});
    return out;
  }

  if (plan == TransitionPlan::Extreme) {
    out.push_back({ps.front(), ps.back()});
    out.push_back({ps.back(), ps.front()});
    return out;
  }

  // Adjacent
  for (size_t i = 1; i < ps.size(); ++i) {
    out.push_back({ps[i - 1], ps[i]});
    out.push_back({ps[i], ps[i - 1]});
  }
  return out;
}

// -----------------------------
// CSV logging
// -----------------------------
static bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

static void csv_write_header(std::ofstream& of) {
  of << "ts_us,device_id,gpu_name,mode,api_kind,mem_fixed_mhz,"
        "from_gpu_mhz,to_gpu_mhz,"
        "api_us,settle_total_us,settle_after_call_us,polls,stable_n,poll_us,timeout_ms,"
        "final_gpu_mhz,"
        "gpu_util_pct,mem_util_pct,pcie_tx_kbps,pcie_rx_kbps,"
        "power_mw,energy_mj,status\n";
}

static void csv_write_row(std::ofstream& of,
                          uint64_t ts_us,
                          int device_id,
                          const std::string& gpu_name,
                          const std::string& mode,
                          const std::string& api_kind,
                          unsigned int mem_fixed_mhz,
                          unsigned int from_gpu,
                          unsigned int to_gpu,
                          const SetClockApiLatencyUs& api,
                          const SettleResult& settle,
                          int stable_n,
                          int poll_us,
                          int timeout_ms,
                          const std::optional<Metrics>& met,
                          const std::optional<unsigned int>& power_mw,
                          const std::optional<unsigned long long>& energy_mj) {
  of << ts_us << ","
     << device_id << ","
     << "\"" << gpu_name << "\"" << ","
     << mode << ","
     << api_kind << ","
     << mem_fixed_mhz << ","
     << from_gpu << "," << to_gpu << ","
     << api.api_us << ","
     << settle.t_total_us << ","
     << settle.t_after_call_us << ","
     << settle.polls << ","
     << stable_n << "," << poll_us << "," << timeout_ms << ","
     << settle.final_gpu_mhz << ",";

  if (met.has_value()) {
    of << met->gpu_util_pct << ","
       << met->mem_util_pct << ","
       << met->pcie_tx_kbps << ","
       << met->pcie_rx_kbps << ",";
  } else {
    of << ",,,,"; // empty
  }

  of << (power_mw.has_value() ? std::to_string(*power_mw) : "") << ","
     << (energy_mj.has_value() ? std::to_string(*energy_mj) : "") << ","
     << settle.status
     << "\n";
  of.flush();
}

// -----------------------------
// Workload config parsing (unchanged)
// -----------------------------
static WorkloadMode parse_mode(const std::string& s) {
  if (s == "idle") return WorkloadMode::Idle;
  if (s == "compute") return WorkloadMode::Compute;
  if (s == "comm") return WorkloadMode::Comm;
  return WorkloadMode::Idle;
}

static ComputeLoadKind parse_compute_kind(const std::string& s) {
  if (s == "compute") return ComputeLoadKind::ComputeOnly;
  if (s == "mem") return ComputeLoadKind::MemoryOnly;
  if (s == "mixed") return ComputeLoadKind::Mixed;
  return ComputeLoadKind::ComputeOnly;
}

static CommCopyKind parse_comm_copy(const std::string& s) {
  if (s == "h2d") return CommCopyKind::H2D;
  if (s == "d2h") return CommCopyKind::D2H;
  if (s == "d2d") return CommCopyKind::D2D;
  if (s == "h2d_d2h") return CommCopyKind::H2D_D2H;
  return CommCopyKind::H2D_D2H;
}

static CommSyncKind parse_comm_sync(const std::string& s) {
  if (s == "event") return CommSyncKind::EventSync;
  if (s == "stream") return CommSyncKind::StreamSync;
  if (s == "spin") return CommSyncKind::EventQuerySpin;
  return CommSyncKind::EventSync;
}

// -----------------------------
// main
// -----------------------------
int main(int argc, char** argv) {
  if (has_arg(argc, argv, "--help") || has_arg(argc, argv, "-h")) {
    print_usage();
    return 0;
  }

  int device_id = get_arg_int(argc, argv, "--device", 0);
  std::string mode_s = get_arg(argc, argv, "--mode", "idle");
  std::string out_path = get_arg(argc, argv, "--out", "dvfs_latency_gpu_only.csv");

  // Longer defaults (credibility)
  int iters = get_arg_int(argc, argv, "--iters", 300);
  int poll_us = get_arg_int(argc, argv, "--poll_us", 2000);
  int stable_n = get_arg_int(argc, argv, "--stable_n", 10);
  int timeout_ms = get_arg_int(argc, argv, "--timeout_ms", 8000);
  int warmup_trans = get_arg_int(argc, argv, "--warmup_trans", 100);
  int warmup_ms = get_arg_int(argc, argv, "--warmup_ms", 10000);

  int metric_window_ms = get_arg_int(argc, argv, "--metric_window_ms", 200);
  int metric_interval_ms = get_arg_int(argc, argv, "--metric_interval_ms", 50);

  int pick_core = get_arg_int(argc, argv, "--pick_core", 6);
  std::string core_list_s = get_arg(argc, argv, "--core_list", "");
  std::string plan_s = get_arg(argc, argv, "--transitions", "adjacent");

  std::string api_s = get_arg(argc, argv, "--api", "auto");
  ClockApiKind api_kind = parse_api_kind(api_s);

  // workload params
  ComputeWorkloadParams cwp;
  cwp.device_id = device_id;
  cwp.kind = parse_compute_kind(get_arg(argc, argv, "--compute_kind", "compute"));
  cwp.blocks_per_sm = get_arg_int(argc, argv, "--blocks_per_sm", 2);
  cwp.fma_iters = get_arg_int(argc, argv, "--fma_iters", 4096);
  cwp.mem_iters = get_arg_int(argc, argv, "--mem_iters", 0);
  int buffer_mb = get_arg_int(argc, argv, "--buffer_mb", 0);
  if (buffer_mb > 0) cwp.buffer_bytes = (size_t)buffer_mb * 1024ull * 1024ull;
  cwp.mem_stride_u4 = get_arg_int(argc, argv, "--mem_stride_u4", 64);

  CommWorkloadParams cmp;
  cmp.device_id = device_id;
  cmp.copy_kind = parse_comm_copy(get_arg(argc, argv, "--comm_copy", "h2d_d2h"));
  cmp.sync_kind = parse_comm_sync(get_arg(argc, argv, "--comm_sync", "event"));
  cmp.bytes_per_copy = (size_t)get_arg_ll(argc, argv, "--comm_bytes", (long long)(8ull * 1024ull * 1024ull));
  cmp.burst_copies = get_arg_int(argc, argv, "--comm_burst", 4);
  cmp.use_two_streams = get_arg_bool(argc, argv, "--comm_two_streams", false);
  cmp.duty_off_us = get_arg_int(argc, argv, "--comm_duty_off_us", 0);

  WorkloadMode mode = parse_mode(mode_s);
  TransitionPlan plan = parse_plan(plan_s);

  // Open NVML + device
  NvmlCtx nv = nvml_open_device(device_id);
  std::fprintf(stderr, "GPU[%d]: %s\n", device_id, nv.name.c_str());
  std::fprintf(stderr, "Clock API (--api): %s\n", api_kind_str(api_kind));

  // Make sure CUDA context exists
  {
    cudaError_t e = cudaSetDevice(device_id);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(e));
      nvml_close();
      return 8;
    }
    cudaFree(0);
  }

  // Read current clocks
  auto cur_g = nvml_get_clock_mhz(nv.dev, NVML_CLOCK_GRAPHICS);
  auto cur_m = nvml_get_clock_mhz(nv.dev, NVML_CLOCK_MEM);
  if (!cur_g.has_value() || !cur_m.has_value()) {
    std::fprintf(stderr, "Failed to read current clocks via NVML.\n");
    nvml_close();
    return 3;
  }

  // Decide mem_fixed:
  // - For app mode, use current applications mem clock if available; else use current clockinfo mem.
  unsigned int mem_fixed = *cur_m;
  if (auto app_m = nvml_get_app_clock_mhz(nv.dev, NVML_CLOCK_MEM); app_m.has_value()) mem_fixed = *app_m;

  // Probe support: locked_gpu + app_clocks (mem lock intentionally ignored)
  bool locked_gpu_ok = false, app_ok = false;

  {
    nvmlReturn_t rG = nvmlDeviceSetGpuLockedClocks(nv.dev, *cur_g, *cur_g);
    if (rG == NVML_ERROR_NO_PERMISSION) {
      std::fprintf(stderr, "ERROR: NO_PERMISSION to set locked GPU clocks. Run as root/admin.\n");
      nvml_close();
      return 4;
    }
    locked_gpu_ok = (rG == NVML_SUCCESS);
    reset_locked_gpu(nv.dev);

    nvmlReturn_t rA = nvmlDeviceSetApplicationsClocks(nv.dev, mem_fixed, *cur_g);
    if (rA == NVML_ERROR_NO_PERMISSION) {
      app_ok = false;
    } else {
      app_ok = (rA == NVML_SUCCESS);
      if (app_ok) reset_app(nv.dev);
    }
  }

  std::fprintf(stderr, "Probe: locked_gpu=%s app=%s (mem_fixed=%u MHz)\n",
               locked_gpu_ok ? "OK" : "NO",
               app_ok ? "OK" : "NO",
               mem_fixed);

  if (!locked_gpu_ok && !app_ok) {
    std::fprintf(stderr, "[ERROR] Neither locked_gpu nor app_clocks is usable.\n");
    nvml_close();
    return 4;
  }

  // Select GPU points:
  // We enumerate supported graphics clocks under mem_fixed (since supported g clocks depend on mem).
  std::vector<unsigned int> g_all = nvml_get_supported_graphics_clocks(nv.dev, mem_fixed);

  if (g_all.empty()) {
    // fallback: try enumerate mems and pick one that yields g clocks
    auto mems = nvml_get_supported_mem_clocks(nv.dev);
    for (auto m : mems) {
      auto gg = nvml_get_supported_graphics_clocks(nv.dev, m);
      if (!gg.empty()) {
        mem_fixed = m;
        g_all = gg;
        std::fprintf(stderr, "[WARN] mem_fixed not usable for graphics clocks enum; fallback mem_fixed=%u\n", mem_fixed);
        break;
      }
    }
  }

  if (g_all.empty()) {
    std::fprintf(stderr, "ERROR: Cannot enumerate supported graphics clocks via NVML.\n");
    nvml_close();
    return 5;
  }

  std::optional<std::vector<unsigned int>> user_core;
  if (!core_list_s.empty()) user_core = parse_u32_list_csv(core_list_s);

  std::vector<unsigned int> gpu_points;
  if (user_core.has_value() && !user_core->empty()) {
    // filter by supported
    for (auto g : *user_core) {
      if (std::binary_search(g_all.begin(), g_all.end(), g)) gpu_points.push_back(g);
    }
    gpu_points = sort_unique(gpu_points);
    if (gpu_points.empty()) {
      gpu_points = pick_evenly(g_all, std::max(2, pick_core));
    }
  } else {
    gpu_points = pick_evenly(g_all, std::max(2, pick_core));
  }

  if (gpu_points.size() < 2) {
    std::fprintf(stderr, "Not enough GPU clock points selected.\n");
    nvml_close();
    return 6;
  }

  std::fprintf(stderr, "Selected GPU points (%zu) @ mem_fixed=%u:\n", gpu_points.size(), mem_fixed);
  for (auto g : gpu_points) std::fprintf(stderr, "  %u\n", g);

  std::vector<std::pair<unsigned int, unsigned int>> transitions = build_transitions(gpu_points, plan);
  if (transitions.empty()) {
    std::fprintf(stderr, "No transitions built.\n");
    nvml_close();
    return 6;
  }
  std::fprintf(stderr, "Transitions: %zu (plan=%s)\n", transitions.size(), plan_s.c_str());

  // Prepare output CSV
  bool new_file = !file_exists(out_path);
  std::ofstream of(out_path, std::ios::out | std::ios::app);
  if (!of.good()) {
    std::fprintf(stderr, "Cannot open output file: %s\n", out_path.c_str());
    nvml_close();
    return 7;
  }
  if (new_file) csv_write_header(of);

  // Start workload
  WorkloadConfig wcfg;
  wcfg.mode = mode;
  wcfg.compute = cwp;
  wcfg.comm = cmp;

  WorkloadHandle wh;
  start_workload(wcfg, &wh);

  // Warmup time
  std::fprintf(stderr, "Warmup sleep: %d ms\n", warmup_ms);
  sleep_ms(warmup_ms);

  // Warmup transitions (not logged)
  if (warmup_trans > 0 && !transitions.empty()) {
    std::fprintf(stderr, "Warmup transitions: %d\n", warmup_trans);
    auto [a, b] = transitions.front();

    SetClockApiLatencyUs api{};
    (void)set_and_settle_gpu_only(nv.dev, a, mem_fixed, poll_us, stable_n, timeout_ms,
                                  api_kind, locked_gpu_ok, app_ok, &api);

    for (int i = 0; i < warmup_trans; ++i) {
      (void)set_and_settle_gpu_only(nv.dev, b, mem_fixed, poll_us, stable_n, timeout_ms,
                                    api_kind, locked_gpu_ok, app_ok, &api);
      (void)set_and_settle_gpu_only(nv.dev, a, mem_fixed, poll_us, stable_n, timeout_ms,
                                    api_kind, locked_gpu_ok, app_ok, &api);
    }
  }

  std::fprintf(stderr, "Measuring... iters per transition=%d\n", iters);

  auto energy0 = nvml_get_energy_mj(nv.dev);

  for (const auto& tr : transitions) {
    unsigned int from = tr.first;
    unsigned int to = tr.second;

    std::fprintf(stderr, "Transition %u -> %u\n", from, to);

    for (int i = 0; i < iters; ++i) {
      // Force REAL transition each iteration: go back to 'from' (best-effort)
      {
        SetClockApiLatencyUs api_tmp{};
        (void)set_and_settle_gpu_only(nv.dev, from, mem_fixed, poll_us, stable_n, timeout_ms,
                                      api_kind, locked_gpu_ok, app_ok, &api_tmp);
      }

      // Quantify workload intensity around the transition (avg window)
      auto met = sample_metrics_avg(nv.dev, metric_window_ms, metric_interval_ms);

      // Measure transition 'to'
      SetClockApiLatencyUs api{};
      SettleResult settle = set_and_settle_gpu_only(nv.dev, to, mem_fixed, poll_us, stable_n, timeout_ms,
                                                    api_kind, locked_gpu_ok, app_ok, &api);

      auto power = nvml_get_power_mw(nv.dev);
      auto energy = nvml_get_energy_mj(nv.dev);

      csv_write_row(of,
                    now_us(),
                    device_id,
                    nv.name,
                    mode_s,
                    api_kind_str(api_kind),
                    mem_fixed,
                    from, to,
                    api,
                    settle,
                    stable_n,
                    poll_us,
                    timeout_ms,
                    met,
                    power,
                    energy);

      if (!settle.ok) {
        std::fprintf(stderr, "  [FAIL] iter=%d status=%s (final_g=%u)\n",
                     i, settle.status.c_str(), settle.final_gpu_mhz);
      }
    }
  }

  destroy_workload(&wh);

  // Reset clocks
  if (api_kind == ClockApiKind::App) {
    reset_app(nv.dev);
  } else if (api_kind == ClockApiKind::Locked) {
    reset_locked_gpu(nv.dev);
  } else {
    // auto: reset both best-effort
    reset_locked_gpu(nv.dev);
    reset_app(nv.dev);
  }

  auto energy1 = nvml_get_energy_mj(nv.dev);
  if (energy0.has_value() && energy1.has_value()) {
    std::fprintf(stderr, "Energy delta (mJ): %llu\n",
                 (unsigned long long)(*energy1 - *energy0));
  }

  nvml_close();
  std::fprintf(stderr, "Done. Output: %s\n", out_path.c_str());
  return 0;
}
