// main.cc
// DVFS latency benchmark (V100 target) - mode: idle / compute / comm
//
// Notes:
//   - Supports LockedClocks (nvmlDeviceSetGpuLockedClocks / nvmlDeviceSetMemoryLockedClocks)
//   - Supports ApplicationsClocks (nvmlDeviceSetApplicationsClocks)
//   - Measures API latency + settle latency by polling (locked: clockinfo, app: applications clock)
//   - Workload is optional (idle = no workload).
//
// Key fixes vs your version:
//   1) Explicitly label which NVML set-call failed (mem/gpu/app) -> CSV status is informative.
//   2) Add --api auto|locked|app. auto tries locked first; on NOT_SUPPORTED falls back to app.
//   3) Each iteration performs a REAL transition (force back to `from` before measuring `to`).
//   4) In app-clock mode, settle uses nvmlDeviceGetApplicationsClock to avoid boost mismatch.
//
//

#include <cmath>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "workload.h"  // from your unified workload interface

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

static std::string nvml_err(nvmlReturn_t r) {
  return std::string(nvmlErrorString(r));
}

#define NVML_CHECK(call) do { \
  nvmlReturn_t _r = (call); \
  if (_r != NVML_SUCCESS) { \
    std::fprintf(stderr, "[NVML] %s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, nvmlErrorString(_r)); \
    std::exit(2); \
  } \
} while(0)

static bool nvml_ok(nvmlReturn_t r) { return r == NVML_SUCCESS; }

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
    size_t idx = (size_t)std::llround(t * (double)(sorted_asc.size() - 1));
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

static void print_usage() {
  std::fprintf(stderr,
    "Usage: dvfs_latency_bench [options]\n"
    "  --device N               GPU index (default 0)\n"
    "  --mode idle|compute|comm (default idle)\n"
    "  --out file.csv           output CSV (default dvfs_latency.csv)\n"
    "  --iters N                datapoints per transition (default 200)\n"
    "  --poll_us U              poll interval for settle check (default 2000)\n"
    "  --stable_n N             require N consecutive polls at target (default 5)\n"
    "  --timeout_ms T           settle timeout per transition (default 2000ms)\n"
    "  --warmup_trans N         warmup transitions (default 20)\n"
    "  --warmup_ms M            warmup time before measuring (default 1000ms)\n"
    "\n"
    "DVFS point selection:\n"
    "  --core_list a,b,c        explicit graphics clocks MHz (optional)\n"
    "  --mem_list  a,b,c        explicit memory clocks MHz (optional)\n"
    "  --pick_core K            if no core_list, pick K points evenly (default 4)\n"
    "  --pick_mem  K            if no mem_list,  pick K points evenly (default 3)\n"
    "  --transitions adjacent|extreme|all (default adjacent)\n"
    "  --order mem_then_gpu|gpu_then_mem (default mem_then_gpu)\n"
    "  --smi_csv PATH           fallback supported clocks CSV (default supported_clocks.csv)\n"
    "                          (format: mem, gr  e.g. \"877, 1530\")\n"
    "\n"
    "Clock API selection:\n"
    "  --api auto|locked|app    (default auto)\n"
    "    auto: try locked first; if NOT_SUPPORTED fallback to applications clocks.\n"
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
  NVML_CHECK(nvmlDeviceGetName(dev, name, sizeof(name)-1));
  c.name = name;

  return c;
}

static void nvml_close() {
  nvmlShutdown();
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

// Robust supported-clocks query (avoid nullptr probe).
static std::vector<unsigned int> nvml_get_supported_mem_clocks(nvmlDevice_t dev) {
  unsigned int n = 64;
  std::vector<unsigned int> mem(n);

  nvmlReturn_t r = nvmlDeviceGetSupportedMemoryClocks(dev, &n, mem.data());
  if (r == NVML_ERROR_INSUFFICIENT_SIZE) {
    mem.resize(n);
    r = nvmlDeviceGetSupportedMemoryClocks(dev, &n, mem.data());
  }

  if (!nvml_ok(r) || n == 0) {
    n = 256;
    mem.assign(n, 0);
    r = nvmlDeviceGetSupportedMemoryClocks(dev, &n, mem.data());
    if (r == NVML_ERROR_INSUFFICIENT_SIZE) {
      mem.resize(n);
      r = nvmlDeviceGetSupportedMemoryClocks(dev, &n, mem.data());
    }
  }

  if (!nvml_ok(r) || n == 0) return {};
  mem.resize(n);
  return sort_unique(mem);
}

static std::vector<unsigned int> nvml_get_supported_graphics_clocks(nvmlDevice_t dev, unsigned int mem_mhz) {
  unsigned int n = 128;
  std::vector<unsigned int> gr(n);

  nvmlReturn_t r = nvmlDeviceGetSupportedGraphicsClocks(dev, mem_mhz, &n, gr.data());
  if (r == NVML_ERROR_INSUFFICIENT_SIZE) {
    gr.resize(n);
    r = nvmlDeviceGetSupportedGraphicsClocks(dev, mem_mhz, &n, gr.data());
  }

  if (!nvml_ok(r) || n == 0) {
    n = 512;
    gr.assign(n, 0);
    r = nvmlDeviceGetSupportedGraphicsClocks(dev, mem_mhz, &n, gr.data());
    if (r == NVML_ERROR_INSUFFICIENT_SIZE) {
      gr.resize(n);
      r = nvmlDeviceGetSupportedGraphicsClocks(dev, mem_mhz, &n, gr.data());
    }
  }

  if (!nvml_ok(r) || n == 0) return {};
  gr.resize(n);
  return sort_unique(gr);
}

struct SetClockOrder {
  enum Kind { MemThenGpu, GpuThenMem } kind = MemThenGpu;
};

struct SetClockApiLatencyUs {
  uint64_t mem_us = 0;     // locked only
  uint64_t gpu_us = 0;     // locked only
  uint64_t total_us = 0;   // locked: sum wall time, app: one call wall time
};

static nvmlReturn_t set_locked_mem(nvmlDevice_t dev, unsigned int mem_mhz, uint64_t* api_us) {
  uint64_t t0 = now_us();
  nvmlReturn_t r = nvmlDeviceSetMemoryLockedClocks(dev, mem_mhz, mem_mhz);
  uint64_t t1 = now_us();
  if (api_us) *api_us = (t1 - t0);
  return r;
}

static nvmlReturn_t set_locked_gpu(nvmlDevice_t dev, unsigned int gpu_mhz, uint64_t* api_us) {
  uint64_t t0 = now_us();
  nvmlReturn_t r = nvmlDeviceSetGpuLockedClocks(dev, gpu_mhz, gpu_mhz);
  uint64_t t1 = now_us();
  if (api_us) *api_us = (t1 - t0);
  return r;
}

static nvmlReturn_t set_app_clocks(nvmlDevice_t dev, unsigned int mem_mhz, unsigned int gpu_mhz, uint64_t* api_us) {
  uint64_t t0 = now_us();
  nvmlReturn_t r = nvmlDeviceSetApplicationsClocks(dev, mem_mhz, gpu_mhz);
  uint64_t t1 = now_us();
  if (api_us) *api_us = (t1 - t0);
  return r;
}

static void reset_locked_clocks(nvmlDevice_t dev) {
  nvmlDeviceResetGpuLockedClocks(dev);
  nvmlDeviceResetMemoryLockedClocks(dev);
}

static void reset_app_clocks(nvmlDevice_t dev) {
  nvmlDeviceResetApplicationsClocks(dev);
}

enum class ClockApiKind : int {
  Auto = 0,
  Locked = 1,
  App = 2,
};

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

struct SettleResult {
  bool ok = false;
  uint64_t t_total_us = 0;           // from start (before set) to stable
  uint64_t t_after_calls_us = 0;     // from after last set call returns to stable
  int polls = 0;
  unsigned int final_gpu_mhz = 0;    // actual clockinfo
  unsigned int final_mem_mhz = 0;    // actual clockinfo
  std::string err;                  // "ok(locked)" / "set_locked_mem:Not Supported" / ...
};

// Decide settle condition:
// - locked: require actual clockinfo == target
// - app: require applications clock == target (more stable across boost)
static SettleResult set_and_settle(nvmlDevice_t dev,
                                   unsigned int target_gpu_mhz,
                                   unsigned int target_mem_mhz,
                                   SetClockOrder order,
                                   int poll_us,
                                   int stable_n,
                                   int timeout_ms,
                                   ClockApiKind api_kind,
                                   SetClockApiLatencyUs* api_lat_out) {
  SettleResult res;
  stable_n = std::max(1, stable_n);
  poll_us  = std::max(0, poll_us);

  uint64_t t_start = now_us();
  uint64_t api_mem = 0, api_gpu = 0, api_one = 0;

  // -----------------------------
  // Set clocks
  // -----------------------------
  auto do_locked = [&](std::string* which)->nvmlReturn_t {
    if (order.kind == SetClockOrder::MemThenGpu) {
      nvmlReturn_t r1 = set_locked_mem(dev, target_mem_mhz, &api_mem);
      if (!nvml_ok(r1)) { if (which) *which = "set_locked_mem"; return r1; }
      nvmlReturn_t r2 = set_locked_gpu(dev, target_gpu_mhz, &api_gpu);
      if (!nvml_ok(r2)) { if (which) *which = "set_locked_gpu"; return r2; }
      return NVML_SUCCESS;
    } else {
      nvmlReturn_t r1 = set_locked_gpu(dev, target_gpu_mhz, &api_gpu);
      if (!nvml_ok(r1)) { if (which) *which = "set_locked_gpu"; return r1; }
      nvmlReturn_t r2 = set_locked_mem(dev, target_mem_mhz, &api_mem);
      if (!nvml_ok(r2)) { if (which) *which = "set_locked_mem"; return r2; }
      return NVML_SUCCESS;
    }
  };

  auto do_app = [&](std::string* which)->nvmlReturn_t {
    if (which) *which = "set_app_clocks";
    return set_app_clocks(dev, target_mem_mhz, target_gpu_mhz, &api_one);
  };

  ClockApiKind used = api_kind;
  std::string which;

  nvmlReturn_t rset = NVML_SUCCESS;

  if (api_kind == ClockApiKind::Locked) {
    rset = do_locked(&which);
    used = ClockApiKind::Locked;
  } else if (api_kind == ClockApiKind::App) {
    rset = do_app(&which);
    used = ClockApiKind::App;
  } else {
    // Auto: try locked first; if NOT_SUPPORTED, fallback to app.
    rset = do_locked(&which);
    used = ClockApiKind::Locked;
    if (rset == NVML_ERROR_NOT_SUPPORTED) {
      // fallback
      rset = do_app(&which);
      used = ClockApiKind::App;
    }
  }

  uint64_t t_after_calls = now_us();

  if (api_lat_out) {
    if (used == ClockApiKind::Locked) {
      api_lat_out->mem_us = api_mem;
      api_lat_out->gpu_us = api_gpu;
      api_lat_out->total_us = (t_after_calls - t_start);
    } else {
      api_lat_out->mem_us = 0;
      api_lat_out->gpu_us = 0;
      api_lat_out->total_us = api_one; // one-call time
    }
  }

  if (!nvml_ok(rset)) {
    res.ok = false;
    res.err = which + ":" + nvml_err(rset);
    return res;
  }

  // -----------------------------
  // Settle polling
  // -----------------------------
  int consec = 0;
  int polls = 0;
  uint64_t deadline = t_start + (uint64_t)timeout_ms * 1000ull;

  while (true) {
    // Always record actual clocks for final_{gpu,mem}
    auto cg_act = nvml_get_clock_mhz(dev, NVML_CLOCK_GRAPHICS);
    auto cm_act = nvml_get_clock_mhz(dev, NVML_CLOCK_MEM);
    if (cg_act.has_value()) res.final_gpu_mhz = *cg_act;
    if (cm_act.has_value()) res.final_mem_mhz = *cm_act;

    bool hit = false;

    if (used == ClockApiKind::Locked) {
      if (cg_act.has_value() && cm_act.has_value() &&
          *cg_act == target_gpu_mhz && *cm_act == target_mem_mhz) {
        hit = true;
      }
    } else {
      // Applications clocks: check the configured app clocks (not the instantaneous clockinfo)
      auto cg_app = nvml_get_app_clock_mhz(dev, NVML_CLOCK_GRAPHICS);
      auto cm_app = nvml_get_app_clock_mhz(dev, NVML_CLOCK_MEM);
      if (cg_app.has_value() && cm_app.has_value() &&
          *cg_app == target_gpu_mhz && *cm_app == target_mem_mhz) {
        hit = true;
      }
    }

    polls++;
    consec = hit ? (consec + 1) : 0;

    uint64_t now = now_us();
    if (consec >= stable_n) {
      res.ok = true;
      res.polls = polls;
      res.t_total_us = now - t_start;
      res.t_after_calls_us = now - t_after_calls;
      res.err = std::string("ok(") + api_kind_str(used) + ")";
      return res;
    }

    if (now >= deadline) {
      res.ok = false;
      res.polls = polls;
      res.t_total_us = now - t_start;
      res.t_after_calls_us = now - t_after_calls;
      res.err = std::string("timeout(") + api_kind_str(used) + ")";
      return res;
    }

    if (poll_us > 0) sleep_us(poll_us);
  }
}

// -----------------------------
// DVFS point / supported DB
// -----------------------------
struct DvfsPoint {
  unsigned int gpu_mhz = 0;
  unsigned int mem_mhz = 0;
};

static std::string point_str(const DvfsPoint& p) {
  std::ostringstream os;
  os << p.gpu_mhz << "@" << p.mem_mhz;
  return os.str();
}

struct SupportedDB {
  // mem -> sorted unique graphics clocks
  std::map<unsigned int, std::vector<unsigned int>> g_by_mem;

  std::vector<unsigned int> mems() const {
    std::vector<unsigned int> out;
    out.reserve(g_by_mem.size());
    for (auto& kv : g_by_mem) out.push_back(kv.first);
    return out;
  }

  size_t size_pairs() const {
    size_t s = 0;
    for (auto& kv : g_by_mem) s += kv.second.size();
    return s;
  }
};

static SupportedDB build_db_from_nvml(nvmlDevice_t dev) {
  SupportedDB db;
  auto mems = nvml_get_supported_mem_clocks(dev);
  for (auto m : mems) {
    auto gs = nvml_get_supported_graphics_clocks(dev, m);
    if (!gs.empty()) db.g_by_mem[m] = gs;
  }
  return db;
}

static bool parse_mem_gr_line(const std::string& line, unsigned int* mem, unsigned int* gr) {
  // formats tolerated:
  //   "877, 1530"
  //   "877,1530"
  //   "877 1530"
  if (line.empty()) return false;
  char* end = nullptr;
  unsigned long m = std::strtoul(line.c_str(), &end, 10);
  if (end == line.c_str()) return false;
  while (*end && (*end == ' ' || *end == ',' || *end == '\t')) ++end;
  if (!*end) return false;
  unsigned long g = std::strtoul(end, nullptr, 10);
  if (m == 0 || g == 0) return false;
  *mem = (unsigned int)m;
  *gr  = (unsigned int)g;
  return true;
}

static SupportedDB build_db_from_smi_csv_file(const std::string& path) {
  SupportedDB db;
  std::ifstream in(path);
  if (!in.good()) return db;
  std::string line;
  while (std::getline(in, line)) {
    unsigned int mem=0, gr=0;
    if (!parse_mem_gr_line(line, &mem, &gr)) continue;
    db.g_by_mem[mem].push_back(gr);
  }
  for (auto& kv : db.g_by_mem) kv.second = sort_unique(kv.second);
  return db;
}

static SupportedDB build_db_from_smi_popen() {
  SupportedDB db;
  const char* cmd = "nvidia-smi --query-supported-clocks=mem,gr --format=csv,noheader,nounits";
  FILE* fp = popen(cmd, "r");
  if (!fp) return db;

  char buf[512];
  while (std::fgets(buf, sizeof(buf), fp)) {
    std::string line(buf);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    unsigned int mem=0, gr=0;
    if (!parse_mem_gr_line(line, &mem, &gr)) continue;
    db.g_by_mem[mem].push_back(gr);
  }
  pclose(fp);

  for (auto& kv : db.g_by_mem) kv.second = sort_unique(kv.second);
  return db;
}

static std::vector<DvfsPoint> select_points_from_db(const SupportedDB& db,
                                                    std::optional<std::vector<unsigned int>> user_core_list,
                                                    std::optional<std::vector<unsigned int>> user_mem_list,
                                                    int pick_core,
                                                    int pick_mem) {
  std::vector<unsigned int> mem_all = db.mems();
  if (mem_all.empty()) return {};

  std::vector<unsigned int> mem_sel;
  if (user_mem_list.has_value() && !user_mem_list->empty()) {
    mem_sel = sort_unique(*user_mem_list);
  } else {
    mem_sel = pick_evenly(mem_all, std::max(1, pick_mem));
  }

  std::vector<DvfsPoint> points;

  for (unsigned int mem_mhz : mem_sel) {
    auto it = db.g_by_mem.find(mem_mhz);
    if (it == db.g_by_mem.end() || it->second.empty()) continue;

    const std::vector<unsigned int>& g_all = it->second;
    std::vector<unsigned int> g_sel;

    if (user_core_list.has_value() && !user_core_list->empty()) {
      std::vector<unsigned int> gl = sort_unique(*user_core_list);
      for (auto g : gl) {
        if (std::binary_search(g_all.begin(), g_all.end(), g)) g_sel.push_back(g);
      }
      g_sel = sort_unique(g_sel);
      if (g_sel.empty()) {
        g_sel = pick_evenly(g_all, std::max(1, pick_core));
      }
    } else {
      g_sel = pick_evenly(g_all, std::max(1, pick_core));
    }

    for (unsigned int g : g_sel) {
      points.push_back({g, mem_mhz});
    }
  }

  std::sort(points.begin(), points.end(), [](const DvfsPoint& a, const DvfsPoint& b){
    if (a.mem_mhz != b.mem_mhz) return a.mem_mhz < b.mem_mhz;
    return a.gpu_mhz < b.gpu_mhz;
  });
  points.erase(std::unique(points.begin(), points.end(), [](const DvfsPoint& a, const DvfsPoint& b){
    return a.gpu_mhz==b.gpu_mhz && a.mem_mhz==b.mem_mhz;
  }), points.end());

  return points;
}

// -----------------------------
// Transition plan
// -----------------------------
enum class TransitionPlan : int {
  Adjacent = 0,
  Extreme  = 1,
  AllPairs = 2,
};

static std::vector<std::pair<DvfsPoint, DvfsPoint>>
build_transitions(const std::vector<DvfsPoint>& points, TransitionPlan plan) {
  std::vector<std::pair<DvfsPoint, DvfsPoint>> out;
  if (points.size() < 2) return out;

  if (plan == TransitionPlan::AllPairs) {
    for (size_t i = 0; i < points.size(); ++i) {
      for (size_t j = 0; j < points.size(); ++j) {
        if (i == j) continue;
        out.push_back({points[i], points[j]});
      }
    }
    return out;
  }

  std::vector<DvfsPoint> ps = points;
  std::sort(ps.begin(), ps.end(), [](const DvfsPoint& a, const DvfsPoint& b){
    if (a.mem_mhz != b.mem_mhz) return a.mem_mhz < b.mem_mhz;
    return a.gpu_mhz < b.gpu_mhz;
  });

  if (plan == TransitionPlan::Extreme) {
    out.push_back({ps.front(), ps.back()});
    out.push_back({ps.back(), ps.front()});
    return out;
  }

  // Adjacent within same mem (gpu changes)
  for (size_t i = 1; i < ps.size(); ++i) {
    if (ps[i].mem_mhz == ps[i-1].mem_mhz) {
      out.push_back({ps[i-1], ps[i]});
      out.push_back({ps[i], ps[i-1]});
    }
  }

  // Adjacent within same gpu (mem changes)
  for (size_t i = 0; i < ps.size(); ++i) {
    for (size_t j = i + 1; j < ps.size(); ++j) {
      if (ps[i].gpu_mhz == ps[j].gpu_mhz && ps[i].mem_mhz != ps[j].mem_mhz) {
        bool adjacent = true;
        for (size_t k = 0; k < ps.size(); ++k) {
          if (ps[k].gpu_mhz != ps[i].gpu_mhz) continue;
          if (ps[k].mem_mhz > ps[i].mem_mhz && ps[k].mem_mhz < ps[j].mem_mhz) { adjacent = false; break; }
          if (ps[k].mem_mhz > ps[j].mem_mhz && ps[k].mem_mhz < ps[i].mem_mhz) { adjacent = false; break; }
        }
        if (adjacent) {
          out.push_back({ps[i], ps[j]});
          out.push_back({ps[j], ps[i]});
        }
      }
    }
  }

  auto key = [](const std::pair<DvfsPoint,DvfsPoint>& e){
    return ((uint64_t)e.first.gpu_mhz << 48) ^
           ((uint64_t)e.first.mem_mhz << 32) ^
           ((uint64_t)e.second.gpu_mhz << 16) ^
           ((uint64_t)e.second.mem_mhz);
  };
  std::sort(out.begin(), out.end(), [&](auto& a, auto& b){ return key(a) < key(b); });
  out.erase(std::unique(out.begin(), out.end(), [&](auto& a, auto& b){ return key(a)==key(b); }), out.end());
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
  of << "ts_us,device_id,gpu_name,mode,"
        "from_gpu_mhz,from_mem_mhz,to_gpu_mhz,to_mem_mhz,order,"
        "api_mem_us,api_gpu_us,api_total_us,"
        "settle_total_us,settle_after_calls_us,polls,stable_n,poll_us,timeout_ms,"
        "final_gpu_mhz,final_mem_mhz,"
        "power_mw,energy_mj,status\n";
}

static void csv_write_row(std::ofstream& of,
                          uint64_t ts_us,
                          int device_id,
                          const std::string& gpu_name,
                          const std::string& mode,
                          const DvfsPoint& from,
                          const DvfsPoint& to,
                          const std::string& order,
                          const SetClockApiLatencyUs& api,
                          const SettleResult& settle,
                          int stable_n,
                          int poll_us,
                          int timeout_ms,
                          const std::optional<unsigned int>& power_mw,
                          const std::optional<unsigned long long>& energy_mj) {
  of << ts_us << ","
     << device_id << ","
     << "\"" << gpu_name << "\"" << ","
     << mode << ","
     << from.gpu_mhz << "," << from.mem_mhz << ","
     << to.gpu_mhz << "," << to.mem_mhz << ","
     << order << ","
     << api.mem_us << "," << api.gpu_us << "," << api.total_us << ","
     << settle.t_total_us << "," << settle.t_after_calls_us << ","
     << settle.polls << "," << stable_n << "," << poll_us << "," << timeout_ms << ","
     << settle.final_gpu_mhz << "," << settle.final_mem_mhz << ","
     << (power_mw.has_value() ? std::to_string(*power_mw) : "") << ","
     << (energy_mj.has_value() ? std::to_string(*energy_mj) : "") << ","
     << settle.err
     << "\n";
  of.flush();
}

// -----------------------------
// Workload config from CLI
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

static TransitionPlan parse_plan(const std::string& s) {
  if (s == "adjacent") return TransitionPlan::Adjacent;
  if (s == "extreme") return TransitionPlan::Extreme;
  if (s == "all") return TransitionPlan::AllPairs;
  return TransitionPlan::Adjacent;
}

static SetClockOrder parse_order(const std::string& s) {
  SetClockOrder o;
  if (s == "gpu_then_mem") o.kind = SetClockOrder::GpuThenMem;
  else o.kind = SetClockOrder::MemThenGpu;
  return o;
}

static std::string order_str(const SetClockOrder& o) {
  return (o.kind == SetClockOrder::MemThenGpu) ? "mem_then_gpu" : "gpu_then_mem";
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
  std::string out_path = get_arg(argc, argv, "--out", "dvfs_latency.csv");

  int iters = get_arg_int(argc, argv, "--iters", 200);
  int poll_us = get_arg_int(argc, argv, "--poll_us", 2000);
  int stable_n = get_arg_int(argc, argv, "--stable_n", 5);
  int timeout_ms = get_arg_int(argc, argv, "--timeout_ms", 2000);
  int warmup_trans = get_arg_int(argc, argv, "--warmup_trans", 20);
  int warmup_ms = get_arg_int(argc, argv, "--warmup_ms", 1000);

  int pick_core = get_arg_int(argc, argv, "--pick_core", 4);
  int pick_mem = get_arg_int(argc, argv, "--pick_mem", 3);

  std::string core_list_s = get_arg(argc, argv, "--core_list", "");
  std::string mem_list_s  = get_arg(argc, argv, "--mem_list", "");

  std::string plan_s = get_arg(argc, argv, "--transitions", "adjacent");
  std::string order_s = get_arg(argc, argv, "--order", "mem_then_gpu");

  std::string smi_csv_path = get_arg(argc, argv, "--smi_csv", "supported_clocks.csv");

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
  SetClockOrder order = parse_order(order_s);

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

  // Read current clocks (for probe)
  auto cur_g = nvml_get_clock_mhz(nv.dev, NVML_CLOCK_GRAPHICS);
  auto cur_m = nvml_get_clock_mhz(nv.dev, NVML_CLOCK_MEM);
  if (!cur_g.has_value() || !cur_m.has_value()) {
    std::fprintf(stderr, "Failed to read current clocks via NVML.\n");
    nvml_close();
    return 3;
  }

  // Permission / support probe: locked gpu + locked mem + app
  bool locked_gpu_ok = false, locked_mem_ok = false, app_ok = false;

  {
    nvmlReturn_t rG = nvmlDeviceSetGpuLockedClocks(nv.dev, *cur_g, *cur_g);
    if (rG == NVML_ERROR_NO_PERMISSION) {
      std::fprintf(stderr, "ERROR: NO_PERMISSION to set locked GPU clocks. Run as root/admin.\n");
      nvml_close();
      return 4;
    }
    locked_gpu_ok = (rG == NVML_SUCCESS);

    nvmlReturn_t rM = nvmlDeviceSetMemoryLockedClocks(nv.dev, *cur_m, *cur_m);
    if (rM == NVML_ERROR_NO_PERMISSION) {
      std::fprintf(stderr, "ERROR: NO_PERMISSION to set locked MEM clocks. Run as root/admin.\n");
      nvml_close();
      return 4;
    }
    locked_mem_ok = (rM == NVML_SUCCESS);

    reset_locked_clocks(nv.dev);

    nvmlReturn_t rA = nvmlDeviceSetApplicationsClocks(nv.dev, *cur_m, *cur_g);
    if (rA == NVML_ERROR_NO_PERMISSION) {
      app_ok = false;
    } else {
      app_ok = (rA == NVML_SUCCESS);
      if (app_ok) reset_app_clocks(nv.dev);
    }
  }

  std::fprintf(stderr, "Probe: locked_gpu=%s locked_mem=%s app=%s\n",
               locked_gpu_ok ? "OK" : "NO",
               locked_mem_ok ? "OK" : "NO",
               app_ok ? "OK" : "NO");

  // If user chose auto, but locked incomplete, fallback preference.
  if (api_kind == ClockApiKind::Auto) {
    if (!(locked_gpu_ok && locked_mem_ok)) {
      if (app_ok) {
        std::fprintf(stderr, "[INFO] Auto API: locked not fully available -> will fallback to app when needed.\n");
      } else {
        std::fprintf(stderr, "[ERROR] Auto API: neither locked nor app is usable.\n");
        nvml_close();
        return 4;
      }
    }
  }

  // User lists
  std::optional<std::vector<unsigned int>> user_core;
  std::optional<std::vector<unsigned int>> user_mem;
  if (!core_list_s.empty()) user_core = parse_u32_list_csv(core_list_s);
  if (!mem_list_s.empty())  user_mem  = parse_u32_list_csv(mem_list_s);

  // Build supported-clocks DB:
  SupportedDB db = build_db_from_nvml(nv.dev);
  if (db.g_by_mem.empty()) {
    std::fprintf(stderr, "[WARN] NVML supported clocks enumeration returned empty. Trying nvidia-smi fallbacks...\n");
    if (file_exists(smi_csv_path)) {
      db = build_db_from_smi_csv_file(smi_csv_path);
      if (!db.g_by_mem.empty()) {
        std::fprintf(stderr, "[INFO] Loaded supported clocks from CSV: %s (pairs=%zu)\n",
                     smi_csv_path.c_str(), db.size_pairs());
      }
    }
    if (db.g_by_mem.empty()) {
      db = build_db_from_smi_popen();
      if (!db.g_by_mem.empty()) {
        std::fprintf(stderr, "[INFO] Loaded supported clocks via popen(nvidia-smi) (pairs=%zu)\n", db.size_pairs());
      }
    }
  }

  if (db.g_by_mem.empty()) {
    std::fprintf(stderr,
      "ERROR: Cannot obtain supported clocks from NVML or nvidia-smi.\n"
      "Try manually generating supported_clocks.csv:\n"
      "  nvidia-smi --query-supported-clocks=mem,gr --format=csv,noheader,nounits > supported_clocks.csv\n"
      "and run with --smi_csv supported_clocks.csv\n");
    nvml_close();
    return 5;
  }

  // Select DVFS points from DB
  std::vector<DvfsPoint> points = select_points_from_db(db, user_core, user_mem, pick_core, pick_mem);
  if (points.size() < 2) {
    std::fprintf(stderr,
      "Not enough DVFS points selected.\n"
      "Tips:\n"
      "  - Try specify both --mem_list and --core_list\n"
      "  - Or increase --pick_mem / --pick_core\n"
      "  - Or provide correct --smi_csv (default supported_clocks.csv)\n");
    auto mems = db.mems();
    if (!mems.empty()) std::fprintf(stderr, "  DB mem range: [%u .. %u], mem count=%zu\n", mems.front(), mems.back(), mems.size());
    nvml_close();
    return 5;
  }

  std::fprintf(stderr, "Selected DVFS points (%zu):\n", points.size());
  for (auto& p : points) std::fprintf(stderr, "  %s\n", point_str(p).c_str());

  std::vector<std::pair<DvfsPoint, DvfsPoint>> transitions = build_transitions(points, plan);
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
  if (warmup_ms > 0) {
    std::fprintf(stderr, "Warmup sleep: %d ms\n", warmup_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(warmup_ms));
  }

  // Warmup DVFS transitions (not logged)
  if (warmup_trans > 0 && !transitions.empty()) {
    std::fprintf(stderr, "Warmup transitions: %d\n", warmup_trans);
    auto [a, b] = transitions.front();

    SetClockApiLatencyUs api{};
    (void)set_and_settle(nv.dev, a.gpu_mhz, a.mem_mhz, order, poll_us, stable_n, timeout_ms, api_kind, &api);

    for (int i = 0; i < warmup_trans; ++i) {
      (void)set_and_settle(nv.dev, b.gpu_mhz, b.mem_mhz, order, poll_us, stable_n, timeout_ms, api_kind, &api);
      (void)set_and_settle(nv.dev, a.gpu_mhz, a.mem_mhz, order, poll_us, stable_n, timeout_ms, api_kind, &api);
    }
  }

  std::fprintf(stderr, "Measuring... iters per transition=%d\n", iters);

  auto energy0 = nvml_get_energy_mj(nv.dev);

  for (const auto& tr : transitions) {
    const DvfsPoint from = tr.first;
    const DvfsPoint to   = tr.second;

    std::fprintf(stderr, "Transition %s -> %s\n", point_str(from).c_str(), point_str(to).c_str());

    for (int i = 0; i < iters; ++i) {
      // IMPORTANT: force real transition each iteration
      {
        SetClockApiLatencyUs api_tmp{};
        (void)set_and_settle(nv.dev, from.gpu_mhz, from.mem_mhz, order,
                             poll_us, stable_n, timeout_ms, api_kind, &api_tmp);
      }

      SetClockApiLatencyUs api{};
      SettleResult settle = set_and_settle(nv.dev, to.gpu_mhz, to.mem_mhz, order,
                                           poll_us, stable_n, timeout_ms, api_kind, &api);

      auto power = nvml_get_power_mw(nv.dev);
      auto energy = nvml_get_energy_mj(nv.dev);

      csv_write_row(of,
                    now_us(),
                    device_id,
                    nv.name,
                    mode_s,
                    from, to,
                    order_str(order),
                    api,
                    settle,
                    stable_n,
                    poll_us,
                    timeout_ms,
                    power,
                    energy);

      // If it is consistently failing, print to stderr too (helps debug immediately)
      if (!settle.ok) {
        std::fprintf(stderr, "  [FAIL] iter=%d status=%s (final_g=%u final_m=%u)\n",
                     i, settle.err.c_str(), settle.final_gpu_mhz, settle.final_mem_mhz);
      }
    }
  }

  destroy_workload(&wh);

  // Reset clocks according to chosen API
  if (api_kind == ClockApiKind::App) {
    reset_app_clocks(nv.dev);
  } else if (api_kind == ClockApiKind::Locked) {
    reset_locked_clocks(nv.dev);
  } else {
    // auto: reset both best-effort
    reset_locked_clocks(nv.dev);
    reset_app_clocks(nv.dev);
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
