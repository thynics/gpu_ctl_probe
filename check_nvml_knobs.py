#!/usr/bin/env python3
import argparse
import ctypes
import os
import sys
from ctypes import c_uint, c_ulonglong, c_int, c_char_p, c_void_p, byref, create_string_buffer

# -------- NVML basics (ctypes) --------
def load_nvml():
    # Most common soname on Linux
    candidates = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    last = None
    for so in candidates:
        try:
            return ctypes.CDLL(so)
        except OSError as e:
            last = e
    raise RuntimeError(f"Failed to load NVML ({candidates}): {last}")

# nvmlClockType_t (from nvml.h; stable across versions)
NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM       = 1
NVML_CLOCK_MEM      = 2
NVML_CLOCK_VIDEO    = 3

# nvmlReturn_t codes we care about (values are stable; but we mostly print nvmlErrorString)
NVML_SUCCESS              = 0
NVML_ERROR_NOT_SUPPORTED  = 3
NVML_ERROR_NO_PERMISSION  = 4

def bind_nvml(nvml):
    nvml.nvmlInit_v2.restype = c_int

    nvml.nvmlShutdown.restype = c_int

    nvml.nvmlErrorString.restype = c_char_p
    nvml.nvmlErrorString.argtypes = [c_int]

    nvml.nvmlDeviceGetHandleByIndex_v2.restype = c_int
    nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [c_int, ctypes.POINTER(c_void_p)]

    nvml.nvmlDeviceGetName.restype = c_int
    nvml.nvmlDeviceGetName.argtypes = [c_void_p, ctypes.c_char_p, c_uint]

    nvml.nvmlDeviceGetClockInfo.restype = c_int
    nvml.nvmlDeviceGetClockInfo.argtypes = [c_void_p, c_int, ctypes.POINTER(c_uint)]

    nvml.nvmlDeviceGetApplicationsClock.restype = c_int
    nvml.nvmlDeviceGetApplicationsClock.argtypes = [c_void_p, c_int, ctypes.POINTER(c_uint)]

    nvml.nvmlDeviceSetGpuLockedClocks.restype = c_int
    nvml.nvmlDeviceSetGpuLockedClocks.argtypes = [c_void_p, c_uint, c_uint]

    nvml.nvmlDeviceResetGpuLockedClocks.restype = c_int
    nvml.nvmlDeviceResetGpuLockedClocks.argtypes = [c_void_p]

    nvml.nvmlDeviceSetMemoryLockedClocks.restype = c_int
    nvml.nvmlDeviceSetMemoryLockedClocks.argtypes = [c_void_p, c_uint, c_uint]

    nvml.nvmlDeviceResetMemoryLockedClocks.restype = c_int
    nvml.nvmlDeviceResetMemoryLockedClocks.argtypes = [c_void_p]

    nvml.nvmlDeviceSetApplicationsClocks.restype = c_int
    nvml.nvmlDeviceSetApplicationsClocks.argtypes = [c_void_p, c_uint, c_uint]

    nvml.nvmlDeviceResetApplicationsClocks.restype = c_int
    nvml.nvmlDeviceResetApplicationsClocks.argtypes = [c_void_p]

    nvml.nvmlDeviceGetPowerUsage.restype = c_int
    nvml.nvmlDeviceGetPowerUsage.argtypes = [c_void_p, ctypes.POINTER(c_uint)]

    nvml.nvmlDeviceGetTotalEnergyConsumption.restype = c_int
    nvml.nvmlDeviceGetTotalEnergyConsumption.argtypes = [c_void_p, ctypes.POINTER(c_ulonglong)]

    # Supported clocks enumeration used by your main.cc
    nvml.nvmlDeviceGetSupportedMemoryClocks.restype = c_int
    nvml.nvmlDeviceGetSupportedMemoryClocks.argtypes = [c_void_p, ctypes.POINTER(c_uint), ctypes.POINTER(c_uint)]

    nvml.nvmlDeviceGetSupportedGraphicsClocks.restype = c_int
    nvml.nvmlDeviceGetSupportedGraphicsClocks.argtypes = [c_void_p, c_uint, ctypes.POINTER(c_uint), ctypes.POINTER(c_uint)]

def errstr(nvml, r):
    s = nvml.nvmlErrorString(r)
    return s.decode("utf-8", errors="replace") if s else f"code={r}"

def classify(r):
    if r == NVML_SUCCESS:
        return "OK"
    if r == NVML_ERROR_NO_PERMISSION:
        return "NO_PERMISSION"
    if r == NVML_ERROR_NOT_SUPPORTED:
        return "NOT_SUPPORTED"
    return "ERROR"

def get_u32(nvml, fn, *args):
    out = c_uint(0)
    r = fn(*args, byref(out))
    return r, int(out.value)

def get_u64(nvml, fn, *args):
    out = c_ulonglong(0)
    r = fn(*args, byref(out))
    return r, int(out.value)

def get_name(nvml, dev):
    buf = create_string_buffer(128)
    r = nvml.nvmlDeviceGetName(dev, buf, 127)
    return r, buf.value.decode("utf-8", errors="replace")

def enum_supported_clocks(nvml, dev, max_buf=2048):
    # Avoid NULL-probe: allocate a large buffer once (like your "robust" approach)
    mem_count = c_uint(max_buf)
    mem_arr = (c_uint * max_buf)()
    r = nvml.nvmlDeviceGetSupportedMemoryClocks(dev, byref(mem_count), mem_arr)
    if r != NVML_SUCCESS:
        return r, []

    mems = [int(mem_arr[i]) for i in range(int(mem_count.value)) if int(mem_arr[i]) > 0]
    mems = sorted(set(mems))
    pairs = []
    # For a few mems, try enumerate graphics clocks (cap to avoid huge output)
    for m in mems[: min(5, len(mems))]:
        gr_count = c_uint(max_buf)
        gr_arr = (c_uint * max_buf)()
        rg = nvml.nvmlDeviceGetSupportedGraphicsClocks(dev, c_uint(m), byref(gr_count), gr_arr)
        if rg != NVML_SUCCESS:
            pairs.append((m, rg, []))
            continue
        grs = [int(gr_arr[i]) for i in range(int(gr_count.value)) if int(gr_arr[i]) > 0]
        grs = sorted(set(grs))
        pairs.append((m, rg, grs))
    return NVML_SUCCESS, mems, pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    nvml = load_nvml()
    bind_nvml(nvml)

    r = nvml.nvmlInit_v2()
    if r != NVML_SUCCESS:
        print(f"[FATAL] nvmlInit_v2: {classify(r)} ({errstr(nvml,r)})")
        return 2

    dev = c_void_p()
    r = nvml.nvmlDeviceGetHandleByIndex_v2(args.device, byref(dev))
    if r != NVML_SUCCESS:
        print(f"[FATAL] nvmlDeviceGetHandleByIndex_v2({args.device}): {classify(r)} ({errstr(nvml,r)})")
        nvml.nvmlShutdown()
        return 3

    rn, name = get_name(nvml, dev)
    if rn != NVML_SUCCESS:
        name = "<unknown>"
    print(f"GPU[{args.device}]: {name}")
    print(f"Run as root: {'YES' if os.geteuid()==0 else 'NO'}")

    # ---- Read current clocks (these are the "get" knobs your settle uses) ----
    rg, cur_g = get_u32(nvml, nvml.nvmlDeviceGetClockInfo, dev, c_int(NVML_CLOCK_GRAPHICS))
    rm, cur_m = get_u32(nvml, nvml.nvmlDeviceGetClockInfo, dev, c_int(NVML_CLOCK_MEM))
    rag, app_g = get_u32(nvml, nvml.nvmlDeviceGetApplicationsClock, dev, c_int(NVML_CLOCK_GRAPHICS))
    ram, app_m = get_u32(nvml, nvml.nvmlDeviceGetApplicationsClock, dev, c_int(NVML_CLOCK_MEM))

    print("\n[READ] ClockInfo / ApplicationsClock")
    print(f"  GetClockInfo(GRAPHICS): {classify(rg)}  value={cur_g} MHz  ({errstr(nvml,rg) if rg!=0 else 'OK'})")
    print(f"  GetClockInfo(MEM)     : {classify(rm)}  value={cur_m} MHz  ({errstr(nvml,rm) if rm!=0 else 'OK'})")
    print(f"  GetApplicationsClock(GRAPHICS): {classify(rag)} value={app_g} MHz ({errstr(nvml,rag) if rag!=0 else 'OK'})")
    print(f"  GetApplicationsClock(MEM)     : {classify(ram)} value={app_m} MHz ({errstr(nvml,ram) if ram!=0 else 'OK'})")

    # ---- Supported clocks enum (main.cc uses this to pick DVFS points) ----
    print("\n[ENUM] Supported clocks (NVML)")
    try:
        re, mems, sample_pairs = enum_supported_clocks(nvml, dev)
        if re != NVML_SUCCESS:
            print(f"  GetSupportedMemoryClocks: {classify(re)} ({errstr(nvml,re)})")
        else:
            print(f"  GetSupportedMemoryClocks: OK  count={len(mems)}  range=[{mems[0]}..{mems[-1]}] MHz")
            for (m, rgc, grs) in sample_pairs:
                if rgc != NVML_SUCCESS:
                    print(f"    mem={m} -> GetSupportedGraphicsClocks: {classify(rgc)} ({errstr(nvml,rgc)})")
                else:
                    if grs:
                        print(f"    mem={m} -> graphics count={len(grs)}  range=[{grs[0]}..{grs[-1]}] MHz")
                    else:
                        print(f"    mem={m} -> graphics count=0")
    except Exception as e:
        print(f"  [ERROR] enum failed: {e}")

    # ---- Power / energy knobs used in CSV ----
    print("\n[READ] Power / Energy")
    rp, p_mw = get_u32(nvml, nvml.nvmlDeviceGetPowerUsage, dev)
    print(f"  GetPowerUsage: {classify(rp)} value={p_mw} mW ({errstr(nvml,rp) if rp!=0 else 'OK'})")

    re, e_mj = get_u64(nvml, nvml.nvmlDeviceGetTotalEnergyConsumption, dev)
    print(f"  GetTotalEnergyConsumption: {classify(re)} value={e_mj} mJ ({errstr(nvml,re) if re!=0 else 'OK'})")

    # ---- SET knobs: locked GPU / locked MEM / applications clocks ----
    print("\n[SET] Clock control knobs (set to current values to minimize disturbance)")
    # locked gpu
    rlg = nvml.nvmlDeviceSetGpuLockedClocks(dev, c_uint(cur_g), c_uint(cur_g))
    print(f"  SetGpuLockedClocks({cur_g},{cur_g}): {classify(rlg)} ({errstr(nvml,rlg)})")
    rrg = nvml.nvmlDeviceResetGpuLockedClocks(dev)
    print(f"  ResetGpuLockedClocks: {classify(rrg)} ({errstr(nvml,rrg)})")

    # locked mem
    rlm = nvml.nvmlDeviceSetMemoryLockedClocks(dev, c_uint(cur_m), c_uint(cur_m))
    print(f"  SetMemoryLockedClocks({cur_m},{cur_m}): {classify(rlm)} ({errstr(nvml,rlm)})")
    rrm = nvml.nvmlDeviceResetMemoryLockedClocks(dev)
    print(f"  ResetMemoryLockedClocks: {classify(rrm)} ({errstr(nvml,rrm)})")

    # app clocks
    rac = nvml.nvmlDeviceSetApplicationsClocks(dev, c_uint(cur_m), c_uint(cur_g))
    print(f"  SetApplicationsClocks(mem={cur_m}, gpu={cur_g}): {classify(rac)} ({errstr(nvml,rac)})")
    rra = nvml.nvmlDeviceResetApplicationsClocks(dev)
    print(f"  ResetApplicationsClocks: {classify(rra)} ({errstr(nvml,rra)})")

    # ---- Summary: exactly what your main.cc cares about ----
    print("\n===== SUMMARY (for main.cc knobs) =====")
    print(f"locked_gpu : {classify(rlg)}")
    print(f"locked_mem : {classify(rlm)}")
    print(f"app_clocks : {classify(rac)}")
    print(f"power_read : {classify(rp)}")
    print(f"energy_read: {classify(re)}")
    print("======================================\n")

    if classify(rlm) == "NOT_SUPPORTED" and classify(rlg) == "OK":
        print("[HINT] Your environment supports locked GPU but NOT locked MEM.")
        print("       - main.cc with --api locked will fail if it always calls SetMemoryLockedClocks.")
        print("       - Use --api app (or auto), OR patch locked mode to GPU-only when mem-lock is unsupported.")

    nvml.nvmlShutdown()
    return 0

if __name__ == "__main__":
    sys.exit(main())
