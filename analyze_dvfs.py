#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_dvfs.py
================
Analyze dvfs_latency_bench CSV and produce:
  - Summary tables (CSV)
  - A markdown report (report.md)
  - Line charts (PNG): ECDFs, tail curves, timeseries drift

Works with the CSV format emitted by your main.cc:
ts_us,device_id,gpu_name,mode,from_gpu_mhz,from_mem_mhz,to_gpu_mhz,to_mem_mhz,order,
api_mem_us,api_gpu_us,api_total_us,settle_total_us,settle_after_calls_us,polls,stable_n,poll_us,timeout_ms,
final_gpu_mhz,final_mem_mhz,power_mw,energy_mj,status

Usage examples:
  python3 analyze_dvfs.py idle.csv --outdir out_idle
  python3 analyze_dvfs.py idle.csv compute.csv comm.csv --outdir out_all
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


PCTS = [0.5, 0.9, 0.95, 0.99]


def _fmt_us(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    if x >= 1000.0:
        return f"{x/1000.0:.3f} ms"
    return f"{x:.1f} us"


def _dir(a: int, b: int) -> str:
    if b > a:
        return "up"
    if b < a:
        return "down"
    return "flat"


def _transition_type(dg: int, dm: int) -> str:
    g = (dg != 0)
    m = (dm != 0)
    if g and m:
        return "both"
    if g:
        return "gpu_only"
    if m:
        return "mem_only"
    return "noop"


def _percentiles(s: pd.Series, pcts=PCTS) -> Dict[str, float]:
    out = {}
    if s is None or s.dropna().empty:
        for p in pcts:
            out[f"p{int(p*100)}"] = float("nan")
        return out
    qs = s.quantile(pcts, interpolation="linear")
    for p, v in zip(pcts, qs.values):
        out[f"p{int(p*100)}"] = float(v)
    return out


def _ecdf_xy(values: pd.Series) -> Tuple[List[float], List[float]]:
    v = values.dropna().astype(float).sort_values().values
    if len(v) == 0:
        return [], []
    y = [(i + 1) / len(v) for i in range(len(v))]
    return v.tolist(), y


def _ensure_cols_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_csvs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d.columns = [str(c).strip() for c in d.columns]
        d["__source__"] = os.path.basename(p)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    num_cols = [
        "ts_us", "device_id",
        "from_gpu_mhz", "from_mem_mhz", "to_gpu_mhz", "to_mem_mhz",
        "api_mem_us", "api_gpu_us", "api_total_us",
        "settle_total_us", "settle_after_calls_us",
        "polls", "stable_n", "poll_us", "timeout_ms",
        "final_gpu_mhz", "final_mem_mhz",
        "power_mw", "energy_mj",
    ]
    _ensure_cols_numeric(df, num_cols)

    for c in ["gpu_name", "mode", "order", "status"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    df["ok"] = (df.get("status", "") == "ok")

    df["delta_gpu_mhz"] = df["to_gpu_mhz"] - df["from_gpu_mhz"]
    df["delta_mem_mhz"] = df["to_mem_mhz"] - df["from_mem_mhz"]
    df["dir_gpu"] = df.apply(lambda r: _dir(int(r["from_gpu_mhz"]), int(r["to_gpu_mhz"])), axis=1)
    df["dir_mem"] = df.apply(lambda r: _dir(int(r["from_mem_mhz"]), int(r["to_mem_mhz"])), axis=1)
    df["transition_type"] = df.apply(
        lambda r: _transition_type(int(r["delta_gpu_mhz"]), int(r["delta_mem_mhz"])),
        axis=1
    )

    df["pair"] = (
        df["from_gpu_mhz"].astype("Int64").astype(str) + "@"
        + df["from_mem_mhz"].astype("Int64").astype(str)
        + "->"
        + df["to_gpu_mhz"].astype("Int64").astype(str) + "@"
        + df["to_mem_mhz"].astype("Int64").astype(str)
    )

    if "energy_mj" in df.columns:
        df = df.sort_values(["device_id", "__source__", "ts_us"], kind="mergesort")
        df["energy_mj_delta"] = df.groupby(["device_id", "__source__"])["energy_mj"].diff()
    else:
        df["energy_mj_delta"] = float("nan")

    return df


def summarize(df: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["rows"] = int(len(df))
    out["rows_ok"] = int(df["ok"].sum())
    out["ok_rate"] = float(df["ok"].mean()) if len(df) else float("nan")
    out["unique_pairs"] = int(df["pair"].nunique())

    if df["ts_us"].notna().any():
        span_s = (df["ts_us"].max() - df["ts_us"].min()) / 1e6
        out["timespan_s"] = float(span_s)
    else:
        out["timespan_s"] = float("nan")

    ok = df[df["ok"]].copy()
    for col in ["api_total_us", "api_gpu_us", "api_mem_us", "settle_after_calls_us", "settle_total_us"]:
        if col in ok.columns:
            out[col] = {
                "mean": float(ok[col].mean()) if ok[col].notna().any() else float("nan"),
                "std": float(ok[col].std()) if ok[col].notna().any() else float("nan"),
                "min": float(ok[col].min()) if ok[col].notna().any() else float("nan"),
                "max": float(ok[col].max()) if ok[col].notna().any() else float("nan"),
                **_percentiles(ok[col]),
            }

    if "power_mw" in ok.columns and ok["power_mw"].notna().any():
        out["power_mw"] = {"mean": float(ok["power_mw"].mean()), **_percentiles(ok["power_mw"])}

    if "energy_mj_delta" in ok.columns and ok["energy_mj_delta"].notna().any():
        ed = ok["energy_mj_delta"]
        ed = ed[ed >= 0]
        if len(ed):
            out["energy_mj_delta"] = {"mean": float(ed.mean()), **_percentiles(ed)}
    return out


def group_table(df: pd.DataFrame, group_cols: List[str], metric: str) -> pd.DataFrame:
    ok = df[df["ok"]].copy()
    if ok.empty:
        return pd.DataFrame()

    g = ok.groupby(group_cols, dropna=False)[metric]
    res = g.agg(["count", "mean", "median"]).reset_index()
    for p in PCTS:
        res[f"p{int(p*100)}"] = g.quantile(p).reset_index(drop=True)

    tot = df.groupby(group_cols, dropna=False).size().reset_index(name="rows_total")
    okc = df[df["ok"]].groupby(group_cols, dropna=False).size().reset_index(name="rows_ok")
    res = res.merge(tot, on=group_cols, how="left").merge(okc, on=group_cols, how="left")
    res["rows_ok"] = res["rows_ok"].fillna(0).astype(int)
    res["fail_rate"] = 1.0 - (res["rows_ok"] / res["rows_total"].clip(lower=1))

    if "p95" in res.columns:
        res = res.sort_values("p95", ascending=False, kind="mergesort")
    return res


def write_markdown_report(outdir: Path, df: pd.DataFrame, gsum: Dict[str, object]) -> None:
    lines = []
    lines.append("# DVFS latency analysis report\n")
    lines.append(f"- Rows: {gsum['rows']} | OK: {gsum['rows_ok']} | OK-rate: {gsum['ok_rate']:.3f}\n")
    if not math.isnan(gsum.get("timespan_s", float('nan'))):
        lines.append(f"- Timespan: {gsum['timespan_s']:.2f} s\n")
    lines.append(f"- Unique transitions (pair): {gsum['unique_pairs']}\n")

    def add_block(name: str):
        blk = gsum.get(name)
        if not isinstance(blk, dict):
            return
        lines.append(f"## {name}\n")
        lines.append(f"- mean: {_fmt_us(blk.get('mean', float('nan')))}\n")
        lines.append(f"- p50/p90/p95/p99: {_fmt_us(blk.get('p50', float('nan')))} / {_fmt_us(blk.get('p90', float('nan')))} / {_fmt_us(blk.get('p95', float('nan')))} / {_fmt_us(blk.get('p99', float('nan')))}\n")
        lines.append(f"- max: {_fmt_us(blk.get('max', float('nan')))}\n")

    add_block("api_total_us")
    add_block("settle_after_calls_us")
    add_block("settle_total_us")

    lines.append("## Derived quantitative indices\n")
    ok = df[df["ok"]].copy()
    if not ok.empty:
        # GPU-dir asymmetry per type
        for typ in ["gpu_only", "mem_only", "both"]:
            sub = ok[ok["transition_type"] == typ]
            if sub.empty:
                continue
            up = sub[sub["dir_gpu"] == "up"]["settle_total_us"]
            down = sub[sub["dir_gpu"] == "down"]["settle_total_us"]
            if not up.dropna().empty and not down.dropna().empty:
                up_p95 = float(up.quantile(0.95))
                down_p95 = float(down.quantile(0.95))
                ratio = up_p95 / max(1e-9, down_p95)
                lines.append(f"- GPU-dir asymmetry (type={typ}) p95(up)/p95(down): {ratio:.3f}\n")

        # Order effect
        if "order" in ok.columns and ok["order"].nunique() >= 2:
            a = ok[ok["order"] == "mem_then_gpu"]["settle_total_us"]
            b = ok[ok["order"] == "gpu_then_mem"]["settle_total_us"]
            if not a.dropna().empty and not b.dropna().empty:
                a95 = float(a.quantile(0.95))
                b95 = float(b.quantile(0.95))
                lines.append(f"- Order effect p95(mem_then_gpu)/p95(gpu_then_mem): {a95/max(1e-9,b95):.3f}\n")

        # Wait-share
        if "api_total_us" in ok.columns and "settle_total_us" in ok.columns:
            r = (ok["settle_total_us"] - ok["api_total_us"]) / ok["settle_total_us"].clip(lower=1e-9)
            lines.append(f"- Wait-share mean={float(r.mean()):.3f}, p95={float(r.quantile(0.95)):.3f}\n")

    bad = df[~df["ok"]]
    if not bad.empty:
        lines.append("## Failures\n")
        top = bad.groupby(["mode", "order", "status"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
        lines.append(top.head(20).to_markdown(index=False))
        lines.append("\n")

    lines.append("## Output files\n")
    lines.append("- summary_tables/: grouped CSV tables (p50/p95/p99, fail_rate)\n")
    lines.append("- figures/: PNG figures (ECDF, tails, drift)\n")

    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_ecdf(outpath: Path, df: pd.DataFrame, metric: str, label_col: str, title: str) -> None:
    ok = df[df["ok"]].copy()
    if ok.empty:
        return

    plt.figure()
    for label, sub in ok.groupby(label_col, dropna=False):
        x, y = _ecdf_xy(sub[metric])
        if len(x) == 0:
            continue
        plt.plot(x, y, label=str(label))
    plt.xlabel(metric)
    plt.ylabel("ECDF")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_tail_curve(outpath: Path, df: pd.DataFrame, metric: str, group_cols: List[str], topk: int, title: str) -> None:
    ok = df[df["ok"]].copy()
    if ok.empty:
        return

    g = ok.groupby(group_cols, dropna=False)[metric].quantile(0.95).reset_index(name="p95")
    g = g.sort_values("p95", ascending=False).head(topk)
    if g.empty:
        return

    x = list(range(1, len(g) + 1))
    y = g["p95"].astype(float).tolist()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(f"ranked groups (top {topk})")
    plt.ylabel(f"p95({metric})")
    plt.title(title)

    labels = []
    for _, row in g.iterrows():
        labels.append(" | ".join([f"{c}={row[c]}" for c in group_cols]))
    for i in range(min(10, len(x))):
        plt.annotate(labels[i], (x[i], y[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_timeseries(outpath: Path, df: pd.DataFrame, metric: str, title: str) -> None:
    ok = df[df["ok"]].copy()
    if ok.empty or ok["ts_us"].isna().all() or metric not in ok.columns:
        return

    ok = ok.sort_values("ts_us", kind="mergesort")
    t0 = ok["ts_us"].iloc[0]
    x = ((ok["ts_us"] - t0) / 1e6).astype(float)
    y = pd.to_numeric(ok[metric], errors="coerce").astype(float)

    plt.figure()
    plt.plot(x, y, linewidth=1.0)
    plt.xlabel("time since start (s)")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="input dvfs_latency_bench csv(s)")
    ap.add_argument("--outdir", default="dvfs_analysis_out", help="output directory")
    ap.add_argument("--topk", type=int, default=30, help="top-k groups for tail plots")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "summary_tables").mkdir(parents=True, exist_ok=True)

    df = load_csvs(args.csv)
    gsum = summarize(df)

    tables: Dict[str, pd.DataFrame] = {}
    tables["by_mode"] = group_table(df, ["mode"], "settle_total_us")
    tables["by_mode_order"] = group_table(df, ["mode", "order"], "settle_total_us")
    tables["by_type_dir"] = group_table(df, ["mode", "order", "transition_type", "dir_gpu", "dir_mem"], "settle_total_us")
    tables["by_pair"] = group_table(df, ["mode", "order", "pair", "transition_type", "dir_gpu", "dir_mem"], "settle_total_us")

    for name, t in tables.items():
        if t is not None and not t.empty:
            t.to_csv(outdir / "summary_tables" / f"{name}.csv", index=False)

    bad = df[~df["ok"]].copy()
    if not bad.empty:
        fail = bad.groupby(["__source__", "mode", "order", "status"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
        fail.to_csv(outdir / "summary_tables" / "failures.csv", index=False)

    write_markdown_report(outdir, df, gsum)

    # figures (line charts)
    plot_ecdf(outdir / "figures" / "ecdf_settle_by_mode.png",
              df, "settle_total_us", "mode",
              "ECDF of settle_total_us by mode")

    plot_ecdf(outdir / "figures" / "ecdf_settle_by_order.png",
              df, "settle_total_us", "order",
              "ECDF of settle_total_us by DVFS set order")

    plot_tail_curve(outdir / "figures" / "tail_worst_pairs_p95.png",
                    df, "settle_total_us",
                    ["mode", "order", "pair"],
                    topk=args.topk,
                    title="Worst transition pairs by p95(settle_total_us)")

    plot_tail_curve(outdir / "figures" / "tail_type_dir_p95.png",
                    df, "settle_total_us",
                    ["mode", "order", "transition_type", "dir_gpu", "dir_mem"],
                    topk=min(args.topk, 24),
                    title="Worst buckets by p95(settle_total_us): type + direction")

    plot_timeseries(outdir / "figures" / "timeseries_settle_total.png",
                    df, "settle_total_us",
                    "settle_total_us over time (drift / instability check)")

    plot_timeseries(outdir / "figures" / "timeseries_power_mw.png",
                    df, "power_mw",
                    "power_mw over time")

    print(f"[OK] Wrote report: {outdir/'report.md'}")
    print(f"[OK] Wrote tables: {outdir/'summary_tables'}")
    print(f"[OK] Wrote figures: {outdir/'figures'}")


if __name__ == "__main__":
    main()
