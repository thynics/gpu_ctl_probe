#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_dvfs_csvs.py

Reads dvfs_{type}_{workload}.csv files (type: compute/comm; workload: low/mid/high),
produces:
- summary tables (CSV)
- plots (PNG)
- a Markdown report

Usage:
  python analyze_dvfs_csvs.py --input_dir . --out_dir dvfs_report

Dependencies:
  pip install pandas numpy matplotlib
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config / Helpers
# -----------------------------

TYPE_SET = {"compute", "comm"}
WORKLOAD_SET = {"low", "mid", "high"}

NUMERIC_COLS = [
    "ts_us",
    "device_id",
    "mem_fixed_mhz",
    "from_gpu_mhz",
    "to_gpu_mhz",
    "api_us",
    "settle_total_us",
    "settle_after_call_us",
    "polls",
    "stable_n",
    "poll_us",
    "timeout_ms",
    "final_gpu_mhz",
    "gpu_util_pct",
    "mem_util_pct",
    "pcie_tx_kbps",
    "pcie_rx_kbps",
    "power_mw",
    "energy_mj",
]

STRING_COLS = ["gpu_name", "mode", "api_kind", "status"]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_type_workload_from_name(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Expect basename like dvfs_{type}_{workload}.csv
    """
    base = os.path.basename(path)
    m = re.match(r"dvfs_([a-zA-Z]+)_([a-zA-Z]+)\.csv$", base)
    if not m:
        return None, None
    t, w = m.group(1).lower(), m.group(2).lower()
    if t not in TYPE_SET or w not in WORKLOAD_SET:
        return None, None
    return t, w


def clean_status(s: pd.Series) -> pd.Series:
    # remove Chinese comma, whitespace, stray quotes
    return (
        s.astype(str)
        .str.replace("，", "", regex=False)
        .str.strip()
        .str.strip('"')
        .replace({"nan": np.nan})
    )


def coerce_numeric(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    # remove common junk (commas, quotes, trailing spaces)
    x = df[col].astype(str).str.replace("，", "", regex=False).str.strip().str.strip('"')
    # If there are parentheses or other trailing tokens, keep the leading numeric part.
    # Example: "1234ms" -> "1234"
    x = x.str.extract(r"^\s*([-+]?\d+(\.\d+)?)", expand=False)[0]
    df[col] = pd.to_numeric(x, errors="coerce")


def robust_read_csv(path: str) -> pd.DataFrame:
    """
    Tries to read CSV even if there are minor format issues.
    """
    # Most of your files should parse fine with default engine.
    # If not, fallback to python engine.
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python")
    return df


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def quantiles(s: pd.Series, qs=(0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    s2 = s.dropna()
    if len(s2) == 0:
        return {f"p{int(q*100)}": np.nan for q in qs}
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(s2.quantile(q))
    out["mean"] = float(s2.mean())
    out["max"] = float(s2.max())
    out["n"] = int(s2.shape[0])
    return out


@dataclass
class GroupKey:
    type: str
    workload: str

    @property
    def label(self) -> str:
        return f"{self.type}_{self.workload}"


# -----------------------------
# Energy interpretation helpers
# -----------------------------

def add_energy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - step_mhz: abs(to - from)
      - reached_target: final_gpu_mhz == to_gpu_mhz (with tolerance)
      - energy_mj_derived: power_mw * settle_total_us / 1000  (mW * ms = mJ)
      - energy_mj_delta: if energy_mj looks like a cumulative counter, compute diffs
    """
    df = df.copy()

    if "from_gpu_mhz" in df.columns and "to_gpu_mhz" in df.columns:
        df["step_mhz"] = (df["to_gpu_mhz"] - df["from_gpu_mhz"]).abs()
    else:
        df["step_mhz"] = np.nan

    if "final_gpu_mhz" in df.columns and "to_gpu_mhz" in df.columns:
        # tolerate small mismatches (e.g., rounding)
        df["reached_target"] = (df["final_gpu_mhz"] - df["to_gpu_mhz"]).abs() <= 1
    else:
        df["reached_target"] = np.nan

    # Derived energy sanity check:
    # power_mw * (settle_total_us/1000) -> mW*ms -> mJ
    if "power_mw" in df.columns and "settle_total_us" in df.columns:
        df["energy_mj_derived"] = df["power_mw"] * (df["settle_total_us"] / 1000.0)
    else:
        df["energy_mj_derived"] = np.nan

    # Try to interpret energy_mj: may be cumulative counter or per-transition delta.
    df["energy_mj_delta"] = np.nan
    if "energy_mj" in df.columns and "ts_us" in df.columns:
        tmp = df.sort_values(["device_id", "ts_us"]).copy()
        tmp["energy_mj_delta"] = tmp.groupby("device_id")["energy_mj"].diff()

        # Drop negative or insane jumps (wrap / reset / noise)
        # "insane" threshold: > 100x median derived energy (if derived available), else > 1e7 mJ
        med_derived = np.nanmedian(tmp.get("energy_mj_derived", pd.Series([np.nan])))
        if np.isfinite(med_derived) and med_derived > 0:
            upper = 100.0 * med_derived
        else:
            upper = 1e7

        tmp.loc[(tmp["energy_mj_delta"] < 0) | (tmp["energy_mj_delta"] > upper), "energy_mj_delta"] = np.nan
        df = tmp.sort_index()  # back to original index order

    # A convenience: "best guess" energy per transition
    # If delta is mostly finite and not crazy, use it; otherwise use derived.
    delta_ok_ratio = np.isfinite(df["energy_mj_delta"]).mean()
    if delta_ok_ratio >= 0.7:
        df["energy_mj_best"] = df["energy_mj_delta"]
        df["energy_source"] = "delta(counter)"
    else:
        df["energy_mj_best"] = df["energy_mj_derived"]
        df["energy_source"] = "derived(power*settle)"

    return df


# -----------------------------
# Main analysis
# -----------------------------

def load_all(input_dir: str, pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {os.path.join(input_dir, pattern)}")

    all_rows = []
    for p in paths:
        t, w = parse_type_workload_from_name(p)
        if t is None or w is None:
            # skip unrelated files
            continue

        df = robust_read_csv(p)

        # ensure expected columns exist
        for c in STRING_COLS:
            if c not in df.columns:
                df[c] = np.nan
        for c in NUMERIC_COLS:
            if c not in df.columns:
                df[c] = np.nan

        # clean strings
        for c in STRING_COLS:
            df[c] = clean_status(df[c])

        # coerce numerics
        for c in NUMERIC_COLS:
            coerce_numeric(df, c)

        df["type"] = t
        df["workload"] = w
        df["file"] = os.path.basename(p)

        df = add_energy_columns(df)

        all_rows.append(df)

    if not all_rows:
        raise FileNotFoundError("Found CSV files, but none matched dvfs_{type}_{workload}.csv naming.")
    return pd.concat(all_rows, ignore_index=True)


def make_summary_tables(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)

    # status counts
    status = (
        df.groupby(["type", "workload", "status"])
        .size()
        .reset_index(name="count")
        .sort_values(["type", "workload", "count"], ascending=[True, True, False])
    )
    status.to_csv(os.path.join(out_dir, "status_counts.csv"), index=False)

    # core quantiles by group
    metrics = ["api_us", "settle_total_us", "settle_after_call_us", "energy_mj_best", "power_mw"]
    rows = []
    for (t, w), g in df.groupby(["type", "workload"]):
        row = {"type": t, "workload": w, "n_total": int(g.shape[0])}
        # success rate proxies
        row["reached_target_rate"] = float(np.nanmean(g["reached_target"].astype(float)))
        row["stable_rate"] = float(np.nanmean((g["stable_n"] >= g["polls"]).astype(float))) if "stable_n" in g else np.nan
        row["timeout_rate"] = float(np.nanmean((g["status"].astype(str).str.contains("timeout", case=False, na=False)).astype(float)))

        for m in metrics:
            q = quantiles(g[m])
            for k, v in q.items():
                row[f"{m}_{k}"] = v
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["type", "workload"])
    summary.to_csv(os.path.join(out_dir, "summary_quantiles.csv"), index=False)

    # transition-level table: by step_mhz buckets
    bins = [0, 25, 50, 100, 200, 400, 800, 2000, np.inf]
    labels = ["0-25", "25-50", "50-100", "100-200", "200-400", "400-800", "800-2000", "2000+"]
    df2 = df.copy()
    df2["step_bucket_mhz"] = pd.cut(df2["step_mhz"], bins=bins, labels=labels, include_lowest=True)

    bucket_rows = []
    for (t, w, b), g in df2.groupby(["type", "workload", "step_bucket_mhz"], dropna=False):
        bucket_rows.append(
            {
                "type": t,
                "workload": w,
                "step_bucket_mhz": str(b),
                "n": int(g.shape[0]),
                "settle_total_us_p50": float(g["settle_total_us"].quantile(0.5)) if g["settle_total_us"].notna().any() else np.nan,
                "settle_total_us_p95": float(g["settle_total_us"].quantile(0.95)) if g["settle_total_us"].notna().any() else np.nan,
                "api_us_p50": float(g["api_us"].quantile(0.5)) if g["api_us"].notna().any() else np.nan,
                "energy_mj_best_p50": float(g["energy_mj_best"].quantile(0.5)) if g["energy_mj_best"].notna().any() else np.nan,
            }
        )
    buckets = pd.DataFrame(bucket_rows).sort_values(["type", "workload", "step_bucket_mhz"])
    buckets.to_csv(os.path.join(out_dir, "by_step_bucket.csv"), index=False)

    # slowest transitions per group
    slow = (
        df.sort_values("settle_total_us", ascending=False)
        .groupby(["type", "workload"], as_index=False)
        .head(30)[
            [
                "type",
                "workload",
                "file",
                "ts_us",
                "device_id",
                "from_gpu_mhz",
                "to_gpu_mhz",
                "final_gpu_mhz",
                "api_us",
                "settle_after_call_us",
                "settle_total_us",
                "polls",
                "stable_n",
                "power_mw",
                "energy_mj_best",
                "status",
            ]
        ]
    )
    slow.to_csv(os.path.join(out_dir, "top_slowest_30.csv"), index=False)


def plot_latency_box(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    groups = []
    labels = []
    for (t, w), g in df.groupby(["type", "workload"]):
        vals = g["settle_total_us"].dropna().values
        if vals.size == 0:
            continue
        groups.append(vals)
        labels.append(f"{t}-{w}")

    if not groups:
        return

    plt.figure()
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.ylabel("settle_total_us (us)")
    plt.title("DVFS settle_total_us distribution (box, no fliers)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_box_settle_total_us.png"), dpi=160)
    plt.close()


def plot_latency_ecdf(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    plt.figure()

    for (t, w), g in df.groupby(["type", "workload"]):
        x, y = ecdf(g["settle_total_us"].values)
        if x.size == 0:
            continue
        plt.plot(x, y, label=f"{t}-{w}")

    plt.xlabel("settle_total_us (us)")
    plt.ylabel("ECDF")
    plt.title("DVFS settle_total_us ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latency_ecdf_settle_total_us.png"), dpi=160)
    plt.close()


def plot_step_vs_latency(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    plt.figure()

    # sample to avoid huge files making unreadable plot
    d = df[["type", "workload", "step_mhz", "settle_total_us"]].dropna()
    if d.empty:
        return
    if len(d) > 20000:
        d = d.sample(20000, random_state=1)

    for (t, w), g in d.groupby(["type", "workload"]):
        plt.scatter(g["step_mhz"], g["settle_total_us"], s=6, alpha=0.25, label=f"{t}-{w}")

    plt.xlabel("|to - from| (MHz)")
    plt.ylabel("settle_total_us (us)")
    plt.title("Step size vs settle_total_us (sampled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step_mhz_vs_settle_total_us_scatter.png"), dpi=160)
    plt.close()


def plot_power_vs_latency(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    plt.figure()

    d = df[["type", "workload", "power_mw", "settle_total_us"]].dropna()
    if d.empty:
        return
    if len(d) > 20000:
        d = d.sample(20000, random_state=2)

    for (t, w), g in d.groupby(["type", "workload"]):
        plt.scatter(g["power_mw"], g["settle_total_us"], s=6, alpha=0.25, label=f"{t}-{w}")

    plt.xlabel("power_mw (mW)")
    plt.ylabel("settle_total_us (us)")
    plt.title("Power vs settle_total_us (sampled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "power_mw_vs_settle_total_us_scatter.png"), dpi=160)
    plt.close()


def plot_comm_pcie_vs_latency(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    comm = df[df["type"] == "comm"].copy()
    if comm.empty:
        return

    # pcie tx/rx vs settle_total_us
    for col in ["pcie_tx_kbps", "pcie_rx_kbps"]:
        plt.figure()
        d = comm[["workload", col, "settle_total_us"]].dropna()
        if d.empty:
            plt.close()
            continue
        if len(d) > 20000:
            d = d.sample(20000, random_state=3)

        for w, g in d.groupby("workload"):
            plt.scatter(g[col], g["settle_total_us"], s=6, alpha=0.25, label=f"comm-{w}")

        plt.xlabel(f"{col}")
        plt.ylabel("settle_total_us (us)")
        plt.title(f"{col} vs settle_total_us (comm, sampled)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_vs_settle_total_us_comm_scatter.png"), dpi=160)
        plt.close()


def write_markdown_report(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)

    summary_path = os.path.join(out_dir, "summary_quantiles.csv")
    status_path = os.path.join(out_dir, "status_counts.csv")
    step_path = os.path.join(out_dir, "by_step_bucket.csv")
    slow_path = os.path.join(out_dir, "top_slowest_30.csv")

    # Load saved tables for easier embedding
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None
    status = pd.read_csv(status_path) if os.path.exists(status_path) else None
    step = pd.read_csv(step_path) if os.path.exists(step_path) else None

    # Find “energy interpretation” per file/group
    energy_source = (
        df.groupby(["type", "workload"])["energy_source"]
        .agg(lambda x: x.value_counts().index[0] if len(x.dropna()) else "unknown")
        .reset_index()
    )

    def df_to_md(d: pd.DataFrame, n: int = 20) -> str:
        if d is None or d.empty:
            return "_(empty)_"
        return d.head(n).to_markdown(index=False)

    lines = []
    lines.append("# DVFS CSV Analysis Report\n")
    lines.append("## Inputs\n")
    files = sorted(df["file"].unique().tolist())
    for f in files:
        lines.append(f"- {f}")
    lines.append("")

    lines.append("## High-level summary (p50/p90/p95/p99)\n")
    if summary is not None and not summary.empty:
        # Keep only a few key columns
        keep_cols = ["type", "workload", "n_total", "reached_target_rate", "timeout_rate",
                     "api_us_p50", "api_us_p95",
                     "settle_total_us_p50", "settle_total_us_p95", "settle_total_us_p99",
                     "settle_after_call_us_p50", "settle_after_call_us_p95",
                     "power_mw_mean", "energy_mj_best_p50"]
        keep_cols = [c for c in keep_cols if c in summary.columns]
        lines.append(df_to_md(summary[keep_cols], n=20))
    else:
        lines.append("_(summary table missing)_")
    lines.append("")

    lines.append("## Status breakdown\n")
    if status is not None and not status.empty:
        lines.append(df_to_md(status, n=60))
    else:
        lines.append("_(status table missing)_")
    lines.append("")

    lines.append("## Step-size bucket view\n")
    if step is not None and not step.empty:
        lines.append(df_to_md(step, n=80))
    else:
        lines.append("_(step bucket table missing)_")
    lines.append("")

    lines.append("## Energy interpretation (best guess)\n")
    lines.append(df_to_md(energy_source, n=20))
    lines.append("")
    lines.append(
        "Notes:\n"
        "- `energy_mj_best` chooses `energy_mj_delta` if `energy_mj` behaves like a cumulative counter; otherwise uses `power_mw * settle_total_us`.\n"
        "- If you *know* the true meaning of `energy_mj`, you can lock this logic.\n"
    )

    lines.append("## Plots\n")
    plots = [
        "latency_box_settle_total_us.png",
        "latency_ecdf_settle_total_us.png",
        "step_mhz_vs_settle_total_us_scatter.png",
        "power_mw_vs_settle_total_us_scatter.png",
        "pcie_tx_kbps_vs_settle_total_us_comm_scatter.png",
        "pcie_rx_kbps_vs_settle_total_us_comm_scatter.png",
    ]
    for p in plots:
        pp = os.path.join(out_dir, p)
        if os.path.exists(pp):
            lines.append(f"- {p}")
    lines.append("")
    lines.append("## Slowest 30 transitions per group\n")
    if os.path.exists(slow_path):
        lines.append(f"- See `{os.path.basename(slow_path)}`\n")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default=".", help="Directory containing dvfs_*.csv")
    ap.add_argument("--out_dir", type=str, default="dvfs_report", help="Output directory")
    ap.add_argument("--pattern", type=str, default="dvfs_*.csv", help="Glob pattern")
    args = ap.parse_args()

    df = load_all(args.input_dir, args.pattern)

    ensure_dir(args.out_dir)

    # Save merged & cleaned
    df.to_csv(os.path.join(args.out_dir, "merged_cleaned.csv"), index=False)

    # Tables
    make_summary_tables(df, args.out_dir)

    # Plots
    plot_latency_box(df, args.out_dir)
    plot_latency_ecdf(df, args.out_dir)
    plot_step_vs_latency(df, args.out_dir)
    plot_power_vs_latency(df, args.out_dir)
    plot_comm_pcie_vs_latency(df, args.out_dir)

    # Report
    write_markdown_report(df, args.out_dir)

    # Console hint
    print(f"[OK] Wrote outputs to: {args.out_dir}")
    print("Key files:")
    print("  - merged_cleaned.csv")
    print("  - summary_quantiles.csv")
    print("  - status_counts.csv")
    print("  - by_step_bucket.csv")
    print("  - top_slowest_30.csv")
    print("  - report.md")
    print("  - *.png plots")


if __name__ == "__main__":
    main()
