#!/usr/bin/env python3
import argparse
import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------
NUM_COLS = [
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

STR_COLS = ["gpu_name", "mode", "api_kind", "status"]

def ensure_outdir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

def safe_to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def quantiles(series: pd.Series, qs=(0.5, 0.9, 0.95, 0.99)):
    s = series.dropna()
    if len(s) == 0:
        return {f"p{int(q*100)}": np.nan for q in qs}
    out = {}
    for q in qs:
        out[f"p{int(q*100)}"] = float(s.quantile(q))
    return out

def basic_stats(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan, **quantiles(series))
    return dict(
        n=int(s.shape[0]),
        mean=float(s.mean()),
        std=float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        min=float(s.min()),
        max=float(s.max()),
        **quantiles(s),
    )

def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_hist(series: pd.Series, title: str, xlabel: str, out_png: Path, bins=60, clip_p99=True):
    s = series.dropna()
    if len(s) == 0:
        return
    if clip_p99:
        hi = s.quantile(0.99)
        s = s[s <= hi]
    plt.figure()
    plt.hist(s.values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    save_fig(out_png)

def plot_cdf(series: pd.Series, title: str, xlabel: str, out_png: Path, clip_p999=True):
    s = series.dropna().sort_values()
    if len(s) == 0:
        return
    if clip_p999:
        hi = s.quantile(0.999)
        s = s[s <= hi]
    y = np.arange(1, len(s)+1) / len(s)
    plt.figure()
    plt.plot(s.values, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True, which="both", axis="both")
    save_fig(out_png)

def plot_timeseries(t_s: pd.Series, y: pd.Series, title: str, ylabel: str, out_png: Path):
    df = pd.DataFrame({"t_s": t_s, "y": y}).dropna()
    if len(df) == 0:
        return
    plt.figure()
    plt.plot(df["t_s"].values, df["y"].values, linewidth=1.0)
    plt.title(title)
    plt.xlabel("time since start (s)")
    plt.ylabel(ylabel)
    save_fig(out_png)

def plot_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, out_png: Path):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) == 0:
        return
    plt.figure()
    plt.scatter(df["x"].values, df["y"].values, s=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(out_png)

def plot_box_by_group(df: pd.DataFrame, group_col: str, value_col: str, title: str, out_png: Path, max_groups=30):
    tmp = df[[group_col, value_col]].dropna()
    if len(tmp) == 0:
        return
    # keep top groups by sample count to avoid unreadable plots
    counts = tmp[group_col].value_counts()
    groups = counts.index.tolist()[:max_groups]
    data = [tmp[tmp[group_col] == g][value_col].values for g in groups]
    plt.figure(figsize=(max(10, len(groups) * 0.45), 5))
    plt.boxplot(data, labels=groups, showfliers=False)
    plt.title(title + (f" (top {len(groups)} groups by N)" if len(groups) < counts.shape[0] else ""))
    plt.xlabel(group_col)
    plt.ylabel(value_col)
    plt.xticks(rotation=45, ha="right")
    save_fig(out_png)

def corr_spearman(df: pd.DataFrame, cols):
    # Spearman is robust for monotonic/nonlinear
    sub = df[cols].dropna()
    if sub.shape[0] < 3:
        return None
    return sub.corr(method="spearman")

# -------------------------
# Main analysis
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input dvfs CSV")
    ap.add_argument("--out", default="dvfs_report", help="output directory")
    ap.add_argument("--ok_only", type=int, default=0, help="1: analyze only status starting with 'ok'")
    ap.add_argument("--max_groups", type=int, default=30, help="max transitions in box plot")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    ensure_outdir(out_dir)

    df = pd.read_csv(csv_path)
    # normalize columns
    for c in STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    df = safe_to_numeric(df, NUM_COLS)

    # derived columns
    if "ts_us" in df.columns:
        t0 = df["ts_us"].min()
        df["t_s"] = (df["ts_us"] - t0) / 1e6
    else:
        df["t_s"] = np.nan

    if "from_gpu_mhz" in df.columns and "to_gpu_mhz" in df.columns:
        df["transition"] = df["from_gpu_mhz"].astype("Int64").astype(str) + "->" + df["to_gpu_mhz"].astype("Int64").astype(str)
    else:
        df["transition"] = "NA"

    # status buckets
    if "status" in df.columns:
        df["is_ok"] = df["status"].str.startswith("ok")
        df["is_timeout"] = df["status"].str.startswith("timeout")
    else:
        df["is_ok"] = True
        df["is_timeout"] = False

    if args.ok_only == 1:
        df_use = df[df["is_ok"]].copy()
    else:
        df_use = df.copy()

    # -------------------------
    # Summaries
    # -------------------------
    total_n = int(df.shape[0])
    use_n = int(df_use.shape[0])

    status_counts = df["status"].value_counts(dropna=False) if "status" in df.columns else pd.Series(dtype=int)
    api_kinds = df["api_kind"].value_counts(dropna=False) if "api_kind" in df.columns else pd.Series(dtype=int)
    modes = df["mode"].value_counts(dropna=False) if "mode" in df.columns else pd.Series(dtype=int)

    key_latency_cols = ["api_us", "settle_after_call_us", "settle_total_us", "polls"]
    metric_cols = ["gpu_util_pct", "mem_util_pct", "pcie_tx_kbps", "pcie_rx_kbps", "power_mw"]

    overall = {}
    for c in key_latency_cols + metric_cols:
        if c in df_use.columns:
            overall[c] = basic_stats(df_use[c])
    overall_df = pd.DataFrame(overall).T.reset_index().rename(columns={"index": "metric"})
    overall_df.to_csv(out_dir / "summary_overall.csv", index=False)

    # per-transition
    by_tr = []
    if "transition" in df_use.columns and use_n > 0:
        g = df_use.groupby("transition", dropna=False)
        for tr, sub in g:
            row = {"transition": tr, "n": int(sub.shape[0])}
            for c in ["api_us", "settle_after_call_us", "settle_total_us", "polls", "gpu_util_pct", "pcie_tx_kbps", "pcie_rx_kbps", "power_mw"]:
                if c in sub.columns:
                    st = basic_stats(sub[c])
                    # keep the most useful stats
                    row.update({
                        f"{c}_mean": st["mean"],
                        f"{c}_p50": st["p50"],
                        f"{c}_p95": st["p95"],
                        f"{c}_p99": st["p99"],
                        f"{c}_max": st["max"],
                    })
            by_tr.append(row)

    by_tr_df = pd.DataFrame(by_tr) if by_tr else pd.DataFrame()

    if not by_tr_df.empty:
        # prefer sort by settle_after_call_us_p99 if available; otherwise fallback
        if "settle_after_call_us_p99" in by_tr_df.columns:
            by_tr_df = by_tr_df.sort_values(
                by=["n", "settle_after_call_us_p99"],
                ascending=[False, False],
            )
        elif "settle_total_us_p99" in by_tr_df.columns:
            by_tr_df = by_tr_df.sort_values(
                by=["n", "settle_total_us_p99"],
                ascending=[False, False],
            )
        elif "api_us_p99" in by_tr_df.columns:
            by_tr_df = by_tr_df.sort_values(
                by=["n", "api_us_p99"],
                ascending=[False, False],
            )
        else:
            by_tr_df = by_tr_df.sort_values(by=["n"], ascending=[False])
    
        by_tr_df.to_csv(out_dir / "summary_by_transition.csv", index=False)

    # -------------------------
    # Plots (each as standalone)
    # -------------------------
    if "settle_after_call_us" in df_use.columns:
        plot_hist(df_use["settle_after_call_us"], "Settle latency (after set-call) histogram", "settle_after_call_us (us)",
                  out_dir / "hist_settle_after_call_us.png", bins=80, clip_p99=True)
        plot_cdf(df_use["settle_after_call_us"], "Settle latency (after set-call) CDF", "settle_after_call_us (us)",
                 out_dir / "cdf_settle_after_call_us.png", clip_p999=True)

    if "settle_total_us" in df_use.columns:
        plot_hist(df_use["settle_total_us"], "Total latency (set + settle) histogram", "settle_total_us (us)",
                  out_dir / "hist_settle_total_us.png", bins=80, clip_p99=True)
        plot_cdf(df_use["settle_total_us"], "Total latency (set + settle) CDF", "settle_total_us (us)",
                 out_dir / "cdf_settle_total_us.png", clip_p999=True)

    if "api_us" in df_use.columns:
        plot_hist(df_use["api_us"], "NVML set-call latency histogram", "api_us (us)",
                  out_dir / "hist_api_us.png", bins=80, clip_p99=True)
        plot_cdf(df_use["api_us"], "NVML set-call latency CDF", "api_us (us)",
                 out_dir / "cdf_api_us.png", clip_p999=True)

    # time series
    if "t_s" in df_use.columns and "settle_after_call_us" in df_use.columns:
        plot_timeseries(df_use["t_s"], df_use["settle_after_call_us"],
                        "Settle latency over time", "settle_after_call_us (us)",
                        out_dir / "ts_settle_after_call_us.png")

    # box by transition (top N)
    if "transition" in df_use.columns and "settle_after_call_us" in df_use.columns:
        plot_box_by_group(df_use, "transition", "settle_after_call_us",
                          "Settle latency by transition", out_dir / "box_settle_by_transition.png",
                          max_groups=args.max_groups)

    # scatter: load proxies vs settle
    if "settle_after_call_us" in df_use.columns:
        for xcol, label in [
            ("gpu_util_pct", "gpu_util_pct (%)"),
            ("mem_util_pct", "mem_util_pct (%)"),
            ("pcie_tx_kbps", "pcie_tx_kbps"),
            ("pcie_rx_kbps", "pcie_rx_kbps"),
            ("power_mw", "power_mw"),
        ]:
            if xcol in df_use.columns:
                plot_scatter(df_use[xcol], df_use["settle_after_call_us"],
                             f"Settle latency vs {xcol}", label, "settle_after_call_us (us)",
                             out_dir / f"scatter_settle_vs_{xcol}.png")

    # correlations
    corr_cols = []
    for c in ["settle_after_call_us", "api_us"] + metric_cols:
        if c in df_use.columns:
            corr_cols.append(c)
    corr = corr_spearman(df_use, corr_cols) if len(corr_cols) >= 3 else None
    if corr is not None:
        corr.to_csv(out_dir / "corr_spearman.csv")

        # heatmap-ish via imshow (no seaborn)
        plt.figure(figsize=(max(6, 0.6*len(corr_cols)), max(5, 0.6*len(corr_cols))))
        plt.imshow(corr.values, vmin=-1, vmax=1)
        plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
        plt.yticks(range(len(corr_cols)), corr_cols)
        plt.colorbar()
        plt.title("Spearman correlation")
        save_fig(out_dir / "corr_spearman.png")

    # -------------------------
    # Markdown report
    # -------------------------
    report_lines = []
    report_lines.append(f"# DVFS CSV Report\n")
    report_lines.append(f"- input: `{csv_path}`")
    report_lines.append(f"- rows: total={total_n}, analyzed={use_n} (ok_only={args.ok_only})\n")

    if len(modes) > 0:
        report_lines.append("## Modes")
        for k, v in modes.items():
            report_lines.append(f"- {k}: {int(v)}")
        report_lines.append("")

    if len(api_kinds) > 0:
        report_lines.append("## API kinds")
        for k, v in api_kinds.items():
            report_lines.append(f"- {k}: {int(v)}")
        report_lines.append("")

    if len(status_counts) > 0:
        report_lines.append("## Status breakdown (all rows)")
        for k, v in status_counts.items():
            report_lines.append(f"- {k}: {int(v)}")
        report_lines.append("")

    def fmt_us(x):
        if x is None or (isinstance(x, float) and (math.isnan(x))):
            return "NA"
        return f"{x/1000.0:.3f} ms" if x >= 1000 else f"{x:.1f} us"

    report_lines.append("## Overall latency stats (analyzed rows)")
    for metric in ["api_us", "settle_after_call_us", "settle_total_us"]:
        if metric in overall:
            st = overall[metric]
            report_lines.append(f"- **{metric}**: n={st['n']}, "
                               f"p50={fmt_us(st['p50'])}, p90={fmt_us(st['p90'])}, p95={fmt_us(st['p95'])}, p99={fmt_us(st['p99'])}, max={fmt_us(st['max'])}")
    report_lines.append("")

    # slow transitions
    if not by_tr_df.empty and "settle_after_call_us_p99" in by_tr_df.columns:
        topk = by_tr_df.sort_values("settle_after_call_us_p99", ascending=False).head(10)
        report_lines.append("## Slowest transitions (by settle_after_call_us p99)")
        for _, r in topk.iterrows():
            report_lines.append(
                f"- {r['transition']} (n={int(r['n'])}): "
                f"p50={fmt_us(r.get('settle_after_call_us_p50', np.nan))}, "
                f"p95={fmt_us(r.get('settle_after_call_us_p95', np.nan))}, "
                f"p99={fmt_us(r.get('settle_after_call_us_p99', np.nan))}, "
                f"max={fmt_us(r.get('settle_after_call_us_max', np.nan))}"
            )
        report_lines.append("")

    # load proxy quick read
    report_lines.append("## Load proxies (interpretation tips)")
    report_lines.append("- `gpu_util_pct`/`mem_util_pct`：NVML 的利用率采样，适合做粗粒度相关性。")
    report_lines.append("- `pcie_tx_kbps`/`pcie_rx_kbps`：更像‘是否有 PCIe 活动’，对 comm 模式/host copies 有意义。")
    report_lines.append("- `power_mw`：是最稳的“负载强度代理变量”（尤其 compute）。\n")

    report_lines.append("## Files generated")
    report_lines.append("- `summary_overall.csv`")
    report_lines.append("- `summary_by_transition.csv`")
    report_lines.append("- `corr_spearman.csv` (if enough data)")
    report_lines.append("- `*.png` plots\n")

    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OK] Wrote outputs to: {out_dir.resolve()}")
    print(f" - report.md")
    print(f" - summary_overall.csv")
    print(f" - summary_by_transition.csv")
    if (out_dir / "corr_spearman.csv").exists():
        print(f" - corr_spearman.csv + corr_spearman.png")
    print(f" - png plots")

if __name__ == "__main__":
    main()
