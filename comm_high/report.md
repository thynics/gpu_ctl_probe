# DVFS CSV Report

- input: `dvfs_comm_high.csv`
- rows: total=3000, analyzed=3000 (ok_only=0)

## Modes
- comm: 3000

## API kinds
- locked: 3000

## Status breakdown (all rows)
- ok(locked): 3000

## Overall latency stats (analyzed rows)
- **api_us**: n=3000, p50=4.595 ms, p90=7.296 ms, p95=9.835 ms, p99=11.818 ms, max=14.365 ms
- **settle_after_call_us**: n=3000, p50=20.654 ms, p90=21.052 ms, p95=21.423 ms, p99=22.586 ms, max=25.731 ms
- **settle_total_us**: n=3000, p50=25.289 ms, p90=28.511 ms, p95=30.783 ms, p99=32.746 ms, max=35.467 ms

## Slowest transitions (by settle_after_call_us p99)
- 975->1252 (n=300): p50=20.634 ms, p95=22.109 ms, p99=24.029 ms, max=25.190 ms
- 412->690 (n=300): p50=20.682 ms, p95=21.424 ms, p99=23.112 ms, max=24.302 ms
- 1252->1530 (n=300): p50=20.631 ms, p95=22.014 ms, p99=22.887 ms, max=24.098 ms
- 690->412 (n=300): p50=20.661 ms, p95=22.026 ms, p99=22.733 ms, max=23.580 ms
- 690->975 (n=300): p50=20.625 ms, p95=21.221 ms, p99=22.693 ms, max=24.163 ms
- 1252->975 (n=300): p50=20.636 ms, p95=21.418 ms, p99=22.375 ms, max=25.731 ms
- 135->412 (n=300): p50=20.701 ms, p95=21.325 ms, p99=22.301 ms, max=22.554 ms
- 412->135 (n=300): p50=20.684 ms, p95=21.182 ms, p99=22.251 ms, max=22.759 ms
- 1530->1252 (n=300): p50=20.612 ms, p95=21.188 ms, p99=22.120 ms, max=23.463 ms
- 975->690 (n=300): p50=20.664 ms, p95=21.131 ms, p99=22.069 ms, max=23.348 ms

## Load proxies (interpretation tips)
- `gpu_util_pct`/`mem_util_pct`：NVML 的利用率采样，适合做粗粒度相关性。
- `pcie_tx_kbps`/`pcie_rx_kbps`：更像‘是否有 PCIe 活动’，对 comm 模式/host copies 有意义。
- `power_mw`：是最稳的“负载强度代理变量”（尤其 compute）。

## Files generated
- `summary_overall.csv`
- `summary_by_transition.csv`
- `corr_spearman.csv` (if enough data)
- `*.png` plots
