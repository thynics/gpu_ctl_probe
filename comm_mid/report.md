# DVFS CSV Report

- input: `dvfs_comm_mid.csv`
- rows: total=3950, analyzed=3950 (ok_only=0)

## Modes
- comm: 3950

## API kinds
- locked: 3950

## Status breakdown (all rows)
- ok(locked): 3950

## Overall latency stats (analyzed rows)
- **api_us**: n=3950, p50=4.074 ms, p90=6.571 ms, p95=8.372 ms, p99=11.296 ms, max=14.174 ms
- **settle_after_call_us**: n=3950, p50=20.295 ms, p90=20.590 ms, p95=20.832 ms, p99=22.054 ms, max=25.173 ms
- **settle_total_us**: n=3950, p50=24.488 ms, p90=27.209 ms, p95=29.032 ms, p99=32.057 ms, max=35.222 ms

## Slowest transitions (by settle_after_call_us p99)
- 1252->1530 (n=300): p50=20.259 ms, p95=21.720 ms, p99=23.572 ms, max=24.430 ms
- 975->690 (n=300): p50=20.323 ms, p95=20.905 ms, p99=22.781 ms, max=25.173 ms
- 975->1252 (n=300): p50=20.236 ms, p95=21.687 ms, p99=22.400 ms, max=23.277 ms
- 1252->975 (n=300): p50=20.271 ms, p95=21.417 ms, p99=22.072 ms, max=23.612 ms
- 412->690 (n=600): p50=20.306 ms, p95=21.044 ms, p99=21.915 ms, max=23.665 ms
- 1530->1252 (n=300): p50=20.252 ms, p95=21.145 ms, p99=21.863 ms, max=22.707 ms
- 690->412 (n=350): p50=20.328 ms, p95=20.592 ms, p99=21.656 ms, max=22.716 ms
- 412->135 (n=600): p50=20.304 ms, p95=20.624 ms, p99=21.567 ms, max=22.504 ms
- 690->975 (n=300): p50=20.257 ms, p95=20.737 ms, p99=21.482 ms, max=22.847 ms
- 135->412 (n=600): p50=20.306 ms, p95=20.584 ms, p99=20.987 ms, max=21.473 ms

## Load proxies (interpretation tips)
- `gpu_util_pct`/`mem_util_pct`：NVML 的利用率采样，适合做粗粒度相关性。
- `pcie_tx_kbps`/`pcie_rx_kbps`：更像‘是否有 PCIe 活动’，对 comm 模式/host copies 有意义。
- `power_mw`：是最稳的“负载强度代理变量”（尤其 compute）。

## Files generated
- `summary_overall.csv`
- `summary_by_transition.csv`
- `corr_spearman.csv` (if enough data)
- `*.png` plots
