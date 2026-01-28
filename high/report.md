# DVFS CSV Report

- input: `dvfs_compute_high.csv`
- rows: total=3000, analyzed=3000 (ok_only=0)

## Modes
- compute: 3000

## API kinds
- locked: 3000

## Status breakdown (all rows)
- ok(locked): 3000

## Overall latency stats (analyzed rows)
- **api_us**: n=3000, p50=4.506 ms, p90=7.647 ms, p95=9.299 ms, p99=11.564 ms, max=15.595 ms
- **settle_after_call_us**: n=3000, p50=20.146 ms, p90=21.652 ms, p95=22.831 ms, p99=27.705 ms, max=29.414 ms
- **settle_total_us**: n=3000, p50=24.838 ms, p90=29.160 ms, p95=31.929 ms, p99=36.664 ms, max=38.558 ms

## Slowest transitions (by settle_after_call_us p99)
- 135->412 (n=300): p50=21.561 ms, p95=27.894 ms, p99=29.129 ms, max=29.414 ms
- 412->135 (n=300): p50=20.819 ms, p95=21.958 ms, p99=25.441 ms, max=29.163 ms
- 412->690 (n=300): p50=21.256 ms, p95=22.571 ms, p99=24.789 ms, max=25.370 ms
- 1530->1252 (n=300): p50=20.021 ms, p95=21.175 ms, p99=22.541 ms, max=24.896 ms
- 975->690 (n=300): p50=20.081 ms, p95=21.000 ms, p99=22.367 ms, max=23.018 ms
- 1252->1530 (n=300): p50=19.991 ms, p95=21.586 ms, p99=22.038 ms, max=23.942 ms
- 1252->975 (n=300): p50=20.026 ms, p95=20.599 ms, p99=21.991 ms, max=24.299 ms
- 690->412 (n=300): p50=20.309 ms, p95=21.615 ms, p99=21.932 ms, max=23.151 ms
- 975->1252 (n=300): p50=20.009 ms, p95=20.950 ms, p99=21.840 ms, max=23.856 ms
- 690->975 (n=300): p50=20.025 ms, p95=20.766 ms, p99=21.710 ms, max=23.772 ms

## Load proxies (interpretation tips)
- `gpu_util_pct`/`mem_util_pct`：NVML 的利用率采样，适合做粗粒度相关性。
- `pcie_tx_kbps`/`pcie_rx_kbps`：更像‘是否有 PCIe 活动’，对 comm 模式/host copies 有意义。
- `power_mw`：是最稳的“负载强度代理变量”（尤其 compute）。

## Files generated
- `summary_overall.csv`
- `summary_by_transition.csv`
- `corr_spearman.csv` (if enough data)
- `*.png` plots
