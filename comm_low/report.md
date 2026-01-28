# DVFS CSV Report

- input: `dvfs_comm_low.csv`
- rows: total=3000, analyzed=3000 (ok_only=0)

## Modes
- comm: 3000

## API kinds
- locked: 3000

## Status breakdown (all rows)
- ok(locked): 3000

## Overall latency stats (analyzed rows)
- **api_us**: n=3000, p50=5.196 ms, p90=8.062 ms, p95=8.878 ms, p99=11.078 ms, max=13.317 ms
- **settle_after_call_us**: n=3000, p50=20.058 ms, p90=20.363 ms, p95=20.653 ms, p99=21.865 ms, max=24.864 ms
- **settle_total_us**: n=3000, p50=25.218 ms, p90=28.225 ms, p95=29.165 ms, p99=32.182 ms, max=34.465 ms

## Slowest transitions (by settle_after_call_us p99)
- 1252->975 (n=300): p50=20.030 ms, p95=20.506 ms, p99=22.803 ms, max=24.011 ms
- 412->690 (n=300): p50=20.076 ms, p95=20.785 ms, p99=22.526 ms, max=24.864 ms
- 975->690 (n=300): p50=20.082 ms, p95=20.615 ms, p99=22.289 ms, max=22.961 ms
- 690->412 (n=300): p50=20.090 ms, p95=20.579 ms, p99=21.898 ms, max=22.858 ms
- 1252->1530 (n=300): p50=19.982 ms, p95=20.619 ms, p99=21.863 ms, max=23.972 ms
- 1530->1252 (n=300): p50=20.023 ms, p95=21.051 ms, p99=21.787 ms, max=23.567 ms
- 135->412 (n=300): p50=20.093 ms, p95=20.635 ms, p99=21.718 ms, max=21.937 ms
- 690->975 (n=300): p50=20.050 ms, p95=20.395 ms, p99=21.666 ms, max=22.863 ms
- 412->135 (n=300): p50=20.125 ms, p95=20.513 ms, p99=21.578 ms, max=22.003 ms
- 975->1252 (n=300): p50=20.005 ms, p95=20.436 ms, p99=21.569 ms, max=22.947 ms

## Load proxies (interpretation tips)
- `gpu_util_pct`/`mem_util_pct`：NVML 的利用率采样，适合做粗粒度相关性。
- `pcie_tx_kbps`/`pcie_rx_kbps`：更像‘是否有 PCIe 活动’，对 comm 模式/host copies 有意义。
- `power_mw`：是最稳的“负载强度代理变量”（尤其 compute）。

## Files generated
- `summary_overall.csv`
- `summary_by_transition.csv`
- `corr_spearman.csv` (if enough data)
- `*.png` plots
