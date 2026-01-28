# DVFS CSV Report

- input: `dvfs_compute_low.csv`
- rows: total=3000, analyzed=3000 (ok_only=0)

## Modes
- compute: 3000

## API kinds
- locked: 3000

## Status breakdown (all rows)
- ok(locked): 3000

## Overall latency stats (analyzed rows)
- **api_us**: n=3000, p50=4.429 ms, p90=7.457 ms, p95=9.304 ms, p99=11.234 ms, max=15.009 ms
- **settle_after_call_us**: n=3000, p50=20.115 ms, p90=21.584 ms, p95=22.618 ms, p99=26.670 ms, max=47.863 ms
- **settle_total_us**: n=3000, p50=24.744 ms, p90=28.285 ms, p95=31.440 ms, p99=36.076 ms, max=54.695 ms

## Slowest transitions (by settle_after_call_us p99)
- 135->412 (n=300): p50=21.393 ms, p95=26.946 ms, p99=28.053 ms, max=28.576 ms
- 412->135 (n=300): p50=20.387 ms, p95=22.711 ms, p99=24.886 ms, max=27.850 ms
- 412->690 (n=300): p50=21.232 ms, p95=22.721 ms, p99=24.755 ms, max=25.586 ms
- 1530->1252 (n=300): p50=20.018 ms, p95=21.338 ms, p99=23.412 ms, max=47.863 ms
- 975->690 (n=300): p50=20.076 ms, p95=20.995 ms, p99=22.452 ms, max=24.143 ms
- 690->412 (n=300): p50=20.128 ms, p95=21.543 ms, p99=22.141 ms, max=22.607 ms
- 1252->1530 (n=300): p50=19.983 ms, p95=21.623 ms, p99=22.118 ms, max=23.201 ms
- 975->1252 (n=300): p50=19.995 ms, p95=20.933 ms, p99=21.783 ms, max=24.317 ms
- 1252->975 (n=300): p50=20.025 ms, p95=20.490 ms, p99=21.664 ms, max=24.115 ms
- 690->975 (n=300): p50=20.032 ms, p95=20.687 ms, p99=21.592 ms, max=22.889 ms

## Load proxies (interpretation tips)
- `gpu_util_pct`/`mem_util_pct`：NVML 的利用率采样，适合做粗粒度相关性。
- `pcie_tx_kbps`/`pcie_rx_kbps`：更像‘是否有 PCIe 活动’，对 comm 模式/host copies 有意义。
- `power_mw`：是最稳的“负载强度代理变量”（尤其 compute）。

## Files generated
- `summary_overall.csv`
- `summary_by_transition.csv`
- `corr_spearman.csv` (if enough data)
- `*.png` plots
