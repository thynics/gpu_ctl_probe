# DVFS CSV Report

- input: `dvfs_latency_gpu_only.csv`
- rows: total=3000, analyzed=3000 (ok_only=0)

## Modes
- idle: 3000

## API kinds
- locked: 3000

## Status breakdown (all rows)
- ok(locked): 3000

## Overall latency stats (analyzed rows)
- **api_us**: n=3000, p50=4.530 ms, p90=7.522 ms, p95=9.369 ms, p99=11.362 ms, max=13.679 ms
- **settle_after_call_us**: n=3000, p50=20.056 ms, p90=20.436 ms, p95=20.859 ms, p99=21.917 ms, max=23.973 ms
- **settle_total_us**: n=3000, p50=24.566 ms, p90=27.862 ms, p95=29.830 ms, p99=31.858 ms, max=34.163 ms

## Slowest transitions (by settle_after_call_us p99)
- 975->1252 (n=300): p50=20.003 ms, p95=21.219 ms, p99=22.371 ms, max=23.973 ms
- 1252->1530 (n=300): p50=19.998 ms, p95=21.629 ms, p99=22.034 ms, max=23.721 ms
- 135->412 (n=300): p50=20.095 ms, p95=20.679 ms, p99=21.866 ms, max=21.963 ms
- 1252->975 (n=300): p50=20.017 ms, p95=20.631 ms, p99=21.808 ms, max=23.853 ms
- 975->690 (n=300): p50=20.071 ms, p95=20.660 ms, p99=21.804 ms, max=23.720 ms
- 690->975 (n=300): p50=20.029 ms, p95=20.734 ms, p99=21.760 ms, max=23.454 ms
- 1530->1252 (n=300): p50=20.020 ms, p95=20.762 ms, p99=21.657 ms, max=23.692 ms
- 690->412 (n=300): p50=20.081 ms, p95=20.938 ms, p99=21.543 ms, max=21.866 ms
- 412->135 (n=300): p50=20.103 ms, p95=20.673 ms, p99=21.496 ms, max=22.682 ms
- 412->690 (n=300): p50=20.085 ms, p95=20.841 ms, p99=21.268 ms, max=22.517 ms

## Load proxies (interpretation tips)
- `gpu_util_pct`/`mem_util_pct`：NVML 的利用率采样，适合做粗粒度相关性。
- `pcie_tx_kbps`/`pcie_rx_kbps`：更像‘是否有 PCIe 活动’，对 comm 模式/host copies 有意义。
- `power_mw`：是最稳的“负载强度代理变量”（尤其 compute）。

## Files generated
- `summary_overall.csv`
- `summary_by_transition.csv`
- `corr_spearman.csv` (if enough data)
- `*.png` plots
