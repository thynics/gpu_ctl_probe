# DVFS CSV Analysis Report

## Inputs

- dvfs_comm_high.csv
- dvfs_comm_low.csv
- dvfs_comm_mid.csv
- dvfs_compute_high.csv
- dvfs_compute_low.csv
- dvfs_compute_mid.csv

## High-level summary (p50/p90/p95/p99)

| type    | workload   |   n_total |   reached_target_rate |   timeout_rate |   api_us_p50 |   api_us_p95 |   settle_total_us_p50 |   settle_total_us_p95 |   settle_total_us_p99 |   settle_after_call_us_p50 |   settle_after_call_us_p95 |   power_mw_mean |   energy_mj_best_p50 |
|:--------|:-----------|----------:|----------------------:|---------------:|-------------:|-------------:|----------------------:|----------------------:|----------------------:|---------------------------:|---------------------------:|----------------:|---------------------:|
| comm    | high       |      3000 |                     1 |              0 |       4594.5 |      9835.05 |               25289   |               30783   |               32745.5 |                    20654   |                    21423   |         32417.7 |                 9459 |
| comm    | low        |      3000 |                     1 |              0 |       5196   |      8878    |               25218.5 |               29165.2 |               32181.8 |                    20058   |                    20653   |         30725.5 |                 8966 |
| comm    | mid        |      3950 |                     1 |              0 |       4073.5 |      8371.85 |               24488.5 |               29031.8 |               32056.6 |                    20295   |                    20832   |         30760   |                 9149 |
| compute | high       |      3000 |                     1 |              0 |       4506.5 |      9299.25 |               24838.5 |               31929.3 |               36664.2 |                    20146.5 |                    22831.2 |         71661.9 |                23380 |
| compute | low        |      3000 |                     1 |              0 |       4429   |      9304.35 |               24744   |               31439.8 |               36076.1 |                    20115   |                    22617.7 |         65365.6 |                21072 |
| compute | mid        |      3000 |                     1 |              0 |       4454   |      9232.1  |               24825.5 |               31922.5 |               36867.8 |                    20139   |                    23168.4 |         71032.9 |                23463 |

## Status breakdown

| type    | workload   | status     |   count |
|:--------|:-----------|:-----------|--------:|
| comm    | high       | ok(locked) |    3000 |
| comm    | low        | ok(locked) |    3000 |
| comm    | mid        | ok(locked) |    3950 |
| compute | high       | ok(locked) |    3000 |
| compute | low        | ok(locked) |    3000 |
| compute | mid        | ok(locked) |    3000 |

## Step-size bucket view

| type    | workload   | step_bucket_mhz   |    n |   settle_total_us_p50 |   settle_total_us_p95 |   api_us_p50 |   energy_mj_best_p50 |
|:--------|:-----------|:------------------|-----:|----------------------:|----------------------:|-------------:|---------------------:|
| comm    | high       | 200-400           | 3000 |               25289   |               30783   |       4594.5 |                 9459 |
| comm    | low        | 200-400           | 3000 |               25218.5 |               29165.2 |       5196   |                 8966 |
| comm    | mid        | 200-400           | 3950 |               24488.5 |               29031.8 |       4073.5 |                 9149 |
| compute | high       | 200-400           | 3000 |               24838.5 |               31929.3 |       4506.5 |                23380 |
| compute | low        | 200-400           | 3000 |               24744   |               31439.8 |       4429   |                21072 |
| compute | mid        | 200-400           | 3000 |               24825.5 |               31922.5 |       4454   |                23463 |

## Energy interpretation (best guess)

| type    | workload   | energy_source   |
|:--------|:-----------|:----------------|
| comm    | high       | delta(counter)  |
| comm    | low        | delta(counter)  |
| comm    | mid        | delta(counter)  |
| compute | high       | delta(counter)  |
| compute | low        | delta(counter)  |
| compute | mid        | delta(counter)  |

Notes:
- `energy_mj_best` chooses `energy_mj_delta` if `energy_mj` behaves like a cumulative counter; otherwise uses `power_mw * settle_total_us`.
- If you *know* the true meaning of `energy_mj`, you can lock this logic.

## Plots

- latency_box_settle_total_us.png
- latency_ecdf_settle_total_us.png
- step_mhz_vs_settle_total_us_scatter.png
- power_mw_vs_settle_total_us_scatter.png
- pcie_tx_kbps_vs_settle_total_us_comm_scatter.png
- pcie_rx_kbps_vs_settle_total_us_comm_scatter.png

## Slowest 30 transitions per group

- See `top_slowest_30.csv`
