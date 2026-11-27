# AnemoCore `v0.3.5` Benchmark Summary

- **Peak RSS:** `28.96 MiB` (Budget `<= 150.0 MiB`)

| Suite | Engine | p50 (ms) | p99 (ms) | Metric |
| :--- | :--- | :--- | :--- | :--- |
| Weibull-MLE (n=2000) | `AnemoCore` | `1.022` | `13.267` | `k=2.23, c=8.56` |
| Wake Analytic Gradients | `AnemoCore` | `0.001` | `0.380` | `grad0=-7.9453e-04` |
| IEC 61400-15-1 Bankable AEP | `AnemoCore` | `0.001` | `0.007` | `P50=23280.0 MWh` |
| P53 Laplace-SARIMAX Dual-Guard | `AnemoCore` | `0.038` | `0.060` | `alarms=0` |
| Power Proxy Regression | `LightGBM` | `67.883` | `125.907` | `baseline` |
| Power Proxy Regression | `NGBoost` | `494.101` | `559.658` | `baseline` |
