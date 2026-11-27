# AnemoCore `v0.3.5` Benchmark Summary

- **Peak RSS:** `275.1 MiB` (Budget `<= 150.0 MiB`)

| Suite | Engine | Fit (ms) | Infer p50 (ms) | Infer p99 (ms) | Metric (Accuracy / Error / Guarantee) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| [Tier I: Synthetic] Weibull-MLE | `AnemoCore` | `0.000` | `0.062` | `0.074` | `k~2.10, Warmup=3 (stable p99)` |
| [Tier I: Synthetic] PowerCurve-Proxy | `AnemoCore` | `0.000` | `0.022` | `0.024` | `RMSE=6.051, MAE=4.364, R2=0.9804` |
| [Tier I: Synthetic] PowerCurve-Proxy | `LightGBM` | `129.048` | `1.335` | `12.492` | `RMSE=1.662, MAE=1.327, R2=0.9985` |
| [Tier I: Synthetic] PowerCurve-Proxy | `NGBoost` | `629.859` | `19.777` | `24.443` | `RMSE=19.378, MAE=18.285, R2=0.7988` |
| [Tier II: Real-World] Kelmarsh-SCADA | `AnemoCore` | `0.000` | `0.062` | `0.063` | `RMSE=567.566, MAE=458.291, R2=0.0029` |
| [Tier II: Real-World] Kelmarsh-SCADA | `LightGBM` | `33.938` | `2.196` | `12.712` | `RMSE=42.624, MAE=23.404, R2=0.9944` |
| [Tier II: Real-World] Kelmarsh-SCADA | `NGBoost` | `1123.533` | `19.559` | `26.213` | `RMSE=270.400, MAE=232.671, R2=0.7737` |
| [Tier II: Real-World] Penmanshiel-SCADA | `AnemoCore` | `0.000` | `0.037` | `0.069` | `RMSE=315.872, MAE=217.095, R2=0.8064` |
| [Tier II: Real-World] Penmanshiel-SCADA | `LightGBM` | `23.436` | `2.333` | `15.135` | `RMSE=28.528, MAE=15.864, R2=0.9984` |
| [Tier II: Real-World] Penmanshiel-SCADA | `NGBoost` | `732.422` | `20.266` | `25.707` | `RMSE=333.537, MAE=270.540, R2=0.7842` |
