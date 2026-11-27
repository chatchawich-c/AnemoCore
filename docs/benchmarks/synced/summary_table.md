# AnemoCore `v0.3.5` Benchmark Summary

- **Peak RSS:** `152.1 MiB` (Budget `<= 150.0 MiB`)

| Suite | Engine | Fit (ms) | Infer p50 (ms) | Infer p99 (ms) | Metric (Accuracy / Error / Guarantee) |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Weibull-MLE | `AnemoCore` | `0.000` | `0.724` | `1.038` | `k~2.10, Warmup=3 (stable p99)` |
| PowerCurve-Proxy | `AnemoCore (HGB-Proxy)` | `448.703` | `1.422` | `2.688` | `RMSE=1.699, MAE=1.356, R2=0.9985` |
| PowerCurve-Proxy | `LightGBM` | `168.278` | `1.778` | `4.248` | `RMSE=1.664, MAE=1.334, R2=0.9985` |
