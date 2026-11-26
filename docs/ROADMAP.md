# Roadmap & Architecture Overview

## Version History

| Version | Highlights |
|--------|------------|
| **v0.1.x – v0.2.x** | Initial wind-statistics library, pure-Rust `#![no_std]` core, WRA I/O, and Python (`PyO3`) / R (`extendr`) bindings. |
| **v0.3.0 – v0.3.3** | ECO-WIND D3.3 thermal models, bearing kinematics (BPFO/BPFI/BSF/FTF), peridynamics, SCADA health-index, heat diffusion, Arrhenius oil life, ZDDP depletion, and `numpy` wheel dependency. |
| **v0.3.4 (Feature-Complete Freeze)** | Consolidated all planned `v0.4.x–v1.0.0` capabilities: **P53 Laplace-SARIMAX** (sparse $k$-NN Laplacian, causal MODWT `db4`, one-sided rectification & parallel dual-guard), **analytic Bastankhah-Porté-Agel wake gradients**, **IEC 61400-15-1 bankable AEP** (P50/P75/P90/P99 + RSS uncertainty), **dual-horizon RL derating** (`ThermalDeratingEnv` for PPO/SAC coupled with P4/CTD), **modular plug-in registry**, 100% Python/R binding parity, and **STM32F401 `#![no_std]` + Renode vHIL CI**. |
