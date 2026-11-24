# Per-Paper Reproducibility Scripts & Zenodo Tagging (`v0.1.0`)

This directory archives standalone Python/R reproducibility scripts and benchmark pipelines per manuscript, linked with [`../CITATION.cff`](../CITATION.cff) and [`../.zenodo.json`](../.zenodo.json) for per-submission Zenodo DOI minting via Pre-release Git Tags (`v0.1.0-paperXX`).

## Active Per-Paper Folders & Zenodo Release Tags

- `v0.1.0-paper53` (`reproducibility/paper_53/`): **Paper 53** (*Reliability Engineering & System Safety*) — Multi-Domain Operator-SARIMAX 9-Model Benchmark Ladder, Causal MODWT (`db4`, $J=3$), Dense & Sparse $k$-NN ($k=6$) Normalized Graph Laplacian Heat Diffusion (`Laplace-SARIMAX`, Theorem 4 & Remarks 1–2 Parallel Dual-Guard), exact Gück et al. (2024) CARE Score across Wind Farms A–C (36 turbines, 95 sub-datasets), $\pm 50\%$ sensitivity sweep, WPHM survival bounds, and dual-market 10,000-loop Monte Carlo risk sizing (native MIBEL `BZN|PT` vs. Nord Pool winter stress $\text{CVaR}_{95}$).
- `v0.1.0-paper52` (`reproducibility/paper_52/`): **Paper 52** (*Energy Conversion and Management*) — 4-layer Tobit + PIT Copula + Froude Geodesic + Weyl-NGBoost + Rank-1 Lattice QMC (`ecowind_core.weyl_convergent_lags`, `rank1_lattice_qmc`) & IEC 61400-15-1:2025 bankability.
- `v0.1.0-paper51` (`reproducibility/paper_51/`): **Paper 51** (*Automatica*) — Symplectic Störmer-Verlet null-geodesic solver + KAM-$\varphi^{-1}$ `jacobi_horizon_guard()` on NTNU 5 MW DS2 under JONSWAP waves.
- `v0.1.0-paper47`..`paper_50`: **Papers 47–50** (Volcano/BHY-FDR + Fisher liquidity horizon `Gap 6`, `PI-eMeans`, 13-model Boosting benchmark, SEM-GB + `bartlett_factor_project()`).
- Supplemental RMT & Geometric Horizons: `paper_15/` (Wigner-GUE 237-sensor denoiser), `paper_35/` (Spectral Zeta Determinant & Finsler Trapping Horizon `Gap 3`), `paper_36/` (Hessian Singularity bearing thermal runaway `Gap 5`).
