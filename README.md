# AnemoCore

AnemoCore (`anemocore`) is a pure‑Rust computational engine for wind resource assessment, three‑dimensional Gaussian‑Curl wake modeling, sectoral AEP, signal processing, and embedded digital twins on micro‑controllers. The library is dual‑licensed under MIT OR Apache‑2.0 and conforms to IEC 61400‑12‑1, CIPM‑2007, ISO 281/16281, ISO 14179, and windIO parity requirements.

The project was initiated during a research exchange at Tallinn University of Technology, Estonia, with funding from the EU Erasmus+ CBHE Project ECO‑WIND (Grant No. ERASMUS‑EDU‑2025‑CBHE‑101177220).

---

## 1. Naming Convention (Hybrid Naming Convention)

* **Display / Prose / Repo:** `AnemoCore`
* **Code Identifiers:** `anemocore` (Rust crate `anemocore`, Python module `anemocore`, R package `anemocore`)
* **WAsP Compatibility:** Supports reading and exporting sectoral wind‑rose `.tab` files via the `microsite` module (trademark‑safe).

---

## 2. Architecture – 16 Modules of `AnemoCore`

### Group A: `#![no_std]` Embedded‑Safe (8 modules — Zero‑Heap, ready for STM32F401)
| Module | Status | Core Functionality |
| :--- | :---: | :--- |
| `fib_qmc.rs` | **Phase 1** | Rank‑1 Fibonacci/Korobov QMC, `weyl_convergent_lags`, Tent Transform, Golden‑Section Search, Streaming Weibull $E[v^3]$ |
| `weibull_est.rs` | **Phase 1** | `FixedWindHistogram` (Zero‑Heap) + 6 estimation equations (EMJ, MoM, EPFM/PDM, Raw MLE, Binned MMLE, Graphical LSM) + Goodness‑of‑Fit ($R^2$, RMSE, KS) |
| `transforms.rs` | **Phase 1** | Goertzel single‑bin DFT, `microfft` 64‑pt FFT/IFFT/Hilbert, Numerical inverse Laplace (Stehfest, Talbot), Tustin $s \to z$ |
| `thermo.rs` | Phase 4 | First‑law thermal network, Arrhenius `CTD_P6`, CIPM‑2007 moist air density, ISO 281 |
| `kam_vsg.rs` | Phase 4 | Two‑mass shaft, four‑stage Foster RC IGBT, KAM golden‑ratio VSG, `jacobi_horizon_guard` |
| `quasicrystal_wpt.rs` | Phase 4 | 1D Fibonacci quasicrystal multiplier‑less shift‑add filter bank |
| `edge_governor.rs` | Phase 5 | 64‑byte aligned lock‑free SPSC ring buffer, branchless Grönwall guard, `bartlett_factor_project`, `stealth_fdia_wind_residual` |
| `pi_emeans.rs` | Phase 4 | Physics‑informed thermodynamic centroid clustering |

### Group B: `std` High‑Performance (8 modules — `rayon` + `faer` + `rustfft`, Peak RSS ≤ 150 MiB on a 2017 MacBook Pro)
| Module | Status | Core Functionality |
| :--- | :---: | :--- |
| `microsite.rs` | **Phase 1** | Monin–Obukhov log‑law, Elliott IBL, `CfdSpeedupGrid` (f32 trilinear + circular‑sector), `.tab` I/O, `TurbineSpec` (IEA‑3.4MW, NREL‑5MW, IEA‑15MW, parametric, auto‑$C_t$), Gross AEP |
| `phyllotaxis_wake.rs` | **Phase 1** | 3D Gaussian‑Curl wake (Bastankhah–Porté‑Agel), RSS superposition, golden‑angle yaw‑staggering, Net AEP & Wake loss (%) |
| `signal_wpt.rs` | Phase 4 | STFT, FSST, SET, Hertzian–Gabor, Rényi entropy, tacholess resampling, $J=3$ WPT |
| `fib_liquidity.rs` | Phase 4 | Roll spread, discrete scale invariance, skew‑Brownian HJB‑VI, Pontryagin LATFC |
| `tribology.rs` | Phase 4 | Hertzian contact, Dowson–Higginson EHL, Roelands–VFT, Lemaitre CDM, Paris–Erdogan |
| `topology_braids.rs` | Phase 4 | 3D Takens embedding, Artin braid group, Alexander polynomial |
| `sheaf_graph.rs` | Phase 4 | Cellular cosheaf Laplacian, first cohomology, matrix product states (MPS) |
| `reg_econ.rs` | Phase 5 | Baron–Myerson audit, ECPR + Pigouvian wake fee, TOTEX PBR, Wasserstein MFG |

---
## Installation

```bash
# Python installation via PyPI
pip install anemocore

# Python installation via GitHub repository
python3 -m pip install "git+https://github.com/chatchawich-c/AnemoCore.git#subdirectory=bindings/python"

# R installation via remotes
R -q -e 'remotes::install_github("chatchawich-c/AnemoCore", subdir = "bindings/r")'

# R-universe (optional)
R -q -e 'install.packages("anemocore", repos = c("https://chatchawich-c.r-universe.dev", "https://cloud.r-project.org"))'
```

## 3. Quickstart & Verification

```bash
# 1. Compile and run all unit tests
cargo test -p anemocore

# 2. Verify no‑std compilation for ARM Cortex‑M4
cargo check -p anemocore --no-default-features --target thumbv7em-none-eabihf

# 3. Run benchmark to measure Peak RSS on a 2017 MacBook Pro
python benchmarks/bench_fib_qmc_mbp2017.py --fib-order 28 --runs 10
```

---

## 4. Authors & Grant Attribution

* **Lead Author:** Chatchawich Chaihong (`ORCID: 0009-0000-7379-5738`, `6610130021@email.psu.ac.th`), Energy Technology Program, Department of Interdisciplinary Engineering, Faculty of Engineering, Prince of Songkla University (PSU), Hat Yai Campus, Songkhla 90110, Thailand.
* **Funding:** Initiated during an academic exchange at Tallinn University of Technology (TalTech), Estonia, under the **EU Erasmus+ CBHE Project ECO‑WIND** (*Enhancing Capacity in Condition‑based Maintenance of Wind Energy*, Grant No. `ERASMUS-EDU-2025-CBHE-101177220`).
* **License:** Dual‑licensed under [MIT](LICENSE-MIT) OR [Apache‑2.0](LICENSE-APACHE).
