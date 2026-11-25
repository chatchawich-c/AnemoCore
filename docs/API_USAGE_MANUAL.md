# AnemoCore v0.2.0 — API & Function Reference Manual

> **Release:** `v0.2.0` | **Rust Crate:** `anemocore` (`anemocore_crate`) | **Python Package:** `anemocore` (`PyO3` / `cp39-abi3`) | **R Package:** `anemocorer` (`extendr`)

---

## Table of Contents

1. [Module 1: `weibull_est` & `wra` — Weibull Estimation, 5-Format WRA I/O & Vertical Wind Shear](#1-weibull_est-wra)
2. [Module 2: `microsite`, `wake` & `oa` — Wind Atlas, Wake Superposition & Bankable AEP](#2-microsite-wake-oa)
3. [Module 3: `thermo` & `tribology` — Drivetrain Thermal-Tribology & Bearing Reliability](#3-thermo-tribology)
4. [Module 4: `math_no_std` — Pure `no_std` Math Functions (Acklam Probit & Gaussian CDF)](#4-math_no_std)
5. [Module 5: `kam_vsg` — KAM Golden‑Ratio VSG, Biquad Notch & 4‑Stage Foster RC](#5-kam_vsg)
6. [Module 6: `quasicrystal_wpt` — Integer Shift‑Add Fibonacci Filter (Zero-FPU for Cortex‑M4)](#6-quasicrystal_wpt)
7. [Module 7: `fib_qmc` — 6‑D Fibonacci‑Korobov QMC & Golden‑Section Optimal Stopping](#7-fib_qmc)
8. [Module 8: `signal_wpt` — Daubechies‑4 3‑Level WPT, Rényi Entropy `H_3` & Hertzian Contact](#8-signal_wpt)
9. [Module 9: `fib_liquidity` — Fibonacci LOB Density, Price SDE & Roll's Effective Spread](#9-fib_liquidity)
10. [Module 10: `phyllotaxis_wake` — Vogel Golden‑Angle Farm, Structure Factor `S(q)` & Wake‑Thermal ODE](#10-phyllotaxis_wake)
11. [Comparison Table of Python (`PyO3`) and R (`extendr`) Functions](#11-comparison-table-of-python-pyo3-and-r-extendr-functions)
12. [Troubleshooting Guide](#12-troubleshooting-guide)

---

## 1. `weibull_est` & `wra`

> Core wind resource assessment (WRA), statistical estimators, and vertical shear suite (v0.2.0).

| Function | Parameters | Return Value | Description |
| :--- | :--- | :--- | :--- |
| `fit_mle_raw(samples)` | `samples: &[f64]` | `Result<WeibullFit, &'static str>` | Maximum Likelihood Estimation (MLE) for 2-parameter Weibull `(k, c)`. |
| `fit_mmle_binned` / `fit_lsm_binned` | Histogram bin counts & boundaries | `WeibullFit` | Modified MLE and Least-Squares Method for binned wind speed data. |
| `fit_mom` / `fit_epfm` / `fit_empirical_justus` | `mean`, `std_dev` / `mean_cube` | `WeibullFit` | Method of Moments, Energy Pattern Factor Method, and Empirical Justus fit. |
| `extrapolate_wind_speed(u_ref, z_ref, z_target, model)` | Heights `[m]`, `ShearModel` enum | `f64` | Extrapolates wind speed using `PowerLaw` ($\alpha = 0.143$), `LogLawNeutral`, `MoninObukhov`, `GryningExtended`, or `JustusMikhailWeibull`. |
| `extrapolate_weibull_justus_mikhail(a_ref, k_ref, z_ref, z_target)` | Scale `a`, shape `k`, heights `[m]` | `(f64, f64)` | Extrapolates Weibull scale and shape parameters vertically. |
| WRA I/O Parsers (`io` submodule) | File path (`&str`) | Struct / Grid | Round-trip support for `.tab` (WAsP), `.gwc`/`.lib`, `.rsf`/`.wrg`, IEA Task 43 JSON, and ESRI ASCII `.asc`/`.dem`. |

---

## 2. `microsite`, `wake` & `oa`

> Wind atlas transformations, `windIO`-verified wake models ($\le 10^{-4}$ parity), and IEC 61400-12-1 operational assessment (v0.2.0).

* **`geostrophic_from_atlas(k, c, z0)` / `atlas_from_geostrophic(speed, direction)`**: Bidirectional geostrophic drag law conversion between surface Weibull parameters and geostrophic wind climate.
* **Wake Deficit & Deflection (`wake`)**: `jensen_deficit`, `bastankhah_deficit` (2014/2016 Gaussian), `turbopark_k`, `jimenez_deflection`, and `superpose(deficits, mode, inflow_ratios)`.
* **Operational Assessment (`oa`)**:
  * `normalize_wind_speed_iec(v, rho_kg_m3)` — IEC 61400-12-1 air-density wind speed normalization.
  * `clean_scada_iec61400_12_1(raw)` & `pava_monotonic_health(h)` — Automated SCADA filtering and Pool Adjacent Violators Algorithm (PAVA) monotonic power-curve health enforcement.
  * `compute_aep_full` / `compute_aep_full_with_model` / `estimate_bankable_aep` — Full Annual Energy Production (AEP) reporting with `PowerCurveModel` (`Tabular`, `Cubic`, `Logistic4P`).

---

## 3. `thermo` & `tribology`

> Drivetrain thermal-tribology physics and bearing life modeling (v0.2.0).

* **`moist_air_density_cipm2007(pressure_pa, temp_k, relative_humidity)`**: Full BIPM CIPM-2007 equation for moist air density $\rho$ ($\text{kg/m}^3$).
* **Thermal Indicators & Cumulative Dose (`thermo`)**:
  * `compute_thermal_stress_indicators` ($P_1$ excursion, $P_2$ rate-of-rise, $P_3$ ambient-adjusted rise, $P_4$ hotspot margin, $P_5$ exceedance increment).
  * `ctd_p6(t_k_series, t_ref_k, ea_j_mol, dt_sec)` — Arrhenius Cumulative Thermal Dose ($P_6$).
  * `sdple_gearbox_power_loss` & `step_gearbox_temperature_transient` — ISO 14179 gear mesh/churning and bearing load/viscous power losses plus transient sump temperature ODE.
* **Lubrication & Bearing Fatigue (`tribology`)**:
  * `dowson_higginson_film` & `ehl_two_stage` — Elastohydrodynamic lubrication (EHL) central/minimum film thickness and $\lambda$-ratio regime classification.
  * `iso281_modified_rating_life_l10m` & `iso16281_tilt_derating_factor` — Modified bearing rating life $L_{10m}$ and misalignment tilt derating.
  * `caputo_peridynamics_damage(slice, dt_sec, alpha, critical_stretch_s0)` — Fractional Caputo peridynamics fatigue damage accumulation.

---

## 4. `math_no_std`

> Fully `#![no_std]` – uses only `libm` (see P24, P34, P46)

| Function | Input Parameter | Return Value | Details & Formula |
| :--- | :--- | :--- | :--- |
| `inv_norm_cdf(p: f64)` | `p` probability `(0, 1)` (clamped to `[1e-15, 1‑1e-15]`) | Standard Z‑score (`f64`) | Replaces `scipy.stats.norm.ppf` with Acklam minimax approximation (relative error `< 1.15e‑9`). |
| `norm_cdf(z: f64)` | `z` Z‑score (`f64`) | `Phi(z) ∈ [0, 1]` | `0.5 * (1 + erf(z / sqrt(2)))`. |
| `wrap_unit_interval(x: f64)` | Any real `x` | Fractional part in `[0.0, 1.0)` | Replaces `np.mod(x, 1.0)` on `#![no_std]` with `x - libm::floor(x)`. |

---

## 5. `kam_vsg`

> Fully `#![no_std]`, O(1) per step, no heap allocation (see P42, P19, P30)

### Core Structs & Enums

* **`ControlMode`**: `GridFollowing` | `StandardVsg` | `FixedNotchVsg` | `KamGoldenRatioVsg`
* **`KamVsgSpec`**:
  * `KamVsgSpec::p42_default()` – 5 MW turbine (`J_r = 3.5e7`, `J_g = 5.3e5`, `K_0 = 8.67e8 N·m/rad`, `eps_K1 = 1.25e8 N·m/rad³`, `R_th = [0.012, 0.024, 0.018, 0.008] K/W`, `C_th = [0.85, 3.2, 14.5, 62.0] J/K`).
  * `KamVsgSpec::p19_foster_preset()` – Foster RC preset for P19 (`TAU_TH = [0.0015, 0.020, 0.150, 1.200] s`, `t_coolant_c = 45.0`, `delta_t_hs_c = 18.0`).
* **`KamVsgState`** – stores `{ theta_s [rad], dtheta_s [rad/s], v_foster: [f64; 4] [°C] }` (initialised with `KamVsgState::zero()`).

### Main Functions

1. `step_kam_vsg(spec, state, mode, rocof, delta_f, t_aero_disturb_nm, dt) -> KamVsgStepOutput`
   * `rocof`: Grid frequency change rate (`Hz/s`).
   * `delta_f`: Grid frequency deviation (`Hz`, e.g., `-0.26` for `< 50 Hz` events).
   * `t_aero_disturb_nm`: Aerodynamic torque disturbance (`N·m`, set `0.0` for no disturbance).
   * `dt`: Sample period (`s`, default `0.001` = 1 kHz).
   * **Returns**: `KamVsgStepOutput` with fields `k_eff`, `omega_n`, `omega_ctrl`, `m_vsg`, `d_vsg`, `p_wt_mw`, `t_em_nm`, `t_elastic_nm`, `ddtheta`, `delta_tj_jc_c`, `t_junction_c`.
2. `step_foster_rc4(v_foster, r_th, tau_th, p_loss_w, dt) -> f64` — Computes exact exponential integration of a 4‑stage thermal chain, returning `delta_T_jc` (`°C`).
3. `BiquadNotchFilter::new(omega_n, zeta_1, zeta_2, dt)` + `.step(u0)` — Second‑order bilinear notch filter with `zeta_1 = 0.05`, `zeta_2 = 0.707` (Paper 19).

```rust
use anemocore::kam_vsg::{step_kam_vsg, BiquadNotchFilter, ControlMode, KamVsgSpec, KamVsgState};

let spec = KamVsgSpec::p42_default();
let mut state = KamVsgState::zero();
let mut notch = BiquadNotchFilter::new(40.75, 0.05, 0.707, 0.001);

for _ in 0..1000 {
    let out = step_kam_vsg(&spec, &mut state, ControlMode::KamGoldenRatioVsg, -0.05, -0.02, 0.0, 0.001);
    let _p_filtered = notch.step(out.p_wt_mw);
}
```

---

## 6. `quasicrystal_wpt`

> Fully `#![no_std]` and **zero floating‑point operations** in the core filter (see P44).

| Function | Parameters | Description |
| :--- | :--- | :--- |
| `mul_phi_inv_shift_add(x: i32) -> i32` | Integer signal `x` (e.g., 12‑24 bit ADC) | Approximates `x * φ⁻¹` using `(x>>1)+(x>>3)-(x>>7)+(x>>10)` (error `< 0.021%`). |
| `mul_phi_inv2_shift_add(x: i32) -> i32` | Integer signal `x` | Computes `x * φ⁻² = x - mul_phi_inv_shift_add(x)`. |
| `shift_add_forward_step(signal, lo, hi)` | `signal: &[i32]` (even length), `lo/hi: &mut [i32]` | Performs one lifting step of a 2‑level integer wavelet transform. |
| `shift_add_inverse_step(lo, hi, recon)` | `lo, hi: &[i32]`, `recon: &mut [i32]` | Reconstructs the signal losslessly (bit‑for‑bit). |
| `shift_add_wpt_3level_band_energy(sig, sa, sb, bands)` | `sig.len()` divisible by 8, `sa/sb` length `≥ sig.len()`, `bands: &mut [u64; 8]` | Decomposes three levels into energy for 8 frequency bands without heap allocation. |

---

## 7. `fib_qmc`

> Fully `#![no_std]`, O(1) RAM – does **not** allocate the full sample lattice in memory (see P46).

| Function | Input | Returns |
| :--- | :--- | :--- |
| `fib_lattice_2d_point(k, n, n_prev)` | Index `k`, Fibonacci numbers `F_n = 4181`, `F_{n‑1} = 2584` | `[u1, u2]` in `[0, 1)²`. |
| `fib_lattice_6d_point(k, n, a_korobov, shift)` | `k`, `4181`, `a_korobov = 719`, `shift: &[f64; 6]` | 6‑D Cranley‑Patterson shifted point. |
| `evaluate_limit_state_6d(u: &[f64; 6])` | 6‑D point in `[0, 1)⁶` | `LimitStateEval { r_cap, s_load, g_val, dtf }`. |
| `value_difference_functional(spot_price, dtf)` | `spot_price [EUR/MWh]`, `dtf ∈ [0, 1]` | Difference `V_cont – V_stop`. |
| `golden_section_stopping_search(dtf, a0, b0, tol, max_iter)` | Example `(0.35, 5.0, 160.0, 1e‑5, 35)` | `StoppingSearchResult { s_star, final_width, iterations }`. |
| `estimate_failure_prob_qmc_o1(shift: &[f64; 6])` | 6‑D random shift vector | `(P_f, mean_DtF)` from all `4,181` points using O(1) RAM. |
| `qmc_weibull_cubic_expectation(k_weib, c_weib, n_points)` | Weibull `(k, c)` and lattice sample count | Quasi-Monte Carlo estimate of $\mathbb{E}[V^3]$. |

---

## 8. `signal_wpt`

> Supports both `#![no_std]` and `std` (see P33, P09, P10).

* `extract_wpt_indicators_vec(v: &[f64]) -> Option<WptIndicators>`
  * **Requirements:** `v.len() >= 8` and divisible by 8.
  * **Outputs (`WptIndicators`):** `p_low_mid` (sub‑bands 1–4), `p_high` (resonance sub‑bands 5–8), `kappa_high` (average kurtosis), `h3_renyi` (Rényi entropy $\alpha = 3$), `band_energies: [f64; 8]`.
* `HertzianBearingSpec::p10_default().instantaneous_mechanics(p_rated, dt)` — Computes Hertzian contact mechanics for AISI 52100 bearing, returning `(k_h [N/m], f_c [Hz], τ_c [s], σ_t_scale [-])`.

---

## 9. `fib_liquidity`

> Requires the `std` feature (default for `anemocore`) (see P43, P08).

* `FibonacciWallConfig::p43_default()` – Fibonacci support levels `[0.236, 0.382, 0.500, 0.618, 0.786]` for price range `[40.0, 110.0] EUR/MWh`.
* `lob_density(i, &cfg)` – Limit Order Book density `ρ_LOB(i)`.
* `roll_effective_spread(bids: &[f64], ask: f64) -> Vec<f64>` & `roll_effective_spread_scalar(bid, ask)` – Calculates Roll (1984) bid‑ask spread.
* `fibonacci_depth(order_book, target_volume)` & `depth_slope_index(volumes, price_deltas, weights)` – Order book depth liquidity indicators.

---

## 10. `phyllotaxis_wake`

> Requires the `std` feature (default for `anemocore`) (see P45, P18).

| Function | Parameters | Description |
| :--- | :--- | :--- |
| `vogel_phyllotaxis_layout(n, r_min, c)` | `(100, 4.5, 90.0)` | Generates turbine coordinates `(Vec<f64>, Vec<f64>)` using golden‑angle `θ_g = 137.508°`. |
| `golden_spiral_layout(n, r_min, c)` | `(100, 4.5, 90.0)` | Returns turbine coordinate pairs `Vec<(f64, f64)>`. |
| `structure_factor_2d(&vx, &vy, &grid)` | Coordinates `vx, vy` and wave‑vector grid | Computes structure factor `S(q)` for Bragg diffraction detection. |
| `golden_yaw_staggering(&x, &y, D, wind_dir_deg, gamma_max_deg)` | Example `gamma_max_deg = 16.0` | Produces golden‑angle yaw‑stagger angles `[deg]` for upstream turbines. |
| `step_aerothermal_wake(&x, &y, &spec, u_inf, wind_dir_deg, t_amb_k, &yaw_deg, &mut t_oil_k_state, dt_s)` | `t_amb_k`, `t_oil_k_state` in **Kelvin**; `dt_s` in seconds (e.g., `600.0`) | Calculates Bastankhah 3‑D wake + Crespo‑Hernández TI + exact exponential gearbox oil thermal ODE. |
| `compute_ctd_hours(&t_oil_k, &spec, dt)` | Time series `T_oil [K]` | Computes **Cumulative Thermal Dose (CTD)** relative to reference temperature `T_ref = 343.15 K`. |

---

## 11. Comparison Table of Python (`PyO3`) and R (`extendr`) Functions

| Capability | Python (`import anemocore`) | High-Level Python Façade | R (`anemocorer`) |
| :--- | :--- | :--- | :--- |
| Weibull Fit (MLE) | `anemocore.fit_weibull(data)` | `WindClimate(data)` | `r_fit_weibull(data)` |
| QMC Cubic Speed $\mathbb{E}[V^3]$ | `anemocore.qmc_weibull_cubic(k, c, n)` | `WindClimate.cubic_expectation(n)` | `r_qmc_weibull_cubic(k, c, n)` |
| Vertical Wind Shear | `anemocore._core.py_extrapolate_wind_speed(...)` | `WindClimate` | `r_extrapolate_wind_speed(...)` |
| WRA File Parsers | `anemocore._core.py_parse_lib` / `py_parse_rsf` | `WindClimate` / `WindFarm` | `r_parse_lib` / `r_parse_rsf` |
| Full Bankable AEP | `anemocore._core.py_compute_aep_full(...)` | `WindFarm` | `r_compute_aep_full(...)` |
| Geostrophic Wind | `anemocore.py_geostrophic_from_atlas(...)` | `WindClimate` | `r_geostrophic_from_atlas(...)` |
| KAM‑VSG 1 Step | `anemocore.py_kam_vsg_step(...)` | `WindFarm` | `r_kam_vsg_step(...)` |
| Thermal Stress ($P_1$–$P_5$) | `anemocore.py_thermal_stress_indicators(...)` | `Drivetrain.run(...)` | `r_thermal_stress_indicators(...)` |
| Arrhenius CTD ($P_6$) | `anemocore.py_arrhenius_ctd_p6(...)` | `Drivetrain.run(...)` | `r_arrhenius_ctd_p6(...)` |
| CIPM‑2007 Air Density | `anemocore.py_moist_air_density_cipm2007(...)` | `Drivetrain.run(...)` | `r_moist_air_density_cipm2007(...)` |
| Gearbox Loss & Temp | `anemocore.py_gearbox_power_loss(...)` | `Drivetrain.run(...)` | `r_gearbox_power_loss(...)` |
| EHL & ISO 281 $L_{10m}$ | `anemocore.py_ehl_two_stage` / `py_iso281_l10m` | `Drivetrain.run(...)` | `r_ehl_two_stage` / `r_iso281_l10m` |
| Caputo Damage | `anemocore.py_caputo_peridynamics_damage(...)` | `Drivetrain.run(...)` | `r_caputo_peridynamics_damage(...)` |

---

## 12. Troubleshooting Guide

1. **Error `Undefined symbols … _Py_*` or missing `libR` when running `cargo test`:**
   Ensure you run `cargo test` from the repository root (`AnemoCore/`) where `default-members = ["crates/anemocore"]` is defined. Avoid `cargo test --workspace` unless you have built the PyO3 bindings via `maturin develop` or set `R_HOME` correctly.
2. **Error `no method named abs/sqrt/exp/powf` in `#![no_std]` mode:**
   In `#![no_std]` modules, replace direct calls like `.abs()`, `.sqrt()`, `.exp()`, `.powf()` on `f64` with `libm::{fabs, sqrt, exp, pow, log, log2}`.
3. **Python PyO3 throws `ValueError: Signal array must be contiguous f64`:**
   When passing a NumPy slice (e.g., `x[::2]`) or a DataFrame column, first call `np.ascontiguousarray(x, dtype=np.float64)` (or `np.int32` for Shift‑Add functions) before invoking the Rust binding.
4. **Temperature units mismatch between `phyllotaxis_wake` and `kam_vsg`:**
   `kam_vsg` uses **Celsius (°C)** for `t_coolant_c` / `t_junction_c`. `phyllotaxis_wake` uses **Kelvin (K)** for `t_amb_k`, `t_oil_k_state`, and `compute_ctd_hours`. Convert accordingly (`K = °C + 273.15`).
