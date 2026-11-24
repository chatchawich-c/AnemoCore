# AnemoCore API & Function Reference Manual (English)

---

## Table of Contents
1. Module 1: `math_no_std` – Pure `no_std` math functions (Acklam Probit & Gaussian CDF)
2. Module 2: `kam_vsg` – KAM Golden‑Ratio VSG, Biquad Notch, and 4‑Stage Foster RC
3. Module 3: `quasicrystal_wpt` – Integer Shift‑Add Fibonacci Filter (no FPU for Cortex‑M4)
4. Module 4: `fib_qmc` – 6‑D Fibonacci‑Korobov QMC & Golden‑Section Optimal Stopping
5. Module 5: `signal_wpt` – Daubechies‑4 3‑Level WPT, Rényi Entropy `H_3`, and Hertzian Contact
6. Module 6: `fib_liquidity` – Fibonacci LOB Density, Price SDE, and Roll's Effective Spread
7. Module 7: `phyllotaxis_wake` – Vogel Golden‑Angle Farm, Structure Factor `S(q)`, Wake‑Thermal ODE, and Arrhenius CTD
8. Comparison Table of Python (`PyO3`) and R (`extendr`) Functions
9. Troubleshooting Guide

---

## 1. `math_no_std`

> Fully `#![no_std]` – uses only `libm` (see P24, P34, P46)

| Function | Input Parameter | Return Value | Details & Formula |
| :--- | :--- | :--- | :--- |
| `inv_norm_cdf(p: f64)` | `p` probability `(0, 1)` (clamped to `[1e-15, 1‑1e-15]`) | Standard Z‑score (`f64`) | Replaces `scipy.stats.norm.ppf` with Acklam minimax approximation (relative error `< 1.15e‑9`). |
| `norm_cdf(z: f64)` | `z` Z‑score (`f64`) | `Phi(z) ∈ [0, 1]` | `0.5 * (1 + erf(z / sqrt(2)))`. |
| `wrap_unit_interval(x: f64)` | Any real `x` | Fractional part in `[0.0, 1.0)` | Replaces `np.mod(x, 1.0)` on `#![no_std]` with `x - libm::floor(x)`. |

---

## 2. `kam_vsg`

> Fully `#![no_std]`, O(1) per step, no heap allocation (see P42, P19, P30)

### Core Structs & Enums
* **`ControlMode`**: `GridFollowing` | `StandardVsg` | `FixedNotchVsg` | `KamGoldenRatioVsg`
* **`KamVsgSpec`**:
  * `KamVsgSpec::p42_default()` – 5 MW turbine (parameters `J_r = 3.5e7`, `J_g = 5.3e5`, `K_0 = 8.67e8 N·m/rad`, `eps_K1 = 1.25e8 N·m/rad³`, `R_th = [0.012, 0.024, 0.018, 0.008] K/W`, `C_th = [0.85, 3.2, 14.5, 62.0] J/K`).
  * `KamVsgSpec::p19_foster_preset()` – Foster RC preset for P19 (`TAU_TH = [0.0015, 0.020, 0.150, 1.200] s`, `t_coolant_c = 45.0`, `delta_t_hs_c = 18.0`).
* **`KamVsgState`** – stores `{ theta_s [rad], dtheta_s [rad/s], v_foster: [f64; 4] [°C] }` (initialised with `KamVsgState::zero()`).

### Main Functions
1. `step_kam_vsg(spec, state, mode, rocof, delta_f, t_aero_disturb_nm, dt) -> KamVsgStepOutput`
   * `rocof`: Grid frequency change rate (`Hz/s`).
   * `delta_f`: Grid frequency deviation (`Hz`, e.g., `‑0.26` for < 50 Hz events). 
   * `t_aero_disturb_nm`: Aerodynamic torque disturbance (`N·m`, set `0.0` for no disturbance).
   * `dt`: Sample period (`s`, default `0.001` = 1 kHz).
   * **Returns**: `KamVsgStepOutput` with fields `k_eff`, `omega_n`, `omega_ctrl`, `m_vsg`, `d_vsg`, `p_wt_mw`, `t_em_nm`, `t_elastic_nm`, `ddtheta`, `delta_tj_jc_c`, `t_junction_c`.
2. `step_foster_rc4(v_foster, r_th, tau_th, p_loss_w, dt) -> f64`
   * Computes exact exponential integration of a 4‑stage thermal chain, returning `delta_T_jc` (`°C`).
3. `BiquadNotchFilter::new(omega_n, zeta_1, zeta_2, dt)` + `.step(u0)`
   * Second‑order notch filter (bilinear) with `zeta_1 = 0.05`, `zeta_2 = 0.707` (Paper 19).

```rust
use ecowind_core::kam_vsg::{step_kam_vsg, BiquadNotchFilter, ControlMode, KamVsgSpec, KamVsgState};

let spec = KamVsgSpec::p42_default();
let mut state = KamVsgState::zero();
let mut notch = BiquadNotchFilter::new(40.75, 0.05, 0.707, 0.001);

for _ in 0..1000 {
    let out = step_kam_vsg(&spec, &mut state, ControlMode::KamGoldenRatioVsg, -0.05, -0.02, 0.0, 0.001);
    let _p_filtered = notch.step(out.p_wt_mw);
}
```

---

## 3. `quasicrystal_wpt`

> Fully `#![no_std]` and **zero floating‑point operations** in the core filter (see P44).

| Function | Parameters | Description |
| :--- | :--- | :--- |
| `mul_phi_inv_shift_add(x: i32) -> i32` | Integer signal `x` (e.g., 12‑24 bit ADC) | Approximates `x * φ⁻¹` using `(x>>1)+(x>>3)-(x>>7)+(x>>10)` (error `< 0.021%`). |
| `mul_phi_inv2_shift_add(x: i32) -> i32` | Integer signal `x` | Computes `x * φ⁻² = x - mul_phi_inv_shift_add(x)`. |
| `shift_add_forward_step(input, low, high) -> usize` | `input: &[i32]` (even length), `low/high: &mut [i32]` | Performs one lifting step of a 2‑level integer wavelet transform. |
| `shift_add_inverse_step(low, high, output) -> usize` | `low, high: &[i32]`, `output: &mut [i32]` | Reconstructs the signal losslessly (bit‑for‑bit). |
| `shift_add_wpt_3level_band_energy(sig, sa, sb, bands) -> usize` | `sig.len()` divisible by 8, `sa/sb` length `≥ sig.len()`, `bands: &mut [u64; 8]` | Decomposes three levels into energy for 8 frequency bands (ordered by true frequency) without heap allocation. |
| `frame_tightness_ratio(a, b) -> f64` | Defaults `FRAME_BOUND_A=2.482`, `FRAME_BOUND_B=2.492` | Returns frame tightness ratio (`~1.0040`). |

---

## 4. `fib_qmc`

> Fully `#![no_std]`, O(1) RAM – does **not** allocate the full sample lattice in memory (see P46).

| Function | Input | Returns |
| :--- | :--- | :--- |
| `fib_lattice_2d_point(k, n, n_prev)` | Index `k`, Fibonacci numbers `F_n = 4181`, `F_{n‑1} = 2584` | `[u1, u2]` in `[0, 1)²`. |
| `fib_lattice_6d_point(k, n, a_korobov, shift)` | `k`, `4181`, `a_korobov = 719`, `shift: &[f64; 6]` | 6‑D Cranley‑Patterson shifted point. |
| `evaluate_limit_state_6d(u: &[f64; 6])` | 6‑D point in `[0, 1)⁶` | `LimitStateEval { r_cap, s_load, g_val, dtf }`. |
| `value_difference_functional(spot_price, dtf)` | `spot_price [EUR/MWh]`, `dtf ∈ [0, 1]` | Difference `V_cont – V_stop`. |
| `golden_section_stopping_search(dtf, a0, b0, tol, max_iter)` | Example `(0.35, 5.0, 160.0, 1e‑5, 35)` | `StoppingSearchResult { s_star, final_width, iterations }`. |
| `estimate_failure_prob_qmc_o1(shift: &[f64; 6])` | 6‑D random shift vector | `(P_f, mean_DtF)` from all `4,181` points using O(1) RAM. |

---

## 5. `signal_wpt`

> Supports both `#![no_std]` (`extract_wpt_indicators_no_std`) and `std` (`extract_wpt_indicators_vec`) (see P33, P09, P10).

* `extract_wpt_indicators_no_std(signal: &[f64], scratch_a: &mut [f64], scratch_b: &mut [f64]) -> Result<WaveletIndicators, WptError>`
  * **Requirements:** `signal.len() ≥ 8` and divisible by 8; `scratch_a.len(), scratch_b.len() ≥ signal.len()`.
  * **Outputs (`WaveletIndicators`):**
    * `p_low_mid`: Energy proportion in low‑mid bands (sub‑bands 1‑4).
    * `p_high`: Energy proportion in high‑frequency resonance bands (sub‑bands 5‑8).
    * `kappa_high`: Average kurtosis (Pearson, `fisher = false`) on sub‑bands 5‑8.
    * `h3_renyi`: Rényi entropy of order `α = 3` (bits).
    * `band_energies: [f64; 8]`.
* `HertzianBearingSpec::p10_default().instantaneous_mechanics(q_n, e_ratio)`
  * Computes Hertzian contact mechanics for AISI 52100 bearing, returning `(k_h [N/m], f_c [Hz], τ_c [s], σ_t_scale [-])`.

---

## 6. `fib_liquidity`

> Requires the `std` feature (default for `ecowind-core`) (see P43, P08).

* `FibonacciWallConfig::p43_default()` – Fibonacci support levels `[0.236, 0.382, 0.500, 0.618, 0.786]` for price range `[40.0, 110.0] EUR/MWh`.
* `lob_density(z, &cfg)` – Limit Order Book density `ρ_LOB(z)`.
* `local_volatility(z, sigma_base, &cfg)` – Net volatility reduced by liquidity wall.
* `step_price_jump_diffusion(price_curr, p_mean, alpha_mr, sigma_base, dt_day, std_normal_draw, jump_eur_mwh, &cfg)` – One‑step SDE for electricity price, returning `(p_next, z, rho_lob)`.
* `roll_effective_spread(prices: &[f64], window: usize) -> Vec<f64>` – Calculates Roll (1984) bid‑ask spread with linear NaN interpolation and automatic fallback to Parkinson (1980) when `Cov ≥ 0` (O(N) prefix‑sum).

---

## 7. `phyllotopsis_wake`

> Requires the `std` feature (default for `ecowind-core`) (see P45, P18).

| Function | Parameters | Description |
| :--- | :--- | :--- |
| `vogel_phyllotaxis_layout(n_turb, s_min_d, rotor_diam_m)` | `(100, 4.5, 90.0)` | Generates turbine coordinates `(Vec<f64>, Vec<f64>)` using golden‑angle `θ_g = 137.508°`. |
| `min_normalized_spacing_d(&x, &y, rotor_diam_m)` | Coordinates `x, y` and rotor diameter `D` | Returns minimum spacing as a multiple of `D`. |
| `structure_factor_2d(&x, &y, &q_vecs)` | Array of wave vectors `[[qx, qy]]` | Computes structure factor `S(q)` for Bragg diffraction detection. |
| `golden_yaw_staggering(&x, &y, D, wind_dir_deg, gamma_max_deg)` | Example `gamma_max_deg = 16.0` | Produces golden‑angle yaw‑stagger angles `[deg]` for upstream turbines. |
| `step_aerothermal_wake(&x, &y, &spec, u_inf, wind_dir_deg, t_amb_k, &yaw_deg, &mut t_oil_k_state, dt_s)` | `t_amb_k`, `t_oil_k_state` in **Kelvin**; `dt_s` in seconds (e.g., `600.0`) | Calculates Bastankhah 3‑D wake + Crespo‑Hernández TI + exact exponential gearbox oil thermal ODE. |
| `compute_ctd_hours(&t_oil_k_series, &spec, dt_s)` | Time series `T_oil [K]` | Computes **Cumulative Thermal Dose (CTD)** relative to reference temperature `T_ref = 343.15 K`. |

---

## 8. Comparison Table of Python (`PyO3`) and R (`extendr`) Functions

| Capability | Python (`import ecowind_core`) | R (`ecowindr`) | Return |
| :--- | :--- | :--- | :--- |
| KAM‑VSG 1 Step | `kam_vsg_step(theta_s, rocof, delta_f, dt)` | `r_kam_vsg_step(theta_s, rocof, delta_f, dt)` | `(p_wt_mw, freq_ratio, t_junction_c)` |
| 3‑Level DB4 WPT | `wpt_db4_indicators(np_f64_1d)` | `r_wpt_db4_indicators(dbl_vec)` | `(p_low_mid, p_high, kappa_high, h3_renyi)` |
| Shift‑Add 8‑Band Energy | `quasicrystal_shift_add_energies(np_i32_1d)` | *(planned for Round 2)* | `[u64; 8]` |
| Shift‑Add Split 1 Stage | `quasicrystal_shift_add_split(np_i32_1d)` | *(planned for Round 2)* | `(np_i32_low, np_i32_high)` |
| 6‑D Limit State | `fib_qmc_limit_state([u0..u5])` | `r_fib_limit_state_6d(u_vec6)` | `(r_cap, s_load, g_val, dtf)` |
| Golden Stopping `S*` | `fib_golden_stopping(dtf)` | `r_fib_golden_stopping(dtf)` | Python: `(s_star, iters)` / R: `c(s_star, iters, width)` |
| O(1) 4,181‑Pt QMC Sweep | `fib_qmc_sweep_o1([s0..s5])` | *(planned for Round 2)* | `(p_f, mean_dtf)` |
| Roll Effective Spread | `market_roll_spread(np_f64_1d, window)` | `r_market_roll_spread(dbl_vec, window)` | Vector length equals input. |
| Fibonacci LOB Density | `market_lob_density(z)` | `r_lob_density(z)` | `ρ_lob` (`f64`). |
| Vogel Wake Snapshot | `phyllotaxis_aerothermal_snapshot(n, u_inf, dir_deg, t_amb_c, dt_s)` | *(planned for Round 2)* | `(mean_p_mw, max_t_oil_c, ctd_hrs)` |

---

## 9. Troubleshooting Guide

1. **Error `Undefined symbols … _Py_*` or missing `libR` when running `cargo test`:**
   * Ensure you run `cargo test` from the repository root (`ecowind-suite/`) where `default-members = ["crates/ecowind-core"]` is defined. Avoid `cargo test --workspace` unless you have built the PyO3 bindings via `maturin develop` or set `R_HOME` correctly.
2. **Error `no method named abs/sqrt/exp/powf` in `#![no_std]` mode:**
   * In `#![no_std]` modules, replace direct calls like `.abs()`, `.sqrt()`, `.exp()`, `.powf()` on `f64` with `libm::{fabs, sqrt, exp, pow, log, log2}`.
3. **Python PyO3 throws `ValueError: Signal array must be contiguous f64`:**
   * When passing a NumPy slice (e.g., `x[::2]`) or a DataFrame column, first call `np.ascontiguousarray(x, dtype=np.float64)` (or `np.int32` for Shift‑Add functions) before invoking the Rust binding.
4. **Temperature units mismatch between `phyllotaxis_wake` and `kam_vsg`:**
   * `kam_vsg` uses **Celsius (°C)** for `t_coolant_c` / `t_junction_c`. `phyllotaxis_wake` uses **Kelvin (K)** for `t_amb_k`, `t_oil_k_state`, and `compute_ctd_hours`. Convert accordingly (`K = °C + 273.15`).
