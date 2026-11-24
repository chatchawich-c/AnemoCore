//! R `extendr` bindings exposing `ecowind-core` mathematical kernels to R scripts
//! (Papers 07, 31, 32, 33) via `rextendr`.

use ecowind_core::fib_liquidity::{lob_density, roll_effective_spread, FibonacciWallConfig};
use ecowind_core::fib_qmc::{evaluate_limit_state_6d, golden_section_stopping_search};
use ecowind_core::kam_vsg::{step_kam_vsg, ControlMode, KamVsgSpec, KamVsgState};
use ecowind_core::signal_wpt::extract_wpt_indicators_vec;
use extendr_api::prelude::*;

/// Computes 3-level Daubechies-4 WPT indicators `(p_low_mid, p_high, kappa_high, h3_renyi)`
/// matching Paper 33 (`W-SEM` / `W-MGSEM`).
#[extendr]
fn r_wpt_db4_indicators(signal: &[f64]) -> Vec<f64> {
    match extract_wpt_indicators_vec(signal) {
        Ok(res) => vec![res.p_low_mid, res.p_high, res.kappa_high, res.h3_renyi],
        Err(_) => vec![f64::NAN; 4],
    }
}

/// Computes Roll's (1984) effective bid-ask spread with Parkinson fallback (`Paper 8/43`).
#[extendr]
fn r_market_roll_spread(prices: &[f64], window: i32) -> Vec<f64> {
    roll_effective_spread(prices, window.max(1) as usize)
}

/// Computes Paper 43 Fibonacci Limit-Order-Book density `rho_LOB(z)`.
#[extendr]
fn r_lob_density(z: f64) -> f64 {
    lob_density(z, &FibonacciWallConfig::p43_default())
}

/// Evaluates Paper 46 Golden-Section optimal stopping price `S*(DtF)`.
#[extendr]
fn r_fib_golden_stopping(dtf: f64) -> Vec<f64> {
    let res = golden_section_stopping_search(dtf, 5.0, 160.0, 1e-5, 35);
    vec![res.s_star, res.iterations as f64, res.final_width]
}

/// Evaluates Paper 46 6D structural limit state `(R_cap, S_load, g_val, DtF)`.
#[extendr]
fn r_fib_limit_state_6d(u: &[f64]) -> Vec<f64> {
    if u.len() < 6 {
        return vec![f64::NAN; 4];
    }
    let arr = [u[0], u[1], u[2], u[3], u[4], u[5]];
    let r = evaluate_limit_state_6d(&arr);
    vec![r.r_cap, r.s_load, r.g_val, r.dtf]
}

/// Evaluates one O(1) KAM Golden-Ratio VSG step -> `(p_wt_mw, freq_ratio, t_junction_c)`.
#[extendr]
fn r_kam_vsg_step(theta_s: f64, rocof: f64, delta_f: f64, dt: f64) -> Vec<f64> {
    let spec = KamVsgSpec::p42_default();
    let mut state = KamVsgState {
        theta_s,
        dtheta_s: 0.0,
        v_foster: [0.0; 4],
    };
    let out = step_kam_vsg(
        &spec,
        &mut state,
        ControlMode::KamGoldenRatioVsg,
        rocof,
        delta_f,
        0.0,
        dt,
    );
    vec![out.p_wt_mw, out.omega_ctrl / out.omega_n, out.t_junction_c]
}

extendr_module! {
    mod ecowindr;
    fn r_wpt_db4_indicators;
    fn r_market_roll_spread;
    fn r_lob_density;
    fn r_fib_golden_stopping;
    fn r_fib_limit_state_6d;
    fn r_kam_vsg_step;
}
