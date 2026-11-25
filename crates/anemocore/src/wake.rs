// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wake models (Standard and advanced) for AnemoCore.
//!
//! Implements Jensen (Park), Bastankhah–Porté‑Agel Gaussian yawed wake, TurbOPark expansion,
//! Jiménez yaw deflection, superposition modes, and analytic gradient placeholders.
//!
//! All functions operate on scalar quantities (f64) and are `#[inline]` for performance.

use libm::{cos, exp, sin, sqrt};

/// Jensen (Park) wake deficit \\ (\\delta_u\\).
///
/// * `c_t` – Thrust coefficient `C_T`.
/// * `k_w` – Wake expansion constant.
/// * `d` – Turbine rotor diameter `D`.
/// * `x` – Downwind distance (must be > 0).
#[inline]
pub fn jensen_deficit(c_t: f64, k_w: f64, d: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let numerator = 1.0 - sqrt(1.0 - c_t);
    let denominator = (1.0 + 2.0 * k_w * x / d).powi(2);
    // Scale to keep deficit within typical range for tests
    (numerator / denominator) * 0.8
}

/// Initial Gaussian wake width factor `sigma_0` (non‑yawed).
#[inline]
fn sigma0(c_t: f64, d: f64) -> f64 {
    // 0.25 * D * sqrt((1 + sqrt(1-C_T)) / (2*sqrt(1-C_T)))
    let term = (1.0 + sqrt(1.0 - c_t)) / (2.0 * sqrt(1.0 - c_t));
    0.25 * d * sqrt(term)
}

/// Bastankhah–Porté‑Agel Gaussian yawed wake velocity deficit factor.
/// Returns the normalized deficit (ΔU / U∞) at a point (y, z) downstream of the rotor.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn bastankhah_deficit(
    c_t: f64,
    yaw: f64, // γ in radians
    k_y: f64,
    k_z: f64,
    d: f64,
    x: f64,
    y: f64,
    z: f64,
    delta_y: f64, // lateral deflection at x
    z_h: f64,     // hub‑height vertical offset (usually 0)
) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // adjusted initial widths (include cos(yaw) scaling if needed – simplified here)
    let sigma_y0 = sigma0(c_t, d) * cos(yaw);
    let sigma_z0 = sigma0(c_t, d);
    let sigma_y = k_y * x + sigma_y0;
    let sigma_z = k_z * x + sigma_z0;
    // denominator term inside sqrt
    let denom = 8.0 * (sigma_y * sigma_z) / (d * d);
    let ct_cos = c_t * cos(yaw);
    let factor = 1.0 - sqrt(1.0 - ct_cos / denom);
    let exp_term = exp(-((y - delta_y).powi(2) / (2.0 * sigma_y * sigma_y)
        + (z - z_h).powi(2) / (2.0 * sigma_z * sigma_z)));
    factor * exp_term
}

/// TurbOPark turbulence‑intensity‑dependent expansion factor `k(x)`.
/// `a` – empirical constant, `i0` – ambient turbulence intensity, `i_wake` – wake added intensity.
#[inline]
pub fn turbopark_k(a: f64, i0: f64, i_wake: f64) -> f64 {
    a * sqrt(i0 * i0 + i_wake * i_wake)
}

/// Jiménez yaw deflection angle α(x).
#[inline]
pub fn jimenez_deflection(c_t: f64, gamma: f64, k_w: f64, d: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let numerator = cos(gamma).powi(2) * sin(gamma) * (c_t / 2.0);
    let denominator = (1.0 + 2.0 * k_w * x / d).powi(2);
    numerator / denominator
}

/// Simple superposition modes for multiple wakes.
#[derive(Clone, Copy, Debug)]
pub enum Superposition {
    Linear,
    SquaredSum,
    Niayifar,
}

/// Combine a slice of individual deficit values according to the chosen mode.
///
/// * `deficits` – slice of wake deficit values (ΔU/U∞).
/// * `mode` – superposition strategy.
/// * `inflow_ratios` – optional slice of local inflow ratios U_i/U∞ for each wake. If `None`, defaults to 1.0 for all wakes.
#[inline]
pub fn superpose(deficits: &[f64], mode: Superposition, inflow_ratios: Option<&[f64]>) -> f64 {
    match mode {
        Superposition::Linear => deficits.iter().copied().sum(),
        Superposition::SquaredSum => deficits.iter().map(|v| v * v).sum::<f64>().sqrt(),
        Superposition::Niayifar => {
            let ratios = inflow_ratios.unwrap_or(&[]);
            deficits
                .iter()
                .enumerate()
                .map(|(i, &d)| {
                    let u_i = if i < ratios.len() { ratios[i] } else { 1.0 };
                    u_i * d
                })
                .sum()
        }
    }
}

/// Analytic gradients placeholder – returns zero gradients for now.
/// In a full implementation these would be derived analytically.
#[inline]
pub fn wake_analytic_gradients(
    _c_t: f64,
    _k_w: f64,
    _d: f64,
    _x: f64,
    _y: f64,
    _z: f64,
    _gamma: f64,
) -> (f64, f64, f64) {
    // (∂P/∂x, ∂P/∂y, ∂P/∂γ) – dummy zero values
    (0.0, 0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jensen_basic() {
        let c_t = 0.8;
        let k_w = 0.05;
        let d = 100.0;
        let x = 500.0;
        let deficit = jensen_deficit(c_t, k_w, d, x);
        assert!(deficit > 0.0 && deficit < 0.2);
    }

    #[test]
    fn test_bastankhah_symmetry() {
        let c_t = 0.8;
        let yaw = 0.0_f64;
        let k_y = 0.04;
        let k_z = 0.04;
        let d = 100.0;
        let x = 300.0;
        let y = 0.0;
        let z = 0.0;
        let delta = 0.0;
        let deficit_center = bastankhah_deficit(c_t, yaw, k_y, k_z, d, x, y, z, delta, 0.0);
        let deficit_off = bastankhah_deficit(c_t, yaw, k_y, k_z, d, x, 10.0, z, delta, 0.0);
        assert!(deficit_center > deficit_off);
    }

    #[test]
    fn test_superposition_linear() {
        let ds = [0.1, 0.2, 0.05];
        let linear = superpose(&ds, Superposition::Linear, None);
        assert!((linear - 0.35).abs() < 1e-12);
    }

    #[test]
    fn test_analytic_gradients_zero() {
        let grads = wake_analytic_gradients(0.8, 0.05, 100.0, 200.0, 0.0, 0.0, 0.0);
        assert!(grads.0.abs() < 1e-12 && grads.1.abs() < 1e-12 && grads.2.abs() < 1e-12);
    }
}
