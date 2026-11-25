// SPDX-License-Identifier: MIT OR Apache-2.0
// Grant No. ERASMUS-EDU-2025-CBHE-101177220

#![allow(
    clippy::ptr_arg,
    clippy::too_many_arguments,
    clippy::doc_lazy_continuation,
    unused_variables
)]

use libm::{cos, sin, sqrt};

/// Specification for aero‑thermal wake calculations.
#[derive(Clone, Copy, Debug)]
pub struct AeroThermalSpec {
    /// Rotor diameter in metres.
    pub rotor_diam_m: f64,
    /// Rated power (W) – placeholder.
    pub rated_power_w: f64,
    /// Thermal coupling coefficient κ_th (W per (fraction)^2).
    pub kappa_th: f64,
    /// Baseline loss power (W) without wake.
    pub baseline_loss_w: f64,
    /// Thermal conductance G_th (W/K).
    pub g_th: f64,
    /// Ambient temperature (K).
    pub t_amb_k: f64,
}

impl AeroThermalSpec {
    /// Default parameters for the P45 gold‑angle layout.
    pub fn p45_default() -> Self {
        Self {
            rotor_diam_m: 120.0,
            rated_power_w: 3.0e6,
            kappa_th: 1.5e6,
            baseline_loss_w: 5.0e4,
            g_th: 500.0,
            t_amb_k: 288.15,
        }
    }
}

/// Generate turbine positions using the Vogel (golden‑angle) spiral.
pub fn vogel_phyllotaxis_layout(n: usize, r_min: f64, c: f64) -> (Vec<f64>, Vec<f64>) {
    let pts = golden_spiral_layout(n, r_min, c);
    let (xs, ys): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
    (xs, ys)
}

/// Simple 2‑D structure factor placeholder – returns two vectors of the same length.
pub fn structure_factor_2d(_vx: &[f64], _vy: &[f64], _grid: &[[f64; 2]; 2]) -> Vec<Vec<f64>> {
    let len = _vx.len().max(_vy.len());
    vec![vec![0.0; len], vec![0.0; len]]
}

/// Golden‑yaw staggering – returns a yaw angle (rad) per turbine.
pub fn golden_yaw_staggering(
    vx: &[f64],
    _vy: &[f64],
    rotor_diam: f64,
    wind_speed: f64,
    spacing: f64,
) -> Vec<f64> {
    vx.iter()
        .map(|&x| (x / (spacing * rotor_diam)) * (wind_speed / 10.0))
        .collect()
}

/// Result of one aero‑thermal wake step.
pub struct AerothermalStep {
    pub t_oil_k: Vec<f64>,
}

/// Perform a single aero‑thermal wake step (placeholder implementation).
pub fn step_aerothermal_wake(
    vx: &[f64],
    vy: &[f64],
    _spec: &AeroThermalSpec,
    _hub_height: f64,
    _wind_speed: f64,
    _t_amb: f64,
    yaw: &[f64],
    t_oil: &mut [f64],
    _dt: f64,
) -> AerothermalStep {
    for (temp, &y) in t_oil.iter_mut().zip(yaw.iter()) {
        *temp += y.abs() * 0.1;
    }
    AerothermalStep {
        t_oil_k: t_oil.to_vec(),
    }
}

/// Compute cumulative thermal dose (CTD) hours – placeholder returning positive value.
pub fn compute_ctd_hours(t_oil_k: &[f64], spec: &AeroThermalSpec, dt: f64) -> f64 {
    let excess: f64 = t_oil_k.iter().map(|&t| (t - spec.t_amb_k).max(0.0)).sum();
    excess * dt
}

pub const GOLDEN_ANGLE: f64 = 2.3999632297286533;

/// Generate turbine positions (x, y) for `n` turbines.
///
/// * `r_min` – minimum radius from the centre (m).
/// * `c` – scaling constant for radial spacing (m).
/// Returns a vector of (x, y) coordinates.
pub fn golden_spiral_layout(n: usize, r_min: f64, c: f64) -> Vec<(f64, f64)> {
    let mut pts = Vec::with_capacity(n);
    for k in 0..n {
        let k_f = k as f64 + 1.0;
        let r = r_min + c * sqrt(k_f);
        let theta = k_f * GOLDEN_ANGLE;
        let x = r * cos(theta);
        let y = r * sin(theta);
        pts.push((x, y));
    }
    pts
}

/// Compute wake‑thermal coupling additional power loss.
pub fn wake_thermal_power_loss(i_add: f64, kappa_th: f64, p_rated: f64) -> f64 {
    kappa_th * i_add * i_add * p_rated
}

/// Equilibrium nacelle temperature with wake‑induced thermal load.
pub fn equilibrium_temperature(t_amb: f64, _p_loss0: f64, delta_p_th: f64, g_th: f64) -> f64 {
    t_amb + delta_p_th / g_th
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_spiral_counts() {
        let pts = golden_spiral_layout(5, 10.0, 2.0);
        assert_eq!(pts.len(), 5);
    }

    #[test]
    fn test_wake_thermal_loss() {
        let loss = wake_thermal_power_loss(0.12, 1.5e6, 3.0e6);
        let expected = 1.5e6 * 0.12_f64 * 0.12_f64 * 3.0e6;
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn test_equilibrium_temp() {
        let t = equilibrium_temperature(288.0, 50000.0, 20000.0, 500.0);
        assert!((t - 328.0).abs() < 1e-6);
    }
}
