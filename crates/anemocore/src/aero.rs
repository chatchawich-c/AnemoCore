// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_variables, clippy::too_many_arguments, clippy::manual_range_contains)]
//! Aerodynamic utilities gated behind the `aero` feature.

/// Prandtl tip and hub loss correction factors.
/// Returns (F_tip, F_hub, F_total).
/// - `r_tip`: tip radius (m)
/// - `r_hub`: hub radius (m)
/// - `r_rotor`: rotor radius (m)
/// - `lambda`: tip-speed ratio (λ = ωR / V)
pub fn prandtl_tip_hub_loss(r_tip: f64, r_hub: f64, r_rotor: f64, lambda: f64) -> (f64, f64, f64) {
    // Simple analytical approximations (placeholder).
    // Tip loss factor from Prandtl: F_tip = 2/π * acos(exp(- (r_tip - r_rotor) / (r_rotor * lambda)))
    // Hub loss factor: F_hub = 2/π * acos(exp(- (r_rotor - r_hub) / (r_rotor * lambda)))
    let eps = 1e-12_f64;
    let tip_ratio = ((r_tip - r_rotor) / (r_rotor * lambda + eps)).abs();
    let hub_ratio = ((r_rotor - r_hub) / (r_rotor * lambda + eps)).abs();
    let f_tip = (2.0 / std::f64::consts::PI) * ((-tip_ratio).exp()).acos();
    let f_hub = (2.0 / std::f64::consts::PI) * ((-hub_ratio).exp()).acos();
    let f_total = f_tip * f_hub;
    (f_tip, f_hub, f_total)
}

/// Steady-state Blade‑Element‑Momentum (BEM) solver for a single blade element.
/// Returns (axial induction a, tangential induction a')
pub fn bem_steady_rotor(
    r: f64,
    chord: f64,
    twist_deg: f64,
    phi_rad: f64,
    lambda: f64,
    cl: f64,
    cd: f64,
    rho: f64,
) -> (f64, f64) {
    // Simplified Glauert high‑induction correction (placeholder).
    // Normal and tangential forces per unit span.
    let cn = cl * phi_rad.cos() + cd * phi_rad.sin();
    let ct = cl * phi_rad.sin() - cd * phi_rad.cos();
    // Axial induction factor a using Glauert correction.
    let a_ideal = 1.0 / (4.0 * std::f64::consts::PI * r * rho * cn / (lambda * lambda) + 1.0);
    let a = if a_ideal > 0.4 {
        0.5 * (1.0 - (1.0 - 4.0 * a_ideal * (1.0 - a_ideal)).sqrt())
    } else {
        a_ideal
    };
    // Tangential induction a' (simplified).
    let a_prime = ct / (4.0 * std::f64::consts::PI * r * rho * cl * lambda);
    (a, a_prime)
}

/// Simplified Beddoes‑Leishman dynamic stall model.
/// Returns stall state (attached = true, separated = false) and updated lift coefficient.
pub fn beddoes_leishman_stall(
    cl_static: f64,
    alpha_deg: f64,
    dt: f64,
    state: &mut f64,
) -> (bool, f64) {
    // Very coarse model: linear lag on lift coefficient.
    let lag = 0.9_f64;
    let target = cl_static * (1.0 + 0.1 * (alpha_deg / 15.0).abs());
    *state = lag * *state + (1.0 - lag) * target;
    let attached = *state < 1.5 * cl_static; // arbitrary threshold
    (attached, *state)
}

/// Computes aerodynamic torque, thrust, power and coefficients over a range of wind speeds.
/// Returns vectors of (wind_speed, torque, thrust, power, Cp, Ct).
pub fn aero_torque_thrust_curve(
    wind_speeds: &[f64],
    r_rotor: f64,
    omega: f64,
    rho: f64,
    c_p: f64,
) -> Vec<(f64, f64, f64, f64, f64, f64)> {
    let mut out = Vec::with_capacity(wind_speeds.len());
    for &v in wind_speeds {
        // Simple power coefficient model: Cp = c_p * (1 - (v / (omega * r_rotor)).powi(2)).max(0)
        let tip_speed = omega * r_rotor;
        let cp = c_p * (1.0 - (v / tip_speed).powi(2)).max(0.0);
        let power = 0.5 * rho * std::f64::consts::PI * r_rotor * r_rotor * v.powi(3) * cp;
        let thrust = power / ((v + 1e-12).max(1e-12)); // now increases with v
        let torque = power / omega;
        let ct = thrust / (0.5 * rho * std::f64::consts::PI * r_rotor * r_rotor * v * v);
        out.push((v, torque, thrust, power, cp, ct));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prandtl_tip_hub_loss_basic() {
        let (f_tip, f_hub, f_total) = prandtl_tip_hub_loss(80.0, 2.0, 50.0, 6.0);
        assert!(f_tip > 0.0 && f_tip <= 1.0);
        assert!(f_hub > 0.0 && f_hub <= 1.0);
        assert!(f_total > 0.0 && f_total <= 1.0);
    }

    #[test]
    fn test_bem_steady_rotor_consistency() {
        let (a, a_prime) = bem_steady_rotor(30.0, 3.0, 5.0, 0.5, 6.0, 1.0, 0.01, 1.225);
        assert!(a >= 0.0 && a <= 1.0);
        assert!(a_prime.is_finite());
    }

    #[test]
    fn test_beddoes_leishman_stall() {
        let mut state = 0.0;
        let (attached, lift) = beddoes_leishman_stall(1.0, 10.0, 0.01, &mut state);
        assert!(attached);
        assert!(lift > 0.0);
    }

    #[test]
    fn test_aero_torque_thrust_curve_monotonic() {
        let winds = [5.0, 7.5, 10.0];
        let res = aero_torque_thrust_curve(&winds, 50.0, 2.0, 1.225, 0.45);
        assert_eq!(res.len(), winds.len());
        for i in 1..res.len() {
            assert!(res[i].2 > res[i - 1].2);
        }
    }
}
