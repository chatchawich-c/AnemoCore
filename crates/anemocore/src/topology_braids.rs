//! `#![no_std]` Geometric Stability, Jacobi Horizon Guard, Symplectic Solvers, and Braid Bounds.
//!
//! Supports **Paper 51** (*Null-Geodesic Control & KAM Diophantine Curvature Stabilization*)
//! along with [`crate::kam_vsg::jacobi_horizon_guard`] for `< 10 µs` WCET execution on ARM Cortex-M4F.

pub use crate::kam_vsg::{jacobi_horizon_guard, JacobiGuardOutput, JacobiHorizonConfig};
use libm::fabs;

/// Structure-preserving 1-DOF Symplectic Störmer–Verlet step `[P51 Eqs. 16-18]`
/// preserving canonical phase-space area in `#![no_std]` fixed stack memory.
#[inline]
pub fn symplectic_stormer_verlet_step(
    q: f64,
    p: f64,
    g_inv: f64,
    grad_v_q: f64,
    grad_v_q_next: f64,
    dt: f64,
) -> (f64, f64) {
    let p_half = p - 0.5 * dt * grad_v_q;
    let q_next = q + dt * g_inv * p_half;
    let p_next = p_half - 0.5 * dt * grad_v_q_next;
    (q_next, p_next)
}

/// Evaluates minimum Diophantine small divisor `inf |k1 + k2 * omega_ratio|` up to `k_max`
/// verifying Hurwitz's lower bound `gamma_max / k = 1 / (sqrt(5) * k)` at `omega_ratio = phi^-1`.
pub fn min_diophantine_small_divisor(k_max: i32, omega_ratio: f64) -> f64 {
    let mut min_div = f64::MAX;
    for k2 in 1..=k_max {
        for k1 in -k_max..=k_max {
            let val = fabs((k1 as f64) + (k2 as f64) * omega_ratio);
            if val < min_div {
                min_div = val;
            }
        }
    }
    min_div
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PHI_INV;
    use libm::sqrt;

    #[test]
    fn p51_golden_ratio_beats_hurwitz_bound() {
        let k_max = 20;
        let div_phi = min_diophantine_small_divisor(k_max, PHI_INV);
        let hurwitz_lower = 1.0 / (sqrt(5.0) * (k_max as f64));
        // Golden ratio achieves strong separation vs resonant rational 2/3 (= 0)
        assert!(div_phi >= 0.9 * hurwitz_lower, "div_phi={div_phi}");
        let div_rational = min_diophantine_small_divisor(k_max, 2.0 / 3.0);
        assert!(div_rational < 1e-12);
    }

    #[test]
    fn p51_symplectic_step_preserves_harmonic_energy() {
        // Simple harmonic oscillator: V(q) = 0.5 * q^2 -> grad_V = q, g_inv = 1.0
        let (mut q, mut p) = (1.0_f64, 0.0_f64);
        let dt = 0.01;
        for _ in 0..1000 {
            let q_next_est = q + dt * (p - 0.5 * dt * q);
            let (qn, pn) = symplectic_stormer_verlet_step(q, p, 1.0, q, q_next_est, dt);
            q = qn;
            p = pn;
        }
        let h_final = 0.5 * (q * q + p * p);
        assert!(fabs(h_final - 0.5) < 1e-4, "h_final={h_final}");
    }
}
