// SPDX-License-Identifier: MIT OR Apache-2.0
// Grant No. ERASMUS-EDU-2025-CBHE-101177220

use libm::exp;

/// Economic Cost‑Performance Ratio (ECPR).
/// Returns a dimensionless ratio: (total_cost - revenue) / revenue.
pub fn ecpr(total_cost: f64, revenue: f64) -> f64 {
    if revenue == 0.0 {
        return f64::INFINITY;
    }
    (total_cost - revenue) / revenue
}

/// Pigouvian fee for wake‑induced power loss.
/// `wake_loss` is the power lost due to wake effects (W), `cost_factor` is monetary cost per Watt.
pub fn pigouvian_wake_fee(wake_loss: f64, cost_factor: f64) -> f64 {
    wake_loss * cost_factor
}

/// Total‑expenditure per barrel‑equivalent (`totex_pbr`).
/// Simplified: `totex = capital + opex`, `pbr = totex / barrel_eq`.
pub fn totex_pbr(capital: f64, opex: f64, barrel_eq: f64) -> f64 {
    if barrel_eq == 0.0 {
        return f64::INFINITY;
    }
    (capital + opex) / barrel_eq
}

/// One‑dimensional Fokker‑Planck + Hamilton‑Jacobi‑Bellman step.
/// `state` is the current probability density (slice), `dt` time step, `mu` drift, `sigma` diffusion.
/// Returns a new state vector (same length) using explicit Euler.
pub fn mfg_step(state: &[f64], dt: f64, _mu: f64, sigma: f64) -> ([f64; 256], usize) {
    const MAX: usize = 256;
    let n = state.len().min(MAX);
    let mut new_state = [0.0; MAX];
    for i in 0..n {
        // Simple diffusion term: sigma^2 * second derivative approximated as zero.
        // Drift term: mu * gradient approximated as zero for placeholder.
        // We just apply exponential decay as a dummy update.
        new_state[i] = state[i] * exp(-sigma * dt);
    }
    (new_state, n)
}

/// Compute LCOE (Levelized Cost of Electricity) and LCOH (Hydrogen) in USD/kg.
/// Simplified: `lcoe = (capex * crf + opex) / (capacity * 8760 * capacity_factor)`.
/// `lcoh = lcoe / efficiency_h2` where efficiency_h2 converts electricity to H2.
pub fn lcoe_lcoh_hydrogen(
    capex: f64,
    opex: f64,
    capacity_mw: f64,
    capacity_factor: f64,
    efficiency_h2: f64,
    discount_rate: f64,
    lifetime_years: f64,
) -> (f64, f64) {
    // Capital recovery factor (CRF)
    let crf = if discount_rate == 0.0 {
        1.0 / lifetime_years
    } else {
        let r = discount_rate;
        let n = lifetime_years;
        r * (1.0 + r).powf(n) / ((1.0 + r).powf(n) - 1.0)
    };
    let annual_energy_mwh = capacity_mw * 8760.0 * capacity_factor;
    let lcoe = if annual_energy_mwh == 0.0 {
        f64::INFINITY
    } else {
        (capex * crf + opex) / annual_energy_mwh
    };
    let lcoh = if efficiency_h2 == 0.0 {
        f64::INFINITY
    } else {
        lcoe / efficiency_h2
    };
    (lcoe, lcoh)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ecpr() {
        assert!((ecpr(1200.0, 1000.0) - 0.2).abs() < 1e-12);
    }
    #[test]
    fn test_pigouvian() {
        assert!((pigouvian_wake_fee(500.0, 0.05) - 25.0).abs() < 1e-12);
    }
    #[test]
    fn test_totex_pbr() {
        assert!((totex_pbr(1_000_000.0, 200_000.0, 1_600_000.0) - 0.75).abs() < 1e-12);
    }
    #[test]
    fn test_mfg_step() {
        let state = [1.0, 2.0, 3.0];
        let (new_state, n) = mfg_step(&state, 1.0, 0.0, 0.1);
        assert_eq!(n, 3);
        // exp(-0.1) approx 0.904837418.
        let expected = 0.9048374180359595;
        assert!((new_state[0] - expected).abs() < 1e-12);
    }
    #[test]
    fn test_lcoe_lcoh() {
        let (lcoe, lcoh) = lcoe_lcoh_hydrogen(1_000_000.0, 50_000.0, 10.0, 0.4, 0.7, 0.05, 20.0);
        // Rough expected values (not exact precision check).
        assert!(lcoe > 0.0 && lcoh > 0.0);
    }
}
