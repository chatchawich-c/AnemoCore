// SPDX-License-Identifier: MIT OR Apache-2.0
//! Economic and risk analysis module (enabled with feature="econ").

/// Compute Levelized Cost of Energy (LCOE).
/// Returns cost in $/MWh.
pub fn compute_lcoe(
    capex: f64,
    fixed_opex: f64,
    variable_opex: f64,
    discount_rate: f64,
    lifetime_years: f64,
    annual_energy_mwh: f64,
) -> f64 {
    // Capital Recovery Factor (CRF)
    let crf = if discount_rate == 0.0 {
        1.0 / lifetime_years
    } else {
        let r = discount_rate;
        let n = lifetime_years;
        r * (1.0 + r).powf(n) / ((1.0 + r).powf(n) - 1.0)
    };
    let annualized_capex = capex * crf;
    if annual_energy_mwh == 0.0 {
        f64::INFINITY
    } else {
        (annualized_capex + fixed_opex + variable_opex) / annual_energy_mwh
    }
}

/// Compute Levelized Cost of Hydrogen (LCOH) by converting electricity cost via efficiency.
/// `efficiency_h2` is fraction of electricity turned into hydrogen (kg/kWh).
pub fn compute_lcoh(
    capex: f64,
    fixed_opex: f64,
    variable_opex: f64,
    discount_rate: f64,
    lifetime_years: f64,
    annual_energy_mwh: f64,
    efficiency_h2: f64,
) -> f64 {
    let lcoe = compute_lcoe(
        capex,
        fixed_opex,
        variable_opex,
        discount_rate,
        lifetime_years,
        annual_energy_mwh,
    );
    if efficiency_h2 == 0.0 {
        f64::INFINITY
    } else {
        // Convert $/MWh to $/kg H2 (1 MWh = 1000 kWh)
        lcoe / (efficiency_h2 * 1000.0)
    }
}

/// Unscheduled O&M cost model.
/// Returns expected annual cost.
pub fn om_unscheduled_cost_model(
    failure_rate_per_year: f64,
    repair_duration_hours: f64,
    mobilization_cost: f64,
    downtime_revenue_loss_per_hour: f64,
) -> f64 {
    // Expected number of failures per year * (mobilization + repair downtime loss)
    failure_rate_per_year
        * (mobilization_cost + repair_duration_hours * downtime_revenue_loss_per_hour)
}

/// Simple deterministic pseudo‑random number generator (LCG).
fn lcg(seed: u64) -> impl Iterator<Item = f64> {
    const A: u64 = 6364136223846793005;
    const C: u64 = 1;
    let mut state = seed;
    std::iter::from_fn(move || {
        state = state.wrapping_mul(A).wrapping_add(C);
        Some((state as f64) / (u64::MAX as f64 + 1.0))
    })
}

/// Coupled Monte‑Carlo risk simulation.
/// Returns a vector of simulated Net Present Values (NPV) in $.
pub fn coupled_monte_carlo_risk(
    seed: u64,
    n_samples: usize,
    wind_stddev: f64,
    drivetrain_failure_rate: f64,
    price_volatility: f64,
) -> Vec<f64> {
    let mut rng = lcg(seed);
    let mut results = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        // Sample wind factor (log‑normal approximation)
        let wind_factor = 1.0 + wind_stddev * (rng.next().unwrap() - 0.5);
        // Sample drivetrain failure multiplier (binary event)
        let failure = if rng.next().unwrap() < drivetrain_failure_rate {
            0.0
        } else {
            1.0
        };
        // Sample electricity price multiplier
        let price_factor = 1.0 + price_volatility * (rng.next().unwrap() - 0.5);
        // Simple NPV model: base NPV * wind * price * failure
        let base_npv = 1_000_000.0; // placeholder deterministic base
        let npv = base_npv * wind_factor * price_factor * failure;
        results.push(npv);
    }
    results
}

/// Compute Value‑at‑Risk (VaR) and Conditional VaR (CVaR) for a set of NPV samples.
/// `confidence` should be in (0.0, 1.0) e.g., 0.95.
pub fn var_cvar_project_npv(samples: &[f64], confidence: f64) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((confidence * (sorted.len() as f64)).ceil() as usize).saturating_sub(1);
    let var = sorted[idx];
    // CVaR = average of worst (1-confidence) tail
    let tail: Vec<f64> = sorted.iter().take(idx + 1).cloned().collect();
    let cvar = tail.iter().sum::<f64>() / (tail.len() as f64);
    (var, cvar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_lcoe_basic() {
        let lcoe = compute_lcoe(1_000_000.0, 10_000.0, 5_000.0, 0.05, 20.0, 8_760.0);
        assert!(lcoe > 0.0);
    }

    #[test]
    fn test_compute_lcoh_basic() {
        let lcoh = compute_lcoh(1_000_000.0, 10_000.0, 5_000.0, 0.05, 20.0, 8_760.0, 0.6);
        assert!(lcoh > 0.0);
    }

    #[test]
    fn test_om_unscheduled_cost_basic() {
        let cost = om_unscheduled_cost_model(0.2, 48.0, 200_000.0, 10_000.0);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_coupled_monte_carlo_risk_determinism() {
        let sims1 = coupled_monte_carlo_risk(42, 10, 0.1, 0.05, 0.2);
        let sims2 = coupled_monte_carlo_risk(42, 10, 0.1, 0.05, 0.2);
        assert_eq!(sims1, sims2);
    }

    #[test]
    fn test_var_cvar() {
        let samples = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let (var, cvar) = var_cvar_project_npv(&samples, 0.8);
        // 80th percentile (index 4) -> 400 after sorting
        assert!((var - 400.0).abs() < 1e-12);
        // CVaR = average of first 4 values (0..4)
        let expected_cvar = (100.0 + 200.0 + 300.0 + 400.0) / 4.0;
        assert!((cvar - expected_cvar).abs() < 1e-12);
    }
}
