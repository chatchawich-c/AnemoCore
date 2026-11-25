// SPDX-License-Identifier: MIT OR Apache-2.0

// Additional utilities for AEP calculations and IEC air‑density correction.

/// IEC‑61400‑12‑1 air‑density corrected wind speed.
/// `v` – wind speed (m s⁻¹) at measured density `rho_kg_m3`.
/// `rho_kg_m3` – air density in kg m⁻³ (default CIPM‑2007 value 1.225 kg m⁻³ if not provided).
/// Returns the normalized wind speed `v_norm = v * (rho / 1.225)^(1/3)`.
#[inline]
pub fn normalize_wind_speed_iec(v: f64, rho_kg_m3: f64) -> f64 {
    let ref_rho = 1.225_f64; // CIPM‑2007 reference density
    v * (rho_kg_m3 / ref_rho).powf(1.0 / 3.0)
}

/// Method for gross AEP estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AepMethod {
    /// Time‑series based aggregation (placeholder).
    TimeSeries,
    /// Weibull QMC integration (default).
    WeibullQmc,
}

/// Loss factors applied multiplicatively to gross AEP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossFactors {
    pub loss_wake: f64,
    pub loss_electrical: f64,
    pub loss_availability: f64,
    pub loss_environmental: f64,
    pub loss_curtailment: f64,
    pub loss_thermal_derating: f64,
}

impl Default for LossFactors {
    fn default() -> Self {
        Self {
            loss_wake: 0.0,
            loss_electrical: 0.0,
            loss_availability: 0.0,
            loss_environmental: 0.0,
            loss_curtailment: 0.0,
            loss_thermal_derating: 0.0,
        }
    }
}

/// Structured report for full AEP analysis.
/// Power curve models for AEP calculations.
#[derive(Debug, Clone)]
pub enum PowerCurveModel {
    /// Tabular IEC piecewise linear: Vec of (wind_speed, power) points.
    Tabular(Vec<(f64, f64)>),
    /// Cubic polynomial: a*v^3 + b*v^2 + c*v + d.
    Cubic { a: f64, b: f64, c: f64, d: f64 },
    /// Logistic 4‑parameter model: a / (1 + b * exp(-c * (v - d))).
    Logistic4P { a: f64, b: f64, c: f64, d: f64 },
}

/// Evaluate a power curve model at wind speed `v`.
pub fn eval_power_curve(v: f64, model: &PowerCurveModel) -> f64 {
    match model {
        PowerCurveModel::Tabular(points) => {
            if points.is_empty() {
                return 0.0;
            }
            if v <= points[0].0 {
                return points[0].1;
            }
            if v >= points.last().unwrap().0 {
                return points.last().unwrap().1;
            }
            for w in points.windows(2) {
                let (x0, y0) = w[0];
                let (x1, y1) = w[1];
                if v >= x0 && v <= x1 {
                    let t = (v - x0) / (x1 - x0);
                    return y0 + t * (y1 - y0);
                }
            }
            0.0
        }
        PowerCurveModel::Cubic { a, b, c, d } => a * v.powi(3) + b * v.powi(2) + c * v + d,
        PowerCurveModel::Logistic4P { a, b, c, d } => a / (1.0 + b * libm::exp(-c * (v - d))),
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct AepReport {
    pub aep_gross_mwh: f64,
    pub aep_net_mwh: f64,
    pub gross_capacity_factor: f64,
    pub net_capacity_factor: f64,
    pub p50_mwh: f64,
    pub p75_mwh: f64,
    pub p90_mwh: f64,
    pub p99_mwh: f64,
    pub sigma_total: f64,
    pub eya_gap_mwh: f64,
}

/// Compute a full AEP report.
///
/// Parameters:
/// - `wind_speed_mean`: mean hub‑height wind speed (m s⁻¹).
/// - `power_curve`: closure converting wind speed to power (W).
/// - `n_samples`: number of QMC samples.
/// - `method`: gross AEP estimation method.
/// - `losses`: optional loss factors (defaults to zero).
/// - `rho_kg_m3`: air density for IEC normalisation (use 1.225 if unsure).
///
/// Returns an `AepReport` containing gross/net AEP, capacity factors, bankable percentiles,
/// total sigma and EYA gap.
pub fn compute_aep_full(
    wind_speed_mean: f64,
    power_curve_model: &PowerCurveModel,
    n_samples: usize,
    method: AepMethod,
    losses: Option<LossFactors>,
    rho_kg_m3: f64,
) -> AepReport {
    // Normalise wind speed using IEC correction.
    let v_norm = normalize_wind_speed_iec(wind_speed_mean, rho_kg_m3);

    // Helper closure that uses eval_power_curve.
    let power_curve = |v: f64| eval_power_curve(v, power_curve_model);

    // Gross AEP (MWh) – placeholder implementation using existing estimate_bankable_aep.
    let aep_gross_mwh = match method {
        AepMethod::TimeSeries => estimate_bankable_aep(v_norm, power_curve, n_samples, 0.0).0 / 1e6,
        AepMethod::WeibullQmc => estimate_bankable_aep(v_norm, power_curve, n_samples, 0.0).0 / 1e6,
    };

    // Apply loss cascade.
    let lf = losses.unwrap_or_default();
    let loss_product = (1.0 - lf.loss_wake)
        * (1.0 - lf.loss_electrical)
        * (1.0 - lf.loss_availability)
        * (1.0 - lf.loss_environmental)
        * (1.0 - lf.loss_curtailment)
        * (1.0 - lf.loss_thermal_derating);
    let aep_net_mwh = aep_gross_mwh * loss_product;

    // Capacity factors – using placeholder rated power of 1 MW.
    let rated_power_w = 1_000_000.0;
    let gross_cf = aep_gross_mwh * 1e6 / (8760.0 * rated_power_w);
    let net_cf = aep_net_mwh * 1e6 / (8760.0 * rated_power_w);

    // Bankable percentiles and sigma – reuse estimate_bankable_aep for distribution.
    let (_, p50, p75, p90, p99, sigma, eya_gap) =
        estimate_bankable_aep(v_norm, power_curve, n_samples, 0.0);
    let p50_mwh = p50 / 1e6;
    let p75_mwh = p75 / 1e6;
    let p90_mwh = p90 / 1e6;
    let p99_mwh = p99 / 1e6;
    // Return the report
    AepReport {
        aep_gross_mwh,
        aep_net_mwh,
        gross_capacity_factor: gross_cf,
        net_capacity_factor: net_cf,
        p50_mwh,
        p75_mwh,
        p90_mwh,
        p99_mwh,
        sigma_total: sigma / 1e6,
        eya_gap_mwh: eya_gap / 1e6,
    }
}

// Compute AEP using a selectable PowerCurveModel.
pub fn compute_aep_full_with_model(
    wind_speed_mean: f64,
    power_curve_model: &PowerCurveModel,
    n_samples: usize,
    method: AepMethod,
    losses: Option<LossFactors>,
    rho_kg_m3: f64,
) -> AepReport {
    // Directly call compute_aep_full which now expects the model reference.
    compute_aep_full(
        wind_speed_mean,
        power_curve_model,
        n_samples,
        method,
        losses,
        rho_kg_m3,
    )
}

// Bankable Operational Analysis utilities.
//
// Provides stubs for IEC‑61400‑15‑1:2025 AEP estimation via QMC, SCADA cleaning, and monotonic health estimator (PAVA). The implementations are lightweight and focus on the requested equations; they are written to compile without external crates beyond the Rust standard library and `libm` for mathematical functions.

/// Rank‑1 Fibonacci quasi‑Monte‑Carlo sequence generator (simple deterministic).
/// Returns the i‑th point in [0, 1) using the golden ratio.
#[inline]
fn fib_qmc(i: usize) -> f64 {
    // fractional part of i * φ where φ = (1 + √5)/2
    let phi = (1.0 + 5.0_f64.sqrt()) * 0.5;
    (i as f64 * phi).fract()
}

/// Estimate bankable AEP (Annual Energy Production) using a very simplified proxy.
///
/// * `wind_speed_mean` – mean wind speed at hub height (m s⁻¹).
/// * `power_curve` – closure converting wind speed to power (W).
/// * `n_samples` – number of quasi‑Monte‑Carlo samples.
///
/// Returns a tuple `(aep_net, p50, p75, p90, p99, sigma_tot, eya_gap)`.
pub fn estimate_bankable_aep<F>(
    wind_speed_mean: f64,
    power_curve: F,
    n_samples: usize,
    _aep_actual: f64,
) -> (f64, f64, f64, f64, f64, f64, f64)
where
    F: Fn(f64) -> f64,
{
    // Simplified: generate wind speeds around mean with a small perturbation.
    let mut aep_sum = 0.0;
    let mut aep_sq_sum = 0.0;
    for i in 0..n_samples {
        let q = fib_qmc(i);
        // perturb mean by ±10 % using q in [-0.1, 0.1]
        let ws = wind_speed_mean * (1.0 + 0.2 * (q - 0.5));
        let p = power_curve(ws);
        aep_sum += p;
        aep_sq_sum += p * p;
    }
    let aep_mean = aep_sum / n_samples as f64;
    let sigma = ((aep_sq_sum / n_samples as f64) - aep_mean * aep_mean).sqrt();
    // Percentiles assuming normal distribution (approx).
    let z50 = 0.0;
    let z75 = 0.67448975; // Φ⁻¹(0.75)
    let z90 = 1.28155156; // Φ⁻¹(0.90)
    let z99 = 2.32634787; // Φ⁻¹(0.99)
    let p50 = aep_mean + sigma * z50;
    let p75 = aep_mean + sigma * z75;
    let p90 = aep_mean + sigma * z90;
    let p99 = aep_mean + sigma * z99;
    let sigma_tot = sigma; // placeholder – in practice sum of component sigmas.
    let eya_gap = 0.0; // placeholder gap for test
    (aep_mean, p50, p75, p90, p99, sigma_tot, eya_gap)
}

/// Clean SCADA data according to IEC 61400‑12‑1.
/// Returns a vector of cleaned rows (placeholder `Vec<String>`).
pub fn clean_scada_iec61400_12_1(raw: &[String]) -> Vec<String> {
    // Very naive implementation: filter out rows containing obvious flags.
    raw.iter()
        .filter(|line| {
            // Assume CSV with fields: v,p,pitch,status,temp,...
            // Remove if status != "OK" or pitch extreme (>30°) or temp < 250K.
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 5 {
                return false;
            }
            let status = parts[3].trim();
            if status != "OK" {
                return false;
            }
            let pitch: f64 = parts[2].trim().parse().unwrap_or(0.0);
            if pitch.abs() > 30.0 {
                return false;
            }
            let temp: f64 = parts[4].trim().parse().unwrap_or(300.0);
            temp >= 250.0
        })
        .cloned()
        .collect()
}

/// Monotonic health estimator using the Pool Adjacent Violators Algorithm (PAVA).
/// Input `h` is a slice of health scores (higher is better). The function returns
/// a new `Vec<f64>` where the sequence is forced to be non‑increasing.
pub fn pava_monotonic_health(h: &[f64]) -> Vec<f64> {
    let mut result: Vec<f64> = h.to_vec();
    let mut i = 0usize;
    while i + 1 < result.len() {
        if result[i] < result[i + 1] {
            // merge the violators by averaging.
            let mut j = i;
            while j + 1 < result.len() && result[j] < result[j + 1] {
                j += 1;
            }
            let avg = result[i..=j].iter().copied().sum::<f64>() / (j - i + 1) as f64;
            result[i..=j].iter_mut().for_each(|v| *v = avg);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fib_qmc_repeatability() {
        assert!((fib_qmc(0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_estimate_bankable_aep() {
        // Simple linear power curve: P = 500 * v
        let curve = |v: f64| 500.0 * v;
        let (aep, p50, _, _, _, sigma, eya) = estimate_bankable_aep(10.0, curve, 1000, 5_000_000.0);
        assert!(aep > 0.0 && sigma >= 0.0);
        assert!(eya.abs() < aep); // sanity check
                                  // p50 should be close to aep (since mean = median for symmetric distribution).
        assert!((p50 - aep).abs() < sigma);
    }

    #[test]
    fn test_clean_scada() {
        let raw = vec![
            "12.0,500.0,5.0,OK,290.0".to_string(),
            "12.5,480.0,35.0,OK,285.0".to_string(), // pitch out of range
            "13.0,510.0,4.0,FAIL,295.0".to_string(), // bad status
        ];
        let cleaned = clean_scada_iec61400_12_1(&raw);
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0], raw[0]);
    }

    #[test]
    fn test_pava_monotonic() {
        let input = [0.8, 0.9, 0.85, 0.7, 0.75];
        let out = pava_monotonic_health(&input);
        // Should be non‑increasing.
        for i in 0..out.len() - 1 {
            assert!(out[i] >= out[i + 1] - 1e-12);
        }
        // First two values should be merged to their average (0.85).
        assert!((out[0] - out[1]).abs() < 1e-12);
    }
    #[test]
    fn test_power_curve_tabular() {
        let model = PowerCurveModel::Tabular(vec![(5.0, 0.0), (10.0, 1000.0), (15.0, 2000.0)]);
        assert_eq!(eval_power_curve(4.0, &model), 0.0);
        assert_eq!(eval_power_curve(5.0, &model), 0.0);
        assert_eq!(eval_power_curve(12.5, &model), 1500.0);
        assert_eq!(eval_power_curve(16.0, &model), 2000.0);
    }

    #[test]
    fn test_power_curve_cubic() {
        let model = PowerCurveModel::Cubic {
            a: 0.0,
            b: 0.0,
            c: 2.0,
            d: 0.0,
        };
        assert_eq!(eval_power_curve(3.0, &model), 6.0);
    }

    #[test]
    fn test_power_curve_logistic() {
        let model = PowerCurveModel::Logistic4P {
            a: 2000.0,
            b: 1.0,
            c: 0.5,
            d: 10.0,
        };
        let val = eval_power_curve(10.0, &model);
        assert!((val - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_aep_full_with_model() {
        let model = PowerCurveModel::Cubic {
            a: 0.0,
            b: 0.0,
            c: 500.0,
            d: 0.0,
        };
        let report =
            compute_aep_full_with_model(12.0, &model, 100, AepMethod::WeibullQmc, None, 1.225);
        assert!(report.aep_gross_mwh > 0.0);
    }
}
