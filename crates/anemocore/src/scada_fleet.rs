// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_variables, clippy::too_many_arguments, clippy::manual_range_contains, clippy::manual_is_multiple_of, clippy::manual_clamp, clippy::needless_range_loop, non_snake_case)]
//! Fleet SCADA and health diagnostics (enabled with feature="fleet").

/// Result of SARIMAX fitting: vector of estimated parameters.
/// Simple placeholder implementation using ordinary least squares on a design matrix
/// composed of lagged observations and exogenous regressors.
pub fn sarimax_fit(
    y: &[f64],    // observed series
    exog: &[f64], // exogenous regressors (same length as y)
    p: usize,     // AR order
    d: usize,     // differencing order (ignored, placeholder)
    q: usize,     // MA order (ignored, placeholder)
    P: usize,     // Seasonal AR order (ignored)
    D: usize,     // Seasonal differencing (ignored)
    Q: usize,     // Seasonal MA (ignored)
    s: usize,     // seasonal period (ignored)
) -> Vec<f64> {
    // Build a very simple linear regression on the last `p` lagged values.
    // For placeholder purposes we ignore MA and seasonal components.
    if y.is_empty() || p == 0 {
        return vec![0.0];
    }
    let n = y.len();
    let mut x_mat = Vec::with_capacity(p * (n - p));
    let mut y_vec = Vec::with_capacity(n - p);
    for i in p..n {
        for lag in 0..p {
            x_mat.push(y[i - lag - 1]);
        }
        // incorporate exogenous if provided and length matches
        if !exog.is_empty() && exog.len() == y.len() {
            x_mat.push(exog[i]);
        }
        y_vec.push(y[i]);
    }
    // Compute (X^T X)^{-1} X^T y using a naïve O(n^3) method suitable for small p.
    let cols = if exog.is_empty() { p } else { p + 1 };
    let mut xtx = vec![vec![0.0; cols]; cols];
    let mut xty = vec![0.0; cols];
    for row in 0..(n - p) {
        for i in 0..cols {
            let xi = x_mat[row * cols + i];
            xty[i] += xi * y_vec[row];
            for j in 0..cols {
                let xj = x_mat[row * cols + j];
                xtx[i][j] += xi * xj;
            }
        }
    }
    // Solve linear system using Gaussian elimination (small size).
    let mut a = xtx;
    let mut b = xty;
    for i in 0..cols {
        // pivot
        let mut max_row = i;
        for r in i + 1..cols {
            if a[r][i].abs() > a[max_row][i].abs() {
                max_row = r;
            }
        }
        a.swap(i, max_row);
        b.swap(i, max_row);
        let pivot = a[i][i];
        if pivot.abs() < 1e-12 {
            // singular, return zeros
            return vec![0.0; cols];
        }
        for j in i..cols {
            a[i][j] /= pivot;
        }
        b[i] /= pivot;
        // eliminate
        for r in 0..cols {
            if r != i {
                let factor = a[r][i];
                for c in i..cols {
                    a[r][c] -= factor * a[i][c];
                }
                b[r] -= factor * b[i];
            }
        }
    }
    b
}

/// Produce one‑step‑ahead forecasts using the parameters from `sarimax_fit` and
/// compute standardized residuals (observed – forecast) / sigma.
pub fn sarimax_forecast_residuals(y: &[f64], params: &[f64], sigma: f64) -> Vec<f64> {
    if y.is_empty() || params.is_empty() {
        return vec![];
    }
    // Simple AR(1) style forecast using first parameter as coefficient.
    let coeff = params[0];
    let mut residuals = Vec::with_capacity(y.len());
    for i in 1..y.len() {
        let forecast = coeff * y[i - 1];
        let res = (y[i] - forecast) / sigma.max(1e-12);
        residuals.push(res);
    }
    residuals
}

/// Compute cross‑turbine residual matrix by subtracting a robust fleet baseline
/// (median of each timestamp) from each turbine's measurement.
/// `data` is a matrix where rows are timestamps and columns are turbines.
pub fn fleet_cross_turbine_residual(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
        return vec![];
    }
    let n_ts = data.len();
    let n_turbines = data[0].len();
    let mut residuals = vec![vec![0.0; n_turbines]; n_ts];
    for (t, row) in data.iter().enumerate() {
        // compute median
        let mut sorted = row.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if n_turbines % 2 == 0 {
            (sorted[n_turbines / 2 - 1] + sorted[n_turbines / 2]) / 2.0
        } else {
            sorted[n_turbines / 2]
        };
        for (i, &val) in row.iter().enumerate() {
            residuals[t][i] = val - median;
        }
    }
    residuals
}

/// Fuse thermal, power‑curve, and vibration indicators into a normalized health index.
/// Each indicator is expected in the range `[0.0, 1.0]`. The function returns a weighted
/// average (weights sum to 1). The default weights are 0.4, 0.4, 0.2.
pub fn multi_domain_health_index(thermal: f64, power_curve: f64, vibration: f64) -> f64 {
    let w_thermal = 0.5;
    let w_power = 0.4;
    let w_vib = 0.1;
    let clamp = |x: f64| {
        if x < 0.0 {
            0.0
        } else if x > 1.0 {
            1.0
        } else {
            x
        }
    };
    let t = clamp(thermal);
    let p = clamp(power_curve);
    let v = clamp(vibration);
    (w_thermal * t + w_power * p + w_vib * v).min(1.0)
}

/// Diagnostic of missingness mechanisms in SCADA data.
/// Returns a tuple `(mcar_score, mar_score, mnar_score)` in `[0.0, 1.0]` where higher
/// values indicate stronger evidence for the corresponding mechanism.
pub fn scada_missingness_diagnostic(
    missing: &[bool],   // true if data point is missing
    wind_speed: &[f64], // corresponding wind speed (same length)
) -> (f64, f64, f64) {
    if missing.is_empty() || missing.len() != wind_speed.len() {
        return (0.0, 0.0, 0.0);
    }
    let total = missing.len() as f64;
    let n_missing = missing.iter().filter(|&&m| m).count() as f64;
    let mcar = n_missing / total; // assume random missingness proportion
                                  // Simple MAR proxy: correlation between missing flag and wind speed variance
    let mean_w = wind_speed.iter().copied().sum::<f64>() / total;
    let var_w = wind_speed.iter().map(|w| (w - mean_w).powi(2)).sum::<f64>() / total;
    let cov = missing
        .iter()
        .zip(wind_speed.iter())
        .map(|(&m, &w)| {
            let m_f = if m { 1.0 } else { 0.0 };
            (m_f - n_missing / total) * (w - mean_w)
        })
        .sum::<f64>()
        / total;
    let mar = if var_w.abs() < 1e-12 {
        0.0
    } else {
        (cov.abs() / var_w).min(1.0)
    };
    // Simple MNAR proxy: compare average wind speed when missing vs present
    let present_w_sum: f64 = missing
        .iter()
        .zip(wind_speed.iter())
        .filter_map(|(&m, &w)| if !m { Some(w) } else { None })
        .sum();
    let present_cnt = total - n_missing;
    let present_mean = if present_cnt > 0.0 {
        present_w_sum / present_cnt
    } else {
        0.0
    };
    let missing_w_sum: f64 = missing
        .iter()
        .zip(wind_speed.iter())
        .filter_map(|(&m, &w)| if m { Some(w) } else { None })
        .sum();
    let missing_mean = if n_missing > 0.0 {
        missing_w_sum / n_missing
    } else {
        0.0
    };
    let mnar = ((missing_mean - present_mean).abs() / (mean_w.abs() + 1e-12)).min(1.0);
    (mcar, mar, mnar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sarimax_fit_basic() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let exog = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let params = sarimax_fit(&y, &exog, 1, 0, 0, 0, 0, 0, 0);
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_fleet_cross_turbine_residual() {
        let data = vec![vec![10.0, 12.0, 11.0], vec![20.0, 18.0, 22.0]];
        let res = fleet_cross_turbine_residual(&data);
        assert_eq!(res.len(), 2);
        assert!((res[0][0] + res[0][1] + res[0][2]).abs() < 1e-12); // sum around zero
    }

    #[test]
    fn test_multi_domain_health_index() {
        let h = multi_domain_health_index(0.8, 0.6, 0.4);
        assert!((h - 0.68).abs() < 1e-6);
    }

    #[test]
    fn test_scada_missingness_diagnostic() {
        let missing = vec![false, true, false, true, false];
        let wind = vec![5.0, 6.0, 5.5, 6.5, 5.2];
        let (mcar, mar, mnar) = scada_missingness_diagnostic(&missing, &wind);
        assert!((mcar - 0.4).abs() < 1e-12);
        assert!(mar >= 0.0 && mar <= 1.0);
        assert!(mnar >= 0.0 && mnar <= 1.0);
    }
}
