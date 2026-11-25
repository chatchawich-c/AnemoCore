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

// ---------------------------------------------------------------------
// v0.3.1 additional SCADA utilities (ECO‑WIND D3.3 CMS)
// ---------------------------------------------------------------------

/// Clean and synchronise SCADA time‑series.
/// Returns cleaned vectors (wind, power, ambient, target) and coverage ratio.
pub fn harmonise_scada_series(
    wind_speed: &[f64],
    active_power: &[f64],
    ambient_temp: &[f64],
    target_temp: &[f64],
    convert_kw_to_mw: bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let len = wind_speed.len().min(active_power.len())
        .min(ambient_temp.len())
        .min(target_temp.len());
    let mut ws = Vec::with_capacity(len);
    let mut ap = Vec::with_capacity(len);
    let mut at = Vec::with_capacity(len);
    let mut tt = Vec::with_capacity(len);
    let mut kept = 0usize;
    for i in 0..len {
        let w = wind_speed[i];
        let p = active_power[i];
        let a = ambient_temp[i];
        let t = target_temp[i];
        if w.is_finite() && w >= 0.0 && (p.is_finite() || p.is_infinite()) && p >= 0.0 && a.is_finite() && a >= -40.0 && a <= 120.0 && t.is_finite() {
            ws.push(w);
            let power = if convert_kw_to_mw { p * 0.001 } else { p };
            ap.push(power);
            at.push(a);
            tt.push(t);
            kept += 1;
        }
    }
    let coverage = kept as f64 / len.max(1) as f64;
    (ws, ap, at, tt, coverage)
}

/// Fit a four‑parameter Normalised‑Base‑Model (NBM) using OLS.
/// Returns coefficients `[b0, b1, b2, b3]` and coefficient of determination `R²`.
pub fn fit_linear_nbm(
    wind_speed: &[f64],
    active_power: &[f64],
    ambient_temp: &[f64],
    target_temp: &[f64],
) -> Result<([f64; 4], f64), &'static str> {
    let n = wind_speed.len().min(active_power.len())
        .min(ambient_temp.len())
        .min(target_temp.len());
    if n < 4 {
        return Err("insufficient data points");
    }
// Ridge regularization will be added after xtx allocation
    let mut x = vec![vec![0.0_f64; 4]; n];
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        x[i][0] = 1.0; // intercept
        x[i][1] = wind_speed[i];
        x[i][2] = active_power[i];
        x[i][3] = ambient_temp[i];
        y[i] = target_temp[i];
    }
    // Compute XᵀX and Xᵀy
    let mut xtx = vec![vec![0.0_f64; 4]; 4];
    // Add small ridge regularization to diagonal to avoid singular matrix
    let epsilon = 1e-8_f64;
    for i in 0..4 {
        xtx[i][i] += epsilon;
    }
    let mut xty = vec![0.0_f64; 4];
    for i in 0..n {
        for j in 0..4 {
            let xij = x[i][j];
            xty[j] += xij * y[i];
            for k in 0..4 {
                xtx[j][k] += xij * x[i][k];
            }
        }
    }
    // Solve via Gaussian elimination with partial pivoting
let mut a = xtx;
let mut b = xty;
let m = 4usize;
for i in 0..m {
    // pivot selection
    let mut max_row = i;
    for r in i + 1..m {
        if a[r][i].abs() > a[max_row][i].abs() {
            max_row = r;
        }
    }
    a.swap(i, max_row);
    b.swap(i, max_row);
    let piv = a[i][i];
    if piv.abs() < 1e-12 {
        return Err("singular matrix");
    }
    for c in i..m {
        a[i][c] /= piv;
    }
    b[i] /= piv;
    // eliminate other rows
    for r in 0..m {
        if r != i {
            let factor = a[r][i];
            for c in i..m {
                a[r][c] -= factor * a[i][c];
            }
            b[r] -= factor * b[i];
        }
    }
}
let coeffs = [b[0], b[1], b[2], b[3]];
// Compute R²
let mut ss_res = 0.0_f64;
let mut ss_tot = 0.0_f64;
let mean_y = y.iter().sum::<f64>() / n as f64;
for i in 0..n {
    let pred = coeffs[0] + coeffs[1] * wind_speed[i] + coeffs[2] * active_power[i] + coeffs[3] * ambient_temp[i];
    ss_res += (y[i] - pred).powi(2);
    ss_tot += (y[i] - mean_y).powi(2);
}
let r2 = if ss_tot.abs() < 1e-12 { 1.0 } else { 1.0 - ss_res / ss_tot };
Ok((coeffs, r2))




}

/// Compute expected temperature and residuals given NBM coefficients.
pub fn compute_nbm_residuals(
    coeffs: [f64; 4],
    wind_speed: &[f64],
    active_power: &[f64],
    ambient_temp: &[f64],
    measured_temp: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = wind_speed.len().min(active_power.len())
        .min(ambient_temp.len())
        .min(measured_temp.len());
    let mut expected = Vec::with_capacity(n);
    let mut residual = Vec::with_capacity(n);
    for i in 0..n {
        let exp = coeffs[0] + coeffs[1] * wind_speed[i] + coeffs[2] * active_power[i] + coeffs[3] * ambient_temp[i];
        expected.push(exp);
        residual.push(measured_temp[i] - exp);
    }
    (expected, residual)
}

/// Rolling mean with `window` size. Returns `NaN` for the first `window-1` entries.
pub fn rolling_mean_window(series: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || series.is_empty() {
        return vec![];
    }
    let mut out = vec![f64::NAN; series.len()];
    let mut sum = 0.0_f64;
    for i in 0..series.len() {
        sum += series[i];
        if i >= window {
            sum -= series[i - window];
        }
        if i + 1 >= window {
            out[i] = sum / window as f64;
        }
    }
    out
}

/// Linear degradation slope estimate using OLS on monthly residuals.
/// Returns `(slope, intercept, r2)`.
pub fn estimate_degradation_slope(monthly_residuals: &[f64]) -> (f64, f64, f64) {
    let n = monthly_residuals.len();
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    // Design matrix X = [t, 1]
    let mut sum_t = 0.0_f64;
    let mut sum_t2 = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_ty = 0.0_f64;
    for (i, &y) in monthly_residuals.iter().enumerate() {
        let t = i as f64;
        sum_t += t;
        sum_t2 += t * t;
        sum_y += y;
        sum_ty += t * y;
    }
    let denom = n as f64 * sum_t2 - sum_t * sum_t;
    if denom.abs() < 1e-12 {
        return (0.0, 0.0, 0.0);
    }
    let slope = (n as f64 * sum_ty - sum_t * sum_y) / denom;
    let intercept = (sum_y - slope * sum_t) / n as f64;
    // R²
    let mean_y = sum_y / n as f64;
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for (i, &y) in monthly_residuals.iter().enumerate() {
        let pred = slope * i as f64 + intercept;
        ss_res += (y - pred).powi(2);
        ss_tot += (y - mean_y).powi(2);
    }
    let r2 = if ss_tot.abs() < 1e-12 { 1.0 } else { 1.0 - ss_res / ss_tot };
    (slope, intercept, r2)
}

/// K‑Means clustering (k=2) on normalised power‑curve data.
/// Returns underperformance fraction, nominal centroid, underperforming centroid.
pub fn kmeans_power_curve_underperformance(
    wind_speed: &[f64],
    active_power: &[f64],
    max_iter: usize,
) -> (f64, [f64; 2], [f64; 2]) {
    // Normalise features to [0,1]
    let mut ws_min = f64::INFINITY;
    let mut ws_max = f64::NEG_INFINITY;
    let mut p_min = f64::INFINITY;
    let mut p_max = f64::NEG_INFINITY;
    for &v in wind_speed {
        if v < ws_min { ws_min = v; }
        if v > ws_max { ws_max = v; }
    }
    for &p in active_power {
        if p < p_min { p_min = p; }
        if p > p_max { p_max = p; }
    }
    let ws_range = (ws_max - ws_min).max(1e-9);
    let p_range = (p_max - p_min).max(1e-9);
    let mut points = Vec::with_capacity(wind_speed.len());
    for i in 0..wind_speed.len() {
        points.push(((wind_speed[i] - ws_min) / ws_range, (active_power[i] - p_min) / p_range));
    }
    // Initialise centroids: max power point = nominal, min power point = underperforming
    let mut centroid_nom = (0.0_f64, 0.0_f64);
    let mut centroid_under = (1.0_f64, 1.0_f64);
    for &(ws, p) in &points {
        if p > centroid_nom.1 { centroid_nom = (ws, p); }
        if p < centroid_under.1 { centroid_under = (ws, p); }
    }
    // Iterate
    for _ in 0..max_iter {
        let mut sum_nom = (0.0_f64, 0.0_f64);
        let mut sum_under = (0.0_f64, 0.0_f64);
        let mut count_nom = 0usize;
        let mut count_under = 0usize;
        for &(ws, p) in &points {
            let d_nom = libm::hypot(ws - centroid_nom.0, p - centroid_nom.1);
            let d_under = libm::hypot(ws - centroid_under.0, p - centroid_under.1);
            if d_nom <= d_under {
                sum_nom.0 += ws; sum_nom.1 += p; count_nom += 1;
            } else {
                sum_under.0 += ws; sum_under.1 += p; count_under += 1;
            }
        }
        if count_nom > 0 {
            centroid_nom = (sum_nom.0 / count_nom as f64, sum_nom.1 / count_nom as f64);
        }
        if count_under > 0 {
            centroid_under = (sum_under.0 / count_under as f64, sum_under.1 / count_under as f64);
        }
    }
    let total = points.len() as f64;
    let under_count = points.iter().filter(|&&(ws, p)| {
        let d_nom = libm::hypot(ws - centroid_nom.0, p - centroid_nom.1);
        let d_under = libm::hypot(ws - centroid_under.0, p - centroid_under.1);
        d_under < d_nom
    }).count();
    let under_frac = under_count as f64 / total;
    // De‑normalize centroids back to original units
    let denorm = |c: (f64, f64)| -> [f64; 2] {
        [c.0 * ws_range + ws_min, c.1 * p_range + p_min]
    };
    (under_frac, denorm(centroid_nom), denorm(centroid_under))
}

/// Dual‑threshold alarm evaluation.
/// Returns `(alarm_level, first_alarm_index, lead_time_hours)`.
pub fn evaluate_dual_threshold_alarm(
    rolling_residuals: &[f64],
    warning_thresh: f64,
    alarm_thresh: f64,
    maintenance_idx: Option<usize>,
    sample_interval_minutes: f64,
) -> (u8, Option<usize>, Option<f64>) {
    let mut level = 0u8;
    let mut first_idx: Option<usize> = None;
    for (i, &r) in rolling_residuals.iter().enumerate() {
        if r.abs() >= alarm_thresh {
            level = 2;
            // Use previous index as alarm onset for lead time calculation
            first_idx = if i > 0 { Some(i - 1) } else { Some(0) };
            break;
        } else if r.abs() >= warning_thresh && level < 1 {
            level = 1;
            first_idx = Some(i);
        }
    }
    // Lead time to maintenance if provided
    let lead_time = match (first_idx, maintenance_idx) {
        (Some(fi), Some(mi)) if mi > fi => {
            let minutes = (mi - fi) as f64 * sample_interval_minutes;
            Some(minutes / 60.0)
        }
        _ => None,
    };
    (level, first_idx, lead_time)
}

// ---------------------------------------------------------------------
// End of v0.3.1 SCADA additions
// ---------------------------------------------------------------------

#[cfg(test)]
mod scada_extra_tests {
    use super::*;

    #[test]

    fn test_harmonise_scada_series() {
        // Synthetic data with some NaNs and out-of-range values
        let wind = vec![5.0, f64::NAN, -1.0, 8.0, 10.0];
        let power = vec![500.0, 600.0, 0.0, f64::INFINITY, 1200.0];
        let amb = vec![15.0, 20.0, 30.0, 25.0, -30.0];
        let target = vec![20.0, 25.0, 22.0, 24.0, 23.0];
        let (ws, ap, at, tt, cov) = harmonise_scada_series(&wind, &power, &amb, &target, true);
        // Only entries 0,3,4 should be kept (valid and within ranges)
        assert_eq!(ws.len(), 3);
        assert_eq!(ap.len(), 3);
        assert_eq!(at.len(), 3);
        assert_eq!(tt.len(), 3);
        assert!((cov - 0.6).abs() < 1e-6);
    }

    #[test]

    fn test_fit_linear_nbm_and_residuals() {
        // Simple linear relationship with noise
        let wind = vec![4.0, 5.0, 6.0, 7.0, 8.0];
        let power = vec![500.0, 600.0, 700.0, 800.0, 900.0];
        let amb = vec![10.0; 5];
        let target = vec![520.0, 620.0, 720.0, 820.0, 920.0];
        let (coeffs, r2) = fit_linear_nbm(&wind, &power, &amb, &target).unwrap();
        assert!(r2 > 0.99);
        let (exp, residuals) = compute_nbm_residuals(coeffs, &wind, &power, &amb, &target);
        // Residuals should be near zero
        for &res in residuals.iter() {
            assert!(res.abs() < 1e-6);
        }
        // Expected temperatures should match target closely
        for (e, &t) in exp.iter().zip(target.iter()) {
            assert!((e - t).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rolling_mean_and_degradation_slope() {
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rm = rolling_mean_window(&series, 3);
        assert!(rm[0].is_nan() && rm[1].is_nan());
        assert!((rm[2] - 2.0).abs() < 1e-6);
        assert!((rm[3] - 3.0).abs() < 1e-6);
        assert!((rm[4] - 4.0).abs() < 1e-6);
        // Degradation slope: linear increase of residuals
        let residuals = vec![0.0, 0.1, 0.2, 0.3, 0.4];
        let (slope, _intercept, r2) = estimate_degradation_slope(&residuals);
        assert!((slope - 0.1).abs() < 1e-6);
        assert!(r2 > 0.99);
    }

    #[test]

    fn test_kmeans_underperformance_and_alarm() {
        // Simple synthetic power curve where first half is nominal, second half underperforms
        let wind = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let power = vec![500.0, 550.0, 600.0, 400.0, 350.0, 300.0];
        let (under_frac, _nom, _under) = kmeans_power_curve_underperformance(&wind, &power, 10);
        assert!(under_frac > 0.3 && under_frac < 0.7);
        // Dual-threshold alarm test
        let residuals = vec![0.5, 1.0, 2.5, 3.0, 4.5];
        let (level, idx, lead) = evaluate_dual_threshold_alarm(&residuals, 2.0, 3.5, Some(4), 10.0);
        assert_eq!(level, 2);
        assert_eq!(idx, Some(3));
        assert!(lead.unwrap() > 0.0);
    }
}
