//! Intraday Electricity Market Liquidity Microstructure & Fibonacci LOB SDE.
//!
//! Unifies Paper 43 (`compute_dsi_fibonacci_liquidity_stopping.py`) Fibonacci limit-order-book
//! density and jump-diffusion volatility dampening with Paper 8 (`run_paper8_local.py`)
//! empirical Roll (1984) effective bid-ask spread and Parkinson (1980) volatility fallback.

use core::f64::consts::PI;

/// Empirical Fibonacci retracement liquidity wall configuration `[P43 L55-57]`.
#[derive(Clone, Debug)]
pub struct FibonacciWallConfig {
    pub z_levels: [f64; 5],
    pub weights: [f64; 5],
    pub bandwidths: [f64; 5],
    pub p_min_roll: f64,
    pub p_max_roll: f64,
}

impl Default for FibonacciWallConfig {
    fn default() -> Self {
        Self::p43_default()
    }
}

impl FibonacciWallConfig {
    pub fn p43_default() -> Self {
        Self {
            z_levels: [0.236, 0.382, 0.500, 0.618, 0.786],
            weights: [0.15, 0.35, 0.18, 0.42, 0.12],
            bandwidths: [0.018, 0.022, 0.015, 0.024, 0.016],
            p_min_roll: 40.0,
            p_max_roll: 110.0,
        }
    }
}

/// Maps raw spot electricity price `[EUR/MWh]` into normalized retracement `z in [0.01, 0.99]`.
#[inline]
pub fn retracement_coord(price: f64, cfg: &FibonacciWallConfig) -> f64 {
    let span = (cfg.p_max_roll - cfg.p_min_roll).max(1e-6);
    ((price - cfg.p_min_roll) / span).clamp(0.01, 0.99)
}

/// Endogenous Limit Order Book (LOB) clustering density `rho_LOB(z)` `[P43 L88]`.
pub fn lob_density(z: f64, cfg: &FibonacciWallConfig) -> f64 {
    let sqrt_2pi = (2.0 * PI).sqrt();
    let mut sum = 1.0;
    for k in 0..5 {
        let delta = cfg.bandwidths[k];
        let diff = z - cfg.z_levels[k];
        sum += (cfg.weights[k] / (sqrt_2pi * delta)) * (-0.5 * (diff / delta).powi(2)).exp();
    }
    sum
}

/// Volatility modulated by endogenous LOB depth (`sigma_local = sigma_base / sqrt(1 + 0.15 * rho_LOB)`) `[P43 L91]`.
#[inline]
pub fn local_volatility(z: f64, sigma_base: f64, cfg: &FibonacciWallConfig) -> f64 {
    sigma_base / (1.0 + 0.15 * lob_density(z, cfg)).sqrt()
}

/// Executes one Euler-Maruyama price SDE step coupled to Fibonacci liquidity walls `[P43 L97-99]`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_price_jump_diffusion(
    price_curr: f64,
    p_mean: f64,
    alpha_mr: f64,
    sigma_base: f64,
    dt_day: f64,
    std_normal_draw: f64,
    jump_eur_mwh: f64,
    cfg: &FibonacciWallConfig,
) -> (f64, f64, f64) {
    let z = retracement_coord(price_curr, cfg);
    let rho = lob_density(z, cfg);
    let sigma_loc = sigma_base / (1.0 + 0.15 * rho).sqrt();
    let dw = std_normal_draw * dt_day.sqrt();
    let dp = alpha_mr * (p_mean - price_curr) * dt_day + sigma_loc * price_curr * dw + jump_eur_mwh;
    let p_next = (price_curr + dp).max(5.0);
    (p_next, z, rho)
}

/// Linear interpolation across `NaN` gaps so Roll spread matches Paper 8 sanitization (`P8 L153-158`).
fn sanitize_prices(prices: &[f64]) -> Option<Vec<f64>> {
    if prices.is_empty() {
        return None;
    }
    let valid_idx: Vec<usize> = prices
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if v.is_nan() { None } else { Some(i) })
        .collect();
    if valid_idx.is_empty() {
        return None;
    }
    let mut out = prices.to_vec();
    if valid_idx.len() == prices.len() {
        return Some(out);
    }
    let first = valid_idx[0];
    let last = *valid_idx.last().unwrap();
    for i in 0..first {
        out[i] = out[first];
    }
    for i in (last + 1)..out.len() {
        out[i] = out[last];
    }
    for win in valid_idx.windows(2) {
        let (a, b) = (win[0], win[1]);
        if b > a + 1 {
            let va = out[a];
            let vb = out[b];
            let span = (b - a) as f64;
            for (k, slot) in out.iter_mut().enumerate().take(b).skip(a + 1) {
                *slot = va + ((k - a) as f64 / span) * (vb - va);
            }
        }
    }
    Some(out)
}

/// Computes Roll's (1984) effective bid-ask spread estimator with Parkinson (1980) positive-covariance
/// fallback (`compute_roll_effective_spread` in Paper 8 `run_paper8_local.py L139-192`).
pub fn roll_effective_spread(prices: &[f64], window: usize) -> Vec<f64> {
    let Some(p) = sanitize_prices(prices) else {
        return vec![0.0; prices.len()];
    };
    let n = p.len();
    let win = window.max(1);

    // 3-point diurnal smoother [0.25, 0.50, 0.25] with edge padding
    let mut p_innov = vec![0.0_f64; n];
    for i in 0..n {
        let left = if i == 0 { p[0] } else { p[i - 1] };
        let mid = p[i];
        let right = if i + 1 >= n { p[n - 1] } else { p[i + 1] };
        p_innov[i] = mid - (0.25 * left + 0.50 * mid + 0.25 * right);
    }

    let mut dp = vec![0.0_f64; n];
    let mut dp_lag = vec![0.0_f64; n];
    for i in 1..n {
        dp[i] = p_innov[i] - p_innov[i - 1];
        dp_lag[i] = dp[i - 1];
    }

    let mut cs_dp = vec![0.0_f64; n + 1];
    let mut cs_lag = vec![0.0_f64; n + 1];
    let mut cs_prod = vec![0.0_f64; n + 1];
    for i in 0..n {
        cs_dp[i + 1] = cs_dp[i] + dp[i];
        cs_lag[i + 1] = cs_lag[i] + dp_lag[i];
        cs_prod[i + 1] = cs_prod[i] + dp[i] * dp_lag[i];
    }

    let park_denom = (4.0 * 2.0_f64.ln()).sqrt();
    let mut s_roll = vec![0.0_f64; n];
    for i in 0..n {
        let start = (i + 1).saturating_sub(win);
        let w_len = i - start + 1;
        if w_len < 4 {
            continue;
        }
        let wf = w_len as f64;
        let m_dp = (cs_dp[i + 1] - cs_dp[start]) / wf;
        let m_lag = (cs_lag[i + 1] - cs_lag[start]) / wf;
        let cov = (cs_prod[i + 1] - cs_prod[start]) / wf - m_dp * m_lag;
        if cov < 0.0 {
            s_roll[i] = 2.0 * (-cov).sqrt();
        } else {
            let mut w_min = f64::INFINITY;
            let mut w_max = f64::NEG_INFINITY;
            for &v in &p_innov[start..=i] {
                if v < w_min {
                    w_min = v;
                }
                if v > w_max {
                    w_max = v;
                }
            }
            s_roll[i] = 0.25 * (w_max - w_min) / park_denom;
        }
    }
    s_roll
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p43_lob_density_matches_python_parity() {
        let cfg = FibonacciWallConfig::p43_default();
        let rho = lob_density(0.618, &cfg);
        assert!((rho - 7.9814899070).abs() < 1e-8);
    }

    #[test]
    fn p08_roll_effective_spread_matches_python_parity() {
        let mut prices = Vec::with_capacity(32);
        for i in 0..32 {
            let alt = if i % 2 == 0 { 1.0 } else { -1.0 };
            prices.push(60.0 + alt * 2.5 + 0.1 * (i as f64));
        }
        let spread = roll_effective_spread(&prices, 24);
        assert!((spread[31] - 9.9467331320).abs() < 1e-7);
    }

    #[test]
    fn p08_parkinson_fallback_handles_positive_serial_cov() {
        // Monotonically accelerating trend yields non-negative serial innovation covariance
        let prices: Vec<f64> = (0..16).map(|i| 50.0 + (i as f64).powi(2) * 0.5).collect();
        let spread = roll_effective_spread(&prices, 8);
        assert!(spread.last().unwrap().is_finite());
        assert!(*spread.last().unwrap() >= 0.0);
    }
}
