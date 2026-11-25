// SPDX-License-Identifier: MIT OR Apache-2.0
// Grant No. ERASMUS-EDU-2025-CBHE-101177220

/// Compute Roll's effective spread for a single bid‑ask pair (simplified).
/// Returns `(ask - bid).abs() / (ask + bid)` or `0.0` if denominator is zero.
pub fn roll_effective_spread_scalar(bid: f64, ask: f64) -> f64 {
    if ask + bid == 0.0 {
        0.0
    } else {
        (ask - bid).abs() / (ask + bid)
    }
}

/// Compute Roll's effective spread for a slice of bid‑ask pairs.
/// Returns a vector of spreads for each pair.
pub fn roll_effective_spread(bids: &[f64], ask: f64) -> Vec<f64> {
    bids.iter()
        .map(|&bid| roll_effective_spread_scalar(bid, ask))
        .collect()
}

/// Fibonacci order‑book depth: returns the depth level (tick index) for a given cumulative volume.
/// `order_book` is a slice of volumes at each depth (in ascending price order).
pub fn fibonacci_depth(order_book: &[f64], target_volume: f64) -> usize {
    let mut cum = 0.0;
    for (i, &vol) in order_book.iter().enumerate() {
        cum += vol;
        if cum >= target_volume {
            return i + 1;
        }
    }
    order_book.len()
}

/// Depth‑Slope Index (DSI) = Σ w_k V_k / Σ w_k |Δp_k|.
/// `volumes` and `price_deltas` must be the same length; `weights` provides w_k.
pub fn depth_slope_index(volumes: &[f64], price_deltas: &[f64], weights: &[f64]) -> f64 {
    let n = volumes.len().min(price_deltas.len()).min(weights.len());
    if n == 0 {
        return 0.0;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        num += weights[i] * volumes[i];
        den += weights[i] * price_deltas[i].abs();
    }
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    #[test]
    fn roll_spread() {
        assert!((roll_effective_spread_scalar(99.0, 101.0) - 0.01).abs() < 1e-12);
    }
    #[test]
    fn fib_depth() {
        let ob = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0];
        assert_eq!(fibonacci_depth(&ob, 4.0), 3);
    }
    #[test]
    fn dsi_calc() {
        let vol = [10.0, 20.0];
        let pd = [0.5, 1.0];
        let w = [1.0, 1.0];
        let dsi = depth_slope_index(&vol, &pd, &w);
        assert!((dsi - (30.0 / 1.5)).abs() < 1e-12);
    }
}

/// Configuration for Fibonacci LOB wall density calculations.
pub struct FibonacciWallConfig {
    /// Example scaling factor for wall density.
    pub density: f64,
    /// Example depth level (unused stub).
    pub depth: usize,
}

impl FibonacciWallConfig {
    /// Default configuration matching P43 specifications.
    pub fn p43_default() -> Self {
        Self {
            density: 0.8,
            depth: 10,
        }
    }
}

/// Compute LOB wall density factor (stub implementation).
pub fn lob_density(i: f64, cfg: &FibonacciWallConfig) -> f64 {
    i * cfg.density
}
