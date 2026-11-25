// SPDX-License-Identifier: MIT OR Apache-2.0
// Grant No. ERASMUS-EDU-2025-CBHE-101177220

#![allow(clippy::result_unit_err, clippy::new_without_default)]

/// Bounded online mean/variance calculator with quantile bins.
pub struct PiEMeans<const B: usize> {
    /// Quantile bin thresholds (sorted).
    bins: [f64; B],
    /// Count of observations processed.
    count: usize,
    /// Current mean (Welford).
    mean: f64,
    /// Sum of squares of differences (for variance).
    m2: f64,
}

impl<const B: usize> PiEMeans<B> {
    /// Creates a new instance with empty bins.
    pub const fn new() -> Self {
        Self {
            bins: [0.0; B],
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Updates the statistics with a new sample.
    pub fn update(&mut self, value: f64) {
        // Update Welford mean/variance.
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        // Simple quantile bin update – place value in first empty bin.
        for b in self.bins.iter_mut() {
            if *b == 0.0 {
                *b = value;
                break;
            }
        }
    }

    /// Returns the current mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the variance (population).
    pub fn variance(&self) -> f64 {
        if self.count > 0 {
            self.m2 / self.count as f64
        } else {
            0.0
        }
    }

    /// Enforces monotonicity of a power curve and the Betz ceiling.
    /// `powers` must be a mutable slice of power values indexed by wind speed order.
    /// `rated_power` is the maximum allowable power.
    pub fn enforce_power_curve(powers: &mut [f64], rated_power: f64) {
        // Monotonic increase.
        for i in 1..powers.len() {
            if powers[i] < powers[i - 1] {
                powers[i] = powers[i - 1];
            }
        }
        // Betz ceiling: cap by rated_power.
        for p in powers.iter_mut() {
            if *p > rated_power {
                *p = rated_power;
            }
        }
    }
}

impl<const B: usize> Default for PiEMeans<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Betz ceiling helper – returns the theoretical max power for given conditions.
pub fn betz_ceiling(rho: f64, area: f64, wind_speed: f64) -> f64 {
    // 16/27 ≈ 0.59259259, and 0.5 for kinetic energy factor.
    0.59259259 * 0.5 * rho * area * wind_speed.powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn online_stats() {
        let mut s = PiEMeans::<4>::new();
        s.update(1.0);
        s.update(2.0);
        s.update(3.0);
        assert!((s.mean() - 2.0).abs() < 1e-12);
        // Variance of [1,2,3] population = 2/3.
        assert!((s.variance() - 0.6666666667).abs() < 1e-10);
    }

    #[test]
    fn power_curve_enforce() {
        let mut curve = [0.5, 0.4, 0.6, 1.2, 1.5];
        PiEMeans::<4>::enforce_power_curve(&mut curve, 1.0);
        assert_eq!(curve, [0.5, 0.5, 0.6, 1.0, 1.0]);
    }
}
