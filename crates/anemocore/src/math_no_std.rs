//! Dependency-free `#![no_std]` special mathematical helpers powered by `libm`.

use core::f64::consts::FRAC_1_SQRT_2;
use libm::{erf, floor, log, sqrt};

/// Wraps a scalar into `[0.0, 1.0)` matching Python `np.mod(x, 1.0)`.
/// Uses `libm::floor` so both negative values and large integers behave identically.
#[inline]
pub fn wrap_unit_interval(x: f64) -> f64 {
    let f = x - floor(x);
    if f >= 1.0 {
        0.0
    } else {
        f
    }
}

/// Standard normal cumulative distribution function Phi(z).
#[inline]
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z * FRAC_1_SQRT_2))
}

/// Inverse standard normal CDF (Probit) via Peter John Acklam's minimax rational
/// approximation (relative error < 1.15e-9 across `(0, 1)` without SciPy or heap).
#[allow(clippy::excessive_precision)]
pub fn inv_norm_cdf(p: f64) -> f64 {
    let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p_clamped < P_LOW {
        let q = sqrt(-2.0 * log(p_clamped));
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    } else if p_clamped <= P_HIGH {
        let q = p_clamped - 0.5;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num / den
    } else {
        let q = sqrt(-2.0 * log(1.0 - p_clamped));
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        -(num / den)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libm::fabs;

    #[test]
    fn probit_matches_scipy_norm_ppf() {
        let cases = [
            (1e-4, -3.719016485456),
            (0.01, -2.326347874041),
            (0.15, -1.036433389494),
            (0.50, 0.0),
            (0.85, 1.036433389494),
            (0.99, 2.326347874041),
            (1.0 - 1e-4, 3.719016485456),
        ];
        for (p, expected) in cases {
            let z = inv_norm_cdf(p);
            assert!(fabs(z - expected) < 1e-8, "p={p}: got {z}, want {expected}");
            assert!(fabs(norm_cdf(z) - p) < 2e-7, "roundtrip failed for p={p}");
        }
    }

    #[test]
    fn wrap_unit_handles_positive_and_negative() {
        assert!(fabs(wrap_unit_interval(1.25) - 0.25) < 1e-12);
        assert!(fabs(wrap_unit_interval(-0.25) - 0.75) < 1e-12);
        assert!(fabs(wrap_unit_interval(3.0)) < 1e-12);
    }
}
