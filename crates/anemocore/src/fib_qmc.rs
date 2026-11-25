#![allow(clippy::excessive_precision, clippy::approx_constant)]
//! Streaming Weibull E[v^3], Kahan Summation Integration, and Golden-Section Search.

use crate::math_no_std::{inv_norm_cdf, wrap_unit_interval};
use core::f64::consts::PI;
use libm::{exp, fabs, log, pow, sqrt};

#[cfg(feature = "std")]
use std::vec::Vec;

pub const PHI: f64 = 1.6180339887498949;
pub const INV_PHI: f64 = 0.6180339887498949;
pub const FIB_N_19: u32 = 4181;
pub const FIB_PREV_18: u32 = 2584;
pub const A_KOROBOV: u64 = 719;

/// Fractional part of f64 in [0, 1)
#[inline]
pub fn frac_f64(x: f64) -> f64 {
    let r = x - libm::floor(x);
    if r < 0.0 {
        r + 1.0
    } else {
        r
    }
}

/// Fractional part of f32 in [0, 1)
#[inline]
pub fn frac_f32(x: f32) -> f32 {
    let r = x - libm::floorf(x);
    if r < 0.0 {
        r + 1.0
    } else {
        r
    }
}

/// Baker / tent transform: 1 - 2*|x - 0.5| mapping [0, 1] to [0, 1] with C0 boundary folding.
#[inline]
pub fn tent_transform(x: f64) -> f64 {
    let u = frac_f64(x);
    1.0 - 2.0 * fabs(u - 0.5)
}

/// Returns Fibonacci pair `(F_n, F_{n-1})` for a given order `n >= 1`.
pub fn fibonacci_pair(order: u32) -> (u64, u64) {
    if order == 0 {
        return (0, 0);
    }
    if order == 1 {
        return (1, 0);
    }
    let mut prev = 0_u64;
    let mut curr = 1_u64;
    for _ in 2..=order {
        let next = curr.saturating_add(prev);
        prev = curr;
        curr = next;
    }
    (curr, prev)
}

/// Evaluates node k of a 2D rank-1 Fibonacci lattice with total points n.
/// Point: (k / n, frac(k * F_{order-1} / n)) or (k / n, frac(k * INV_PHI)).
#[inline]
pub fn rank1_fibonacci_2d(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let u1 = (k as f64) / (n as f64);
    let u2 = frac_f64((k as f64) * INV_PHI);
    (u1, u2)
}

/// Multi-dimensional Rank-1 Lattice node generation in zero-heap `#![no_std]`.
#[inline]
pub fn rank1_lattice_qmc<const D: usize>(k: usize, n: usize, generators: &[u64; D]) -> [f64; D] {
    let mut out = [0.0; D];
    if n == 0 {
        return out;
    }
    let nf = n as f64;
    let k64 = k as u64;
    let n64 = n as u64;
    for d in 0..D {
        let rem = (k64 * generators[d]) % n64;
        out[d] = (rem as f64) / nf;
    }
    out
}

/// Iterator over 2D Fibonacci lattice points in [0, 1)^2.
pub struct FibonacciLattice2DIter {
    k: usize,
    n: usize,
}

impl FibonacciLattice2DIter {
    pub fn new(n: usize) -> Self {
        Self { k: 0, n }
    }
}

impl Iterator for FibonacciLattice2DIter {
    type Item = (f64, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.k < self.n {
            let pt = rank1_fibonacci_2d(self.k, self.n);
            self.k += 1;
            Some(pt)
        } else {
            None
        }
    }
}

/// Numerical integration over [0, 1]^2 using 2D Fibonacci QMC with Kahan compensated summation.
pub fn integrate_fibonacci_2d<F: Fn(f64, f64) -> f64>(n: usize, f: F) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut c = 0.0; // Kahan compensation
    for (x, y) in FibonacciLattice2DIter::new(n) {
        let val = f(x, y);
        let y_kahan = val - c;
        let t = sum + y_kahan;
        c = (t - sum) - y_kahan;
        sum = t;
    }
    sum / (n as f64)
}

/// Adaptive 2D Fibonacci integration across consecutive Fibonacci orders until error < tol.
pub fn adaptive_fibonacci_integrate_2d<F: Fn(f64, f64) -> f64>(
    f: F,
    tol: f64,
    max_order: u32,
) -> (f64, usize) {
    let mut prev_val = 0.0;
    let mut chosen_n = 55;
    for order in 10..=max_order {
        let (fn_val, _) = fibonacci_pair(order);
        let n = fn_val as usize;
        let est = integrate_fibonacci_2d(n, &f);
        if order > 10 && fabs(est - prev_val) < tol {
            return (est, n);
        }
        prev_val = est;
        chosen_n = n;
    }
    (prev_val, chosen_n)
}

/// 1D Golden-Section 1D minimum search in [a, b].
/// Returns `(x_min, f_min, iterations)`.
pub fn golden_section_min<F: Fn(f64) -> f64>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> (f64, f64, usize) {
    let mut lam = b - INV_PHI * (b - a);
    let mut mu = a + INV_PHI * (b - a);
    let mut f_lam = f(lam);
    let mut f_mu = f(mu);
    let mut iter = 0;

    while (b - a) >= tol && iter < max_iter {
        iter += 1;
        if f_lam < f_mu {
            b = mu;
            mu = lam;
            f_mu = f_lam;
            lam = b - INV_PHI * (b - a);
            f_lam = f(lam);
        } else {
            a = lam;
            lam = mu;
            f_lam = f_mu;
            mu = a + INV_PHI * (b - a);
            f_mu = f(mu);
        }
    }
    let x_opt = 0.5 * (a + b);
    (x_opt, f(x_opt), iter)
}

/// Quasi-Monte Carlo streaming evaluation of Weibull cubic expectation E[v^3]
/// using rank-1 Fibonacci lattice with tent transform and Kahan summation.
pub fn qmc_weibull_cubic_expectation(k_weib: f64, c_weib: f64, n_points: usize) -> f64 {
    if n_points == 0 || k_weib <= 0.0 || c_weib <= 0.0 {
        return 0.0;
    }
    let inv_k = 1.0 / k_weib;
    let mut sum = 0.0;
    let mut c_kahan = 0.0;

    for k in 0..n_points {
        let u_raw = frac_f64((k as f64) * INV_PHI);
        let u_tent = tent_transform(u_raw).clamp(1e-7, 1.0 - 1e-7);
        // Inverse Weibull CDF: v = c * (-ln(1 - u))^(1/k)
        let v = c_weib * pow(-log(1.0 - u_tent), inv_k);
        let v3 = v * v * v;

        let y = v3 - c_kahan;
        let t = sum + y;
        c_kahan = (t - sum) - y;
        sum = t;
    }
    sum / (n_points as f64)
}

/// Fills contiguous float slice with flat 2D Fibonacci coordinates [x0, y0, x1, y1, ...].
/// Ideal for zero-copy numpy / C interop.
pub fn fill_fibonacci_2d_f32(out: &mut [f32], n: usize) {
    let pairs = n.min(out.len() / 2);
    let nf = n as f32;
    let inv_phi_32 = INV_PHI as f32;
    for k in 0..pairs {
        let x = (k as f32) / nf;
        let y = frac_f32((k as f32) * inv_phi_32);
        out[2 * k] = x;
        out[2 * k + 1] = y;
    }
}

// -------------------------------------------------------------------------
// Existing Limit-State and Failure Probability Functions
// -------------------------------------------------------------------------

#[inline]
pub fn fib_lattice_2d_point(k: u32, n: u32, n_prev: u32) -> [f64; 2] {
    let nf = n as f64;
    let u1 = (k as f64) / nf;
    let u2 = wrap_unit_interval(((k as u64 * n_prev as u64) as f64) / nf);
    [u1, u2]
}

#[inline]
pub fn fib_lattice_6d_point(k: u32, n: u32, a_korobov: u64, shift: &[f64; 6]) -> [f64; 6] {
    let n64 = n as u64;
    let k64 = k as u64;
    let nf = n as f64;
    let mut z_j: u64 = 1;
    let mut out = [0.0_f64; 6];
    for d in 0..6 {
        let base = ((k64 * z_j) % n64) as f64 / nf;
        out[d] = wrap_unit_interval(base + shift[d]);
        z_j = (z_j * a_korobov) % n64;
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct LimitStateEval {
    pub r_cap: f64,
    pub s_load: f64,
    pub g_val: f64,
    pub dtf: f64,
}

pub fn evaluate_limit_state_6d(u: &[f64; 6]) -> LimitStateEval {
    let mut z = [0.0_f64; 6];
    for i in 0..6 {
        z[i] = inv_norm_cdf(u[i].clamp(1e-4, 1.0 - 1e-4));
    }
    let r_cap = 55.0 * exp(0.12 * z[4] - 0.08 * z[5]);
    let thermal_factor = exp(0.045 * (z[2] * 5.0 + 65.0 - 70.0));
    let stress_ampl = 42.0 * exp(0.22 * z[1] + 0.15 * z[0]);
    let initial_flaw = 0.5 * exp(0.25 * z[3]);
    let s_load = 18.5 + 0.32 * stress_ampl * thermal_factor * sqrt(PI * initial_flaw);
    let g_val = r_cap - s_load - 12.8;
    let dtf = ((g_val + 5.0) / 35.0).clamp(0.0, 1.0);
    LimitStateEval {
        r_cap,
        s_load,
        g_val,
        dtf,
    }
}

#[inline]
pub fn value_difference_functional(spot_price: f64, dtf: f64) -> f64 {
    let c_om = 45_000.0;
    let c_cat = 220_000.0;
    let weekly_gen = 130.87;
    let weekly_rev = spot_price * weekly_gen;
    let p_cat = 0.025 * exp(4.0 * (1.0 - dtf)) * 1e-2;
    let v_cont = weekly_rev - 850.0 - p_cat * c_cat + 0.999 * (85_000.0 * dtf);
    let v_stop = 85_000.0 - c_om - 3500.0 * (1.0 - dtf);
    v_cont - v_stop
}

#[derive(Clone, Copy, Debug)]
pub struct StoppingSearchResult {
    pub s_star: f64,
    pub final_width: f64,
    pub iterations: usize,
}

pub fn golden_section_stopping_search(
    dtf: f64,
    a0: f64,
    b0: f64,
    tol: f64,
    max_iter: usize,
) -> StoppingSearchResult {
    let (s_star, _, iterations) = golden_section_min(
        |p| {
            let v = value_difference_functional(p, dtf);
            v * v
        },
        a0,
        b0,
        tol,
        max_iter,
    );
    StoppingSearchResult {
        s_star,
        final_width: (b0 - a0) * libm::pow(INV_PHI, iterations as f64),
        iterations,
    }
}

pub fn estimate_failure_prob_qmc_o1(shift: &[f64; 6]) -> (f64, f64) {
    let mut fail_count: u32 = 0;
    let mut dtf_sum = 0.0;
    for k in 0..FIB_N_19 {
        let u = fib_lattice_6d_point(k, FIB_N_19, A_KOROBOV, shift);
        let res = evaluate_limit_state_6d(&u);
        if res.g_val <= 0.0 {
            fail_count += 1;
        }
        dtf_sum += res.dtf;
    }
    let nf = FIB_N_19 as f64;
    ((fail_count as f64) / nf, dtf_sum / nf)
}

pub fn weyl_convergent_lags_fill(out: &mut [u32]) {
    let mut a = 1_u32;
    let mut b = 1_u32;
    for (idx, slot) in out.iter_mut().enumerate() {
        if idx < 2 {
            *slot = 1;
        } else {
            let next = a.saturating_add(b);
            a = b;
            b = next;
            *slot = b;
        }
    }
}

#[cfg(feature = "std")]
pub fn weyl_convergent_lags(k_lags: usize) -> Vec<u32> {
    let mut out = vec![0_u32; k_lags];
    weyl_convergent_lags_fill(&mut out);
    out
}

#[inline]
pub fn rank1_lattice_point(n_idx: u32, n_points: u32, generator: u64, shift: f64) -> f64 {
    if n_points == 0 {
        return 0.0;
    }
    let rem = ((n_idx as u64) * generator) % (n_points as u64);
    wrap_unit_interval((rem as f64) / (n_points as f64) + shift)
}

#[cfg(feature = "std")]
pub fn rank1_lattice_qmc_vec(n_points: u32, generator: u64, shift: f64) -> Vec<f64> {
    let mut pts = Vec::with_capacity(n_points as usize);
    for n in 0..n_points {
        pts.push(rank1_lattice_point(n, n_points, generator, shift));
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_pair_generation() {
        assert_eq!(fibonacci_pair(1), (1, 0));
        assert_eq!(fibonacci_pair(2), (1, 1));
        assert_eq!(fibonacci_pair(7), (13, 8));
        assert_eq!(fibonacci_pair(19), (4181, 2584));
    }

    #[test]
    fn fibonacci_integration_unit_square() {
        // Integrate f(x, y) = x * y over [0, 1]^2, analytical solution = 0.25
        let res = integrate_fibonacci_2d(4181, |x, y| x * y);
        assert!(fabs(res - 0.25) < 1e-4, "res={res}");
    }

    #[test]
    fn qmc_weibull_cubic_matches_analytical() {
        // Analytical E[v^3] = c^3 * Gamma(1 + 3/k)
        // For k = 2.0, c = 8.0: Gamma(1 + 1.5) = Gamma(2.5) = 1.5 * 0.5 * sqrt(pi) ~ 1.329340388
        // E[v^3] = 512 * 1.329340388 ~ 680.6222787
        let analytical = 8.0_f64.powi(3) * crate::weibull_est::gamma_pos(1.0 + 3.0 / 2.0);
        let qmc = qmc_weibull_cubic_expectation(2.0, 8.0, 4181);
        let rel_err = fabs(qmc - analytical) / analytical;
        assert!(
            rel_err < 0.02,
            "analytical={analytical}, qmc={qmc}, rel_err={rel_err}"
        );
    }

    #[test]
    fn fill_fibonacci_2d_f32_works() {
        let mut buf = [0.0_f32; 8];
        fill_fibonacci_2d_f32(&mut buf, 4);
        assert_eq!(buf[0], 0.0);
        assert!(buf[1] >= 0.0);
    }
}
