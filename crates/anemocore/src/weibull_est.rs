//! #![no_std] Pure-Rust Zero-Heap Weibull Parameter Estimation, Goodness-of-Fit,
//! and Wind Power Density Analysis.
//!
//! Provides 6 estimators:
//! 1. Empirical Method of Justus (EMJ)
//! 2. Exact Method of Moments (MoM via Newton-Raphson)
//! 3. Energy Pattern Factor Method (EPFM / PDM)
//! 4. Maximum Likelihood Estimation on Raw Samples (Raw MLE)
//! 5. Modified Maximum Likelihood Estimation on Binned Histogram (MMLE)
//! 6. Graphical Least-Squares Method (LSM)
//!
//! Plus Lanczos Gamma function, FixedWindHistogram stack buffer, and CDF GoF (R^2, RMSE, KS).
#![allow(
    unused_imports,
    clippy::excessive_precision,
    clippy::doc_lazy_continuation,
    clippy::needless_range_loop
)]
use core::f64::consts::PI;
use libm::{exp, fabs, log, pow, sin, sqrt};

/// Lanczos approximation for the Gamma function Gamma(z) for z > 0.
/// Uses g = 7, N = 9 coefficients.
pub fn gamma_pos(z: f64) -> f64 {
    if z < 0.5 {
        return PI / (sin(PI * z) * gamma_pos(1.0 - z));
    }
    let z_adj = z - 1.0;
    const P: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.138571095836526,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let mut x = P[0];
    for (i, &p_val) in P.iter().enumerate().skip(1) {
        x += p_val / (z_adj + (i as f64));
    }
    let t = z_adj + 7.5;
    sqrt(2.0 * PI) * pow(t, z_adj + 0.5) * exp(-t) * x
}

/// Cumulative Distribution Function (CDF) of Weibull: F(v; k, c) = 1 - exp(-(v/c)^k)
#[inline]
pub fn weibull_cdf(v: f64, k: f64, c: f64) -> f64 {
    if v <= 0.0 || c <= 0.0 || k <= 0.0 {
        return 0.0;
    }
    1.0 - exp(-pow(v / c, k))
}

/// Probability Density Function (PDF) of Weibull: f(v; k, c) = (k/c) * (v/c)^(k-1) * exp(-(v/c)^k)
#[inline]
pub fn weibull_pdf(v: f64, k: f64, c: f64) -> f64 {
    if v <= 0.0 || c <= 0.0 || k <= 0.0 {
        return 0.0;
    }
    let z = v / c;
    (k / c) * pow(z, k - 1.0) * exp(-pow(z, k))
}

/// Wind Power Density (WPD in W/m^2): 0.5 * rho * c^3 * Gamma(1 + 3/k)
#[inline]
pub fn weibull_mean_wpd(k: f64, c: f64, rho: f64) -> f64 {
    if k <= 0.0 || c <= 0.0 {
        return 0.0;
    }
    0.5 * rho * pow(c, 3.0) * gamma_pos(1.0 + 3.0 / k)
}

/// Mean wind speed E[v] = c * Gamma(1 + 1/k)
#[inline]
pub fn weibull_mean_speed(k: f64, c: f64) -> f64 {
    if k <= 0.0 || c <= 0.0 {
        0.0
    } else {
        c * gamma_pos(1.0 + 1.0 / k)
    }
}

/// Fixed-capacity stack histogram for wind speed observations.
/// Filters NaN, negative values, and calm wind (< calm_threshold).
#[derive(Clone, Copy, Debug)]
pub struct FixedWindHistogram<const BINS: usize> {
    pub counts: [u32; BINS],
    pub bin_edges: [f64; BINS],
    pub bin_width: f64,
    pub total_samples: u32,
    pub valid_samples: u32,
    pub calm_samples: u32,
}

impl<const BINS: usize> FixedWindHistogram<BINS> {
    /// Creates a histogram covering [0.0, max_speed] with BINS intervals.
    pub fn new(max_speed: f64) -> Self {
        let bin_width = if BINS > 0 {
            max_speed / (BINS as f64)
        } else {
            1.0
        };
        let mut bin_edges = [0.0; BINS];
        for (i, edge) in bin_edges.iter_mut().enumerate().take(BINS) {
            *edge = i as f64 * bin_width;
        }
        Self {
            counts: [0; BINS],
            bin_edges,
            bin_width,
            total_samples: 0,
            valid_samples: 0,
            calm_samples: 0,
        }
    }

    /// Feeds observations into histogram, filtering calm (< calm_threshold) and invalid numbers.
    pub fn feed_slice(&mut self, samples: &[f64], calm_threshold: f64) {
        for &s in samples {
            self.total_samples += 1;
            if s.is_nan() || s < 0.0 {
                continue;
            }
            if s < calm_threshold {
                self.calm_samples += 1;
                continue;
            }
            let idx = (s / self.bin_width) as usize;
            if idx < BINS {
                self.counts[idx] += 1;
                self.valid_samples += 1;
            } else if BINS > 0 {
                self.counts[BINS - 1] += 1;
                self.valid_samples += 1;
            }
        }
    }

    /// Bin midpoint for bin i
    #[inline]
    pub fn bin_midpoint(&self, i: usize) -> f64 {
        self.bin_edges[i] + 0.5 * self.bin_width
    }
}

/// Goodness-of-fit evaluation metrics (R^2, RMSE, Kolmogorov-Smirnov statistic D).
#[derive(Clone, Copy, Debug)]
pub struct GoFResult {
    pub r_squared: f64,
    pub rmse: f64,
    pub ks_statistic: f64,
}

/// Evaluates goodness-of-fit of Weibull (k, c) against empirical histogram CDF.
pub fn evaluate_gof_histogram<const B: usize>(
    hist: &FixedWindHistogram<B>,
    k: f64,
    c: f64,
) -> GoFResult {
    if hist.valid_samples == 0 || k <= 0.0 || c <= 0.0 {
        return GoFResult {
            r_squared: 0.0,
            rmse: 1.0,
            ks_statistic: 1.0,
        };
    }
    let n = hist.valid_samples as f64;
    let mut cum_count = 0.0;
    let mut sse = 0.0;
    let mut sum_emp = 0.0;
    let mut sum_sq_emp = 0.0;
    let mut max_ks: f64 = 0.0;
    let mut active_bins = 0.0;

    for i in 0..B {
        cum_count += hist.counts[i] as f64;
        let emp_cdf = cum_count / n;
        let v_upper = hist.bin_edges[i] + hist.bin_width;
        let th_cdf = weibull_cdf(v_upper, k, c);

        let diff = fabs(emp_cdf - th_cdf);
        if diff > max_ks {
            max_ks = diff;
        }

        let sq_err = (emp_cdf - th_cdf) * (emp_cdf - th_cdf);
        sse += sq_err;
        sum_emp += emp_cdf;
        sum_sq_emp += emp_cdf * emp_cdf;
        active_bins += 1.0;
    }

    let rmse = sqrt(sse / active_bins);
    let mean_emp = sum_emp / active_bins;
    let sst = sum_sq_emp - active_bins * mean_emp * mean_emp;
    let r_squared = if sst > 1e-12 {
        (1.0 - sse / sst).clamp(0.0, 1.0)
    } else {
        1.0
    };

    GoFResult {
        r_squared,
        rmse,
        ks_statistic: max_ks,
    }
}

// =========================================================================
// 6 WEIBULL PARAMETER ESTIMATORS
// =========================================================================

/// Result struct containing shape k and scale c.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeibullFit {
    pub k: f64,
    pub c: f64,
}

/// 1. Empirical Method of Justus (EMJ):
///    k = (std / mean)^(-1.086)
///    c = mean / Gamma(1 + 1/k)
pub fn fit_empirical_justus(mean: f64, std_dev: f64) -> WeibullFit {
    if mean <= 0.0 || std_dev <= 0.0 {
        return WeibullFit {
            k: 2.0,
            c: mean.max(1.0),
        };
    }
    let k = pow(std_dev / mean, -1.086);
    let c = mean / gamma_pos(1.0 + 1.0 / k);
    WeibullFit { k, c }
}

/// 2. Exact Method of Moments (MoM):
///    Solves (sigma/mu)^2 = Gamma(1 + 2/k) / Gamma(1 + 1/k)^2 - 1 for k via Newton-Raphson,
///    then c = mu / Gamma(1 + 1/k).
pub fn fit_mom(mean: f64, std_dev: f64) -> WeibullFit {
    if mean <= 0.0 || std_dev <= 0.0 {
        return fit_empirical_justus(mean, std_dev);
    }
    let target = (std_dev / mean) * (std_dev / mean);
    // Initial guess from Justus
    let mut k = pow(std_dev / mean, -1.086).clamp(0.5, 20.0);

    for _ in 0..30 {
        let g1 = gamma_pos(1.0 + 1.0 / k);
        let g2 = gamma_pos(1.0 + 2.0 / k);
        let val = (g2 / (g1 * g1)) - 1.0;
        let diff = val - target;
        if fabs(diff) < 1e-7 {
            break;
        }
        // Numerical derivative
        let dk = 1e-4 * k;
        let k_up = k + dk;
        let g1_up = gamma_pos(1.0 + 1.0 / k_up);
        let g2_up = gamma_pos(1.0 + 2.0 / k_up);
        let val_up = (g2_up / (g1_up * g1_up)) - 1.0;
        let deriv = (val_up - val) / dk;
        if fabs(deriv) > 1e-12 {
            k = (k - diff / deriv).clamp(0.2, 50.0);
        } else {
            break;
        }
    }
    let c = mean / gamma_pos(1.0 + 1.0 / k);
    WeibullFit { k, c }
}

/// 3. Energy Pattern Factor Method (EPFM / PDM):
///    Epf = E[v^3] / (E[v])^3 = Gamma(1 + 3/k) / Gamma(1 + 1/k)^3
///    Solves for k via Newton-Raphson with Justus initialization: k = 1 + 3.69 / Epf^2.
pub fn fit_epfm(mean: f64, mean_cube: f64) -> WeibullFit {
    if mean <= 0.0 || mean_cube <= 0.0 {
        return WeibullFit {
            k: 2.0,
            c: mean.max(1.0),
        };
    }
    let epf = mean_cube / (mean * mean * mean);
    if epf < 1.0 {
        return fit_empirical_justus(mean, 0.5 * mean);
    }
    let mut k = (1.0 + 3.69 / (epf * epf)).clamp(0.5, 30.0);
    for _ in 0..30 {
        let g1 = gamma_pos(1.0 + 1.0 / k);
        let g3 = gamma_pos(1.0 + 3.0 / k);
        let val = g3 / (g1 * g1 * g1);
        let diff = val - epf;
        if fabs(diff) < 1e-7 {
            break;
        }
        let dk = 1e-4 * k;
        let k_up = k + dk;
        let g1_up = gamma_pos(1.0 + 1.0 / k_up);
        let g3_up = gamma_pos(1.0 + 3.0 / k_up);
        let val_up = g3_up / (g1_up * g1_up * g1_up);
        let deriv = (val_up - val) / dk;
        if fabs(deriv) > 1e-12 {
            k = (k - diff / deriv).clamp(0.2, 50.0);
        } else {
            break;
        }
    }
    let c = mean / gamma_pos(1.0 + 1.0 / k);
    WeibullFit { k, c }
}

/// 4. Maximum Likelihood Estimation on Raw Samples (Raw MLE):
/// Solves: 1/k + (1/N)*sum(ln v) - (sum(v^k * ln v) / sum(v^k)) = 0
///
/// c = ((1/N) * sum(v^k))^(1/k)
pub fn fit_mle_raw(samples: &[f64]) -> Result<WeibullFit, &'static str> {
    let mut n = 0.0;
    let mut sum_v = 0.0;
    let mut sum_v2 = 0.0;
    let mut sum_ln_v = 0.0;

    for &s in samples {
        if s > 0.0 && !s.is_nan() {
            n += 1.0;
            sum_v += s;
            sum_v2 += s * s;
            sum_ln_v += log(s);
        }
    }
    if n < 2.0 {
        return Err("Insufficient positive samples for MLE");
    }
    let mean = sum_v / n;
    let var = (sum_v2 / n) - (mean * mean);
    let std_dev = if var > 0.0 { sqrt(var) } else { 0.5 * mean };

    let mut k = fit_empirical_justus(mean, std_dev).k.clamp(0.5, 20.0);

    for _ in 0..40 {
        let mut sum_vk = 0.0;
        let mut sum_vk_ln = 0.0;
        let mut sum_vk_ln2 = 0.0;

        for &s in samples {
            if s > 0.0 && !s.is_nan() {
                let ln_s = log(s);
                let vk = pow(s, k);
                sum_vk += vk;
                sum_vk_ln += vk * ln_s;
                sum_vk_ln2 += vk * ln_s * ln_s;
            }
        }
        if sum_vk <= 0.0 {
            break;
        }
        let g = 1.0 / k + (sum_ln_v / n) - (sum_vk_ln / sum_vk);
        if fabs(g) < 1e-7 {
            break;
        }
        let g_prime =
            -1.0 / (k * k) - (sum_vk_ln2 * sum_vk - sum_vk_ln * sum_vk_ln) / (sum_vk * sum_vk);
        if fabs(g_prime) > 1e-12 {
            k = (k - g / g_prime).clamp(0.2, 50.0);
        } else {
            break;
        }
    }

    let mut sum_vk_final = 0.0;
    for &s in samples {
        if s > 0.0 && !s.is_nan() {
            sum_vk_final += pow(s, k);
        }
    }
    let c = pow(sum_vk_final / n, 1.0 / k);
    Ok(WeibullFit { k, c })
}

/// 5. Modified Maximum Likelihood Estimation on Binned Histogram (Binned MMLE):
/// Uses bin midpoints v_i and frequencies f_i.
pub fn fit_mmle_binned<const B: usize>(
    hist: &FixedWindHistogram<B>,
) -> Result<WeibullFit, &'static str> {
    if hist.valid_samples < 2 {
        return Err("Insufficient samples in histogram for MMLE");
    }
    let n = hist.valid_samples as f64;
    let mut sum_fi_ln_vi = 0.0;
    let mut mean_v = 0.0;
    let mut mean_v2 = 0.0;

    for i in 0..B {
        let f = hist.counts[i] as f64;
        if f > 0.0 {
            let v = hist.bin_midpoint(i);
            if v > 0.0 {
                sum_fi_ln_vi += f * log(v);
                mean_v += f * v;
                mean_v2 += f * v * v;
            }
        }
    }
    mean_v /= n;
    mean_v2 /= n;
    let std_v = sqrt((mean_v2 - mean_v * mean_v).max(1e-4));
    let mut k = fit_empirical_justus(mean_v, std_v).k.clamp(0.5, 20.0);

    for _ in 0..40 {
        let mut sum_vk = 0.0;
        let mut sum_vk_ln = 0.0;
        let mut sum_vk_ln2 = 0.0;

        for i in 0..B {
            let f = hist.counts[i] as f64;
            if f > 0.0 {
                let v = hist.bin_midpoint(i);
                if v > 0.0 {
                    let ln_v = log(v);
                    let vk = f * pow(v, k);
                    sum_vk += vk;
                    sum_vk_ln += vk * ln_v;
                    sum_vk_ln2 += vk * ln_v * ln_v;
                }
            }
        }
        if sum_vk <= 0.0 {
            break;
        }
        let g = 1.0 / k + (sum_fi_ln_vi / n) - (sum_vk_ln / sum_vk);
        if fabs(g) < 1e-7 {
            break;
        }
        let g_prime =
            -1.0 / (k * k) - (sum_vk_ln2 * sum_vk - sum_vk_ln * sum_vk_ln) / (sum_vk * sum_vk);
        if fabs(g_prime) > 1e-12 {
            k = (k - g / g_prime).clamp(0.2, 50.0);
        } else {
            break;
        }
    }

    let mut sum_vk_final = 0.0;
    for i in 0..B {
        let f = hist.counts[i] as f64;
        if f > 0.0 {
            let v = hist.bin_midpoint(i);
            if v > 0.0 {
                sum_vk_final += f * pow(v, k);
            }
        }
    }
    let c = pow(sum_vk_final / n, 1.0 / k);
    Ok(WeibullFit { k, c })
}

/// 6. Graphical Least-Squares Method (LSM) on Binned CDF:
/// Linear regression on: ln(-ln(1 - F_i)) = k * ln(v_i) - k * ln(c)
/// Y = a * X + b, with a = k, b = -k * ln(c) => c = exp(-b / k).
pub fn fit_lsm_binned<const B: usize>(
    hist: &FixedWindHistogram<B>,
) -> Result<WeibullFit, &'static str> {
    if hist.valid_samples < 3 {
        return Err("Insufficient samples for LSM");
    }
    let n = hist.valid_samples as f64;
    let mut cum = 0.0;
    let mut count_pts = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..B {
        cum += hist.counts[i] as f64;
        let f = cum / n;
        // Avoid singularities at F = 0 and F = 1
        if f > 0.001 && f < 0.999 {
            let v = hist.bin_edges[i] + hist.bin_width;
            if v > 0.0 {
                let x = log(v);
                let y = log(-log(1.0 - f));
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
                count_pts += 1.0;
            }
        }
    }

    if count_pts < 3.0 {
        return Err("Too few points within (0.001, 0.999) CDF bounds for LSM");
    }

    let denom = count_pts * sum_xx - sum_x * sum_x;
    if fabs(denom) < 1e-12 {
        return Err("Degenerate linear regression denominator in LSM");
    }
    let slope = (count_pts * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / count_pts;

    let k = slope.max(0.2);
    let c = exp(-intercept / k);
    Ok(WeibullFit { k, c })
}

/// Container for all 6 estimator outputs.
#[derive(Clone, Copy, Debug)]
pub struct AllWeibullEstimates {
    pub emj: WeibullFit,
    pub mom: WeibullFit,
    pub epfm: WeibullFit,
    pub raw_mle: Option<WeibullFit>,
    pub mmle: Option<WeibullFit>,
    pub lsm: Option<WeibullFit>,
}

/// Convenience runner to compute all available estimators.
pub fn fit_all_six<const B: usize>(
    mean: f64,
    std_dev: f64,
    mean_cube: f64,
    raw_samples: Option<&[f64]>,
    hist: Option<&FixedWindHistogram<B>>,
) -> AllWeibullEstimates {
    let emj = fit_empirical_justus(mean, std_dev);
    let mom = fit_mom(mean, std_dev);
    let epfm = fit_epfm(mean, mean_cube);
    let raw_mle = raw_samples.and_then(|s| fit_mle_raw(s).ok());
    let mmle = hist.and_then(|h| fit_mmle_binned(h).ok());
    let lsm = hist.and_then(|h| fit_lsm_binned(h).ok());

    AllWeibullEstimates {
        emj,
        mom,
        epfm,
        raw_mle,
        mmle,
        lsm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanczos_gamma_matches_known_values() {
        assert!(fabs(gamma_pos(1.0) - 1.0) < 1e-9);
        assert!(fabs(gamma_pos(2.0) - 1.0) < 1e-9);
        assert!(fabs(gamma_pos(3.0) - 2.0) < 1e-9);
        assert!(fabs(gamma_pos(4.0) - 6.0) < 1e-9);
        assert!(fabs(gamma_pos(5.0) - 24.0) < 1e-8);
        assert!(fabs(gamma_pos(0.5) - sqrt(PI)) < 1e-8);
    }

    #[test]
    fn weibull_cdf_pdf_consistency() {
        let k = 2.0;
        let c = 8.5;
        let v = 8.5;
        // At v = c, CDF = 1 - e^(-1) ~ 0.6321205588
        let cdf = weibull_cdf(v, k, c);
        assert!(fabs(cdf - (1.0 - exp(-1.0))) < 1e-9);
        let pdf = weibull_pdf(v, k, c);
        assert!(pdf > 0.0);
    }

    #[test]
    fn test_six_estimators_on_synthetic_weibull() {
        // True parameters: k = 2.1, c = 8.2
        let k_true = 2.1;
        let c_true = 8.2;
        let mut samples = [0.0; 200];
        let mut hist = FixedWindHistogram::<30>::new(25.0);

        // Deterministic synthetic samples via inverse CDF
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        let mut sum3 = 0.0;
        for i in 0..200 {
            let u = (i as f64 + 0.5) / 200.0;
            // v = c * (-ln(1-u))^(1/k)
            let v = c_true * pow(-log(1.0 - u), 1.0 / k_true);
            samples[i] = v;
            sum += v;
            sum2 += v * v;
            sum3 += v * v * v;
        }
        hist.feed_slice(&samples, 0.1);

        let mean = sum / 200.0;
        let var = (sum2 / 200.0) - (mean * mean);
        let std_dev = sqrt(var);
        let mean_cube = sum3 / 200.0;

        let all = fit_all_six(mean, std_dev, mean_cube, Some(&samples), Some(&hist));

        // Assert all methods recover k within [1.8, 2.4] and c within [7.5, 9.0]
        assert!((1.8..2.4).contains(&all.emj.k), "emj.k={}", all.emj.k);
        assert!((7.5..9.0).contains(&all.emj.c), "emj.c={}", all.emj.c);

        assert!((1.8..2.4).contains(&all.mom.k), "mom.k={}", all.mom.k);
        assert!((7.5..9.0).contains(&all.mom.c), "mom.c={}", all.mom.c);

        assert!((1.8..2.4).contains(&all.epfm.k), "epfm.k={}", all.epfm.k);
        assert!((7.5..9.0).contains(&all.epfm.c), "epfm.c={}", all.epfm.c);

        if let Some(mle) = all.raw_mle {
            assert!((1.9..2.3).contains(&mle.k), "raw_mle.k={}", mle.k);
            assert!((7.8..8.6).contains(&mle.c), "raw_mle.c={}", mle.c);
        } else {
            panic!("raw_mle should succeed");
        }

        if let Some(mmle) = all.mmle {
            assert!((1.8..2.4).contains(&mmle.k), "mmle.k={}", mmle.k);
            assert!((7.5..9.0).contains(&mmle.c), "mmle.c={}", mmle.c);
        } else {
            panic!("mmle should succeed");
        }

        if let Some(lsm) = all.lsm {
            assert!((1.8..2.4).contains(&lsm.k), "lsm.k={}", lsm.k);
            assert!((7.5..9.0).contains(&lsm.c), "lsm.c={}", lsm.c);
        } else {
            panic!("lsm should succeed");
        }

        let gof = evaluate_gof_histogram(&hist, all.emj.k, all.emj.c);
        assert!(gof.r_squared > 0.95, "r_squared={}", gof.r_squared);
        assert!(gof.rmse < 0.05, "rmse={}", gof.rmse);
        assert!(gof.ks_statistic < 0.15, "ks={}", gof.ks_statistic);
    }
}
