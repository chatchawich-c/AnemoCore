//! #![no_std] Daubechies-4 Wavelet Packet Transform (J = 3, 8 sub-bands), Rényi Entropy H_3,
//! Sub-Band Kurtosis, and Hertzian Bearing Contact Mechanics.
//!
//! Direct `#![no_std]` port of Paper 33 (`compute_wsem_ds1_ds5.py`) and Paper 10
//! (`HertzianBearingPhysics` in `run_paper10_local.py`).

use core::f64::consts::PI;
use libm::{log2, pow, sqrt};

/// Orthogonal Daubechies-4 (`db4`) low-pass quadrature mirror filter coefficients `[P33 L34-37]`.
pub const DB4_LO: [f64; 8] = [
    -0.010597401785,
    0.032883011667,
    0.030841381836,
    -0.187034811719,
    -0.027983769417,
    0.630880767930,
    0.714846570553,
    0.230377813309,
];

/// Conjugate quadrature mirror high-pass filter `hi[k] = (-1)^k * lo[7 - k]`.
pub const DB4_HI: [f64; 8] = [
    0.230377813309,
    -0.714846570553,
    0.630880767930,
    0.027983769417,
    -0.187034811719,
    -0.030841381836,
    0.032883011667,
    0.010597401785,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WptError {
    InvalidSignalLength,
    BufferTooSmall,
}

#[derive(Clone, Copy, Debug)]
pub struct WaveletIndicators {
    pub p_low_mid: f64,
    pub p_high: f64,
    pub kappa_high: f64,
    pub h3_renyi: f64,
    pub band_energies: [f64; 8],
}

/// Exact match for `np.convolve(node, filt, mode="same")[::2]`.
/// For an even 8-tap kernel (`M = 8`), `np.convolve(..., mode="same")` uses center offset `(M - 1) / 2 = 3`.
#[inline]
fn convolve_same_downsample2(node: &[f64], filt: &[f64; 8], out: &mut [f64]) {
    let n = node.len();
    let half = n / 2;
    const OFFSET: isize = 3;
    for (j, out_val) in out.iter_mut().enumerate().take(half) {
        let k = (j * 2) as isize;
        let mut acc = 0.0;
        for (m, &w) in filt.iter().enumerate() {
            let idx = k + OFFSET - (m as isize);
            if idx >= 0 && (idx as usize) < n {
                acc += node[idx as usize] * w;
            }
        }
        *out_val = acc;
    }
}

/// Non-Fisher (Pearson) kurtosis across a sub-band slice matching `scipy.stats.kurtosis(d, fisher=False)`.
fn pearson_kurtosis(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for &v in x {
        sum += v;
    }
    let mean = sum / n;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &v in x {
        let d = v - mean;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    let mu2 = m2 / n;
    if mu2 <= 1e-24 {
        0.0
    } else {
        (m4 / n) / (mu2 * mu2)
    }
}

/// Executes 3-level (`8` frequency-ordered sub-bands) `db4` WPT without heap allocation.
/// `signal.len()` must be `>= 8` and divisible by `8`; `scratch_a` and `scratch_b` must have `len >= signal.len()`.
pub fn extract_wpt_indicators_no_std(
    signal: &[f64],
    scratch_a: &mut [f64],
    scratch_b: &mut [f64],
) -> Result<WaveletIndicators, WptError> {
    let n = signal.len();
    if n < 8 || !n.is_multiple_of(8) {
        return Err(WptError::InvalidSignalLength);
    }
    if scratch_a.len() < n || scratch_b.len() < n {
        return Err(WptError::BufferTooSmall);
    }

    scratch_a[..n].copy_from_slice(signal);

    for level in 0..3 {
        let blocks = 1usize << level;
        let block_len = n / blocks;
        let half = block_len / 2;
        for b in 0..blocks {
            let offset = b * block_len;
            let src = &scratch_a[offset..offset + block_len];
            let (first_half, second_half) =
                scratch_b[offset..offset + block_len].split_at_mut(half);
            // Paley-to-frequency ordering `[P33 L50-53]`
            if b % 2 == 0 {
                convolve_same_downsample2(src, &DB4_LO, first_half);
                convolve_same_downsample2(src, &DB4_HI, second_half);
            } else {
                convolve_same_downsample2(src, &DB4_HI, first_half);
                convolve_same_downsample2(src, &DB4_LO, second_half);
            }
        }
        scratch_a[..n].copy_from_slice(&scratch_b[..n]);
    }

    let sub_len = n / 8;
    let mut band_energies = [0.0_f64; 8];
    let mut total_e = 0.0;
    for (idx, e) in band_energies.iter_mut().enumerate() {
        let start = idx * sub_len;
        let mut s = 0.0;
        for &v in &scratch_a[start..start + sub_len] {
            s += v * v;
        }
        *e = s;
        total_e += s;
    }

    let denom = total_e + 1e-12;
    let mut probs = [0.0_f64; 8];
    let mut sum_p3 = 0.0;
    for i in 0..8 {
        probs[i] = band_energies[i] / denom;
        let p = probs[i] + 1e-12;
        sum_p3 += p * p * p;
    }

    let p_low_mid = probs[1] + probs[2] + probs[3];
    let p_high = probs[5] + probs[6] + probs[7];
    let h3_renyi = -0.5 * log2(sum_p3);
    let kappa_high = (pearson_kurtosis(&scratch_a[5 * sub_len..6 * sub_len])
        + pearson_kurtosis(&scratch_a[6 * sub_len..7 * sub_len])
        + pearson_kurtosis(&scratch_a[7 * sub_len..8 * sub_len]))
        / 3.0;

    Ok(WaveletIndicators {
        p_low_mid,
        p_high,
        kappa_high,
        h3_renyi,
        band_energies,
    })
}

/// First-principles Hertzian bearing contact mechanics (`Paper 10`: AISI 52100 bearing steel).
#[derive(Clone, Copy, Debug)]
pub struct HertzianBearingSpec {
    pub e0_pa: f64,
    pub nu: f64,
    pub r_star_m: f64,
    pub m_eff_kg: f64,
    pub q0_n: f64,
    pub e_star0_pa: f64,
}

impl HertzianBearingSpec {
    pub fn p10_default() -> Self {
        let e0_pa = 210.0e9;
        let nu = 0.30;
        Self {
            e0_pa,
            nu,
            r_star_m: 0.0125,
            m_eff_kg: 0.045,
            q0_n: 15_000.0,
            e_star0_pa: e0_pa / (2.0 * (1.0 - nu * nu)),
        }
    }

    /// Computes instantaneous `(k_h [N/m], f_c [Hz], tau_c [s], sigma_t_scale [-])` `[P10 L115-132]`.
    #[inline]
    pub fn instantaneous_mechanics(&self, q_n: f64, e_ratio: f64) -> (f64, f64, f64, f64) {
        let q_safe = q_n.clamp(500.0, 2.5e5);
        let e_star = self.e_star0_pa * e_ratio;
        let k_h = pow(6.0 * self.r_star_m * e_star * e_star * q_safe, 1.0 / 3.0);
        let f_c = sqrt(k_h / self.m_eff_kg) / (2.0 * PI);
        let tau_c = 1.0 / (2.0 * f_c);
        let sigma_t_scale = pow(self.q0_n / q_safe, 1.0 / 6.0) * pow(1.0 / e_ratio, 1.0 / 3.0);
        (k_h, f_c, tau_c, sigma_t_scale)
    }
}

/// Ergonomic allocation wrapper available when `std` feature is enabled.
#[cfg(feature = "std")]
pub fn extract_wpt_indicators_vec(signal: &[f64]) -> Result<WaveletIndicators, WptError> {
    let mut sa = vec![0.0; signal.len()];
    let mut sb = vec![0.0; signal.len()];
    extract_wpt_indicators_no_std(signal, &mut sa, &mut sb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libm::{cos, fabs, sin};

    #[test]
    fn p33_db4_wpt_matches_numpy_parity() {
        let mut sig = [0.0_f64; 64];
        for (i, v) in sig.iter_mut().enumerate() {
            let t = (i as f64) / 64.0;
            *v = sin(8.0 * PI * t) + 0.3 * cos(24.0 * PI * t);
        }
        let mut sa = [0.0_f64; 64];
        let mut sb = [0.0_f64; 64];
        let res = extract_wpt_indicators_no_std(&sig, &mut sa, &mut sb).unwrap();
        assert!(fabs(res.p_low_mid - 0.2966570345) < 1e-8);
        assert!(fabs(res.p_high - 0.0007641162) < 1e-8);
        assert!(fabs(res.h3_renyi - 0.7740334671) < 1e-8);
    }

    #[test]
    fn p10_hertzian_contact_matches_python_parity() {
        let spec = HertzianBearingSpec::p10_default();
        let (kh0, fc0, _tau_c, scale) = spec.instantaneous_mechanics(spec.q0_n, 1.0);
        assert!(fabs(kh0 - 246_499_539.328726) < 1e-4);
        assert!(fabs(fc0 - 11_779.3663978061) < 1e-6);
        assert!(fabs(scale - 1.0) < 1e-12);
    }
}
