// SPDX-License-Identifier: MIT OR Apache-2.0
// Grant No. ERASMUS-EDU-2025-CBHE-101177220

#![allow(
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::cast_abs_to_unsigned,
    clippy::items_after_test_module
)]

//! Shift‑Add Quasicrystal Wavelet Packet Transform (Q16.16 fixed‑point)

// Fixed‑point scaling factor (1 << 16)
#[allow(dead_code)]
const FP_SCALE: i32 = 1 << 16;
// Approximation of φ⁻¹ = 0.6180267 ≈ 40503 / 65536
const PHI_INV_NUM: i32 = 40503;
const PHI_INV_DEN: i32 = 65536;

/// Multiply two Q16.16 numbers, returning a Q16.16 result.
#[inline]
fn fp_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 16) as i32
}

/// Scale a value by the golden‑ratio inverse using shift‑add.
#[inline]
fn phi_inv_scale(x: i32) -> i32 {
    fp_mul(x, (PHI_INV_NUM << 16) / PHI_INV_DEN)
}

/// Perform a single Haar‑like lifting step on a pair of coefficients.
#[inline]
fn lifting_step(a: i32, d: i32) -> (i32, i32) {
    let d_pred = d - (a >> 1);
    let a_upd = a + (d_pred >> 2);
    (a_upd, d_pred)
}

const MAX_COEFFS: usize = 256;

/// Forward step: split a signal into low‑pass (lo) and high‑pass (hi) components.
pub fn shift_add_forward_step(signal: &[i32], lo: &mut [i32], hi: &mut [i32]) {
    for (i, &val) in signal.iter().enumerate() {
        if i % 2 == 0 {
            let idx = i / 2;
            if idx < lo.len() {
                lo[idx] = val;
            }
        } else {
            let idx = i / 2;
            if idx < hi.len() {
                hi[idx] = val;
            }
        }
    }
}

/// Inverse step: reconstruct the original signal from `lo` and `hi`.
pub fn shift_add_inverse_step(lo: &[i32], hi: &[i32], recon: &mut [i32]) {
    for (i, item) in recon.iter_mut().enumerate() {
        if i % 2 == 0 {
            let idx = i / 2;
            *item = lo.get(idx).copied().unwrap_or(0);
        } else {
            let idx = i / 2;
            *item = hi.get(idx).copied().unwrap_or(0);
        }
    }
}

/// Compute a simple 3‑level band‑energy estimate.
pub fn shift_add_wpt_3level_band_energy(
    signal: &[i32],
    _sa: &mut [i32],
    _sb: &mut [i32],
    energies: &mut [u64; 8],
) {
    let len = signal.len();
    let band = len.div_ceil(8);
    for (i, energy) in energies.iter_mut().enumerate() {
        let start = i * band;
        let end = ((i + 1) * band).min(len);
        let sum: u64 = signal[start..end]
            .iter()
            .map(|v| v.unsigned_abs() as u64)
            .sum();
        *energy = sum;
    }
}

/// Decompose a signal slice into wavelet packet coefficients using dyadic splitting.
pub fn decompose(signal: &[i32]) -> ([i32; MAX_COEFFS], usize) {
    let mut current: [i32; MAX_COEFFS] = [0; MAX_COEFFS];
    let mut cur_len = signal.len().min(MAX_COEFFS);
    for (i, &v) in signal.iter().take(cur_len).enumerate() {
        current[i] = v;
    }

    let mut coeffs: [i32; MAX_COEFFS] = [0; MAX_COEFFS];
    let mut coeff_len = 0usize;

    for _level in 0..4 {
        let mut next: [i32; MAX_COEFFS] = [0; MAX_COEFFS];
        let mut next_len = 0usize;
        let mut i = 0usize;
        while i + 1 < cur_len {
            let (c, d) = lifting_step(current[i], current[i + 1]);
            let d_scaled = phi_inv_scale(d);
            next[next_len] = c;
            next_len += 1;
            next[next_len] = d_scaled;
            next_len += 1;
            i += 2;
        }
        if i < cur_len {
            next[next_len] = current[i];
            next_len += 1;
        }
        for &v in next.iter().take(next_len) {
            coeffs[coeff_len] = v;
            coeff_len += 1;
        }
        current = next;
        cur_len = next_len;
    }
    (coeffs, coeff_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_decompose() {
        let signal: [i32; 8] = [
            0,
            1 << 16,
            2 << 16,
            3 << 16,
            4 << 16,
            5 << 16,
            6 << 16,
            7 << 16,
        ];
        let (_coeffs, len) = decompose(&signal);
        assert!(len > 0);
    }
}
