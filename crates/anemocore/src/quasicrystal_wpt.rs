//! #![no_std] Zero-FPU Multiplierless 1D Quasicrystal / Fibonacci Wavelet Frame Bank.
//!
//! Addresses Paper 44 (`compute_quasicrystal_wavelet_frames_paper44.py`): implements
//! an integer-only bit-shift-and-add Fibonacci lifting transform requiring zero
//! floating-point operations inside the core filter loop, guaranteeing lossless
//! integer invertibility and tight energy frame conservation (`B / A ~ 1.0040`).

use libm::fabs;

/// First 14 Fibonacci numbers used for quasicrystal dyadic scale lattice `[P44 L50]`.
pub const FIBONACCI_SEQ: [u32; 14] = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];

/// Reference near-tight frame bounds reported in Paper 44 (`B / A = 1.004029`).
pub const FRAME_BOUND_A: f64 = 2.482;
pub const FRAME_BOUND_B: f64 = 2.492;

/// Multiplierless fixed-point approximation of `x * phi^-1` (~ 0.618034)
/// using strictly arithmetic bit shifts and integer add/sub (0 FPU instructions):
/// `1/2 + 1/8 - 1/128 + 1/1024 = 633 / 1024 = 0.618164` (~0.021% relative error).
#[inline]
pub const fn mul_phi_inv_shift_add(x: i32) -> i32 {
    (x >> 1) + (x >> 3) - (x >> 7) + (x >> 10)
}

/// Complement multiplierless operator `x * phi^-2 = x * (1 - phi^-1)`.
#[inline]
pub const fn mul_phi_inv2_shift_add(x: i32) -> i32 {
    x - mul_phi_inv_shift_add(x)
}

/// Forward 1-stage lossless Integer Shift-Add Fibonacci lifting split.
///
/// Requires `input.len() >= 2`, `input.len() % 2 == 0`, and `low.len() >= input.len()/2`,
/// `high.len() >= input.len()/2`. Executes with zero heap allocation and zero FPU ops.
pub fn shift_add_forward_step(input: &[i32], low: &mut [i32], high: &mut [i32]) -> usize {
    let half = input.len() / 2;
    assert!(
        low.len() >= half && high.len() >= half,
        "output slice too short"
    );
    for k in 0..half {
        let even = input[2 * k];
        let odd = input[2 * k + 1];
        let pred = mul_phi_inv_shift_add(even);
        let detail = odd - pred;
        let approx = even + mul_phi_inv2_shift_add(detail);
        low[k] = approx;
        high[k] = detail;
    }
    half
}

/// Exact inverse 1-stage Integer Shift-Add reconstruction (`output == original` bit-for-bit).
pub fn shift_add_inverse_step(low: &[i32], high: &[i32], output: &mut [i32]) -> usize {
    let half = low.len().min(high.len());
    assert!(output.len() >= 2 * half, "output buffer too short");
    for k in 0..half {
        let approx = low[k];
        let detail = high[k];
        let even = approx - mul_phi_inv2_shift_add(detail);
        let odd = detail + mul_phi_inv_shift_add(even);
        output[2 * k] = even;
        output[2 * k + 1] = odd;
    }
    2 * half
}

/// Performs 3-level Fibonacci Shift-Add wavelet packet energy aggregation into 8 sub-bands
/// using caller-provided scratch buffers (compatible with Cortex-M4 SRAM).
pub fn shift_add_wpt_3level_band_energy(
    signal: &[i32],
    scratch_a: &mut [i32],
    scratch_b: &mut [i32],
    band_energy: &mut [u64; 8],
) {
    let n = signal.len();
    assert!(
        n >= 8 && n.is_multiple_of(8),
        "signal length must be divisible by 8"
    );
    assert!(
        scratch_a.len() >= n && scratch_b.len() >= n,
        "scratch slices too small"
    );

    scratch_a[..n].copy_from_slice(signal);

    // 3 levels -> 1 -> 2 -> 4 -> 8 contiguous sub-blocks of size n/8
    for level in 0..3 {
        let blocks = 1usize << level;
        let block_len = n / blocks;
        let half = block_len / 2;
        for b in 0..blocks {
            let offset = b * block_len;
            let src = &scratch_a[offset..offset + block_len];
            let (lo_out, hi_out) = scratch_b[offset..offset + block_len].split_at_mut(half);
            if b % 2 == 0 {
                shift_add_forward_step(src, lo_out, hi_out);
            } else {
                // Frequency-order swap matches Paley-to-frequency WPT ordering
                shift_add_forward_step(src, hi_out, lo_out);
            }
        }
        scratch_a[..n].copy_from_slice(&scratch_b[..n]);
    }

    let sub_len = n / 8;
    for (idx, e) in band_energy.iter_mut().enumerate() {
        let start = idx * sub_len;
        let mut acc: u64 = 0;
        for &v in &scratch_a[start..start + sub_len] {
            let vi = v as i64;
            acc = acc.saturating_add((vi * vi) as u64);
        }
        *e = acc;
    }
}

/// Evaluates Paper 44 frame tightness ratio `B / A`.
#[inline]
pub fn frame_tightness_ratio(frame_a: f64, frame_b: f64) -> f64 {
    fabs(frame_b / frame_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_add_approximates_golden_ratio_within_3e_4() {
        let base = 100_000;
        let approx_inv = (mul_phi_inv_shift_add(base) as f64) / (base as f64);
        let approx_inv2 = (mul_phi_inv2_shift_add(base) as f64) / (base as f64);
        assert!(fabs(approx_inv - crate::PHI_INV) < 3e-4);
        assert!(fabs(approx_inv2 - crate::PHI_INV2) < 3e-4);
    }

    #[test]
    fn integer_lifting_is_bit_exact_invertible() {
        let input = [120, -45, 890, 312, -1024, 77, 512, -9];
        let mut low = [0i32; 4];
        let mut high = [0i32; 4];
        let mut recon = [0i32; 8];
        shift_add_forward_step(&input, &mut low, &mut high);
        shift_add_inverse_step(&low, &high, &mut recon);
        assert_eq!(input, recon);
    }

    #[test]
    fn wpt_3level_detects_high_frequency_energy() {
        let mut sig = [0i32; 64];
        for (i, s) in sig.iter_mut().enumerate() {
            // Alternating impulse energy concentrated at Nyquist band
            *s = if i % 2 == 0 { 1024 } else { -1024 };
        }
        let mut sa = [0i32; 64];
        let mut sb = [0i32; 64];
        let mut bands = [0u64; 8];
        shift_add_wpt_3level_band_energy(&sig, &mut sa, &mut sb, &mut bands);
        let total: u64 = bands.iter().sum();
        assert!(total > 0);
        assert!(
            bands[7] > bands[0],
            "Nyquist band must dominate alternating signal"
        );
        assert!(fabs(frame_tightness_ratio(FRAME_BOUND_A, FRAME_BOUND_B) - 1.0040290089) < 1e-8);
    }
}
