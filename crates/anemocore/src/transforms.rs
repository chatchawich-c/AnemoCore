#![allow(
    unused_imports,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil
)]
//!
//! Provides:
//! - Complex number arithmetic `C64`
//! - Goertzel single-bin DFT and magnitude
//! - Numerical Inverse Laplace transforms (Gaver-Stehfest, Fixed Talbot contour, Rational Transfer Functions)
//! - Tustin bilinear transformation (s -> z)
//! - microfft 64-point FFT and Hilbert envelope (no_std)
//! - Optional `std_fft` submodule powered by `rustfft` for general FFT, Hann PSD, and Hilbert envelope.

use core::f64::consts::PI;
use core::ops::{Add, Div, Mul};
use libm::{atan2, cos, exp, fabs, sin, sqrt};

#[cfg(feature = "std")]
use std::vec::Vec;

/// Complex 64-bit float representation for zero-heap `#![no_std]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn norm(self) -> f64 {
        sqrt(self.norm_sq())
    }

    #[inline]
    pub fn arg(self) -> f64 {
        atan2(self.im, self.re)
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}

impl core::ops::Add for C64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl core::ops::Sub for C64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl core::ops::Mul for C64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl core::ops::Div for C64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let d = rhs.norm_sq();
        if d == 0.0 {
            Self::new(0.0, 0.0)
        } else {
            Self::new(
                (self.re * rhs.re + self.im * rhs.im) / d,
                (self.im * rhs.re - self.re * rhs.im) / d,
            )
        }
    }
}

/// Goertzel single-bin DFT detector for target frequency index `k_bin` (0 <= k_bin < N).
pub fn goertzel_dft(samples: &[f64], k_bin: f64) -> C64 {
    let n = samples.len();
    if n == 0 {
        return C64::new(0.0, 0.0);
    }
    let omega = (2.0 * PI * k_bin) / (n as f64);
    let coeff = 2.0 * cos(omega);

    let mut s_prev = 0.0;
    let mut s_prev2 = 0.0;

    for &x in samples {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }

    let re = s_prev - s_prev2 * cos(omega);
    let im = -s_prev2 * sin(omega);
    C64::new(re, im)
}

/// Goertzel magnitude detector for target frequency.
#[inline]
pub fn goertzel_magnitude(samples: &[f64], k_bin: f64) -> f64 {
    goertzel_dft(samples, k_bin).norm()
}

/// Gaver-Stehfest Inverse Laplace transform for N = 10 coefficients.
/// Computes f(t) from Laplace transform F(s).
pub fn inv_laplace_stehfest<F: Fn(f64) -> f64>(t: f64, f_laplace: F) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    const LN2: f64 = core::f64::consts::LN_2;
    // Stehfest weights for N = 10
    const V10: [f64; 10] = [
        0.08333333333333333,
        -32.083333333333336,
        1279.0,
        -15623.666666666666,
        84244.16666666666,
        -236957.5,
        375911.6666666667,
        -340071.6666666667,
        164062.5,
        -32812.5,
    ];
    let a = LN2 / t;
    let mut sum = 0.0;
    for (i, &v) in V10.iter().enumerate() {
        let s = ((i + 1) as f64) * a;
        sum += v * f_laplace(s);
    }
    sum * a
}

/// Fixed Talbot numerical Inverse Laplace transform with M integration steps.
/// Implements the Abate & Valko (2004) parabolic contour.
/// F(s) accepts complex s = (re, im) and returns complex (re, im).
pub fn inv_laplace_talbot<F: Fn(f64, f64) -> (f64, f64)>(
    t: f64,
    n_steps: usize,
    f_laplace: F,
) -> f64 {
    if t <= 0.0 || n_steps == 0 {
        return 0.0;
    }
    let m = n_steps as f64;
    let r = 2.0 * m / (5.0 * t);
    let mut sum = 0.0;

    // k = 0
    let s0_re = r;
    let (f0_re, _) = f_laplace(s0_re, 0.0);
    sum += 0.5 * exp(r * t) * f0_re;

    for k in 1..n_steps {
        let theta = (k as f64) * PI / m;
        let cot_th = cos(theta) / sin(theta);
        let s_re = r * theta * cot_th;
        let s_im = r * theta;

        // sigma(theta) = theta + (theta*cot(theta) - 1)*cot(theta)
        // factor = 1 + i * sigma(theta)
        let sigma = theta + (theta * cot_th - 1.0) * cot_th;

        let (f_re, f_im) = f_laplace(s_re, s_im);
        let exp_st_re = exp(s_re * t) * cos(s_im * t);
        let exp_st_im = exp(s_re * t) * sin(s_im * t);

        // Term: exp(s*t) * F(s) * (1 + i * sigma)
        // 1. prod1 = exp(s*t) * F(s)
        let p1_re = exp_st_re * f_re - exp_st_im * f_im;
        let p1_im = exp_st_re * f_im + exp_st_im * f_re;

        // 2. prod1 * (1 + i * sigma)
        // real part = p1_re * 1 - p1_im * sigma
        let term_re = p1_re - p1_im * sigma;
        sum += term_re;
    }

    (r / m) * sum
}

/// Evaluates Horner polynomial for real or complex numbers.
fn eval_poly_complex(coeffs: &[f64], s: C64) -> C64 {
    let mut acc = C64::new(0.0, 0.0);
    for &c in coeffs.iter().rev() {
        acc = acc.mul(s).add(C64::new(c, 0.0));
    }
    acc
}

/// Inverts rational transfer function H(s) = P(s) / Q(s) into time domain h(t) via Talbot contour.
pub fn inv_laplace_talbot_rational(t: f64, num_coeffs: &[f64], den_coeffs: &[f64]) -> f64 {
    inv_laplace_talbot(t, 32, |s_re, s_im| {
        let s = C64::new(s_re, s_im);
        let num = eval_poly_complex(num_coeffs, s);
        let den = eval_poly_complex(den_coeffs, s);
        let res = num.div(den);
        (res.re, res.im)
    })
}

/// Tustin bilinear transformation (s -> z) for first-order low-pass filter:
/// H(s) = 1 / (tau * s + 1)
/// Discrete filter: y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
/// Returns (b0, b1, a1) normalized with a0 = 1.
pub fn tustin_first_order(tau: f64, dt: f64) -> (f64, f64, f64) {
    if tau <= 0.0 || dt <= 0.0 {
        return (1.0, 0.0, 0.0);
    }
    let alpha = 2.0 * tau / dt;
    let a0 = alpha + 1.0;
    let b0 = 1.0 / a0;
    let b1 = 1.0 / a0;
    let a1 = (1.0 - alpha) / a0;
    (b0, b1, a1)
}

/// 64-point Inverse FFT in `#![no_std]` using `microfft`.
pub fn ifft_complex_64_nostd(input: &mut [microfft::Complex32; 64]) {
    // Conjugate, FFT, conjugate, scale by 1/64
    for item in input.iter_mut() {
        item.im = -item.im;
    }
    let _ = microfft::complex::cfft_64(input);
    for item in input.iter_mut() {
        item.re /= 64.0;
        item.im = -item.im / 64.0;
    }
}

/// 64-point Hilbert envelope detector for embedded `#![no_std]`.
pub fn hilbert_envelope_64_nostd(signal_64: &[f32; 64], out_envelope: &mut [f32; 64]) {
    let mut buf = [microfft::Complex32 { re: 0.0, im: 0.0 }; 64];
    for i in 0..64 {
        buf[i].re = signal_64[i];
    }
    let _ = microfft::complex::cfft_64(&mut buf);
    // Double positive frequencies, zero negative
    buf.iter_mut()
        .enumerate()
        .skip(1)
        .take(31)
        .for_each(|(_, c)| {
            c.re *= 2.0;
            c.im *= 2.0;
        });
    buf.iter_mut().enumerate().skip(33).for_each(|(_, c)| {
        c.re = 0.0;
        c.im = 0.0;
    });
    ifft_complex_64_nostd(&mut buf);
    for i in 0..64 {
        out_envelope[i] = libm::sqrtf(buf[i].re * buf[i].re + buf[i].im * buf[i].im);
    }
}

// =========================================================================
// STD FFT SUBMODULE (POWERED BY RUSTFFT)
// =========================================================================

#[cfg(feature = "std")]
pub mod std_fft {
    use super::*;
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    /// Computes forward 1D FFT of any length.
    pub fn fft_forward(input: &[f64]) -> Vec<C64> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        let mut buffer: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buffer);

        buffer.iter().map(|c| C64::new(c.re, c.im)).collect()
    }

    /// Computes inverse 1D FFT of any length with 1/N scaling.
    pub fn fft_inverse(input: &[C64]) -> Vec<f64> {
        let n = input.len();
        if n == 0 {
            return Vec::new();
        }
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n);

        let mut buffer: Vec<Complex<f64>> =
            input.iter().map(|c| Complex::new(c.re, c.im)).collect();
        ifft.process(&mut buffer);

        let inv_n = 1.0 / (n as f64);
        buffer.iter().map(|c| c.re * inv_n).collect()
    }

    /// Hann-windowed Power Spectral Density (PSD).
    /// Returns `(frequencies, psd)` for positive frequencies up to Nyquist (fs/2).
    pub fn hann_psd(signal: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        if n == 0 || fs <= 0.0 {
            return (Vec::new(), Vec::new());
        }

        // Apply Hann window
        let mut windowed = Vec::with_capacity(n);
        let mut s2 = 0.0;
        for (i, &x) in signal.iter().enumerate() {
            let w = 0.5 * (1.0 - cos(2.0 * PI * (i as f64) / (n as f64)));
            s2 += w * w;
            windowed.push(x * w);
        }

        let spectrum = fft_forward(&windowed);
        let n_half = n / 2 + 1;
        let mut freqs = Vec::with_capacity(n_half);
        let mut psd = Vec::with_capacity(n_half);

        let norm = 1.0 / (fs * s2.max(1e-12));

        for i in 0..n_half {
            let f = (i as f64) * fs / (n as f64);
            freqs.push(f);

            let power = spectrum[i].norm_sq();
            let p = if i == 0 || i == n_half - 1 {
                power * norm
            } else {
                2.0 * power * norm
            };
            psd.push(p);
        }

        (freqs, psd)
    }

    /// Hilbert analytic signal envelope and instantaneous phase.
    pub fn hilbert_envelope_and_phase(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        let mut buffer: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut buffer);

        // Analytic signal multiplier in frequency domain
        if n % 2 == 0 {
            for i in 1..n / 2 {
                buffer[i] *= 2.0;
            }
            for i in (n / 2 + 1)..n {
                buffer[i] = Complex::new(0.0, 0.0);
            }
        } else {
            for i in 1..(n + 1) / 2 {
                buffer[i] *= 2.0;
            }
            for i in ((n + 1) / 2)..n {
                buffer[i] = Complex::new(0.0, 0.0);
            }
        }

        ifft.process(&mut buffer);

        let inv_n = 1.0 / (n as f64);
        let mut env = Vec::with_capacity(n);
        let mut phase = Vec::with_capacity(n);

        for c in buffer {
            let re = c.re * inv_n;
            let im = c.im * inv_n;
            env.push(sqrt(re * re + im * im));
            phase.push(atan2(im, re));
        }

        (env, phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goertzel_detects_single_tone() {
        let n = 128;
        let mut signal = [0.0; 128];
        let freq_bin = 8.0;
        for i in 0..n {
            signal[i] = cos(2.0 * PI * freq_bin * (i as f64) / (n as f64));
        }

        let mag_target = goertzel_magnitude(&signal, freq_bin);
        let mag_off = goertzel_magnitude(&signal, 15.0);

        // Peak at target bin should be N/2 = 64
        assert!(fabs(mag_target - 64.0) < 1e-4, "mag_target={mag_target}");
        assert!(mag_off < 1e-4, "mag_off={mag_off}");
    }

    #[test]
    fn inv_laplace_stehfest_exponential_decay() {
        // Laplace of exp(-2*t) is 1 / (s + 2)
        // At t = 1.0, f(1.0) = exp(-2.0) ~ 0.1353352832
        let est = inv_laplace_stehfest(1.0, |s| 1.0 / (s + 2.0));
        assert!(fabs(est - exp(-2.0)) < 1e-3, "est={est}");
    }

    #[test]
    fn inv_laplace_talbot_rational_first_order() {
        // H(s) = 1 / (s + 2) -> h(t) = exp(-2*t)
        // At t = 1.0, exact = exp(-2.0)
        let est = inv_laplace_talbot_rational(1.0, &[1.0], &[2.0, 1.0]);
        assert!(fabs(est - exp(-2.0)) < 1e-4, "est={est}");
    }

    #[test]
    fn tustin_step_response_dc_gain() {
        let (b0, b1, a1) = tustin_first_order(1.0, 0.1);
        // DC gain at z = 1: (b0 + b1) / (1 + a1) must equal 1.0
        let dc = (b0 + b1) / (1.0 + a1);
        assert!(fabs(dc - 1.0) < 1e-12, "dc={dc}");
    }

    #[cfg(feature = "std")]
    #[test]
    fn std_fft_roundtrip_parity() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spec = std_fft::fft_forward(&original);
        let roundtrip = std_fft::fft_inverse(&spec);
        for (a, b) in original.iter().zip(roundtrip.iter()) {
            assert!(fabs(a - b) < 1e-12);
        }
    }
}
