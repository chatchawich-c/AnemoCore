//! `#![no_std]` Edge Governor & Latent Factor Score Projection Kernels.
//!
//! Includes [`bartlett_factor_project`] from **Paper 50 (`SEM-GB`)**, computing the
//! `237 -> 12` standardized Bartlett factor-score projection `eta = W_FS * ((x - mu) .* inv_std)`
//! with zero heap allocation (`< 0.45 µs` target budget on edge hardware).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GovernorError {
    DimensionMismatch,
}

/// Projects a raw telemetry slice `x` (`P` channels, e.g. `237`) onto a standardized
/// latent structural bottleneck `latent_out` (`K` factors, e.g. `12`) via row-major
/// Bartlett factor-score weights `w_fs` of shape `K x P` (`[P50 Table 4]`).
///
/// Computes `latent_out[k] = sum_j w_fs[k * P + j] * (x[j] - mu[j]) * inv_std[j]`
/// strictly in `#![no_std]` without heap allocation.
#[inline]
pub fn bartlett_factor_project(
    x: &[f64],
    mu: &[f64],
    inv_std: &[f64],
    w_fs: &[f64],
    latent_out: &mut [f64],
) -> Result<usize, GovernorError> {
    let p = x.len();
    let k_dim = latent_out.len();
    if p == 0 || k_dim == 0 || mu.len() != p || inv_std.len() != p || w_fs.len() != k_dim * p {
        return Err(GovernorError::DimensionMismatch);
    }

    for (k, out_slot) in latent_out.iter_mut().enumerate() {
        let row = &w_fs[k * p..(k + 1) * p];
        let mut acc = 0.0;
        for j in 0..p {
            let z_j = (x[j] - mu[j]) * inv_std[j];
            acc += row[j] * z_j;
        }
        *out_slot = acc;
    }
    Ok(k_dim)
}

/// Telemetry integrity result from Paper 26 Virtual Wind Speed Stealth-FDIA Observer.
#[derive(Clone, Copy, Debug)]
pub struct WindFdiaStatus {
    pub v_meas_mps: f64,
    pub v_proxy_mps: f64,
    pub abs_residual_mps: f64,
    pub is_compromised: bool,
}

/// Checks instantaneous wind telemetry integrity `|v_meas(t) - v_proxy(t)| > threshold_mps`
/// (`Paper 26 Stage 3`, healthy threshold `< 1.5 m/s`) in `#![no_std]` `O(1)` time.
#[inline]
pub fn stealth_fdia_wind_residual(
    v_meas_mps: f64,
    v_proxy_mps: f64,
    threshold_mps: f64,
) -> WindFdiaStatus {
    let abs_residual_mps = libm::fabs(v_meas_mps - v_proxy_mps);
    WindFdiaStatus {
        v_meas_mps,
        v_proxy_mps,
        abs_residual_mps,
        is_compromised: abs_residual_mps > threshold_mps,
    }
}

/// Evaluates farm-wide Stealth FDIA flags across a slice without heap allocation,
/// returning the count of compromised turbines.
pub fn detect_stealth_fdia_wind_slice(
    v_meas: &[f64],
    v_proxy: &[f64],
    threshold_mps: f64,
    compromised_flags: &mut [bool],
) -> Result<usize, GovernorError> {
    let n = v_meas.len();
    if v_proxy.len() != n || compromised_flags.len() < n {
        return Err(GovernorError::DimensionMismatch);
    }
    let mut flagged = 0_usize;
    for i in 0..n {
        let hit = libm::fabs(v_meas[i] - v_proxy[i]) > threshold_mps;
        compromised_flags[i] = hit;
        if hit {
            flagged += 1;
        }
    }
    Ok(flagged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libm::fabs;

    #[test]
    fn p50_bartlett_projection_237x12_no_std() {
        const P: usize = 237;
        const K: usize = 12;
        let mut x = [0.0_f64; P];
        let mu = [10.0_f64; P];
        let inv_std = [0.5_f64; P];
        let mut w_fs = [0.0_f64; K * P];

        for (j, val) in x.iter_mut().enumerate() {
            *val = 10.0 + ((j % 3) as f64) * 2.0; // z_j = 0, 1, 2 cycling 79 times
        }
        for k in 0..K {
            for j in 0..P {
                w_fs[k * P + j] = ((k + 1) as f64) * 0.01;
            }
        }

        let mut latent = [0.0_f64; K];
        let written = bartlett_factor_project(&x, &mu, &inv_std, &w_fs, &mut latent).unwrap();
        assert_eq!(written, K);
        // sum_{j=0}^{236} z_j = 79 * (0 + 1 + 2) = 237
        for (k, &val) in latent.iter().enumerate() {
            let expected = ((k + 1) as f64) * 0.01 * 237.0;
            assert!(
                fabs(val - expected) < 1e-10,
                "k={k}: got {val}, expected {expected}"
            );
        }
    }

    #[test]
    fn rejects_mismatched_matrix_shape() {
        let x = [1.0; 4];
        let mu = [0.0; 4];
        let inv_std = [1.0; 3];
        let w_fs = [1.0; 8];
        let mut out = [0.0; 2];
        assert_eq!(
            bartlett_factor_project(&x, &mu, &inv_std, &w_fs, &mut out),
            Err(GovernorError::DimensionMismatch)
        );
    }

    #[test]
    fn p26_stealth_fdia_wind_observer_detects_compromised_node() {
        let healthy = stealth_fdia_wind_residual(12.56, 12.31, 1.5);
        assert!(!healthy.is_compromised);
        assert!(fabs(healthy.abs_residual_mps - 0.25) < 1e-12);

        // Paper 26 T12 Stealth FDIA: attacker injects 4.20 m/s vs ~10.80 m/s physical proxy
        let attacked = stealth_fdia_wind_residual(4.20, 10.80, 1.5);
        assert!(attacked.is_compromised);
        assert!(fabs(attacked.abs_residual_mps - 6.60) < 1e-12);

        let v_meas = [12.56, 4.20, 9.10];
        let v_proxy = [12.31, 10.80, 9.02];
        let mut flags = [false; 3];
        let count = detect_stealth_fdia_wind_slice(&v_meas, &v_proxy, 1.5, &mut flags).unwrap();
        assert_eq!(count, 1);
        assert_eq!(flags, [false, true, false]);
    }
}
