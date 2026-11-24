//! `#![no_std]` Physics-Informed Exponential Moving Average Clustering (**PI-eMeans**).
//!
//! Direct port of Paper 48 (`compute_pi_emeans_clustering.py`) featuring:
//! - Diagonal Thermodynamic Metric Tensor `W_th` resolving Euclidean blindness
//! - Meta scaling law decay rate `alpha* = c_0 * ((K * B) / N)^gamma`
//! - First-law energy residual Cyber-Shield + KL divergence drift acceleration
//! - Fixed stack-only `O(K * D)` state footprint (~1.1 kB SRAM, zero heap allocation)

use libm::{exp, fabs, pow};

pub const PI_EMEANS_K: usize = 8;
pub const PI_EMEANS_D: usize = 16;

/// Physical specification, diagonal thermodynamic metric tensor `W_th`,
/// and cyber-shield parameters `[P48 L150-160, L252-255]`.
#[derive(Clone, Copy, Debug)]
pub struct PiEmeansSpec {
    pub w_diag: [f64; PI_EMEANS_D],
    pub beta_fdia: f64,
    pub tau_th_kw: f64,
    pub kappa_drift: f64,
}

impl PiEmeansSpec {
    /// Returns Paper 48 default 16D thermodynamic metric tensor and cyber-shield constants.
    pub const fn p48_default() -> Self {
        Self {
            w_diag: [
                0.25, // 0: wind speed (downweighted)
                0.25, // 1: active power (downweighted)
                1.0,  // 2: power coefficient C_p
                1.0,  // 3: moist air density rho_moist
                4.5,  // 4: gearbox bearing temperature T_gb
                3.5,  // 5: generator winding temperature T_gen
                1.0,  // 6: oil sump temperature T_oil
                4.0,  // 7: normalized delta T_oil / (v + v0)
                5.0,  // 8: Arrhenius thermal stress ln(1 + CTD_P6)
                1.0,  // 9: pitch angle
                1.0,  // 10: rotor speed
                1.0,  // 11: tower fore-aft acceleration
                4.0,  // 12: high-speed shaft vibration
                1.5,  // 13: nacelle ambient temperature
                3.0,  // 14: first-law thermal residual r_th
                1.0,  // 15: sensor clipping indicator
            ],
            beta_fdia: 0.25,
            tau_th_kw: 15.0,
            kappa_drift: 0.15,
        }
    }
}

/// Persistent `O(K * D)` cluster centroid state in fixed SRAM (zero heap allocation).
#[derive(Clone, Copy, Debug)]
pub struct PiEmeansState {
    pub centers: [[f64; PI_EMEANS_D]; PI_EMEANS_K],
    pub update_counts: [u32; PI_EMEANS_K],
}

impl PiEmeansState {
    /// Initializes cluster centroids cleanly from a caller-provided seed slice.
    pub const fn new(centers: [[f64; PI_EMEANS_D]; PI_EMEANS_K]) -> Self {
        Self {
            centers,
            update_counts: [0; PI_EMEANS_K],
        }
    }

    /// Evaluates one streaming sample in deterministic `O(K * D)` worst-case execution time.
    pub fn step_sample(
        &mut self,
        z: &[f64; PI_EMEANS_D],
        abs_r_th_kw: f64,
        d_kl: f64,
        alpha_base: f64,
        spec: &PiEmeansSpec,
    ) -> PiEmeansStepOut {
        let mut best_k = 0_usize;
        let mut best_dist = thermodynamic_sq_dist(z, &self.centers[0], &spec.w_diag);
        for (k, mu) in self.centers.iter().enumerate().skip(1) {
            let d2 = thermodynamic_sq_dist(z, mu, &spec.w_diag);
            if d2 < best_dist {
                best_dist = d2;
                best_k = k;
            }
        }

        let alpha_eff = cyber_guarded_alpha(alpha_base, abs_r_th_kw, d_kl, spec);
        let one_minus = 1.0 - alpha_eff;
        for (c_j, &z_j) in self.centers[best_k].iter_mut().zip(z.iter()) {
            *c_j = one_minus * (*c_j) + alpha_eff * z_j;
        }
        self.update_counts[best_k] = self.update_counts[best_k].saturating_add(1);

        PiEmeansStepOut {
            regime_id: best_k,
            min_thermo_dist_sq: best_dist,
            alpha_eff,
            cyber_suppressed: abs_r_th_kw > spec.tau_th_kw,
        }
    }

    /// Executes one Paper 48 mini-batch EMA update without heap allocation.
    pub fn step_mini_batch(
        &mut self,
        batch_z: &[[f64; PI_EMEANS_D]],
        mean_abs_r_th_kw: f64,
        d_kl: f64,
        alpha_base: f64,
        spec: &PiEmeansSpec,
    ) -> (f64, usize) {
        if batch_z.is_empty() {
            return (0.0, 0);
        }
        let mut sums = [[0.0_f64; PI_EMEANS_D]; PI_EMEANS_K];
        let mut counts = [0_u32; PI_EMEANS_K];

        for z in batch_z {
            let mut best_k = 0_usize;
            let mut best_dist = thermodynamic_sq_dist(z, &self.centers[0], &spec.w_diag);
            for (k, mu) in self.centers.iter().enumerate().skip(1) {
                let d2 = thermodynamic_sq_dist(z, mu, &spec.w_diag);
                if d2 < best_dist {
                    best_dist = d2;
                    best_k = k;
                }
            }
            counts[best_k] += 1;
            for (acc, &val) in sums[best_k].iter_mut().zip(z.iter()) {
                *acc += val;
            }
        }

        let alpha_eff = cyber_guarded_alpha(alpha_base, mean_abs_r_th_kw, d_kl, spec);
        let one_minus = 1.0 - alpha_eff;
        let mut updated_clusters = 0_usize;
        for k in 0..PI_EMEANS_K {
            if counts[k] > 0 {
                let inv_c = 1.0 / (counts[k] as f64);
                for (c_j, &s_j) in self.centers[k].iter_mut().zip(sums[k].iter()) {
                    *c_j = one_minus * (*c_j) + alpha_eff * (s_j * inv_c);
                }
                self.update_counts[k] = self.update_counts[k].saturating_add(counts[k]);
                updated_clusters += 1;
            }
        }
        (alpha_eff, updated_clusters)
    }
}

/// Output telemetry for a single deterministic edge sample update.
#[derive(Clone, Copy, Debug)]
pub struct PiEmeansStepOut {
    pub regime_id: usize,
    pub min_thermo_dist_sq: f64,
    pub alpha_eff: f64,
    pub cyber_suppressed: bool,
}

/// Computes Meta Scaling Law EMA decay `alpha* = c_0 * ((K * B) / N)^gamma` `[P48 L215-217]`.
#[inline]
pub fn meta_scaling_alpha(
    c0: f64,
    gamma: f64,
    k_regimes: usize,
    batch_size: usize,
    n_total: usize,
) -> f64 {
    if n_total == 0 {
        return 0.0;
    }
    let ratio = ((k_regimes * batch_size) as f64) / (n_total as f64);
    c0 * pow(ratio.max(0.0), gamma)
}

/// Diagonal weighted thermodynamic distance `d_M^2 = (z - mu)^T W_th (z - mu)` `[P48 L265-268]`.
#[inline]
pub fn thermodynamic_sq_dist(
    z: &[f64; PI_EMEANS_D],
    mu: &[f64; PI_EMEANS_D],
    w_diag: &[f64; PI_EMEANS_D],
) -> f64 {
    let mut sum = 0.0;
    for i in 0..PI_EMEANS_D {
        let diff = z[i] - mu[i];
        sum += diff * diff * w_diag[i];
    }
    sum
}

/// Cyber-guarded non-stationary EMA step `alpha_eff = alpha_base * exp(-beta * max(0, |r_th| - tau_th)) * (1 + kappa * |D_KL|)` `[P48 L280-283]`.
#[inline]
pub fn cyber_guarded_alpha(
    alpha_base: f64,
    mean_abs_r_th_kw: f64,
    d_kl: f64,
    spec: &PiEmeansSpec,
) -> f64 {
    let excess = (fabs(mean_abs_r_th_kw) - spec.tau_th_kw).max(0.0);
    let cyber_penalty = exp(-spec.beta_fdia * excess);
    let drift_accel = 1.0 + spec.kappa_drift * fabs(d_kl);
    (alpha_base * cyber_penalty * drift_accel).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p48_meta_scaling_alpha_matches_python_parity() {
        let alpha = meta_scaling_alpha(1.2, 0.5, 8, 1024, 100_000);
        // 1.2 * sqrt(8192 / 100000) = 0.3434600413439677
        assert!(fabs(alpha - 0.343_460_041_344) < 1e-11);
    }

    #[test]
    fn p48_cyber_shield_suppresses_fdia_drift() {
        let spec = PiEmeansSpec::p48_default();
        let clean = cyber_guarded_alpha(0.34, 4.2, 0.05, &spec);
        let attacked = cyber_guarded_alpha(0.34, 42.0, 0.05, &spec);
        // Under 42 kW first-law violation: exp(-0.25 * (42 - 15)) = exp(-6.75) ~ 0.00117
        assert!(clean > 0.34);
        assert!(attacked < clean * 0.002);
    }

    #[test]
    fn p48_streaming_sample_and_batch_update_cleanly() {
        let spec = PiEmeansSpec::p48_default();
        let mut init = [[0.0_f64; PI_EMEANS_D]; PI_EMEANS_K];
        for (k, row) in init.iter_mut().enumerate() {
            row[4] = 40.0 + (k as f64) * 5.0; // distinguish centers by T_gb axis
        }
        let mut state = PiEmeansState::new(init);

        let mut sample = [0.0_f64; PI_EMEANS_D];
        sample[4] = 74.2; // nearest k = 7 (center 75.0)
        let out = state.step_sample(&sample, 3.5, 0.0, 0.25, &spec);
        assert_eq!(out.regime_id, 7);
        assert!(!out.cyber_suppressed);
        assert!(fabs(state.centers[7][4] - (0.75 * 75.0 + 0.25 * 74.2)) < 1e-12);

        let batch = [sample; 4];
        let (_alpha_eff, n_upd) = state.step_mini_batch(&batch, 38.5, 0.02, 0.25, &spec);
        assert_eq!(n_upd, 1);
    }
}
