//! Golden-Angle Phyllotaxis Farm Geometry, Spatial Structure Factor `S(q)`,
//! 3D Bastankhah Gaussian-Curl Wake, Crespo-Hernández Added Turbulence,
//! Exact-Exponential Stiff-ODE Gearbox Thermal Model, and Arrhenius CTD.
//!
//! Direct port of Paper 45 (`compute_phyllotaxis_yaw_staggering_paper45.py`) and
//! unified with Paper 18 (`run_paper18_wake_thermal_coupling.py`).

use crate::GOLDEN_ANGLE_RAD;
use core::f64::consts::PI;

/// Physical wind turbine aerodynamic & IEC 61400-4 gearbox thermal specification.
#[derive(Clone, Copy, Debug)]
pub struct AeroThermalSpec {
    pub rotor_diam_m: f64,
    pub p_rated_w: f64,
    pub u_rated_ms: f64,
    pub u_cutin_ms: f64,
    pub u_cutout_ms: f64,
    pub c_t0: f64,
    pub k_wake: f64,
    pub i_ambient: f64,
    pub c_th_j_per_k: f64,
    pub h_conv0_w_m2k: f64,
    pub a_surf_m2: f64,
    pub p_churn_w: f64,
    pub eta_mesh: f64,
    pub kappa_shear: f64,
    pub e_a_j_mol: f64,
    pub r_gas: f64,
    pub t_ref_k: f64,
}

impl Default for AeroThermalSpec {
    fn default() -> Self {
        Self::p45_default()
    }
}

impl AeroThermalSpec {
    /// Paper 45 2.05-MW cluster baseline parameters `[P45 L33-53]`.
    pub fn p45_default() -> Self {
        Self {
            rotor_diam_m: 90.0,
            p_rated_w: 2.05e6,
            u_rated_ms: 11.5,
            u_cutin_ms: 3.0,
            u_cutout_ms: 25.0,
            c_t0: 0.80,
            k_wake: 0.04,
            i_ambient: 0.08,
            c_th_j_per_k: 4.5e5,
            h_conv0_w_m2k: 95.0,
            a_surf_m2: 16.0,
            p_churn_w: 18_500.0,
            eta_mesh: 0.982,
            kappa_shear: 0.15,
            e_a_j_mol: 68_000.0,
            r_gas: 8.314,
            t_ref_k: 343.15,
        }
    }
}

/// Output arrays from one discrete time step of the coupled aero-thermal solver.
#[derive(Clone, Debug)]
pub struct AeroThermalStep {
    pub u_local: Vec<f64>,
    pub p_aero_w: Vec<f64>,
    pub t_oil_k: Vec<f64>,
    pub i_eff: Vec<f64>,
}

/// Generates Vogel Golden-Angle Phyllotaxis spiral layout `(x, y)` in meters `[P45 L67-73]`.
pub fn vogel_phyllotaxis_layout(
    n_turb: usize,
    s_min_d: f64,
    rotor_diam_m: f64,
) -> (Vec<f64>, Vec<f64>) {
    let eta_hex = 0.9069;
    let c_pack = (s_min_d * rotor_diam_m) / (PI * eta_hex).sqrt();
    let mut x = Vec::with_capacity(n_turb);
    let mut y = Vec::with_capacity(n_turb);
    for idx in 1..=n_turb {
        let k = idx as f64;
        let r = c_pack * (k - 0.5).sqrt();
        let th = (k * GOLDEN_ANGLE_RAD).rem_euclid(2.0 * PI);
        x.push(r * th.cos());
        y.push(r * th.sin());
    }
    (x, y)
}

/// Computes minimum inter-turbine Euclidean distance normalized by rotor diameter `D` `[P45 L94-98]`.
pub fn min_normalized_spacing_d(x: &[f64], y: &[f64], rotor_diam_m: f64) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return f64::INFINITY;
    }
    let mut min_sq = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let d2 = dx * dx + dy * dy;
            if d2 < min_sq {
                min_sq = d2;
            }
        }
    }
    min_sq.sqrt() / rotor_diam_m
}

/// Computes 2D spatial structure factor `S(q) = |sum_j exp(i q . r_j)|^2 / N` `[P45 L112-123]`.
pub fn structure_factor_2d(x: &[f64], y: &[f64], q_vecs: &[[f64; 2]]) -> Vec<f64> {
    let n = x.len().min(y.len());
    let nf = n.max(1) as f64;
    q_vecs
        .iter()
        .map(|&[qx, qy]| {
            let mut re = 0.0;
            let mut im = 0.0;
            for i in 0..n {
                let phase = qx * x[i] + qy * y[i];
                re += phase.cos();
                im += phase.sin();
            }
            (re * re + im * im) / nf
        })
        .collect()
}

/// Evaluates Golden-Angle Dynamic Yaw Staggering setpoints `[deg]` for brownfield farms `[P45 L291-302]`.
pub fn golden_yaw_staggering(
    x: &[f64],
    y: &[f64],
    rotor_diam_m: f64,
    wind_dir_deg: f64,
    gamma_max_deg: f64,
) -> Vec<f64> {
    let n = x.len().min(y.len());
    let th_w = wind_dir_deg.to_radians();
    let cos_w = th_w.cos();
    let sin_w = th_w.sin();
    let x_w: Vec<f64> = (0..n).map(|i| cos_w * x[i] + sin_w * y[i]).collect();

    let mut yaw = vec![0.0_f64; n];
    for k in 0..n {
        let downwind = x_w
            .iter()
            .filter(|&&xw_j| (xw_j - x_w[k]) > 0.8 * rotor_diam_m)
            .count();
        if downwind >= 2 {
            let weight = ((downwind as f64) / 3.0).tanh();
            yaw[k] = gamma_max_deg * weight * (((k + 1) as f64) * GOLDEN_ANGLE_RAD + th_w).sin();
        }
    }
    yaw
}

/// Advances one aero-thermal wake + exact exponential gearbox oil ODE step (`[P45 L150-227]`).
/// Mutates `t_oil_k_state` in place so callers can stream zero-copy across arbitrary time series.
#[allow(clippy::too_many_arguments)]
pub fn step_aerothermal_wake(
    x: &[f64],
    y: &[f64],
    spec: &AeroThermalSpec,
    u_inf: f64,
    wind_dir_deg: f64,
    t_amb_k: f64,
    yaw_deg: &[f64],
    t_oil_k_state: &mut [f64],
    dt_s: f64,
) -> AeroThermalStep {
    let n = x.len().min(y.len()).min(t_oil_k_state.len());
    let th_w = wind_dir_deg.to_radians();
    let (cos_w, sin_w) = (th_w.cos(), th_w.sin());

    let mut x_w = Vec::with_capacity(n);
    let mut y_w = Vec::with_capacity(n);
    for i in 0..n {
        x_w.push(cos_w * x[i] + sin_w * y[i]);
        y_w.push(-sin_w * x[i] + cos_w * y[i]);
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x_w[a].total_cmp(&x_w[b]));

    let d = spec.rotor_diam_m;
    let d2 = d * d;
    let sqrt8_inv = 1.0 / 8.0_f64.sqrt();
    let a_i = 0.5 * (1.0 - (1.0 - spec.c_t0).sqrt());

    let mut u_local = vec![u_inf; n];
    let mut i_add = vec![0.0_f64; n];

    for idx_rank in 0..n {
        let j = order[idx_rank];
        let mut deficit_sq_sum = 0.0;
        let mut max_i_add = 0.0_f64;

        for &i in order.iter().take(idx_rank) {
            let dx = x_w[j] - x_w[i];
            if dx > 0.5 * d {
                let gamma_i = yaw_deg.get(i).copied().unwrap_or(0.0).to_radians();
                let sigma_y = spec.k_wake * dx + (d * sqrt8_inv) * gamma_i.cos();
                let sigma_z = spec.k_wake * dx + d * sqrt8_inv;
                let delta_def = if gamma_i.abs() > 1e-4 {
                    0.5 * gamma_i * dx * 0.38
                } else {
                    0.0
                };
                let dy = y_w[j] - y_w[i] - delta_def;
                let c_t_eff = spec.c_t0 * libm::pow(gamma_i.cos(), 3.0);
                let arg = (1.0 - c_t_eff / (8.0 * sigma_y * sigma_z / d2)).clamp(0.0, 1.0);
                let amp = 1.0 - arg.sqrt();
                let gauss = (-0.5 * (dy * dy) / (sigma_y * sigma_y)).exp();
                let du = amp * gauss;
                deficit_sq_sum += du * du;

                let i_add_ij =
                    0.73 * a_i.powf(0.8325) * spec.i_ambient.powf(0.0325) * (dx / d).powf(-0.32);
                if i_add_ij > max_i_add {
                    max_i_add = i_add_ij;
                }
            }
        }
        u_local[j] = (u_inf * (1.0 - deficit_sq_sum.sqrt())).clamp(0.1, u_inf);
        i_add[j] = max_i_add;
    }

    let mut p_aero_w = Vec::with_capacity(n);
    let mut t_oil_k = Vec::with_capacity(n);
    let mut i_eff = Vec::with_capacity(n);

    for k in 0..n {
        let u = u_local[k];
        let gamma_rad = yaw_deg.get(k).copied().unwrap_or(0.0).to_radians();
        let raw_p = if u < spec.u_cutin_ms || u > spec.u_cutout_ms {
            0.0
        } else {
            let frac = (u - spec.u_cutin_ms) / (spec.u_rated_ms - spec.u_cutin_ms);
            (spec.p_rated_w * libm::pow(frac, 3.0)).min(spec.p_rated_w)
        };
        let p_a = raw_p * gamma_rad.cos().powf(1.88);
        let i_e = (spec.i_ambient * spec.i_ambient + i_add[k] * i_add[k]).sqrt();
        let q_gen = spec.p_churn_w
            + (1.0 - spec.eta_mesh) * p_a
            + spec.kappa_shear * (i_e * i_e) * spec.p_rated_w;
        let h_conv = spec.h_conv0_w_m2k * (0.30 + 0.70 * (u / spec.u_rated_ms).powf(0.8));
        let g_th = h_conv * spec.a_surf_m2;
        let tau_th = spec.c_th_j_per_k / g_th;
        let t_eq = t_amb_k + q_gen / g_th;
        let t_next = t_eq + (t_oil_k_state[k] - t_eq) * (-dt_s / tau_th).exp();
        t_oil_k_state[k] = t_next;

        p_aero_w.push(p_a);
        t_oil_k.push(t_next);
        i_eff.push(i_e);
    }

    AeroThermalStep {
        u_local,
        p_aero_w,
        t_oil_k,
        i_eff,
    }
}

/// Arrhenius Cumulative Thermal Dose (equivalent hours at `T_ref`) `[P45 L243-246]`.
pub fn compute_ctd_hours(t_oil_k_series: &[f64], spec: &AeroThermalSpec, dt_s: f64) -> f64 {
    let sum_rate: f64 = t_oil_k_series
        .iter()
        .map(|&tk| {
            let exp_arg = ((spec.e_a_j_mol / spec.r_gas) * (1.0 / spec.t_ref_k - 1.0 / tk))
                .clamp(-50.0, 50.0);
            exp_arg.exp()
        })
        .sum();
    (sum_rate * dt_s) / 3600.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p45_vogel_layout_matches_python_parity() {
        let spec = AeroThermalSpec::p45_default();
        let (x, y) = vogel_phyllotaxis_layout(100, 4.5, spec.rotor_diam_m);
        assert!((x[0] - (-125.10371073)).abs() < 1e-6);
        assert!((y[0] - 114.60524696).abs() < 1e-6);
        let min_d = min_normalized_spacing_d(&x, &y, spec.rotor_diam_m);
        assert!((min_d - 4.12170594).abs() < 1e-6);
    }

    #[test]
    fn p45_aligned_wake_and_ctd_match_python_parity() {
        let spec = AeroThermalSpec::p45_default();
        let x = [0.0, 450.0];
        let y = [0.0, 0.0];
        let yaw = [0.0, 0.0];
        let mut t_oil = [288.15 + 45.0, 288.15 + 45.0];
        let step = step_aerothermal_wake(&x, &y, &spec, 9.5, 0.0, 288.15, &yaw, &mut t_oil, 600.0);

        assert!((step.u_local[1] - 7.7972490051).abs() < 1e-8);
        assert!((step.i_eff[1] - 0.1592908022).abs() < 1e-8);
        assert!((step.p_aero_w[1] - 368_530.866498).abs() < 1e-3);
        assert!((step.t_oil_k[1] - 318.3351823433).abs() < 1e-8);

        let ctd_10m = compute_ctd_hours(&[353.15], &spec, 600.0);
        assert!((ctd_10m - 0.3273143630).abs() < 1e-8);
    }
}
