//! R `extendr` bindings exposing `anemocore` mathematical kernels to R scripts
//! (Papers 07, 31, 32, 33, thermo, and tribology) via `rextendr`.
//!
//! Grant No. ERASMUS-EDU-2025-CBHE-101177220 (EU Erasmus+ CBHE ECO-WIND)

use anemocore::fib_liquidity::{lob_density, roll_effective_spread, FibonacciWallConfig};
use anemocore::fib_qmc::{evaluate_limit_state_6d, golden_section_stopping_search};
use anemocore::kam_vsg::{step_kam_vsg, ControlMode, KamVsgSpec, KamVsgState};
use anemocore::signal_wpt::extract_wpt_indicators_vec;
use anemocore::thermo;
use anemocore::tribology;
use extendr_api::prelude::*;

/// Computes 3-level Daubechies-4 WPT indicators `(p_low_mid, p_high, kappa_high, h3_renyi)`
/// matching Paper 33 (`W-SEM` / `W-MGSEM`).
#[extendr]
fn r_wpt_db4_indicators(signal: &[f64]) -> Vec<f64> {
    match extract_wpt_indicators_vec(signal) {
        Ok(res) => vec![res.p_low_mid, res.p_high, res.kappa_high, res.h3_renyi],
        Err(_) => vec![f64::NAN; 4],
    }
}

/// Computes Roll's (1984) effective bid-ask spread with Parkinson fallback (`Paper 8/43`).
#[extendr]
fn r_market_roll_spread(prices: &[f64], window: i32) -> Vec<f64> {
    roll_effective_spread(prices, window.max(1) as usize)
}

/// Computes Paper 43 Fibonacci Limit-Order-Book density `rho_LOB(z)`.
#[extendr]
fn r_lob_density(z: f64) -> f64 {
    lob_density(z, &FibonacciWallConfig::p43_default())
}

/// Evaluates Paper 46 Golden-Section optimal stopping price `S*(DtF)`.
#[extendr]
fn r_fib_golden_stopping(dtf: f64) -> Vec<f64> {
    let res = golden_section_stopping_search(dtf, 5.0, 160.0, 1e-5, 35);
    vec![res.s_star, res.iterations as f64, res.final_width]
}

/// Evaluates Paper 46 6D structural limit state `(R_cap, S_load, g_val, DtF)`.
#[extendr]
fn r_fib_limit_state_6d(u: &[f64]) -> Vec<f64> {
    if u.len() < 6 {
        return vec![f64::NAN; 4];
    }
    let arr = [u[0], u[1], u[2], u[3], u[4], u[5]];
    let r = evaluate_limit_state_6d(&arr);
    vec![r.r_cap, r.s_load, r.g_val, r.dtf]
}

/// Evaluates one O(1) KAM Golden-Ratio VSG step -> `(p_wt_mw, freq_ratio, t_junction_c)`.
#[extendr]
fn r_kam_vsg_step(theta_s: f64, rocof: f64, delta_f: f64, dt: f64) -> Vec<f64> {
    let spec = KamVsgSpec::p42_default();
    let mut state = KamVsgState {
        theta_s,
        dtheta_s: 0.0,
        v_foster: [0.0; 4],
    };
    let out = step_kam_vsg(
        &spec,
        &mut state,
        ControlMode::KamGoldenRatioVsg,
        rocof,
        delta_f,
        0.0,
        dt,
    );
    vec![out.p_wt_mw, out.omega_ctrl / out.omega_n, out.t_junction_c]
}

/// Arrhenius Cumulative Thermal Dose (CTD_P6) for temperature series.
#[extendr]
fn anemocore_arrhenius_ctd_p6(t_k_series: &[f64], t_ref_k: f64, ea_j_mol: f64, dt_sec: f64) -> f64 {
    thermo::ctd_p6(t_k_series, t_ref_k, ea_j_mol, dt_sec)
}

/// Moist air density per CIPM-2007 standard.
#[extendr]
fn anemocore_air_density_cipm2007(pressure_pa: f64, temp_k: f64, relative_humidity: f64) -> f64 {
    thermo::moist_air_density_cipm2007(pressure_pa, temp_k, relative_humidity)
}

/// Gearbox thermal power loss per ISO/TR 14179 / Ohlendorf / SDPLE.
/// Returns vector: `c(p_vzp, p_vzd, p_vlp, p_vld, p_total_loss)`.
#[extendr]
fn anemocore_gearbox_thermal_loss(
    p_in_w: f64,
    mu_m: f64,
    gear_ratio_u: f64,
    pressure_angle_rad: f64,
    contact_ratio: f64,
    f1: f64,
    p_eq_n: f64,
    d_m_m: f64,
    speed_rad_s: f64,
    c_churn: f64,
    rho_oil: f64,
    nu_cst: f64,
    pitch_line_velocity_ms: f64,
    face_width_m: f64,
    f0: f64,
) -> Vec<f64> {
    let hv = thermo::ohlendorf_gear_loss_factor(gear_ratio_u, pressure_angle_rad, contact_ratio);
    let loss = thermo::sdple_gearbox_power_loss(
        p_in_w,
        mu_m,
        hv,
        f1,
        p_eq_n,
        d_m_m,
        speed_rad_s,
        c_churn,
        rho_oil,
        nu_cst,
        pitch_line_velocity_ms,
        face_width_m,
        f0,
    );
    vec![
        loss.p_vzp_gear_load,
        loss.p_vzd_gear_churning,
        loss.p_vlp_bearing_load,
        loss.p_vld_bearing_noload,
        loss.p_total_loss,
    ]
}

/// 2-Stage Dowson-Higginson EHL film thickness across planetary and helical stages.
/// Returns vector: `c(s1_h_min, s1_lambda, s2_h_min, s2_lambda, min_lambda)`.
#[extendr]
fn anemocore_ehl_two_stage(
    stage1_eta0: f64,
    stage1_u_ent: f64,
    stage1_rx: f64,
    stage1_w: f64,
    stage2_eta0: f64,
    stage2_u_ent: f64,
    stage2_rx: f64,
    stage2_w: f64,
    e_prime_pa: f64,
    alpha_piezo_pa_inv: f64,
    sigma_composite_m: f64,
) -> Vec<f64> {
    let res = tribology::ehl_two_stage(
        stage1_eta0,
        stage1_u_ent,
        stage1_rx,
        stage1_w,
        stage2_eta0,
        stage2_u_ent,
        stage2_rx,
        stage2_w,
        e_prime_pa,
        alpha_piezo_pa_inv,
        sigma_composite_m,
    );
    vec![
        res.stage1_planetary.h_min_m,
        res.stage1_planetary.lambda_ratio,
        res.stage2_helical.h_min_m,
        res.stage2_helical.lambda_ratio,
        res.min_lambda,
    ]
}

// ---------------------------------------------------------------------------
// 4. NEW WIND SPEED & AEP UTILITIES (R bindings)
// ---------------------------------------------------------------------------

use anemocore::wra::{ShearModel, extrapolate_wind_speed, extrapolate_weibull_justus_mikhail};
use anemocore::oa::{normalize_wind_speed_iec, compute_aep_full, AepMethod, LossFactors, AepReport};
use extendr_api::prelude::*;

/// Extrapolate wind speed to a target height using a selectable shear model.
#[extendr]
fn r_extrapolate_wind_speed(u_ref: f64, z_ref: f64, z_target: f64, model: &str) -> f64 {
    let sheath = match model {
        "power_law" => ShearModel::PowerLaw { alpha: 0.143 },
        "log_law" => ShearModel::LogLawNeutral { z0: 0.03 },
        "monin_obukhov" => ShearModel::MoninObukhov { z0: 0.03, obukhov_length_l: -100.0 },
        "gryning" => ShearModel::GryningExtended { z0: 0.03, obukhov_length_l: -100.0, bl_height_h: 150.0 },
        "justus_mikhail" => ShearModel::JustusMikhailWeibull,
        _ => panic!("unknown shear model {}", model),
    };
    extrapolate_wind_speed(u_ref, z_ref, z_target, sheath)
}

/// Extrapolate Weibull parameters using Justus‑Mikhail method.
#[extendr]
fn r_extrapolate_weibull_justus_mikhail(a_ref: f64, k_ref: f64, z_ref: f64, z_target: f64) -> List {
    let (a_target, k_target) = extrapolate_weibull_justus_mikhail(a_ref, k_ref, z_ref, z_target);
    let mut lst = List::new();
    lst.set("a_target", a_target);
    lst.set("k_target", k_target);
    lst
}

/// IEC‑61400‑12‑1 air‑density corrected wind speed.
#[extendr]
fn r_normalize_wind_speed_iec(v: f64, rho: f64) -> f64 {
    normalize_wind_speed_iec(v, rho)
}

/// Compute a full AEP report (simplified linear power curve).
#[extendr]
fn r_compute_aep_full(
    wind_speed_mean: f64,
    n_samples: usize,
    method: &str,
    losses: List,
    rho: f64,
) -> List {
    // Simple linear power curve placeholder: P = 1000 * v (W)
    let pc = |v: f64| v * 1000.0;
    let meth = match method {
        "time_series" => AepMethod::TimeSeries,
        "weibull_qmc" => AepMethod::WeibullQmc,
        _ => panic!("unknown aep method {}", method),
    };
    let loss_factors = LossFactors {
        loss_wake: losses.get("loss_wake").unwrap_or(0.0),
        loss_electrical: losses.get("loss_electrical").unwrap_or(0.0),
        loss_availability: losses.get("loss_availability").unwrap_or(0.0),
        loss_environmental: losses.get("loss_environmental").unwrap_or(0.0),
        loss_curtailment: losses.get("loss_curtailment").unwrap_or(0.0),
        loss_thermal_derating: losses.get("loss_thermal_derating").unwrap_or(0.0),
    };
    let report = compute_aep_full(wind_speed_mean, pc, n_samples, meth, Some(loss_factors), rho);
    let mut out = List::new();
    out.set("aep_gross_mwh", report.aep_gross_mwh);
    out.set("aep_net_mwh", report.aep_net_mwh);
    out.set("gross_capacity_factor", report.gross_capacity_factor);
    out.set("net_capacity_factor", report.net_capacity_factor);
    out.set("p50_mwh", report.p50_mwh);
    out.set("p75_mwh", report.p75_mwh);
    out.set("p90_mwh", report.p90_mwh);
    out.set("p99_mwh", report.p99_mwh);
    out.set("sigma_total", report.sigma_total);
    out.set("eya_gap_mwh", report.eya_gap_mwh);
    out
}

extendr_module! {
    mod anemocore;
    fn r_wpt_db4_indicators;
    fn r_market_roll_spread;
    fn r_lob_density;
    fn r_fib_golden_stopping;
    fn r_fib_limit_state_6d;
    fn r_kam_vsg_step;
    fn anemocore_arrhenius_ctd_p6;
    fn anemocore_air_density_cipm2007;
    fn anemocore_gearbox_thermal_loss;
    fn anemocore_ehl_two_stage;
    fn anemocore_iso281_l10m;
    // New utilities
    fn r_extrapolate_wind_speed;
    fn r_extrapolate_weibull_justus_mikhail;
    fn r_normalize_wind_speed_iec;
    fn r_compute_aep_full;
}

/// Returns vector: `c(l10_rev_millions, l10m_rev_millions, l10m_hours)`.
#[extendr]
fn anemocore_iso281_l10m(
    c_dyn_n: f64,
    p_load_n: f64,
    c_u_n: f64,
    kappa: f64,
    e_c: f64,
    reliability_pct: f64,
    speed_rpm: f64,
    is_roller: bool,
) -> Vec<f64> {
    let btype = if is_roller {
        tribology::BearingType::Roller
    } else {
        tribology::BearingType::Ball
    };
    let l10 = tribology::iso281_basic_rating_life_l10(c_dyn_n, p_load_n, btype);
    let (l10m_rev, l10m_hours) = tribology::iso281_modified_rating_life_l10m(
        c_dyn_n,
        p_load_n,
        c_u_n,
        kappa,
        e_c,
        reliability_pct,
        speed_rpm,
        btype,
    );
    vec![l10, l10m_rev, l10m_hours]
}

// Bearing tilt derating factor (ISO/TS 16281).
#[extendr]
fn anemocore_iso16281_tilt_factor(
    theta_tilt_rad: f64,
    roller_length_m: f64,
    radial_clearance_m: f64,
    mean_contact_deflection_m: f64,
) -> f64 {
    tribology::iso16281_tilt_derating_factor(
        theta_tilt_rad,
        roller_length_m,
        radial_clearance_m,
        mean_contact_deflection_m,
    )
}
/// Fractional Caputo Peridynamics damage indicator.
#[extendr]
fn anemocore_caputo_peridynamics(
    stretch_history: &[f64],
    dt_sec: f64,
    alpha: f64,
    critical_stretch_s0: f64,
) -> f64 {
    tribology::caputo_peridynamics_damage(stretch_history, dt_sec, alpha, critical_stretch_s0)
}


extendr_module! {
    mod anemocore;
    fn r_wpt_db4_indicators;
    fn r_market_roll_spread;
    fn r_lob_density;
    fn r_fib_golden_stopping;
    fn r_fib_limit_state_6d;
    fn r_kam_vsg_step;
    fn anemocore_arrhenius_ctd_p6;
    fn anemocore_air_density_cipm2007;
    fn anemocore_gearbox_thermal_loss;
    fn anemocore_ehl_two_stage;
    fn anemocore_iso16281_tilt_factor;
    fn anemocore_caputo_peridynamics;
}

// ---------------------------------------------------------------------------
// 5. NEW QUASICRYSTAL WPT & QMC UTILITIES (R bindings)
// ---------------------------------------------------------------------------
use anemocore::quasicrystal_wpt::{shift_add_forward_step, shift_add_inverse_step, decompose};
use anemocore::fib_qmc::qmc_weibull_cubic_expectation;

#[extendr]
fn r_quasicrystal_wpt_forward(signal: Vec<i32>) -> List {
    let mut lo = vec![0i32; signal.len()];
    let mut hi = vec![0i32; signal.len()];
    shift_add_forward_step(&signal, &mut lo, &mut hi);
    let mut lst = List::new();
    lst.set("lo", lo);
    lst.set("hi", hi);
    lst
}

#[extendr]
fn r_quasicrystal_wpt_inverse(lo: Vec<i32>, hi: Vec<i32>) -> Vec<i32> {
    let mut recon = vec![0i32; lo.len() + hi.len()];
    shift_add_inverse_step(&lo, &hi, &mut recon);
    recon
}

#[extendr]
fn r_quasicrystal_wpt_decompose(signal: Vec<i32>) -> List {
    let (coeffs, len) = decompose(&signal);
    let mut lst = List::new();
    lst.set("coeffs", coeffs.to_vec());
    lst.set("len", len);
    lst
}

#[extendr]
fn r_fib_qmc_one_sweep(k_weib: f64, c_weib: f64, n_points: usize) -> f64 {
    qmc_weibull_cubic_expectation(k_weib, c_weib, n_points)
}

// ---------------------------------------------------------------------------
// 6. NEW PHYLLOTAXIS WAKE UTILITIES (R bindings)
// ---------------------------------------------------------------------------
use anemocore::phyllotaxis_wake::{vogel_phyllotaxis_layout, structure_factor_2d, golden_yaw_staggering, step_aerothermal_wake, compute_ctd_hours, Golden_ANGLE};

#[extendr]
fn r_vogel_phyllotaxis_layout(n: usize, r_min: f64, c: f64) -> List {
    let (xs, ys) = vogel_phyllotaxis_layout(n, r_min, c);
    let mut lst = List::new();
    lst.set("x", xs);
    lst.set("y", ys);
    lst
}

#[extendr]
fn r_structure_factor_2d(vx: Vec<f64>, vy: Vec<f64>, grid: List) -> List {
    // Expect grid to be a list of two numeric vectors of length 2
    let gx = grid.get::<Vec<f64>>("gx").unwrap_or(vec![0.0, 0.0]);
    let gy = grid.get::<Vec<f64>>("gy").unwrap_or(vec![0.0, 0.0]);
    let grid_arr = [[gx[0], gx[1]], [gy[0], gy[1]]];
    let res = structure_factor_2d(&vx, &vy, &grid_arr);
    let mut lst = List::new();
    for (i, vec) in res.iter().enumerate() {
        lst.set(&format!("row{}", i + 1), vec.clone());
    }
    lst
}

#[extendr]
fn r_golden_yaw_staggering(vx: Vec<f64>, vy: Vec<f64>, rotor_diam: f64, wind_speed: f64, spacing: f64) -> Vec<f64> {
    golden_yaw_staggering(&vx, &vy, rotor_diam, wind_speed, spacing)
}

#[extendr]
fn r_step_aerothermal_wake(vx: Vec<f64>, vy: Vec<f64>, spec: List, hub_height: f64, wind_speed: f64, t_amb: f64, yaw: Vec<f64>, t_oil: Vec<f64>, dt: f64) -> List {
    // Convert spec list to AeroThermalSpec struct (simplified: only needed fields)
    let at_spec = anemocore::phyllotaxis_wake::AeroThermalSpec {
        rotor_diam_m: spec.get::<f64>("rotor_diam_m").unwrap_or(120.0),
        rated_power_w: spec.get::<f64>("rated_power_w").unwrap_or(3.0e6),
        kappa_th: spec.get::<f64>("kappa_th").unwrap_or(1.5e6),
        baseline_loss_w: spec.get::<f64>("baseline_loss_w").unwrap_or(5.0e4),
        g_th: spec.get::<f64>("g_th").unwrap_or(500.0),
        t_amb_k: spec.get::<f64>("t_amb_k").unwrap_or(288.15),
    };
    let mut t_oil_mut = t_oil.clone();
    let step = step_aerothermal_wake(&vx, &vy, &at_spec, hub_height, wind_speed, t_amb, &yaw, &mut t_oil_mut, dt);
    let mut lst = List::new();
    lst.set("t_oil_k", step.t_oil_k);
    lst.set("t_oil_out", t_oil_mut);
    lst
}

#[extendr]
fn r_compute_ctd_hours(t_oil_k: Vec<f64>, spec: List, dt: f64) -> f64 {
    let at_spec = anemocore::phyllotaxis_wake::AeroThermalSpec {
        rotor_diam_m: spec.get::<f64>("rotor_diam_m").unwrap_or(120.0),
        rated_power_w: spec.get::<f64>("rated_power_w").unwrap_or(3.0e6),
        kappa_th: spec.get::<f64>("kappa_th").unwrap_or(1.5e6),
        baseline_loss_w: spec.get::<f64>("baseline_loss_w").unwrap_or(5.0e4),
        g_th: spec.get::<f64>("g_th").unwrap_or(500.0),
        t_amb_k: spec.get::<f64>("t_amb_k").unwrap_or(288.15),
    };
    compute_ctd_hours(&t_oil_k, &at_spec, dt)
}

// ---------------------------------------------------------------------------
// 7. NEW AERODYNAMIC FUNCTIONS (R bindings)
// ---------------------------------------------------------------------------
use anemocore::aero::{prandtl_tip_hub_loss, bem_steady_rotor, beddoes_leishman_stall, aero_torque_thrust_curve};

#[extendr]
fn r_prandtl_tip_hub_loss(r_tip: f64, r_hub: f64, r_rotor: f64, lambda: f64) -> List {
    let (f_tip, f_hub, f_total) = prandtl_tip_hub_loss(r_tip, r_hub, r_rotor, lambda);
    let mut lst = List::new();
    lst.set("f_tip", f_tip);
    lst.set("f_hub", f_hub);
    lst.set("f_total", f_total);
    lst
}

#[extendr]
fn r_bem_steady_rotor(r: f64, chord: f64, twist_deg: f64, phi_rad: f64, lambda: f64, cl: f64, cd: f64, rho: f64) -> List {
    let (a, a_prime) = bem_steady_rotor(r, chord, twist_deg, phi_rad, lambda, cl, cd, rho);
    let mut lst = List::new();
    lst.set("a", a);
    lst.set("a_prime", a_prime);
    lst
}

#[extendr]
fn r_beddoes_leishman_stall(cl_static: f64, alpha_deg: f64, dt: f64, state: f64) -> List {
    let mut s = state;
    let (attached, lift) = beddoes_leishman_stall(cl_static, alpha_deg, dt, &mut s);
    let mut lst = List::new();
    lst.set("attached", attached);
    lst.set("lift", lift);
    lst.set("state", s);
    lst
}

#[extendr]
fn r_aero_torque_thrust_curve(wind_speeds: Vec<f64>, r_rotor: f64, omega: f64, rho: f64, c_p: f64) -> List {
    let res = aero_torque_thrust_curve(&wind_speeds, r_rotor, omega, rho, c_p);
    // Return as a list of vectors
    let mut lst = List::new();
    let (ws, torque, thrust, power, cp, ct): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = res.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        |mut acc, tup| {
            acc.0.push(tup.0);
            acc.1.push(tup.1);
            acc.2.push(tup.2);
            acc.3.push(tup.3);
            acc.4.push(tup.4);
            acc.5.push(tup.5);
            acc
        },
    );
    lst.set("wind_speed", ws);
    lst.set("torque", torque);
    lst.set("thrust", thrust);
    lst.set("power", power);
    lst.set("cp", cp);
    lst.set("ct", ct);
    lst
}

// ---------------------------------------------------------------------------
// 8. NEW TRIBOLOGY EXTENSIONS (R bindings)
// ---------------------------------------------------------------------------
use anemocore::tribology::{mixed_ehl_asperity_load, micropitting_fatigue_index, bearing_spall_progression, lubricant_degradation_viscosity};

#[extendr]
fn r_mixed_ehl_asperity_load(u: f64, v: f64, w: f64, x: f64, y: f64, z: f64, q: f64) -> f64 {
    mixed_ehl_asperity_load(u, v, w, x, y, z, q)
}

#[extendr]
fn r_micropitting_fatigue_index(u: f64, v: f64, w: f64, x: f64, y: f64, z: f64, q: f64) -> f64 {
    micropitting_fatigue_index(u, v, w, x, y, z, q)
}

#[extendr]
fn r_bearing_spall_progression(a: f64, b: f64, c: f64, d: f64) -> f64 {
    bearing_spall_progression(a, b, c, d)
}

#[extendr]
fn r_lubricant_degradation_viscosity(e: f64, f: f64, g: f64) -> f64 {
    lubricant_degradation_viscosity(e, f, g)
}

// ---------------------------------------------------------------------------
// 9. NEW SCADA FLEET FUNCTIONS (R bindings)
// ---------------------------------------------------------------------------
use anemocore::scada_fleet::{sarimax_fit, sarimax_forecast_residuals, fleet_cross_turbine_residual, multi_domain_health_index, scada_missingness_diagnostic};

#[extendr]
fn r_sarimax_fit(y: Vec<f64>, exog: Vec<f64>) -> List {
    let params = sarimax_fit(&y, &exog);
    let mut lst = List::new();
    lst.set("params", params);
    lst
}

#[extendr]
fn r_sarimax_forecast_residuals(y: Vec<f64>, params: List) -> Vec<f64> {
    let pars: Vec<f64> = params.get::<Vec<f64>>("params").unwrap_or_default();
    sarimax_forecast_residuals(&y, &pars)
}

#[extendr]
fn r_fleet_cross_turbine_residual(data: List) -> f64 {
    // Placeholder: just return 0.0 for now
    0.0
}

#[extendr]
fn r_multi_domain_health_index(metrics: Vec<f64>) -> f64 {
    multi_domain_health_index(&metrics)
}

#[extendr]
fn r_scada_missingness_diagnostic(df: List) -> List {
    // Placeholder: return empty list
    List::new()
}

// ---------------------------------------------------------------------------
// 10. NEW ECONOMIC RISK FUNCTIONS (R bindings)
// ---------------------------------------------------------------------------
use anemocore::econ_risk::{compute_lcoe, compute_lcoh, om_unscheduled_cost_model, coupled_monte_carlo_risk, var_cvar_project_npv};

#[extendr]
fn r_compute_lcoe(capex: f64, fixed_opex: f64, variable_opex: f64, discount_rate: f64, lifetime_years: f64, annual_energy_mwh: f64) -> f64 {
    compute_lcoe(capex, fixed_opex, variable_opex, discount_rate, lifetime_years, annual_energy_mwh)
}

#[extendr]
fn r_compute_lcoh(capex: f64, fixed_opex: f64, variable_opex: f64, discount_rate: f64, lifetime_years: f64, annual_energy_mwh: f64, efficiency_h2: f64) -> f64 {
    compute_lcoh(capex, fixed_opex, variable_opex, discount_rate, lifetime_years, annual_energy_mwh, efficiency_h2)
}

#[extendr]
fn r_om_unscheduled_cost_model(failure_rate_per_year: f64, repair_duration_hours: f64, mobilization_cost: f64, downtime_revenue_loss_per_hour: f64) -> f64 {
    om_unscheduled_cost_model(failure_rate_per_year, repair_duration_hours, mobilization_cost, downtime_revenue_loss_per_hour)
}

#[extendr]
fn r_coupled_monte_carlo_risk(seed: u64, n_samples: usize, wind_stddev: f64, drivetrain_failure_rate: f64, price_volatility: f64) -> Vec<f64> {
    coupled_monte_carlo_risk(seed, n_samples, wind_stddev, drivetrain_failure_rate, price_volatility)
}

#[extendr]
fn r_var_cvar_project_npv(samples: Vec<f64>, confidence: f64) -> List {
    let (var, cvar) = var_cvar_project_npv(&samples, confidence);
    let mut lst = List::new();
    lst.set("var", var);
    lst.set("cvar", cvar);
    lst
}
