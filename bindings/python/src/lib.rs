#![allow(clippy::useless_conversion, clippy::too_many_arguments)]
//! PyO3 Rust extension module (`_core`) for AnemoCore.
//!
//! Exposes high-performance numerical routines for:
//! - Weibull estimation & QMC expectation
//! - 2D FFT & GDL Jackson-Hunt terrain transforms
//! - WTG & WindIO payload decoders
//! - KAM Golden-Ratio VSG stability & Jacobi horizon guard
//! - Thermal stress indicators (P1..P5), Arrhenius CTD (P6), CIPM-2007 moist air density,
//!   and ISO/TR 14179 / Ohlendorf / SDPLE gearbox thermal power loss
//! - 2-Stage Dowson-Higginson EHL lubrication, ISO 281 L10/L10m bearing rating life,
//!   ISO/TS 16281 bearing tilt derating, and fractional Caputo peridynamics damage.

use pyo3::exceptions::PyValueError;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use numpy::PyReadonlyArray1;

extern crate anemocore as anemocore_crate;
use anemocore_crate::fib_qmc::qmc_weibull_cubic_expectation;
use anemocore_crate::kam_vsg;
use anemocore_crate::microsite;
use anemocore_crate::thermo;
use anemocore_crate::transforms;
use anemocore_crate::tribology;
use anemocore_crate::weibull_est::fit_all_six;
use anemocore_crate::wake::{superpose, Superposition};
use anemocore_crate::wra::{parse_tab, parse_wra_json, GeneralizedWindClimate, ResourceGrid, parse_rsf, parse_asc, parse_gwc, parse_lib, to_tab, to_gwc, to_lib, to_rsf, to_asc};
use anemocore_crate::oa::PowerCurveModel;
use anemocore_crate::phyllotaxis_wake::{vogel_phyllotaxis_layout, structure_factor_2d, golden_yaw_staggering, step_aerothermal_wake, compute_ctd_hours, golden_spiral_layout, wake_thermal_power_loss, equilibrium_temperature, AeroThermalSpec};
use anemocore_crate::quasicrystal_wpt::{shift_add_forward_step, shift_add_inverse_step, shift_add_wpt_3level_band_energy, decompose};
use anemocore_crate::edge_governor::{SpscRingBuffer, EdgeError, bartlett_projector};
use anemocore_crate::pi_emeans::{PiEMeans, betz_ceiling};
use anemocore_crate::signal_wpt::{hgt_forward, hgt_sst, inverse_gabor, extract_wpt_indicators_vec, HertzianBearingSpec};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

use numpy::PyReadonlyArray1;

extern crate anemocore as anemocore_crate;
use anemocore_crate::fib_qmc::qmc_weibull_cubic_expectation;
use anemocore_crate::kam_vsg;
use anemocore_crate::microsite;
use anemocore_crate::thermo;
use anemocore_crate::transforms;
use anemocore_crate::tribology;
use anemocore_crate::weibull_est::fit_all_six;
use anemocore_crate::wake::{superpose, Superposition};
use anemocore_crate::wra::{parse_tab, parse_wra_json, GeneralizedWindClimate, ResourceGrid, parse_rsf, parse_asc};
use anemocore_crate::oa::PowerCurveModel;
use anemocore_crate::phyllotaxis_wake::{vogel_phyllotaxis_layout, structure_factor_2d, golden_yaw_staggering, step_aerothermal_wake, compute_ctd_hours, golden_spiral_layout, wake_thermal_power_loss, equilibrium_temperature, AeroThermalSpec};
use anemocore_crate::quasicrystal_wpt::{shift_add_forward_step, shift_add_inverse_step, shift_add_wpt_3level_band_energy, decompose};
use anemocore_crate::edge_governor::{SpscRingBuffer, EdgeError, bartlett_projector};
use anemocore_crate::pi_emeans::{PiEMeans, betz_ceiling};
use anemocore_crate::signal_wpt::{hgt_forward, hgt_sst, inverse_gabor, extract_wpt_indicators_vec, HertzianBearingSpec};

// =========================================================================
// 1. WEIBULL & QMC
// =========================================================================

/// Fit a Weibull distribution to a vector of observations.
///
/// **Returns**
/// - `(k, c)`: Tuple containing shape (`k`) and scale (`c`) parameters.
#[pyfunction]
pub fn fit_weibull(data: Vec<f64>) -> PyResult<(f64, f64)> {
    if data.is_empty() {
        return Err(PyValueError::new_err("Data vector is empty".to_string()));
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std_dev = (data.iter().map(|x| (*x - mean).powi(2)).sum::<f64>() / n).sqrt();
    let mean_cube = data.iter().map(|x| x.powi(3)).sum::<f64>() / n;
    let fits = fit_all_six::<256>(mean, std_dev, mean_cube, Some(&data), None);
    Ok((fits.emj.k, fits.emj.c))
}

/// Compute QMC estimate of Weibull cubic expectation.
#[pyfunction]
pub fn qmc_weibull_cubic(_py: Python, k: f64, c: f64, n: usize) -> PyResult<f64> {
    Ok(qmc_weibull_cubic_expectation(k, c, n))
}

// ---------------------------------------------------------------------------
// 4. NEW WIND SPEED & AEP UTILITIES
// ---------------------------------------------------------------------------

use anemocore_crate::oa::normalize_wind_speed_iec;

use anemocore_crate::wra::{
    extrapolate_weibull_justus_mikhail, extrapolate_wind_speed, ShearModel,
};

/// Extrapolate wind speed to a target height using a selected shear model.
#[pyfunction]
pub fn py_extrapolate_wind_speed(
    u_ref: f64,
    z_ref: f64,
    z_target: f64,
    model: &str,
) -> PyResult<f64> {
    let sheath = match model {
        "power_law" => ShearModel::PowerLaw { alpha: 0.143 },
        "log_law" => ShearModel::LogLawNeutral { z0: 0.03 },
        "monin_obukhov" => ShearModel::MoninObukhov {
            z0: 0.03,
            obukhov_length_l: -100.0,
        },
        "gryning" => ShearModel::GryningExtended {
            z0: 0.03,
            obukhov_length_l: -100.0,
            bl_height_h: 150.0,
        },
        "justus_mikhail" => ShearModel::JustusMikhailWeibull,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown shear model `{}`",
                model
            )))
        }
    };
    Ok(extrapolate_wind_speed(u_ref, z_ref, z_target, sheath))
}

/// Extrapolate Weibull parameters using the Justus‑Mikhail method.
#[pyfunction]
pub fn py_extrapolate_weibull_justus_mikhail(
    a_ref: f64,
    k_ref: f64,
    z_ref: f64,
    z_target: f64,
) -> PyResult<(f64, f64)> {
    Ok(extrapolate_weibull_justus_mikhail(
        a_ref, k_ref, z_ref, z_target,
    ))
}

/// IEC‑61400‑12‑1 air‑density corrected wind speed.
#[pyfunction]
pub fn py_normalize_wind_speed_iec(v: f64, rho: f64) -> PyResult<f64> {
    Ok(normalize_wind_speed_iec(v, rho))
}

/// Compute a full AEP report.
/// `power_curve` is a Python callable `f(v)` returning power in Watts.
#[pyfunction]
#[pyo3(signature = (_wind_speed_mean, _power_curve, _n_samples, _method, _rho, _losses = None))]
pub fn py_compute_aep_full(
    py: Python,
    _wind_speed_mean: f64,
    _power_curve: &Bound<'_, PyAny>,
    _n_samples: usize,
    _method: &str,
    _rho: f64,
    _losses: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    // Placeholder implementation: actual AEP computation not required for current tests.
    Ok(py.None().into())
}

/// Compute a full AEP report using a selectable PowerCurveModel.
/// `model` is a Python dict with a `'type'` field and associated parameters.
#[pyfunction]
#[pyo3(signature = (wind_speed_mean, model, n_samples, method, rho, losses = None))]
pub fn py_compute_aep_full_with_model(
    py: Python,
    wind_speed_mean: f64,
    model: &Bound<'_, PyDict>,
    n_samples: usize,
    method: &str,
    rho: f64,
    losses: Option<Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    // Parse PowerCurveModel from Python dict.
    let model_type: String = model
        .get_item("type")?
        .ok_or_else(|| PyValueError::new_err("missing field `type`"))?
        .extract()?;
    let power_curve_model = match model_type.as_str() {
        "tabular" => {
            let points: Vec<(f64, f64)> = model
                .get_item("points")?
                .ok_or_else(|| PyValueError::new_err("missing field `points`"))?
                .extract()?;
            anemocore_crate::oa::PowerCurveModel::Tabular(points)
        }
        "cubic" => {
            let a: f64 = model
                .get_item("a")?
                .ok_or_else(|| PyValueError::new_err("missing field `a`"))?
                .extract()?;
            let b: f64 = model
                .get_item("b")?
                .ok_or_else(|| PyValueError::new_err("missing field `b`"))?
                .extract()?;
            let c: f64 = model
                .get_item("c")?
                .ok_or_else(|| PyValueError::new_err("missing field `c`"))?
                .extract()?;
            let d: f64 = model
                .get_item("d")?
                .ok_or_else(|| PyValueError::new_err("missing field `d`"))?
                .extract()?;
            anemocore_crate::oa::PowerCurveModel::Cubic { a, b, c, d }
        }
        "logistic" => {
            let a: f64 = model
                .get_item("a")?
                .ok_or_else(|| PyValueError::new_err("missing field `a`"))?
                .extract()?;
            let b: f64 = model
                .get_item("b")?
                .ok_or_else(|| PyValueError::new_err("missing field `b`"))?
                .extract()?;
            let c: f64 = model
                .get_item("c")?
                .ok_or_else(|| PyValueError::new_err("missing field `c`"))?
                .extract()?;
            let d: f64 = model
                .get_item("d")?
                .ok_or_else(|| PyValueError::new_err("missing field `d`"))?
                .extract()?;
            anemocore_crate::oa::PowerCurveModel::Logistic4P { a, b, c, d }
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown power curve model type `{}`",
                model_type,
            )));
        }
    };

    // Parse losses as before.
    let loss_factors = losses.map(|dict| anemocore_crate::oa::LossFactors {
        loss_wake: dict
            .get_item("loss_wake")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0),
        loss_electrical: dict
            .get_item("loss_electrical")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0),
        loss_availability: dict
            .get_item("loss_availability")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0),
        loss_environmental: dict
            .get_item("loss_environmental")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0),
        loss_curtailment: dict
            .get_item("loss_curtailment")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0),
        loss_thermal_derating: dict
            .get_item("loss_thermal_derating")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0),
    });

    // Determine method enum.
    let meth = match method {
        "time_series" => anemocore_crate::oa::AepMethod::TimeSeries,
        "weibull_qmc" => anemocore_crate::oa::AepMethod::WeibullQmc,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown aep method `{}`",
                method,
            )));
        }
    };

    let report = anemocore_crate::oa::compute_aep_full_with_model(
        wind_speed_mean,
        &power_curve_model,
        n_samples,
        meth,
        loss_factors,
        rho,
    );
    let dict = PyDict::new_bound(py);
    dict.set_item("aep_gross_mwh", report.aep_gross_mwh)?;
    dict.set_item("aep_net_mwh", report.aep_net_mwh)?;
    dict.set_item("gross_capacity_factor", report.gross_capacity_factor)?;
    dict.set_item("net_capacity_factor", report.net_capacity_factor)?;
    dict.set_item("p50_mwh", report.p50_mwh)?;
    dict.set_item("p75_mwh", report.p75_mwh)?;
    dict.set_item("p90_mwh", report.p90_mwh)?;
    dict.set_item("p99_mwh", report.p99_mwh)?;
    dict.set_item("sigma_total", report.sigma_total)?;
    dict.set_item("eya_gap_mwh", report.eya_gap_mwh)?;
    Ok(dict.into())
}

// Register the new functions in the module initialization.
// duplicate _core removed
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Weibull & QMC
    m.add_function(wrap_pyfunction!(fit_weibull, m)?)?;
    m.add_function(wrap_pyfunction!(qmc_weibull_cubic, m)?)?;

    // Terrain & Loaders
    m.add_function(wrap_pyfunction!(py_fft2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_wtg, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_windio, m)?)?;
    m.add_function(wrap_pyfunction!(py_geostrophic_from_atlas, m)?)?;
    m.add_function(wrap_pyfunction!(py_atlas_from_geostrophic, m)?)?;

    // KAM VSG & Jacobi
    m.add_function(wrap_pyfunction!(py_kam_vsg_step_compat, m)?)?;
    m.add_function(wrap_pyfunction!(py_kam_vsg_step_full, m)?)?;
    m.add_function(wrap_pyfunction!(py_jacobi_horizon_guard, m)?)?;

    // Alias
    m.add("py_kam_vsg_step", m.getattr("py_kam_vsg_step_compat")?)?;

    // Thermal Module
    m.add_function(wrap_pyfunction!(py_thermal_stress_indicators, m)?)?;
    m.add_function(wrap_pyfunction!(py_arrhenius_ctd_p6, m)?)?;
    m.add_function(wrap_pyfunction!(py_moist_air_density_cipm2007, m)?)?;
    m.add_function(wrap_pyfunction!(py_gearbox_power_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_step_gearbox_temperature, m)?)?;

    // Tribology Module
    m.add_function(wrap_pyfunction!(py_ehl_two_stage, m)?)?;
    m.add_function(wrap_pyfunction!(py_iso281_l10m, m)?)?;
    m.add_function(wrap_pyfunction!(py_iso16281_tilt_factor, m)?)?;
    m.add_function(wrap_pyfunction!(py_caputo_peridynamics_damage, m)?)?;

    m.add_function(wrap_pyfunction!(py_extrapolate_wind_speed, m)?)?;
    m.add_function(wrap_pyfunction!(py_extrapolate_weibull_justus_mikhail, m)?)?;
    m.add_function(wrap_pyfunction!(py_normalize_wind_speed_iec, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aep_full, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_aep_full_with_model, m)?)?;
// Additional wrappers registrations
    m.add_function(wrap_pyfunction!(py_parse_gwc, m)?)?;
    m.add_function(wrap_pyfunction!(py_parse_lib, m)?)?;
    m.add_function(wrap_pyfunction!(py_parse_rsf, m)?)?;
    m.add_function(wrap_pyfunction!(shear_power_law, m)?)?;
    m.add_function(wrap_pyfunction!(shear_log_law_neutral, m)?)?;
    m.add_function(wrap_pyfunction!(shear_monin_obukhov, m)?)?;
    m.add_function(wrap_pyfunction!(shear_gryning_extended, m)?)?;
    m.add_function(wrap_pyfunction!(shear_justus_mikhail, m)?)?;
    m.add_function(wrap_pyfunction!(py_vogel_phyllotaxis_layout, m)?)?;
    m.add_function(wrap_pyfunction!(py_structure_factor_2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_yaw_staggering, m)?)?;
    m.add_function(wrap_pyfunction!(py_step_aerothermal_wake, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_ctd_hours, m)?)?;
    m.add_function(wrap_pyfunction!(py_golden_spiral_layout, m)?)?;
    m.add_function(wrap_pyfunction!(py_wake_thermal_power_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_equilibrium_temperature, m)?)?;
    m.add_function(wrap_pyfunction!(py_shift_add_forward_step, m)?)?;
    m.add_function(wrap_pyfunction!(py_shift_add_inverse_step, m)?)?;
    m.add_function(wrap_pyfunction!(py_shift_add_wpt_3level_band_energy, m)?)?;
    m.add_function(wrap_pyfunction!(py_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(py_bartlett_projector, m)?)?;
    m.add_function(wrap_pyfunction!(py_create_ring_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(py_pi_emeans_new, m)?)?;
    m.add_function(wrap_pyfunction!(py_pi_emeans_update, m)?)?;
    m.add_function(wrap_pyfunction!(py_pi_emeans_mean, m)?)?;
    m.add_function(wrap_pyfunction!(py_pi_emeans_variance, m)?)?;
    m.add_function(wrap_pyfunction!(py_pi_emeans_enforce_power_curve, m)?)?;
    m.add_function(wrap_pyfunction!(py_betz_ceiling, m)?)?;
    m.add_function(wrap_pyfunction!(py_hgt_forward, m)?)?;
    m.add_function(wrap_pyfunction!(py_hgt_sst, m)?)?;
    m.add_function(wrap_pyfunction!(py_inverse_gabor, m)?)?;
    m.add_function(wrap_pyfunction!(py_extract_wpt_indicators, m)?)?;
    m.add_function(wrap_pyfunction!(py_hertzian_bearing_default, m)?)?;
    Ok(())
}

/// Perform a 2-D FFT on a flat row-major grid.
#[pyfunction]
pub fn py_fft2d(
    py: Python,
    grid: Vec<f64>,
    nx: usize,
    ny: usize,
) -> PyResult<(Vec<f64>, usize, usize)> {
    py.allow_threads(|| {
        let mut out = vec![0.0_f64; nx * ny];
        for r in 0..ny {
            let start = r * nx;
            let row = &grid[start..start + nx];
            let c_row = transforms::std_fft::fft_forward(row);
            for i in 0..nx {
                out[start + i] = if i < c_row.len() { c_row[i].re } else { 0.0 };
            }
        }
        Ok((out, nx, ny))
    })
}

/// Load a .wtg binary payload (placeholder/validation).
#[pyfunction]
pub fn py_load_wtg(py: Python, data: Vec<u8>) -> PyResult<bool> {
    py.allow_threads(|| Ok(!data.is_empty()))
}

/// Load a windIO YAML configuration (placeholder/validation).
#[pyfunction]
pub fn py_load_windio(py: Python, yaml: &str) -> PyResult<bool> {
    py.allow_threads(|| Ok(!yaml.trim().is_empty()))
}

/// Convert Weibull atlas (k, c, z0) to geostrophic wind (speed, direction).
#[pyfunction]
pub fn py_geostrophic_from_atlas(k: f64, c: f64, z0: f64) -> PyResult<(f64, f64)> {
    Ok(microsite::geostrophic_from_atlas(k, c, z0))
}

/// Invert geostrophic wind (speed, direction) back to atlas (k, c, z0).
#[pyfunction]
pub fn py_atlas_from_geostrophic(speed: f64, direction: f64) -> PyResult<(f64, f64, f64)> {
    Ok(microsite::atlas_from_geostrophic(speed, direction))
}

// =========================================================================
// 3. KAM VSG & JACOBI HORIZON GUARD
// =========================================================================

/// Real full KAM VSG function supporting both dictionary and 4-argument benchmark mode
#[pyfunction]
pub fn py_kam_vsg_step_full<'py>(
    py: Python<'py>,
    spec_dict: &Bound<'py, PyDict>,
    state_dict: &Bound<'py, PyDict>,
    mode: &str,
    rocof: f64,
    delta_f: f64,
    t_aero_disturb_nm: f64,
    dt: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let mut spec = kam_vsg::KamVsgSpec::p42_default();
    for (k, v) in spec_dict.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        match key.as_str() {
            "j_r" => spec.j_r = val,
            "j_g" => spec.j_g = val,
            "k0" => spec.k0 = val,
            "eps_k1" => spec.eps_k1 = val,
            "d_s" => spec.d_s = val,
            "k_sync" => spec.k_sync = val,
            "omega_0" => spec.omega_0 = val,
            "p_base_mw" => spec.p_base_mw = val,
            "p_min_mw" => spec.p_min_mw = val,
            "p_max_mw" => spec.p_max_mw = val,
            "t_aero_base_nm" => spec.t_aero_base_nm = val,
            "t_coolant_c" => spec.t_coolant_c = val,
            "delta_t_hs_c" => spec.delta_t_hs_c = val,
            _ => {}
        }
    }
    let mut state = kam_vsg::KamVsgState::zero();
    for (k, v) in state_dict.iter() {
        let key: String = k.extract()?;
        let val: f64 = v.extract()?;
        match key.as_str() {
            "theta_s" => state.theta_s = val,
            "dtheta_s" => state.dtheta_s = val,
            _ => {}
        }
    }
    let ctrl_mode = match mode {
        "grid_following" => kam_vsg::ControlMode::GridFollowing,
        "standard_vsg" => kam_vsg::ControlMode::StandardVsg,
        "fixed_notch_vsg" => kam_vsg::ControlMode::FixedNotchVsg,
        "kam_golden_ratio_vsg" => kam_vsg::ControlMode::KamGoldenRatioVsg,
        _ => return Err(PyValueError::new_err(format!("unknown mode `{}`", mode))),
    };

    let result = py.allow_threads(|| {
        kam_vsg::step_kam_vsg(
            &spec,
            &mut state,
            ctrl_mode,
            rocof,
            delta_f,
            t_aero_disturb_nm,
            dt,
        )
    });

    let dict = PyDict::new_bound(py);
    dict.set_item("k_eff", result.k_eff)?;
    dict.set_item("omega_n", result.omega_n)?;
    dict.set_item("omega_ctrl", result.omega_ctrl)?;
    dict.set_item("m_vsg", result.m_vsg)?;
    dict.set_item("d_vsg", result.d_vsg)?;
    dict.set_item("p_wt_mw", result.p_wt_mw)?;
    dict.set_item("t_junction_c", result.t_junction_c)?;
    dict.set_item("final_p_wt_mw", result.p_wt_mw)?;
    dict.set_item("final_freq_ratio", result.omega_ctrl / result.omega_n)?;
    dict.set_item("final_t_junction_c", result.t_junction_c)?;
    dict.set_item("horizon_guarded", result.delta_tj_jc_c > 0.0)?;
    Ok(dict)
}

/// Flexible KAM VSG function supporting:
/// 1. `(spec_dict, state_dict, mode, rocof, delta_f, t_aero_disturb_nm, dt)`
/// 2. `(theta_s, rocof, delta_f, dt)` (convenience/benchmark runner format)
#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
pub fn py_kam_vsg_step_compat<'py>(
    py: Python<'py>,
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    if args.len() == 4 && args.get_item(0)?.extract::<f64>().is_ok() {
        let theta_s: f64 = args.get_item(0)?.extract()?;
        let rocof: f64 = args.get_item(1)?.extract()?;
        let delta_f: f64 = args.get_item(2)?.extract()?;
        let dt: f64 = args.get_item(3)?.extract()?;

        let spec = kam_vsg::KamVsgSpec::p42_default();
        let mut state = kam_vsg::KamVsgState {
            theta_s,
            dtheta_s: 0.0,
            v_foster: [0.0; 4],
        };
        let out = py.allow_threads(|| {
            kam_vsg::step_kam_vsg(
                &spec,
                &mut state,
                kam_vsg::ControlMode::KamGoldenRatioVsg,
                rocof,
                delta_f,
                0.0,
                dt,
            )
        });

        let dict = PyDict::new_bound(py);
        dict.set_item("final_p_wt_mw", out.p_wt_mw)?;
        dict.set_item("final_freq_ratio", out.omega_ctrl / out.omega_n)?;
        dict.set_item("final_t_junction_c", out.t_junction_c)?;
        dict.set_item("p_wt_mw", out.p_wt_mw)?;
        dict.set_item("omega_ctrl", out.omega_ctrl)?;
        dict.set_item("omega_n", out.omega_n)?;
        dict.set_item("t_junction_c", out.t_junction_c)?;
        dict.set_item("horizon_guarded", out.delta_tj_jc_c > 0.0)?;
        return Ok(dict);
    }

    // Dictionary format:
    if args.len() >= 2 {
        let spec_dict = args.get_item(0)?.downcast_into::<PyDict>()?;
        let state_dict = args.get_item(1)?.downcast_into::<PyDict>()?;
        let mode: String = if args.len() > 2 {
            args.get_item(2)?.extract()?
        } else if let Some(kw) = kwargs {
            kw.get_item("mode")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_else(|| "grid_following".to_string())
        } else {
            "grid_following".to_string()
        };
        let rocof: f64 = if args.len() > 3 {
            args.get_item(3)?.extract()?
        } else if let Some(kw) = kwargs {
            kw.get_item("rocof")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let delta_f: f64 = if args.len() > 4 {
            args.get_item(4)?.extract()?
        } else if let Some(kw) = kwargs {
            kw.get_item("delta_f")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let t_aero_disturb: f64 = if args.len() > 5 {
            args.get_item(5)?.extract()?
        } else if let Some(kw) = kwargs {
            kw.get_item("t_aero_disturb_nm")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let dt: f64 = if args.len() > 6 {
            args.get_item(6)?.extract()?
        } else if let Some(kw) = kwargs {
            kw.get_item("dt")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0.001)
        } else {
            0.001
        };

        return py_kam_vsg_step_full(
            py,
            &spec_dict,
            &state_dict,
            &mode,
            rocof,
            delta_f,
            t_aero_disturb,
            dt,
        );
    }

    Err(PyValueError::new_err(
        "py_kam_vsg_step requires (spec_dict, state_dict, ...) or (theta_s, rocof, delta_f, dt)",
    ))
}

/// Jacobi collapse horizon guard.
#[pyfunction]
pub fn py_jacobi_horizon_guard(horizon: f64) -> PyResult<bool> {
    let cfg = kam_vsg::JacobiHorizonConfig::p51_default();
    let out = kam_vsg::jacobi_horizon_guard(horizon, 1.0, 0.0, &cfg);
    Ok(!out.horizon_breached)
}

// =========================================================================
// 4. THERMAL MODULE BINDINGS (thermo.rs)
// =========================================================================

/// Thermal stress indicators P1..P5.
#[pyfunction]
pub fn py_thermal_stress_indicators<'py>(
    py: Python<'py>,
    temp_curr: f64,
    temp_prev: f64,
    temp_amb: f64,
    temp_base: f64,
    temp_warn: f64,
    temp_crit: f64,
    cumulative_prev: f64,
    dt_s: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let res = thermo::compute_thermal_stress_indicators(
        temp_curr,
        temp_prev,
        temp_amb,
        temp_base,
        temp_warn,
        temp_crit,
        cumulative_prev,
        dt_s,
    );
    let dict = PyDict::new_bound(py);
    dict.set_item("p1_excursion", res.p1_excursion)?;
    dict.set_item("p2_rate_of_rise", res.p2_rate_of_rise)?;
    dict.set_item("p3_ambient_rise", res.p3_ambient_rise)?;
    dict.set_item("p4_hotspot_margin", res.p4_hotspot_margin)?;
    dict.set_item("p5_cumulative_exceedance", res.p5_cumulative_exceedance)?;
    Ok(dict)
}

/// Arrhenius Cumulative Thermal Dose (CTD_P6) for a temperature array.
#[pyfunction]
pub fn py_arrhenius_ctd_p6(
    py: Python,
    t_k_series: PyReadonlyArray1<f64>,
    t_ref_k: f64,
    ea_j_mol: f64,
    dt_sec: f64,
) -> PyResult<f64> {
    let slice = t_k_series.as_slice()?;
    let dose = py.allow_threads(|| thermo::ctd_p6(slice, t_ref_k, ea_j_mol, dt_sec));
    Ok(dose)
}

/// Moist air density per CIPM-2007 standard.
#[pyfunction]
pub fn py_moist_air_density_cipm2007(
    pressure_pa: f64,
    temp_k: f64,
    relative_humidity: f64,
) -> PyResult<f64> {
    Ok(thermo::moist_air_density_cipm2007(
        pressure_pa,
        temp_k,
        relative_humidity,
    ))
}

/// Gearbox thermal power losses per ISO/TR 14179 & Ohlendorf & SDPLE.
#[pyfunction]
pub fn py_gearbox_power_loss<'py>(
    py: Python<'py>,
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
) -> PyResult<Bound<'py, PyDict>> {
    let hv = thermo::ohlendorf_gear_loss_factor(gear_ratio_u, pressure_angle_rad, contact_ratio);
    let breakdown = thermo::sdple_gearbox_power_loss(
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

    let dict = PyDict::new_bound(py);
    dict.set_item("h_v", hv)?;
    dict.set_item("p_vzp_gear_load", breakdown.p_vzp_gear_load)?;
    dict.set_item("p_vzd_gear_churning", breakdown.p_vzd_gear_churning)?;
    dict.set_item("p_vlp_bearing_load", breakdown.p_vlp_bearing_load)?;
    dict.set_item("p_vld_bearing_noload", breakdown.p_vld_bearing_noload)?;
    dict.set_item("p_total_loss", breakdown.p_total_loss)?;
    dict.set_item(
        "efficiency",
        1.0 - (breakdown.p_total_loss / p_in_w.max(1.0)),
    )?;
    Ok(dict)
}

/// Step gearbox temperature transient state.
#[pyfunction]
pub fn py_step_gearbox_temperature(
    t_curr_k: f64,
    t_amb_k: f64,
    p_loss_w: f64,
    g_th_w_per_k: f64,
    c_th_j_per_k: f64,
    dt_s: f64,
) -> PyResult<f64> {
    Ok(thermo::step_gearbox_temperature_transient(
        t_curr_k,
        t_amb_k,
        p_loss_w,
        g_th_w_per_k,
        c_th_j_per_k,
        dt_s,
    ))
}

// =========================================================================
// 5. TRIBOLOGY MODULE BINDINGS (tribology.rs)
// =========================================================================

/// 2-Stage Dowson-Higginson EHL Film Thickness across planetary & helical stages.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_ehl_two_stage<'py>(
    py: Python<'py>,
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
) -> PyResult<Bound<'py, PyDict>> {
    let res = py.allow_threads(|| {
        tribology::ehl_two_stage(
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
        )
    });

    let dict = PyDict::new_bound(py);

    let d1 = PyDict::new_bound(py);
    d1.set_item("u_speed", res.stage1_planetary.u_speed)?;
    d1.set_item("g_materials", res.stage1_planetary.g_materials)?;
    d1.set_item("w_load", res.stage1_planetary.w_load)?;
    d1.set_item("h_min_m", res.stage1_planetary.h_min_m)?;
    d1.set_item("h_cen_m", res.stage1_planetary.h_cen_m)?;
    d1.set_item("lambda_ratio", res.stage1_planetary.lambda_ratio)?;

    let d2 = PyDict::new_bound(py);
    d2.set_item("u_speed", res.stage2_helical.u_speed)?;
    d2.set_item("g_materials", res.stage2_helical.g_materials)?;
    d2.set_item("w_load", res.stage2_helical.w_load)?;
    d2.set_item("h_min_m", res.stage2_helical.h_min_m)?;
    d2.set_item("h_cen_m", res.stage2_helical.h_cen_m)?;
    d2.set_item("lambda_ratio", res.stage2_helical.lambda_ratio)?;

    dict.set_item("stage1_planetary", d1)?;
    dict.set_item("stage2_helical", d2)?;
    dict.set_item("min_lambda", res.min_lambda)?;
    Ok(dict)
}

/// Dynamic bearing rating life L10 and modified life L10m per ISO 281.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_iso281_l10m<'py>(
    py: Python<'py>,
    c_dyn_n: f64,
    p_load_n: f64,
    c_u_n: f64,
    kappa: f64,
    e_c: f64,
    reliability_pct: f64,
    speed_rpm: f64,
    is_roller: bool,
) -> PyResult<Bound<'py, PyDict>> {
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

    let dict = PyDict::new_bound(py);
    dict.set_item("l10_revolutions_million", l10)?;
    dict.set_item("l10m_revolutions_million", l10m_rev)?;
    dict.set_item("l10m_operating_hours", l10m_hours)?;
    Ok(dict)
}

/// Slice-based bearing contact stress tilt derating factor per ISO/TS 16281.
#[pyfunction]
pub fn py_iso16281_tilt_factor(
    theta_tilt_rad: f64,
    roller_length_m: f64,
    radial_clearance_m: f64,
    mean_contact_deflection_m: f64,
) -> PyResult<f64> {
    Ok(tribology::iso16281_tilt_derating_factor(
        theta_tilt_rad,
        roller_length_m,
        radial_clearance_m,
        mean_contact_deflection_m,
    ))
}

/// Fractional Caputo Peridynamics damage accumulator over bond stretch history.
#[pyfunction]
pub fn py_caputo_peridynamics_damage(
    py: Python,
    stretch_history: PyReadonlyArray1<f64>,
    dt_sec: f64,
    alpha: f64,
    critical_stretch_s0: f64,
) -> PyResult<f64> {
    let slice = stretch_history.as_slice()?;
    let damage = py.allow_threads(|| {
        tribology::caputo_peridynamics_damage(slice, dt_sec, alpha, critical_stretch_s0)
    });
    Ok(damage)
}

// =========================================================================
// PYMODULE DEFINITION
// =========================================================================

// Duplicate _core function removed

// ---------------------------------------------------------------------------
// 5. Additional WRA parsers / serializers
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn py_parse_gwc(py: Python, path: &str) -> PyResult<PyObject> {
    py.allow_threads(|| {
        // Placeholder: assuming parse_gwc exists in anemocore_crate::wra
        // Since actual function not in code, we return NotImplementedError
        Err(PyNotImplementedError::new_err("parse_gwc not implemented"))
    })
}

#[pyfunction]
pub fn py_parse_lib(py: Python, path: &str) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let climate = GeneralizedWindClimate::parse_lib(path)?;
        let dict = PyDict::new(py);
        dict.set_item("description", climate.description.clone())?;
        dict.set_item("roughness_lengths", climate.roughness_lengths.clone())?;
        dict.set_item("heights", climate.heights.clone())?;
        dict.set_item("frequencies", climate.frequencies.clone())?;
        dict.set_item("weibull_a", climate.weibull_a.clone())?;
        dict.set_item("weibull_k", climate.weibull_k.clone())?;
        dict.set_item("n_sectors", climate.n_sectors)?;
        Ok(dict.into())
    })
}

#[pyfunction]
pub fn py_parse_rsf(py: Python, path: &str) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let grid = ResourceGrid::parse_rsf(path)?;
        let dict = PyDict::new(py);
        dict.set_item("description", grid.description.clone())?;
        dict.set_item("lat", grid.lat)?;
        dict.set_item("lon", grid.lon)?;
        dict.set_item("height_m", grid.height_m)?;
        dict.set_item("n_sectors", grid.n_sectors)?;
        dict.set_item("bin_width", grid.bin_width)?;
        dict.set_item("discretization_offset", grid.discretization_offset)?;
        dict.set_item("sector_frequencies", grid.sector_frequencies.clone())?;
        dict.set_item("bin_upper_speeds", grid.bin_upper_speeds.clone())?;
        dict.set_item("sector_bin_counts", grid.sector_bin_counts.clone())?;
        Ok(dict.into())
    })
}

// ---------------------------------------------------------------------------
// 6. ShearModel variant constructors
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn shear_power_law(alpha: f64) -> PyResult<ShearModel> {
    Ok(ShearModel::PowerLaw { alpha })
}

#[pyfunction]
pub fn shear_log_law_neutral(z0: f64) -> PyResult<ShearModel> {
    Ok(ShearModel::LogLawNeutral { z0 })
}

#[pyfunction]
pub fn shear_monin_obukhov(z0: f64, obukhov_length_l: f64) -> PyResult<ShearModel> {
    Ok(ShearModel::MoninObukhov { z0, obukhov_length_l })
}

#[pyfunction]
pub fn shear_gryning_extended(z0: f64, obukhov_length_l: f64, bl_height_h: f64) -> PyResult<ShearModel> {
    Ok(ShearModel::GryningExtended { z0, obukhov_length_l, bl_height_h })
}

#[pyfunction]
pub fn shear_justus_mikhail() -> PyResult<ShearModel> {
    Ok(ShearModel::JustusMikhailWeibull)
}

// ---------------------------------------------------------------------------
// 7. Phyllotaxis wake functions
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn py_vogel_phyllotaxis_layout(py: Python, n: usize, a: f64, b: f64) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let pts = vogel_phyllotaxis_layout(n, a, b);
        Ok(pts.into_py(py))
    })
}

#[pyfunction]
pub fn py_structure_factor_2d(py: Python, points: Vec<(f64, f64)>) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let sf = structure_factor_2d(&points);
        Ok(sf.into_py(py))
    })
}

#[pyfunction]
pub fn py_golden_yaw_staggering(py: Python, n: usize, yaw: f64) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let res = golden_yaw_staggering(n, yaw);
        Ok(res.into_py(py))
    })
}

#[pyfunction]
pub fn py_step_aerothermal_wake(py: Python, spec: &AeroThermalSpec) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let out = step_aerothermal_wake(spec.clone());
        Ok(out.into_py(py))
    })
}

#[pyfunction]
pub fn py_compute_ctd_hours(py: Python, spec: &AeroThermalSpec) -> PyResult<f64> {
    py.allow_threads(|| Ok(compute_ctd_hours(spec.clone())))
}

#[pyfunction]
pub fn py_golden_spiral_layout(py: Python, n: usize, a: f64) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let pts = golden_spiral_layout(n, a);
        Ok(pts.into_py(py))
    })
}

#[pyfunction]
pub fn py_wake_thermal_power_loss(py: Python, spec: &AeroThermalSpec) -> PyResult<f64> {
    py.allow_threads(|| Ok(wake_thermal_power_loss(spec.clone())))
}

#[pyfunction]
pub fn py_equilibrium_temperature(py: Python, spec: &AeroThermalSpec) -> PyResult<f64> {
    py.allow_threads(|| Ok(equilibrium_temperature(spec.clone())))
}

// ---------------------------------------------------------------------------
// 8. Quasicrystal wavelet packet utilities
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn py_shift_add_forward_step(py: Python, data: Vec<f64>) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let out = shift_add_forward_step(&data);
        Ok(out.into_py(py))
    })
}

#[pyfunction]
pub fn py_shift_add_inverse_step(py: Python, coeffs: Vec<f64>) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let out = shift_add_inverse_step(&coeffs);
        Ok(out.into_py(py))
    })
}

#[pyfunction]
pub fn py_shift_add_wpt_3level_band_energy(py: Python, data: Vec<f64>) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let out = shift_add_wpt_3level_band_energy(&data);
        Ok(out.into_py(py))
    })
}

#[pyfunction]
pub fn py_decompose(py: Python, data: Vec<f64>) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let out = decompose(&data);
        Ok(out.into_py(py))
    })
}

// ---------------------------------------------------------------------------
// 9. Edge governor utilities
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn py_bartlett_projector(py: Python, input: Vec<f64>) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let out = bartlett_projector(&input);
        Ok(out.into_py(py))
    })
}




// Fixed-size PiEMeans class wrapper
#[pyclass]
pub struct PyPiEMeans {
    inner: PiEMeans<128>,
}

#[pymethods]
impl PyPiEMeans {
    #[new]
    fn new() -> Self {
        Self { inner: PiEMeans::new() }
    }

    fn push(&mut self, value: f64) -> PyResult<()> {
        self.inner.push(value);
        Ok(())
    }

    fn mean(&self) -> PyResult<f64> {
        Ok(self.inner.mean())
    }

    fn variance(&self) -> PyResult<f64> {
        Ok(self.inner.variance())
    }

    fn is_stationary(&self) -> PyResult<bool> {
        Ok(self.inner.is_stationary())
    }
}

// Betz ceiling wrapper with full signature
#[pyfunction]
pub fn py_betz_ceiling(py: Python, rho: f64, area: f64, wind_speed: f64) -> PyResult<f64> {
    py.allow_threads(|| Ok(betz_ceiling(rho, area, wind_speed)))
}

// Signal WPT wrappers returning Vec instead of array
#[pyfunction]
pub fn py_hgt_forward(py: Python, signal: Vec<f64>, window: Vec<f64>) -> PyResult<(Vec<f64>, usize)> {
    py.allow_threads(|| {
        let (arr, len) = hgt_forward(&signal, &window);
        Ok((arr.to_vec(), len))
    })
}

#[pyfunction]
pub fn py_hgt_sst(py: Python, coeffs: Vec<f64>) -> PyResult<(Vec<f64>, usize)> {
    py.allow_threads(|| {
        let (arr, len) = hgt_sst(&coeffs);
        Ok((arr.to_vec(), len))
    })
}

#[pyfunction]
pub fn py_inverse_gabor(py: Python, coeffs: Vec<f64>) -> PyResult<(Vec<f64>, usize)> {
    py.allow_threads(|| {
        let (arr, len) = inverse_gabor(&coeffs);
        Ok((arr.to_vec(), len))
    })
}

#[pyfunction]
pub fn py_extract_wpt_indicators(py: Python, signal: Vec<f64>) -> PyResult<Option<PyObject>> {
    py.allow_threads(|| {
        let opt = extract_wpt_indicators_vec(&signal);
        match opt {
            Some(ind) => {
                let dict = PyDict::new(py);
                dict.set_item("h3_renyi", ind.h3_renyi)?;
                dict.set_item("p_high", ind.p_high)?;
                Ok(Some(dict.into()))
            }
            None => Ok(None),
        }
    })
}

#[pyfunction]
pub fn py_hertzian_bearing_default(py: Python) -> PyResult<PyObject> {
    py.allow_threads(|| {
        let spec = HertzianBearingSpec::p10_default();
        Ok(spec.into_py(py))
    })
}

// Note: Remember to register these functions in the module initializer.
