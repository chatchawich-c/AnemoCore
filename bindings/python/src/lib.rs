//! Simplified Python bindings for the AnemoCore core library using PyO3.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use std::collections::HashMap;

extern crate anemocore as anemocore_crate;
use anemocore_crate::fib_qmc::qmc_weibull_cubic_expectation;
use anemocore_crate::kam_vsg;
use anemocore_crate::transforms;
use anemocore_crate::weibull_est::fit_all_six;

/// Fit a Weibull distribution to a vector of data.
///
/// **Args**
/// - `data`: Vector of `f64` wind speed observations.
///
/// **Returns**
/// - `(k, c)`: Tuple containing the shape (`k`) and scale (`c`) parameters.
///
/// **Example**
/// ```python
/// from anemocore import fit_weibull
/// k, c = fit_weibull([1.0, 2.5, 3.0])
/// ```
    if data.is_empty() {
        return Err(PyErr::new::<PyValueError, _>(
            "Data vector is empty".to_string(),
        ));
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let std_dev = (data.iter().map(|x| (*x - mean).powi(2)).sum::<f64>() / n).sqrt();
    let mean_cube = data.iter().map(|x| x.powi(3)).sum::<f64>() / n;
    let fits = fit_all_six::<256>(mean, std_dev, mean_cube, Some(&data), None);
    Ok((fits.emj.k, fits.emj.c))
}

/// Compute QMC estimate of Weibull cubic expectation.
///
/// **Args**
/// - `k`: shape parameter of the Weibull distribution.
/// - `c`: scale parameter of the Weibull distribution.
/// - `n`: number of quasi‑Monte‑Carlo samples.
///
/// **Returns**
/// - Expected value as `f64`.
///
/// **Example**
/// ```python
/// from anemocore import qmc_weibull_cubic
/// val = qmc_weibull_cubic(2.0, 5.0, 10000)
/// ```
#[pyfunction]
fn qmc_weibull_cubic(_py: Python, k: f64, c: f64, n: usize) -> PyResult<f64> {
    Ok(qmc_weibull_cubic_expectation(k, c, n))
}
    Ok(qmc_weibull_cubic_expectation(k, c, n))
}

// ---------------------------- New bindings ----------------------------
/// Perform a simple 2‑D FFT on a flat row‑major grid.
///
/// **Args**
/// - `grid`: Flat vector of length `nx * ny` containing the input data.
/// - `nx`: Number of points in the x‑direction.
/// - `ny`: Number of points in the y‑direction.
///
/// **Returns**
/// - `(out, nx, ny)`: Tuple where `out` is a flat vector of the transformed data (real part for the first element of each row, imaginary part for the rest), and `nx`, `ny` are echoed for convenience.
///
/// **Example**
/// ```python
/// from anemocore import py_fft2d
/// out, nx, ny = py_fft2d([1.0, 2.0, 3.0, 4.0], nx=2, ny=2)
/// ```
#[pyfunction]
fn py_fft2d(
    _py: Python,
    grid: Vec<f64>,
    nx: usize,
    ny: usize,
) -> PyResult<(Vec<f64>, usize, usize)> {
    // Release the GIL while we perform the heavy work.
    _py.allow_threads(|| {
        // Very naive implementation: apply the existing micro_fft64 on each row.
        // In a real scenario you would call a proper 2‑D FFT routine.
        let mut out = vec![0.0_f64; nx * ny];
        for r in 0..ny {
            let start = r * nx;
            let row = &grid[start..start + nx];
            // micro_fft64 returns a complex number; we store real then imag.
            let (c, _len, _nx) = transforms::micro_fft64(row);
            for i in 0..nx {
                out[start + i] = if i == 0 { c.re } else { c.im };
            }
        }
        Ok((out, nx, ny))
    })
}

/// Load a .wtg binary payload (placeholder).
///
/// **Args**
/// - `data`: Byte vector containing the `.wtg` payload.
///
/// **Returns**
/// - `bool`: `true` if the payload is non‑empty.
///
/// **Example**
/// ```python
/// from anemocore import py_load_wtg
/// ok = py_load_wtg(b"\x00\x01\x02")
/// ```
#[pyfunction]
fn py_load_wtg(_py: Python, data: Vec<u8>) -> PyResult<bool> {
    _py.allow_threads(|| Ok(!data.is_empty()))
}
    _py.allow_threads(|| Ok(!data.is_empty()))
}

/// Load a windIO YAML configuration (placeholder).
///
/// **Args**
/// - `yaml`: YAML string containing windIO configuration.
///
/// **Returns**
/// - `bool`: `true` if the string is non‑empty after trimming.
///
/// **Example**
/// ```python
/// from anemocore import py_load_windio
/// ok = py_load_windio("""
/// wind_speed: 12.5
/// turbine: ...
/// """)
/// ```
#[pyfunction]
fn py_load_windio(_py: Python, yaml: &str) -> PyResult<bool> {
    _py.allow_threads(|| Ok(!yaml.trim().is_empty()))
}
    _py.allow_threads(|| Ok(!yaml.trim().is_empty()))
}

/// KAM VSG step wrapper.
#[pyfunction]
fn py_kam_vsg_step(
    _py: Python,
    spec_dict: &Bound<'_, PyDict>,
    state_dict: &Bound<'_, PyDict>,
    mode: &str,
    rocof: f64,
    delta_f: f64,
    t_aero_disturb_nm: f64,
    dt: f64,
) -> PyResult<PyObject> {
    // Build a default spec – actual implementation would read values from spec_dict.
    let mut spec = kam_vsg::KamVsgSpec::p42_default();
    // Populate spec fields from the dict.
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
    // Build a default state.
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
    // Choose control mode.
    let ctrl_mode = match mode {
        "grid_following" => kam_vsg::ControlMode::GridFollowing,
        "standard_vsg" => kam_vsg::ControlMode::StandardVsg,
        "fixed_notch_vsg" => kam_vsg::ControlMode::FixedNotchVsg,
        "kam_golden_ratio_vsg" => kam_vsg::ControlMode::KamGoldenRatioVsg,
        _ => {
            return Err(PyErr::new::<PyValueError, _>(format!("unknown mode `{}`", mode)))
        }
    };
    // Run the step while releasing the GIL.
    let result = _py.allow_threads(|| {
        kam_vsg::step_kam_vsg(&spec, &mut state, ctrl_mode, rocof, delta_f, t_aero_disturb_nm, dt)
    });
    // Convert the output struct into a Python dict.
    let dict = PyDict::new(_py);
    dict.set_item("k_eff", result.k_eff)?;
    dict.set_item("omega_n", result.omega_n)?;
    dict.set_item("omega_ctrl", result.omega_ctrl)?;
    dict.set_item("m_vsg", result.m_vsg)?;
    dict.set_item("d_vsg", result.d_vsg)?;
    dict.set_item("p_wt_mw", result.p_wt_mw)?;
    dict.set_item("t_junction_c", result.t_junction_c)?;
    dict.set_item("horizon_guarded", result.delta_tj_jc_c > 0.0)?;
    Ok(dict.into())
}

// Duplicate implementation removed – the correct version above is used.
}

#[pymodule]
fn anemocore(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_weibull, m)?)?;
    m.add_function(wrap_pyfunction!(qmc_weibull_cubic, m)?)?;
    // New bindings
    m.add_function(wrap_pyfunction!(py_fft2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_wtg, m)?)?;
    m.add_function(wrap_pyfunction!(py_load_windio, m)?)?;
    m.add_function(wrap_pyfunction!(py_kam_vsg_step, m)?)?;
    Ok(())
}
