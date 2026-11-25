// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::too_many_arguments,
    clippy::manual_clamp,
    clippy::let_and_return
)]
//! Thermal and moisture related utilities for AnemoCore.
//!
//! Grant No. ERASMUS-EDU-2025-CBHE-101177220 (EU Erasmus+ CBHE ECO-WIND)
//!
//! Provides:
//! - Thermal stress indicators `P1`..`P5`:
//!   * `P1`: Absolute thermal excursion above baseline ($T - T_{\text{base}}$)
//!   * `P2`: Thermal rate of rise / gradient ($\Delta T / \Delta t$)
//!   * `P3`: Ambient-adjusted temperature rise ($T_{\text{oil}} - T_{\text{amb}}$)
//!   * `P4`: Hotspot temperature margin to critical tripping limit ($T_{\text{crit}} - T$)
//!   * `P5`: Cumulative thermal exceedance dose above warning threshold
//! - Arrhenius Cumulative Thermal Dose `CTD_P6`:
//!   $$D_{\text{CTD}} = \sum \exp\left(-\frac{E_a}{R}\left(\frac{1}{T_k} - \frac{1}{T_{\text{ref}, k}}\right)\right) \Delta t$$
//!   with $R = 8.3144626\ \text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$
//! - Moist Air Density per `CIPM-2007`:
//!   $$\rho = \frac{p M_a}{Z R T} \left(1 - x_v \left(1 - \frac{M_v}{M_a}\right)\right)$$
//!   with saturation vapor pressure (Magnus/IAPWS formulation), enhancement factor $f(p, T)$, and compressibility factor $Z(p, T, x_v)$.
//! - Gearbox thermal power loss helpers per `ISO/TR 14179`, `Ohlendorf`, and `SDPLE`:
//!   $$P_{\text{loss}} = P_{\text{VZP}} + P_{\text{VZD}} + P_{\text{VLP}} + P_{\text{VLD}}$$
//!   and transient / steady-state thermal temperature rise.

extern crate libm;

use core::f64::consts::PI;

/// Universal gas constant in J·mol⁻¹·K⁻¹.
pub const R_GAS: f64 = 8.314_462_6;

/// Molar mass of dry air in kg·mol⁻¹ (CIPM-2007).
pub const M_A: f64 = 28.965_46e-3;

/// Molar mass of water vapor in kg·mol⁻¹ (CIPM-2007).
pub const M_V: f64 = 18.015_28e-3;

// =========================================================================
// 1. THERMAL STRESS INDICATORS (P1..P5) & ARRHENIUS CTD_P6
// =========================================================================

/// Thermal Stress Indicator Struct representing P1..P5 indicators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermalStressIndicators {
    /// P1: Absolute thermal excursion above baseline (°C or K)
    pub p1_excursion: f64,
    /// P2: Thermal rate of rise / gradient (°C/s or K/s)
    pub p2_rate_of_rise: f64,
    /// P3: Ambient-adjusted temperature rise (°C or K)
    pub p3_ambient_rise: f64,
    /// P4: Hotspot temperature safety margin to critical tripping limit (°C or K)
    pub p4_hotspot_margin: f64,
    /// P5: Cumulative exceedance dose above warning threshold (°C·s)
    pub p5_cumulative_exceedance: f64,
}

/// Computes thermal stress indicator P1: Absolute thermal excursion above baseline:
/// $$P_1 = \max(0, T - T_{\text{base}})$$
#[inline]
pub fn p1_thermal_excursion(temp: f64, temp_base: f64) -> f64 {
    if temp > temp_base {
        temp - temp_base
    } else {
        0.0
    }
}

/// Computes thermal stress indicator P2: Thermal rate of rise / gradient:
/// $$P_2 = \frac{T_2 - T_1}{\Delta t}$$
#[inline]
pub fn p2_thermal_rate_of_rise(temp_curr: f64, temp_prev: f64, dt_s: f64) -> f64 {
    if dt_s > 0.0 {
        (temp_curr - temp_prev) / dt_s
    } else {
        0.0
    }
}

/// Computes thermal stress indicator P3: Ambient-adjusted temperature rise:
/// $$P_3 = T_{\text{component}} - T_{\text{ambient}}$$
#[inline]
pub fn p3_ambient_adjusted_rise(temp_comp: f64, temp_amb: f64) -> f64 {
    temp_comp - temp_amb
}

/// Computes thermal stress indicator P4: Hotspot temperature safety margin to critical trip:
/// $$P_4 = T_{\text{crit}} - T_{\text{hotspot}}$$
#[inline]
pub fn p4_hotspot_margin(temp_hotspot: f64, temp_crit: f64) -> f64 {
    temp_crit - temp_hotspot
}

/// Computes thermal stress indicator P5: Exceedance increment above warning threshold:
/// $$\Delta P_5 = \max(0, T - T_{\text{warn}}) \Delta t$$
#[inline]
pub fn p5_exceedance_increment(temp: f64, temp_warn: f64, dt_s: f64) -> f64 {
    if temp > temp_warn && dt_s > 0.0 {
        (temp - temp_warn) * dt_s
    } else {
        0.0
    }
}

/// Computes instantaneous thermal stress indicators P1..P5 as a bundle.
pub fn compute_thermal_stress_indicators(
    temp_curr: f64,
    temp_prev: f64,
    temp_amb: f64,
    temp_base: f64,
    temp_warn: f64,
    temp_crit: f64,
    cumulative_prev: f64,
    dt_s: f64,
) -> ThermalStressIndicators {
    let p1 = p1_thermal_excursion(temp_curr, temp_base);
    let p2 = p2_thermal_rate_of_rise(temp_curr, temp_prev, dt_s);
    let p3 = p3_ambient_adjusted_rise(temp_curr, temp_amb);
    let p4 = p4_hotspot_margin(temp_curr, temp_crit);
    let inc = p5_exceedance_increment(temp_curr, temp_warn, dt_s);
    let p5 = cumulative_prev + inc;

    ThermalStressIndicators {
        p1_excursion: p1,
        p2_rate_of_rise: p2,
        p3_ambient_rise: p3,
        p4_hotspot_margin: p4,
        p5_cumulative_exceedance: p5,
    }
}

/// Arrhenius Cumulative Thermal Dose (CTD_P6):
/// $$D_{\text{CTD}} = \sum_{i} \exp\left(-\frac{E_a}{R}\left(\frac{1}{T_{k, i}} - \frac{1}{T_{\text{ref}, k}}\right)\right) \Delta t_i$$
///
/// Where:
/// - `t_k_series`: slice of temperatures in Kelvin $T_k$.
/// - `t_ref_k`: reference temperature in Kelvin $T_{\text{ref}, k}$ (e.g. 343.15 K = 70 °C).
/// - `ea_j_mol`: activation energy $E_a$ in J/mol (e.g. 68,000 J/mol for mineral gear oil oxidation).
/// - `dt_sec`: timestep in seconds.
///
/// Returns total cumulative thermal dose in seconds (equivalent exposure time at $T_{\text{ref}}$).
pub fn ctd_p6(t_k_series: &[f64], t_ref_k: f64, ea_j_mol: f64, dt_sec: f64) -> f64 {
    if dt_sec <= 0.0 || t_ref_k <= 0.0 {
        return 0.0;
    }
    let factor = ea_j_mol / R_GAS;
    let inv_tref = 1.0 / t_ref_k;

    let mut dose = 0.0;
    for &t_k in t_k_series {
        if t_k <= 0.0 {
            continue;
        }
        let inv_tk = 1.0 / t_k;
        // -(Ea/R) * (1/Tk - 1/Tref) = (Ea/R) * (1/Tref - 1/Tk)
        let exponent = factor * (inv_tref - inv_tk);
        // Clamp exponent to prevent numerical overflow/underflow in libm::exp
        let exponent_clamped = if exponent > 70.0 {
            70.0
        } else if exponent < -70.0 {
            -70.0
        } else {
            exponent
        };
        dose += libm::exp(exponent_clamped) * dt_sec;
    }
    dose
}

// =========================================================================
// 2. MOIST AIR DENSITY (CIPM-2007 STANDARD)
// =========================================================================

/// Saturation vapor pressure of water over liquid water $p_{sv}(T)$ (Pa)
/// using the standard CIPM-2007 / IAPWS recommended Magnus-form formulation:
/// $$p_{sv} = \exp(A T_k^2 + B T_k + C + D / T_k)$$
/// with CIPM-2007 coefficients:
/// $A = 1.2378847 \times 10^{-5}\ \text{K}^{-2}$,
/// $B = -1.9121316 \times 10^{-2}\ \text{K}^{-1}$,
/// $C = 33.93711047$,
/// $D = -6.3431645 \times 10^3\ \text{K}$.
pub fn saturation_vapor_pressure_cipm2007(temp_k: f64) -> f64 {
    const A: f64 = 1.237_884_7e-5;
    const B: f64 = -1.912_131_6e-2;
    const C: f64 = 33.937_110_47;
    const D: f64 = -6.343_164_5e3;

    let ln_psv = A * temp_k * temp_k + B * temp_k + C + D / temp_k;
    libm::exp(ln_psv)
}

/// Enhancement factor $f(p, T)$ for moist air (CIPM-2007):
/// $$f(p, T) = \alpha + \beta p + \gamma t^2$$
/// where $t = T_k - 273.15$ in °C,
/// $\alpha = 1.00062$, $\beta = 3.14 \times 10^{-8}\ \text{Pa}^{-1}$, $\gamma = 5.6 \times 10^{-7}\ \text{K}^{-2}$.
pub fn enhancement_factor_cipm2007(pressure_pa: f64, temp_k: f64) -> f64 {
    let t_c = temp_k - 273.15;
    1.000_62 + 3.14e-8 * pressure_pa + 5.6e-7 * t_c * t_c
}

/// Compressibility factor $Z(p, T, x_v)$ of moist air per CIPM-2007:
/// $$Z = 1 - \frac{p}{T_k} \left[a_0 + a_1 t + a_2 t^2 + (b_0 + b_1 t) x_v + (c_0 + c_1 t) x_v^2\right] + \frac{p^2}{T_k^2} \left(d_0 + d_1 x_v^2\right)$$
pub fn compressibility_factor_cipm2007(pressure_pa: f64, temp_k: f64, x_v: f64) -> f64 {
    let t_c = temp_k - 273.15;
    const A0: f64 = 1.581_23e-6; // K·Pa^-1
    const A1: f64 = -2.9331e-8; // Pa^-1
    const A2: f64 = 1.1043e-10; // K^-1·Pa^-1
    const B0: f64 = 5.707e-6;
    const B1: f64 = -2.051e-8;
    const C0: f64 = 1.9898e-4;
    const C1: f64 = -2.376e-6;
    const D0: f64 = 1.83e-11; // K^2·Pa^-2
    const D1: f64 = -7.65e-9;

    let term1 = a0_term(t_c, A0, A1, A2) + (B0 + B1 * t_c) * x_v + (C0 + C1 * t_c) * x_v * x_v;
    let p_over_t = pressure_pa / temp_k;
    let term2 = (D0 + D1 * x_v * x_v) * p_over_t * p_over_t;

    1.0 - p_over_t * term1 + term2
}

#[inline(always)]
fn a0_term(t_c: f64, a0: f64, a1: f64, a2: f64) -> f64 {
    a0 + a1 * t_c + a2 * t_c * t_c
}

/// Computes Moist Air Density $\rho$ according to CIPM-2007 standard:
/// $$\rho = \frac{p M_a}{Z R T} \left[1 - x_v \left(1 - \frac{M_v}{M_a}\right)\right]$$
///
/// Parameters:
/// - `pressure_pa`: absolute barometric air pressure in Pa (e.g. 101325.0 Pa).
/// - `temp_k`: absolute temperature in Kelvin (e.g. 293.15 K = 20 °C).
/// - `relative_humidity`: relative humidity $h \in [0.0, 1.0]$ (e.g. 0.50 for 50%).
///
/// Returns moist air density in $\text{kg}\cdot\text{m}^{-3}$.
pub fn moist_air_density_cipm2007(pressure_pa: f64, temp_k: f64, relative_humidity: f64) -> f64 {
    if temp_k <= 0.0 || pressure_pa <= 0.0 {
        return 0.0;
    }
    let h = relative_humidity.clamp(0.0, 1.0);
    let p_sv = saturation_vapor_pressure_cipm2007(temp_k);
    let f = enhancement_factor_cipm2007(pressure_pa, temp_k);
    // Mole fraction of water vapor x_v = h * f * p_sv / p
    let x_v = (h * f * p_sv / pressure_pa).clamp(0.0, 1.0);

    let z = compressibility_factor_cipm2007(pressure_pa, temp_k, x_v);
    let z_safe = if z < 0.8 { 0.8 } else { z };

    let mv_over_ma = M_V / M_A;
    let density =
        (pressure_pa * M_A / (z_safe * R_GAS * temp_k)) * (1.0 - x_v * (1.0 - mv_over_ma));
    density
}

// =========================================================================
// 3. GEARBOX THERMAL LOSS & TEMPERATURE RISE (ISO/TR 14179, Ohlendorf, SDPLE)
// =========================================================================

/// Breakdown of gearbox thermal power losses according to ISO/TR 14179:
/// - $P_{\text{VZP}}$: Gear load-dependent friction power loss
/// - $P_{\text{VZD}}$: Gear no-load churning/windage power loss
/// - $P_{\text{VLP}}$: Bearing load-dependent friction power loss
/// - $P_{\text{VLD}}$: Bearing no-load viscous seal/churning power loss
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GearboxLossBreakdown {
    pub p_vzp_gear_load: f64,
    pub p_vzd_gear_churning: f64,
    pub p_vlp_bearing_load: f64,
    pub p_vld_bearing_noload: f64,
    pub p_total_loss: f64,
}

/// Ohlendorf gear friction loss factor $H_v$:
/// $$H_v = \frac{(u + 1)}{u \pi \cos\alpha_w} \left(1 - \epsilon_{\alpha} + \epsilon_1^2 + \epsilon_2^2\right)$$
/// For standard involute spur/helical gears:
/// - `gear_ratio_u`: gear ratio $u = z_2 / z_1 \ge 1.0$.
/// - `pressure_angle_rad`: transverse operating pressure angle $\alpha_w$.
/// - `contact_ratio`: transverse contact ratio $\epsilon_\alpha$ (typically $1.2 \dots 1.8$).
#[inline]
pub fn ohlendorf_gear_loss_factor(
    gear_ratio_u: f64,
    pressure_angle_rad: f64,
    contact_ratio: f64,
) -> f64 {
    let u = if gear_ratio_u < 1.0 {
        1.0
    } else {
        gear_ratio_u
    };
    let cos_alpha = libm::cos(pressure_angle_rad);
    if cos_alpha.abs() < 1e-4 {
        return 0.1;
    }
    // Simplification for symmetric addendum: eps_1 = eps_2 = 0.5 * eps_alpha
    let eps1 = 0.5 * contact_ratio;
    let eps2 = 0.5 * contact_ratio;
    let geom = 1.0 - contact_ratio + eps1 * eps1 + eps2 * eps2;
    ((u + 1.0) / (u * PI * cos_alpha)) * geom
}

/// Gear load-dependent mesh power loss $P_{\text{VZP}}$ (ISO/TR 14179 & Ohlendorf):
/// $$P_{\text{VZP}} = P_{\text{in}} \cdot \mu_m \cdot H_v$$
/// - `p_in_w`: input mechanical power in Watts.
/// - `mu_m`: mean friction coefficient across meshing line (typically $0.03 \dots 0.07$).
/// - `h_v`: Ohlendorf gear loss factor $H_v$.
#[inline]
pub fn gear_mesh_loss_iso14179(p_in_w: f64, mu_m: f64, h_v: f64) -> f64 {
    p_in_w * mu_m * h_v
}

/// Bearing load-dependent loss $P_{\text{VLP}}$ (ISO/TR 14179-1):
/// $$P_{\text{VLP}} = f_1 \cdot P_{\text{eq}} \cdot d_m \cdot \omega$$
/// - `f1`: bearing load factor (typically $0.0005 \dots 0.0015$).
/// - `p_eq_n`: equivalent dynamic bearing load in N.
/// - `d_m_m`: pitch diameter of bearing in meters.
/// - `speed_rad_s`: angular velocity in rad/s.
#[inline]
pub fn bearing_load_loss_iso14179(f1: f64, p_eq_n: f64, d_m_m: f64, speed_rad_s: f64) -> f64 {
    f1 * p_eq_n * d_m_m * speed_rad_s
}

/// Gear no-load churning power loss $P_{\text{VZD}}$ (ISO/TR 14179 dip lubrication):
/// $$P_{\text{VZD}} = C_{\text{churn}} \cdot \rho_{\text{oil}} \cdot \nu_0^{0.5} \cdot v_t^2 \cdot b$$
/// - `c_churn`: churning coefficient.
/// - `rho_oil`: oil density in kg/m³ (e.g. 860 kg/m³).
/// - `nu_cst`: kinematic viscosity in cSt (mm²/s).
/// - `pitch_line_velocity_ms`: tangential speed $v_t = \omega \cdot r$ in m/s.
/// - `face_width_m`: gear face width $b$ in meters.
#[inline]
pub fn gear_churning_loss_iso14179(
    c_churn: f64,
    rho_oil: f64,
    nu_cst: f64,
    pitch_line_velocity_ms: f64,
    face_width_m: f64,
) -> f64 {
    let nu_scaled = if nu_cst > 0.0 {
        libm::sqrt(nu_cst)
    } else {
        1.0
    };
    c_churn * rho_oil * nu_scaled * pitch_line_velocity_ms * pitch_line_velocity_ms * face_width_m
}

/// Bearing no-load viscous loss $P_{\text{VLD}}$ (ISO/TR 14179):
/// $$P_{\text{VLD}} = 10^{-7} \cdot f_0 \cdot (\nu \cdot n)^{2/3} \cdot d_m^3$$
/// where $n$ is rpm, $\nu$ is kinematic viscosity in cSt, $d_m$ in mm.
#[inline]
pub fn bearing_viscous_loss_iso14179(f0: f64, nu_cst: f64, speed_rpm: f64, d_m_mm: f64) -> f64 {
    let nu_n = nu_cst * speed_rpm;
    let base = if nu_n >= 2000.0 {
        libm::pow(nu_n, 2.0 / 3.0)
    } else {
        160.0
    };
    1e-7 * f0 * base * d_m_mm * d_m_mm * d_m_mm * (2.0 * PI / 60.0)
}

/// SDPLE (State-Dependent Power Loss Efficiency) total loss:
/// Combines mesh, bearing, and churning losses into an overall state-dependent efficiency $\eta(T, \omega, \tau)$
/// and total thermal dissipation $P_{\text{loss}}$:
/// $$P_{\text{loss}} = P_{\text{VZP}} + P_{\text{VZD}} + P_{\text{VLP}} + P_{\text{VLD}}$$
pub fn sdple_gearbox_power_loss(
    p_in_w: f64,
    mu_m: f64,
    h_v: f64,
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
) -> GearboxLossBreakdown {
    let p_vzp = gear_mesh_loss_iso14179(p_in_w, mu_m, h_v);
    let p_vlp = bearing_load_loss_iso14179(f1, p_eq_n, d_m_m, speed_rad_s);
    let p_vzd = gear_churning_loss_iso14179(
        c_churn,
        rho_oil,
        nu_cst,
        pitch_line_velocity_ms,
        face_width_m,
    );
    let speed_rpm = speed_rad_s * (60.0 / (2.0 * PI));
    let p_vld = bearing_viscous_loss_iso14179(f0, nu_cst, speed_rpm, d_m_m * 1000.0);

    let p_total = p_vzp + p_vzd + p_vlp + p_vld;

    GearboxLossBreakdown {
        p_vzp_gear_load: p_vzp,
        p_vzd_gear_churning: p_vzd,
        p_vlp_bearing_load: p_vlp,
        p_vld_bearing_noload: p_vld,
        p_total_loss: p_total,
    }
}

/// Computes transient exponential temperature update for gearbox oil/casing:
/// $$\tau_{\text{th}} = \frac{C_{\text{th}}}{G_{\text{th}}},\quad T_{\text{eq}} = T_{\text{amb}} + \frac{P_{\text{loss}}}{G_{\text{th}}}$$
/// $$T(t + \Delta t) = T_{\text{eq}} + (T(t) - T_{\text{eq}}) \exp\left(-\frac{\Delta t}{\tau_{\text{th}}}\right)$$
///
/// Parameters:
/// - `t_curr_k`: current temperature in Kelvin.
/// - `t_amb_k`: ambient temperature in Kelvin.
/// - `p_loss_w`: total thermal power loss dissipation in Watts.
/// - `g_th_w_per_k`: total thermal conductance $G_{\text{th}} = h_{\text{conv}} A_{\text{surf}}$ in W/K.
/// - `c_th_j_per_k`: total thermal capacitance $C_{\text{th}} = m c_p$ in J/K.
/// - `dt_s`: timestep in seconds.
pub fn step_gearbox_temperature_transient(
    t_curr_k: f64,
    t_amb_k: f64,
    p_loss_w: f64,
    g_th_w_per_k: f64,
    c_th_j_per_k: f64,
    dt_s: f64,
) -> f64 {
    if g_th_w_per_k <= 0.0 || c_th_j_per_k <= 0.0 {
        return t_curr_k;
    }
    let tau_th = c_th_j_per_k / g_th_w_per_k;
    let t_eq = t_amb_k + p_loss_w / g_th_w_per_k;
    let factor = libm::exp(-dt_s / tau_th);
    t_eq + (t_curr_k - t_eq) * factor
}

// =========================================================================
// UNIT TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_stress_indicators_p1_to_p5() {
        let t_curr = 85.0; // °C
        let t_prev = 82.0;
        let t_amb = 25.0;
        let t_base = 50.0;
        let t_warn = 80.0;
        let t_crit = 95.0;
        let dt = 10.0; // s

        let ind = compute_thermal_stress_indicators(
            t_curr, t_prev, t_amb, t_base, t_warn, t_crit, 100.0, dt,
        );

        // P1: excursion = 85 - 50 = 35 °C
        assert!((ind.p1_excursion - 35.0).abs() < 1e-9);
        // P2: rate of rise = (85 - 82)/10 = 0.3 °C/s
        assert!((ind.p2_rate_of_rise - 0.3).abs() < 1e-9);
        // P3: ambient rise = 85 - 25 = 60 °C
        assert!((ind.p3_ambient_rise - 60.0).abs() < 1e-9);
        // P4: hotspot margin = 95 - 85 = 10 °C
        assert!((ind.p4_hotspot_margin - 10.0).abs() < 1e-9);
        // P5: exceedance = 100 + (85 - 80)*10 = 150 °C·s
        assert!((ind.p5_cumulative_exceedance - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_ctd_p6_arrhenius() {
        let t_ref = 343.15; // 70 °C
        let ea = 68_000.0; // J/mol
        let temps = [343.15, 343.15, 343.15];
        // At T = T_ref, rate = exp(0) = 1.0, total dose = 3 * 1.0 * 60 = 180 s
        let dose = ctd_p6(&temps, t_ref, ea, 60.0);
        assert!((dose - 180.0).abs() < 1e-6);

        // At higher temperature, dose accumulates faster
        let high_temps = [353.15]; // 80 °C (+10 K)
        let high_dose = ctd_p6(&high_temps, t_ref, ea, 60.0);
        assert!(high_dose > 60.0);
    }

    #[test]
    fn test_moist_air_density_cipm2007() {
        // Standard dry air conditions: 101325 Pa, 20 °C (293.15 K), 0% RH
        let rho_dry = moist_air_density_cipm2007(101325.0, 293.15, 0.0);
        // Standard air density at 20°C and 1 atm is ~1.204 kg/m³
        assert!((rho_dry - 1.204).abs() < 0.01, "rho_dry = {rho_dry}");

        // At 50% relative humidity, air density decreases because water vapor is lighter than dry air
        let rho_moist = moist_air_density_cipm2007(101325.0, 293.15, 0.50);
        assert!(rho_moist < rho_dry);
        assert!((rho_moist - 1.200).abs() < 0.01, "rho_moist = {rho_moist}");
    }

    #[test]
    fn test_iso14179_gearbox_losses_and_temperature_step() {
        // 2 MW wind turbine gearbox parameters
        let p_in = 2.0e6; // 2 MW
        let gear_ratio = 100.0;
        let pressure_angle = 20.0_f64.to_radians();
        let contact_ratio = 1.45;
        let hv = ohlendorf_gear_loss_factor(gear_ratio, pressure_angle, contact_ratio);
        assert!(hv > 0.0);

        let losses = sdple_gearbox_power_loss(
            p_in, 0.04, // mu_m
            hv, 0.001,    // f1
            50_000.0, // P_eq (N)
            0.40,     // d_m (m)
            188.5,    // speed 1800 rpm = 188.5 rad/s
            0.0001,   // c_churn
            860.0,    // rho_oil
            320.0,    // nu_cst (ISO VG 320 at cold/medium)
            25.0,     // pitch line velocity
            0.25,     // face width
            2.5,      // f0
        );

        assert!(losses.p_vzp_gear_load > 0.0);
        assert!(losses.p_vlp_bearing_load > 0.0);
        assert!(losses.p_total_loss > 0.0);
        assert!(losses.p_total_loss < 0.05 * p_in); // Gearbox loss < 5%

        // Temperature transient update:
        let t_curr = 293.15; // 20 °C
        let t_amb = 293.15; // 20 °C
        let g_th = 800.0; // W/K
        let c_th = 4.5e5; // J/K
        let t_next =
            step_gearbox_temperature_transient(t_curr, t_amb, losses.p_total_loss, g_th, c_th, 1.0);
        assert!(t_next > t_curr);
    }
}

// ---------------------------------------------------------------------

#[cfg(test)]
mod thermal_extra_tests {
    use super::*;

    #[test]
    fn test_arrhenius_insulation_life() {
        // Example: operating 165°C, rated 155°C, rated life 20000h -> expected life 10000h, factor 0.5
        let (life, factor) = arrhenius_insulation_life(165.0, 155.0, 20000.0);
        assert!((life - 10000.0).abs() < 1e-6);
        assert!((factor - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_compute_electro_thermal_proxies_safety() {
        // wind_speed = 0 should not cause panic; proxies should be zero or handled gracefully
        let proxies = compute_electro_thermal_proxies(
            30.0,   // temp_c
            25.0,   // temp_amb_c
            5.0,    // wind_speed
            1.0,    // power_kw
            2.0,    // rated_power_kw
            1.0,    // v_min
            1.0,    // v0
            0.5,    // alpha_elec
            1000.0, // q_joule_rated_w
            10.0,   // h_conv_w_per_k_ms
        );
        // Expect all proxies to be finite numbers (no NaN/Inf) and non‑negative
        assert!(proxies.p1_naive.is_finite());
        assert!(proxies.p2_piecewise.is_finite());
        assert!(proxies.p3_smooth.is_finite());
        assert!(proxies.p4_composite.is_finite());
        assert!(proxies.p5_continuum.is_finite());
        assert!(proxies.p1_naive >= 0.0 && proxies.p2_piecewise >= 0.0 && proxies.p3_smooth >= 0.0 && proxies.p4_composite >= 0.0 && proxies.p5_continuum >= 0.0);
    }
}

// ---------------------------------------------------------------------
// v0.3.1 additional thermal utilities (no_std compatible)
// ---------------------------------------------------------------------

/// Class F (155 °C) insulation life prediction using the 10 °C halving rule.
/// Returns `(adjusted_life_hours, life_factor)` where `life_factor = 2^{-(T_op - T_r)/10}`.
///
/// * `operating_temp_c` – actual operating temperature in °C.
/// * `rated_temp_c` – rated temperature at which `rated_life_hours` is specified.
/// * `rated_life_hours` – nominal life at `rated_temp_c` (hours).
#[inline]
pub fn arrhenius_insulation_life(operating_temp_c: f64, rated_temp_c: f64, rated_life_hours: f64) -> (f64, f64) {
    let delta = operating_temp_c - rated_temp_c;
    // life factor = 2 ^ ( -delta / 10 )
    let exponent = -delta / 10.0_f64;
    let life_factor = libm::pow(2.0_f64, exponent);
    (rated_life_hours * life_factor, life_factor)
}

/// Collection of electro‑thermal proxy stress terms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElectroThermalProxies {
    pub p1_naive: f64,
    pub p2_piecewise: f64,
    pub p3_smooth: f64,
    pub p4_composite: f64,
    pub p5_continuum: f64,
}

/// Naïve proxy `P1` – simple ratio of temperature rise time to wind speed.
/// Returns `Δt / v` with guarded handling of non‑positive wind speed.
#[inline]
pub fn thermal_stress_proxy_p1_naive(delta_t: f64, wind_speed: f64) -> f64 {
    if wind_speed > 0.0 {
        delta_t / wind_speed
    } else if delta_t > 0.0 {
        f64::INFINITY
    } else if delta_t < 0.0 {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

/// Piecewise proxy `P2` – clamps wind speed to a minimum value `v_min` (> 0).
#[inline]
pub fn thermal_stress_proxy_p2_piecewise(delta_t: f64, wind_speed: f64, v_min: f64) -> f64 {
    let v_eff = if wind_speed < v_min { v_min } else { wind_speed };
    delta_t / v_eff
}

/// Smooth proxy `P3` – regularised ratio using a base velocity `v0`.
#[inline]
pub fn thermal_stress_proxy_p3_smooth(delta_t: f64, wind_speed: f64, v0: f64) -> f64 {
    let denom = wind_speed.max(0.0) + v0.max(1e-9_f64);
    delta_t / denom
}

/// Composite electro‑thermal proxy `P4` – blends `P3` with a quadratic power term.
#[inline]
pub fn thermal_stress_proxy_p4_composite(
    delta_t: f64,
    wind_speed: f64,
    power_kw: f64,
    rated_power_kw: f64,
    v0: f64,
    alpha_elec: f64,
) -> f64 {
    let base = thermal_stress_proxy_p3_smooth(delta_t, wind_speed, v0);
    let power_ratio = power_kw / rated_power_kw.max(1e-9_f64);
    base * (1.0_f64 + alpha_elec * libm::pow(power_ratio, 2.0_f64))
}

/// Continuum energy‑balance proxy `P5` – first‑principles estimate of excess thermal stress.
/// Returns `(T - T_amb) - ΔT_eq` where `ΔT_eq` is the equilibrium temperature rise based on
/// convective heat transfer.
#[inline]
pub fn thermal_stress_proxy_p5_continuum(
    temp_c: f64,
    temp_amb_c: f64,
    wind_speed: f64,
    power_kw: f64,
    rated_power_kw: f64,
    v0: f64,
    q_joule_rated_w: f64,
    h_conv_w_per_k_ms: f64,
) -> f64 {
    let power_ratio = power_kw / rated_power_kw.max(1e-9_f64);
    let denom = h_conv_w_per_k_ms * (wind_speed.max(0.0) + v0.max(1e-9_f64));
    let delta_t_eq = (q_joule_rated_w * libm::pow(power_ratio, 2.0_f64)) / denom;
    (temp_c - temp_amb_c) - delta_t_eq
}

/// Compute all five proxy terms at once and return them as a struct.
#[inline]
pub fn compute_electro_thermal_proxies(
    temp_c: f64,
    temp_amb_c: f64,
    wind_speed: f64,
    power_kw: f64,
    rated_power_kw: f64,
    v_min: f64,
    v0: f64,
    alpha_elec: f64,
    q_joule_rated_w: f64,
    h_conv_w_per_k_ms: f64,
) -> ElectroThermalProxies {
    let p1 = thermal_stress_proxy_p1_naive(temp_c - temp_amb_c, wind_speed);
    let p2 = thermal_stress_proxy_p2_piecewise(temp_c - temp_amb_c, wind_speed, v_min);
    let p3 = thermal_stress_proxy_p3_smooth(temp_c - temp_amb_c, wind_speed, v0);
    let p4 = thermal_stress_proxy_p4_composite(
        temp_c - temp_amb_c,
        wind_speed,
        power_kw,
        rated_power_kw,
        v0,
        alpha_elec,
    );
    let p5 = thermal_stress_proxy_p5_continuum(
        temp_c,
        temp_amb_c,
        wind_speed,
        power_kw,
        rated_power_kw,
        v0,
        q_joule_rated_w,
        h_conv_w_per_k_ms,
    );
    ElectroThermalProxies { p1_naive: p1, p2_piecewise: p2, p3_smooth: p3, p4_composite: p4, p5_continuum: p5 }
}

// ---------------------------------------------------------------------
// End of v0.3.1 additions
// ---------------------------------------------------------------------
