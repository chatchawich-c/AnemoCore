#![allow(
    clippy::too_many_arguments,
    clippy::manual_clamp,
    clippy::let_and_return
)]
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tribology, Contact Mechanics, EHL Lubrication, and ISO 281 / ISO/TS 16281 Bearing Life for AnemoCore.
//!
//! Grant No. ERASMUS-EDU-2025-CBHE-101177220 (EU Erasmus+ CBHE ECO-WIND)
//!
//! Provides:
//! - 2-Stage Dowson–Higginson EHL film thickness & wear regime indicator (`ehl_two_stage`):
//!   Dimensionless speed $U$, materials $G$, load $W$, minimum film thickness:
//!   $$h_{\text{min}} = 2.65 \cdot R_x \cdot U^{0.70} \cdot G^{0.54} \cdot W^{-0.13}$$
//!   and film thickness ratio $\Lambda = h_{\text{min}} / \sigma_{\text{composite}}$ across Stage-1 (planetary) and Stage-2 (parallel helical) gears.
//! - Dynamic Rating Life per `ISO 281` ($L_{10}$ and modified rating life $L_{10m}$):
//!   $$L_{10} = \left(\frac{C}{P}\right)^p,\quad L_{10m} = a_1 \cdot a_{\text{ISO}} \cdot L_{10}$$
//!   where $p = 3$ for ball bearings, $p = 10/3$ for roller bearings, with ISO lubrication/cleanliness factor $a_{\text{ISO}}(\kappa, e_c \cdot C_u / P)$.
//! - Bearing Tilt / Angular Misalignment Factor per `ISO/TS 16281`:
//!   Slice-based contact stress model derating factor under misalignment angle $\theta_{\text{tilt}}$ and equivalent load.
//! - Fractional Caputo Peridynamics Damage Indicator:
//!   Discrete L1 Grünwald–Letnikov / Caputo fractional derivative of order $\alpha \in (0, 1)$ over nonlocal bond stretch history $s_{ij}(t)$
//!   to evaluate sub-surface micro-crack and fatigue damage accumulation.

extern crate libm;

use std::f64::consts::PI;

// =========================================================================
// 1. 2-STAGE DOWSON-HIGGINSON EHL FILM THICKNESS & WEAR INDICATOR
// =========================================================================

/// Dimensionless Elastohydrodynamic Lubrication (EHL) parameters and film thickness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EhlFilmState {
    /// Dimensionless speed parameter $U = \frac{\eta_0 u}{E' R_x}$
    pub u_speed: f64,
    /// Dimensionless materials parameter $G = \alpha E'$
    pub g_materials: f64,
    /// Dimensionless load parameter $W = \frac{w'}{E' R_x}$
    pub w_load: f64,
    /// Minimum EHL film thickness $h_{\text{min}}$ in meters
    pub h_min_m: f64,
    /// Central EHL film thickness $h_{\text{cen}}$ in meters
    pub h_cen_m: f64,
    /// Lubrication regime ratio $\Lambda = h_{\text{min}} / \sigma_{\text{composite}}$
    pub lambda_ratio: f64,
}

/// 2-Stage Gearbox EHL Analysis across Stage 1 (planetary) and Stage 2 (helical parallel).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoStageEhlResult {
    /// Stage 1 (Low-speed planetary mesh) EHL state
    pub stage1_planetary: EhlFilmState,
    /// Stage 2 (Intermediate/high-speed helical mesh) EHL state
    pub stage2_helical: EhlFilmState,
    /// Combined minimum lubrication safety index
    pub min_lambda: f64,
}

/// Computes Dowson–Higginson minimum and central film thickness:
/// $$h_{\text{min}} = 2.65 \cdot R_x \cdot U^{0.70} \cdot G^{0.54} \cdot W^{-0.13}$$
/// $$h_{\text{cen}} = 2.69 \cdot R_x \cdot U^{0.67} \cdot G^{0.53} \cdot W^{-0.067}$$
///
/// Parameters:
/// - `eta0_pa_s`: Dynamic viscosity at atmospheric pressure and operating temperature (Pa·s)
/// - `u_entrainment_ms`: Entrainment velocity $u = \frac{u_1 + u_2}{2}$ (m/s)
/// - `r_equiv_x_m`: Equivalent radius of curvature in rolling direction $R_x = \frac{R_1 R_2}{R_1 + R_2}$ (m)
/// - `w_load_per_len_n_m`: Normal load per unit face width $w' = F_N / b$ (N/m)
/// - `e_prime_pa`: Reduced elastic modulus $E' = \frac{2}{\frac{1 - \nu_1^2}{E_1} + \frac{1 - \nu_2^2}{E_2}}$ (Pa)
/// - `alpha_piezo_pa_inv`: Pressure-viscosity coefficient $\alpha$ ($\text{Pa}^{-1}$, e.g. $1.5 \dots 2.2 \times 10^{-8}$)
/// - `sigma_composite_m`: Composite RMS surface roughness $\sigma = \sqrt{\sigma_1^2 + \sigma_2^2}$ (m)
pub fn dowson_higginson_film(
    eta0_pa_s: f64,
    u_entrainment_ms: f64,
    r_equiv_x_m: f64,
    w_load_per_len_n_m: f64,
    e_prime_pa: f64,
    alpha_piezo_pa_inv: f64,
    sigma_composite_m: f64,
) -> EhlFilmState {
    let r_x = if r_equiv_x_m <= 1e-6 {
        1e-6
    } else {
        r_equiv_x_m
    };
    let e_p = if e_prime_pa <= 1e6 { 1e6 } else { e_prime_pa };
    let u_ent = if u_entrainment_ms < 1e-4 {
        1e-4
    } else {
        u_entrainment_ms
    };
    let w_load = if w_load_per_len_n_m < 1.0 {
        1.0
    } else {
        w_load_per_len_n_m
    };

    let u_dim = (eta0_pa_s * u_ent) / (e_p * r_x);
    let g_dim = alpha_piezo_pa_inv * e_p;
    let w_dim = w_load / (e_p * r_x);

    // h_min = 2.65 * R_x * U^0.70 * G^0.54 * W^-0.13
    let h_min =
        2.65 * r_x * libm::pow(u_dim, 0.70) * libm::pow(g_dim, 0.54) * libm::pow(w_dim, -0.13);

    // h_cen = 2.69 * R_x * U^0.67 * G^0.53 * W^-0.067
    let h_cen =
        2.69 * r_x * libm::pow(u_dim, 0.67) * libm::pow(g_dim, 0.53) * libm::pow(w_dim, -0.067);

    let sigma = if sigma_composite_m <= 1e-9 {
        1e-9
    } else {
        sigma_composite_m
    };
    let lambda = h_min / sigma;

    EhlFilmState {
        u_speed: u_dim,
        g_materials: g_dim,
        w_load: w_dim,
        h_min_m: h_min,
        h_cen_m: h_cen,
        lambda_ratio: lambda,
    }
}

/// Evaluates 2-Stage Dowson–Higginson EHL across:
/// 1. Low-Speed Planetary Stage (high torque, low speed, heavy boundary/mixed wear risk)
/// 2. Intermediate/High-Speed Helical Stage (moderate torque, high entrainment, thermal thinning risk)
pub fn ehl_two_stage(
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
) -> TwoStageEhlResult {
    let s1 = dowson_higginson_film(
        stage1_eta0,
        stage1_u_ent,
        stage1_rx,
        stage1_w,
        e_prime_pa,
        alpha_piezo_pa_inv,
        sigma_composite_m,
    );
    let s2 = dowson_higginson_film(
        stage2_eta0,
        stage2_u_ent,
        stage2_rx,
        stage2_w,
        e_prime_pa,
        alpha_piezo_pa_inv,
        sigma_composite_m,
    );

    let min_lam = if s1.lambda_ratio < s2.lambda_ratio {
        s1.lambda_ratio
    } else {
        s2.lambda_ratio
    };

    TwoStageEhlResult {
        stage1_planetary: s1,
        stage2_helical: s2,
        min_lambda: min_lam,
    }
}

// =========================================================================
// 2. DYNAMIC RATING LIFE PER ISO 281 (L10 & L10m)
// =========================================================================

/// Bearing type for life rating exponent $p$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BearingType {
    /// Point contact ball bearing ($p = 3$)
    Ball,
    /// Line contact roller bearing ($p = 10/3 \approx 3.333$)
    Roller,
}

impl BearingType {
    #[inline]
    pub fn exponent(self) -> f64 {
        match self {
            BearingType::Ball => 3.0,
            BearingType::Roller => 10.0 / 3.0,
        }
    }
}

/// Computes basic dynamic rating life $L_{10}$ in millions of revolutions:
/// $$L_{10} = \left(\frac{C}{P}\right)^p$$
/// - `c_dyn_rating_n`: Basic dynamic load rating $C$ (N)
/// - `p_equiv_load_n`: Equivalent dynamic bearing load $P$ (N)
/// - `btype`: Ball or Roller bearing
#[inline]
pub fn iso281_basic_rating_life_l10(
    c_dyn_rating_n: f64,
    p_equiv_load_n: f64,
    btype: BearingType,
) -> f64 {
    if p_equiv_load_n <= 0.0 || c_dyn_rating_n <= 0.0 {
        return 0.0;
    }
    let ratio = c_dyn_rating_n / p_equiv_load_n;
    libm::pow(ratio, btype.exponent())
}

/// Reliability factor $a_1$ per ISO 281 (Table 1):
/// - 90% reliability: $a_1 = 1.00$
/// - 95% reliability: $a_1 = 0.64$
/// - 96% reliability: $a_1 = 0.55$
/// - 97% reliability: $a_1 = 0.47$
/// - 98% reliability: $a_1 = 0.37$
/// - 99% reliability: $a_1 = 0.25$
pub fn iso281_reliability_factor_a1(reliability_percent: f64) -> f64 {
    if reliability_percent >= 99.0 {
        0.25
    } else if reliability_percent >= 98.0 {
        0.37
    } else if reliability_percent >= 97.0 {
        0.47
    } else if reliability_percent >= 96.0 {
        0.55
    } else if reliability_percent >= 95.0 {
        0.64
    } else {
        1.00
    }
}

/// Computes ISO 281 lubrication and contamination modification factor $a_{\text{ISO}}$:
/// $$a_{\text{ISO}} = f\left(\frac{e_c C_u}{P}, \kappa\right)$$
/// Parametric fit approximating ISO 281 curves:
/// - `kappa`: viscosity ratio $\kappa = \nu / \nu_1$ (typically $0.1 \dots 4.0$).
/// - `e_c`: contamination factor ($0.1$ severe contamination, $0.5$ normal cleanliness, $1.0$ extreme cleanliness).
/// - `c_u_n`: fatigue load limit $C_u$ in N.
/// - `p_load_n`: equivalent dynamic load $P$ in N.
pub fn iso281_a_iso_factor(kappa: f64, e_c: f64, c_u_n: f64, p_load_n: f64) -> f64 {
    if p_load_n <= 0.0 || c_u_n <= 0.0 {
        return 0.1;
    }
    let k_clamped = kappa.clamp(0.1, 4.0);
    let ec_clamped = e_c.clamp(0.05, 1.0);
    let load_param = (ec_clamped * c_u_n) / p_load_n;

    // ISO 281 empirical equation:
    // a_ISO = 0.1 + [ 1 - (1.5859 - 1.3993/(k^0.05438))^2.5 * load_param^-0.4 ]^-1
    // Simplified robust continuous closed-form:
    let k_factor = libm::pow(k_clamped, 0.45);
    let a_iso = 0.1 + 2.5 * (k_factor * load_param) / (1.0 + 0.8 * load_param);
    a_iso.clamp(0.1, 50.0)
}

/// Modified dynamic rating life $L_{10m}$ per ISO 281:
/// $$L_{10m} = a_1 \cdot a_{\text{ISO}} \cdot L_{10}$$
/// In millions of revolutions and converted to operating hours.
pub fn iso281_modified_rating_life_l10m(
    c_dyn_n: f64,
    p_load_n: f64,
    c_u_n: f64,
    kappa: f64,
    e_c: f64,
    reliability_pct: f64,
    speed_rpm: f64,
    btype: BearingType,
) -> (f64, f64) {
    let l10 = iso281_basic_rating_life_l10(c_dyn_n, p_load_n, btype);
    let a1 = iso281_reliability_factor_a1(reliability_pct);
    let a_iso = iso281_a_iso_factor(kappa, e_c, c_u_n, p_load_n);
    let l10m_rev = a1 * a_iso * l10;

    let l10m_hours = if speed_rpm > 0.0 {
        (l10m_rev * 1e6) / (60.0 * speed_rpm)
    } else {
        0.0
    };

    (l10m_rev, l10m_hours)
}

// =========================================================================
// 3. BEARING TILT & MISALIGNMENT FACTOR PER ISO/TS 16281
// =========================================================================

/// Slice-based bearing contact stress derating factor under angular tilt/misalignment $\theta_{\text{tilt}}$:
/// per ISO/TS 16281:
/// For a roller of effective length $L_{\text{we}}$, tilting introduces edge stress concentration:
/// $$\sigma(x) = \sigma_0 \cdot \left[1 + \beta_{\text{edge}} \left(\frac{E^* L_{\text{we}} \theta_{\text{tilt}}}{2 \delta_0}\right) \left(\frac{2x}{L_{\text{we}}}\right)\right]$$
/// Resulting in life reduction factor $f_{\text{tilt}} = \left(\frac{1}{L_{\text{we}}} \int \left(\frac{\sigma(x)}{\sigma_0}\right)^{p} dx\right)^{-1}$.
///
/// Parameters:
/// - `theta_tilt_rad`: relative tilt/misalignment angle across bearing rings in radians.
/// - `roller_length_m`: effective roller length $L_{\text{we}}$ in meters.
/// - `radial_clearance_m`: internal radial clearance $s_r$ in meters.
/// - `mean_contact_deflection_m`: elastic contact deformation $\delta_0$ in meters.
///
/// Returns life reduction multiplier $f_{\text{tilt}} \in (0.0, 1.0]$.
pub fn iso16281_tilt_derating_factor(
    theta_tilt_rad: f64,
    roller_length_m: f64,
    radial_clearance_m: f64,
    mean_contact_deflection_m: f64,
) -> f64 {
    let theta = theta_tilt_rad.abs();
    if theta < 1e-6 {
        return 1.0;
    }
    let l_eff = if roller_length_m <= 1e-4 {
        1e-4
    } else {
        roller_length_m
    };
    let delta0 = if mean_contact_deflection_m <= 1e-6 {
        1e-6
    } else {
        mean_contact_deflection_m
    };

    // Geometric tilt displacement at roller ends
    let edge_deflection = 0.5 * l_eff * theta;
    // Net effective deflection taking into account clearance
    let effective_edge = if edge_deflection > radial_clearance_m {
        edge_deflection - radial_clearance_m
    } else {
        0.0
    };

    // Stress concentration factor parameter xi = effective_edge / delta_0
    let xi = effective_edge / delta0;

    // Numerical integration across 10 slices:
    let n_slices = 10;
    let dx = 1.0 / (n_slices as f64);
    let mut sum_p = 0.0;
    // Roller life exponent p = 10/3 (ISO 281)
    let p = 10.0 / 3.0;

    for i in 0..n_slices {
        // x_norm in [-1.0, 1.0]
        let x_norm = -1.0 + (i as f64 + 0.5) * 2.0 * dx;
        let stress_ratio = 1.0 + xi * x_norm;
        let s_clamped = if stress_ratio > 0.0 {
            stress_ratio
        } else {
            0.0
        };
        sum_p += libm::pow(s_clamped, p) * dx;
    }

    if sum_p <= 0.0 {
        1.0
    } else {
        let f_tilt = libm::pow(sum_p, -1.0);
        f_tilt.clamp(0.01, 1.0)
    }
}

// =========================================================================
// 4. FRACTIONAL CAPUTO PERIDYNAMICS DAMAGE INDICATOR
// =========================================================================

/// Computes the discrete L1 Grünwald–Letnikov / Caputo fractional derivative of order $\alpha \in (0, 1)$
/// for nonlocal bond stretch history $s_{ij}(t_n)$:
/// $$^C D_{0, t}^\alpha s(t_N) = \frac{1}{\Gamma(2 - \alpha)} \sum_{k=0}^{N-1} \frac{s(t_{k+1}) - s(t_k)}{\Delta t} \left[(N - k)^{1 - \alpha} - (N - k - 1)^{1 - \alpha}\right] \Delta t^{1 - \alpha}$$
///
/// Parameters:
/// - `stretch_history`: time history of bond stretch / strain $s(t_k)$ (dimensionless, e.g. $\Delta L / L$).
/// - `dt_sec`: sampling interval $\Delta t$ in seconds.
/// - `alpha`: fractional order $\alpha \in (0.01, 0.99)$ governing hereditary memory decay.
/// - `critical_stretch_s0`: critical bond stretch threshold $s_0$ (damage begins when $s > s_0$).
///
/// Returns accumulated fractional damage index $D_{\alpha} \ge 0.0$.
pub fn caputo_peridynamics_damage(
    stretch_history: &[f64],
    dt_sec: f64,
    alpha: f64,
    critical_stretch_s0: f64,
) -> f64 {
    let n = stretch_history.len();
    if n < 2 || dt_sec <= 0.0 {
        return 0.0;
    }
    let a = alpha.clamp(0.01, 0.99);
    let s0 = if critical_stretch_s0 <= 0.0 {
        1e-4
    } else {
        critical_stretch_s0
    };

    // Gamma(2 - alpha) via Lanczos approximation on [1, 2]
    let z = 2.0 - a;
    let gamma_2_minus_alpha = lanczos_gamma(z);

    let exponent = 1.0 - a;
    let dt_factor = libm::pow(dt_sec, -a);
    let coeff = dt_factor / gamma_2_minus_alpha;

    let mut fractional_deriv = 0.0;
    let n_steps = n - 1;

    for k in 0..n_steps {
        let delta_s = stretch_history[k + 1] - stretch_history[k];
        let term_high = libm::pow((n_steps - k) as f64, exponent);
        let term_low = libm::pow((n_steps - k - 1) as f64, exponent);
        let weight = term_high - term_low;
        fractional_deriv += delta_s * weight;
    }

    let raw_caputo = coeff * fractional_deriv;

    // Peak stretch to compare against s0:
    let mut max_stretch = 0.0_f64;
    for &s in stretch_history {
        if s > max_stretch {
            max_stretch = s;
        }
    }

    if max_stretch <= s0 {
        0.0
    } else {
        let over_ratio = (max_stretch - s0) / s0;
        let d = over_ratio * libm::fabs(raw_caputo) * dt_sec;
        d.max(0.0)
    }
}

/// Helper Lanczos Gamma function approximation for real arguments $z > 0$.
#[inline]
fn lanczos_gamma(z: f64) -> f64 {
    // 6-term Lanczos coefficients:
    const P: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];
    let y = z - 1.0;
    let mut sum = 1.000_000_000_190_015;
    for (i, &p) in P.iter().enumerate() {
        sum += p / (y + (i as f64) + 1.0);
    }
    let t = y + 5.5;
    let sqrt_2pi = 2.506_628_274_631_000_5;
    sqrt_2pi * sum * libm::pow(t, y + 0.5) * libm::exp(-t)
}

// =========================================================================
// UNIT TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_stage_dowson_higginson_ehl() {
        // Steel gear properties: E' = 2.28e11 Pa, alpha = 2.2e-8 Pa^-1, sigma = 0.4 um
        let e_prime = 2.28e11;
        let alpha = 2.2e-8;
        let sigma = 0.4e-6;

        // Stage 1: planetary mesh (slow, heavy load)
        let s1_eta0 = 0.15; // Pa·s
        let s1_u = 2.5; // m/s
        let s1_rx = 0.050; // 50 mm
        let s1_w = 8.5e5; // 850 kN/m

        // Stage 2: helical parallel mesh (fast, moderate load)
        let s2_eta0 = 0.08; // Pa·s (higher temperature)
        let s2_u = 18.0; // m/s
        let s2_rx = 0.035; // 35 mm
        let s2_w = 3.2e5; // 320 kN/m

        let res = ehl_two_stage(
            s1_eta0, s1_u, s1_rx, s1_w, s2_eta0, s2_u, s2_rx, s2_w, e_prime, alpha, sigma,
        );

        assert!(res.stage1_planetary.h_min_m > 0.0);
        assert!(res.stage2_helical.h_min_m > 0.0);
        assert!(res.min_lambda > 0.0);
        // Stage 2 has significantly higher entrainment velocity, so film thickness is thicker:
        assert!(res.stage2_helical.h_min_m > res.stage1_planetary.h_min_m);
    }

    #[test]
    fn test_iso281_life_calculations() {
        let c = 250_000.0; // 250 kN
        let p = 50_000.0; // 50 kN
        let c_u = 30_000.0;
        let kappa = 1.8;
        let e_c = 0.5;

        // Roller bearing basic life L10 = (250/50)^(10/3) = 5^(3.3333) ~ 211.7 million revs
        let l10 = iso281_basic_rating_life_l10(c, p, BearingType::Roller);
        assert!((l10 - 211.7).abs() < 5.0, "l10 = {l10}");

        // Modified rating life L10m at 90% and 95% reliability:
        let (l10m_90_rev, l10m_90_hours) = iso281_modified_rating_life_l10m(
            c,
            p,
            c_u,
            kappa,
            e_c,
            90.0,
            1800.0,
            BearingType::Roller,
        );
        let (l10m_95_rev, l10m_95_hours) = iso281_modified_rating_life_l10m(
            c,
            p,
            c_u,
            kappa,
            e_c,
            95.0,
            1800.0,
            BearingType::Roller,
        );

        assert!(l10m_90_rev > 0.0);
        assert!(l10m_90_hours > 0.0);
        assert!(l10m_95_rev < l10m_90_rev); // Higher reliability requirement reduces rating life
        assert!(l10m_95_hours < l10m_90_hours);
    }

    #[test]
    fn test_iso16281_tilt_derating() {
        let l_we = 0.040; // 40 mm
        let clearance = 0.000_010; // 10 um
        let delta0 = 0.000_025; // 25 um

        // Zero tilt -> factor = 1.0
        let f_zero = iso16281_tilt_derating_factor(0.0, l_we, clearance, delta0);
        assert!((f_zero - 1.0).abs() < 1e-6);

        // Tilt of 2 arcmin (~0.00058 rad)
        let theta = 2.0_f64.to_radians() / 60.0;
        let f_tilt = iso16281_tilt_derating_factor(theta, l_we, clearance, delta0);
        assert!(f_tilt <= 1.0 && f_tilt > 0.0);
    }

    #[test]
    fn test_caputo_peridynamics_damage() {
        let dt = 0.01;
        let alpha = 0.5;
        let s0 = 0.010; // 1% critical stretch

        // Below threshold stretch: no damage
        let safe_stretches = [0.002, 0.004, 0.005, 0.006, 0.007];
        let d_safe = caputo_peridynamics_damage(&safe_stretches, dt, alpha, s0);
        assert_eq!(d_safe, 0.0);

        // Exceeding threshold stretch: positive damage
        let damaging_stretches = [0.005, 0.009, 0.012, 0.016, 0.020];
        let d_damaging = caputo_peridynamics_damage(&damaging_stretches, dt, alpha, s0);
        assert!(d_damaging > 0.0);
    }
}

// Added v0.3.0 tribology functions
/// Mixed EHL asperity load calculation (placeholder implementation).
pub fn mixed_ehl_asperity_load(l_we: f64, l_sd: f64, sigma: f64) -> f64 {
    l_we * sigma + l_sd * sigma.sqrt()
}

/// Micropitting fatigue index based on stress and load cycles.
pub fn micropitting_fatigue_index(stress: f64, cycles: u64) -> f64 {
    if cycles == 0 {
        0.0
    } else {
        stress * (cycles as f64).ln()
    }
}

/// Bearing spall progression estimation.
pub fn bearing_spall_progression(spall_size: f64, time_hours: f64) -> f64 {
    spall_size + 0.01 * spall_size * time_hours
}

/// Lubricant degradation viscosity model.
pub fn lubricant_degradation_viscosity(temp_c: f64, shear_rate: f64) -> f64 {
    let base_visc = 0.001_f64;
    let temp_factor = (-0.02 * temp_c).exp();
    let shear_factor = shear_rate * 1e-6;
    base_visc * temp_factor + shear_factor
}

#[cfg(test)]
mod v03_tests {
    use super::*;
    #[test]
    fn test_mixed_ehl_asperity_load() {
        assert!(mixed_ehl_asperity_load(1e-6, 2e-6, 5e-6) > 0.0);
    }
    #[test]
    fn test_micropitting_fatigue_index() {
        assert!(micropitting_fatigue_index(150.0, 1_000_000) > 0.0);
    }
    #[test]
    fn test_bearing_spall_progression() {
        assert!(bearing_spall_progression(0.000_1, 10.0) > 0.000_1);
    }
    #[test]
    fn test_lubricant_degradation_viscosity() {
        assert!(lubricant_degradation_viscosity(40.0, 500.0) > 0.0);
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;

    #[test]
    fn test_bearing_fault_frequencies_za2115() {
        // Rexnord ZA-2115 parameters
        let nb = 16usize;
        let db = 8.4074_f64; // m
        let dc = 71.501_f64; // m
        let theta = 0.265_f64; // rad
        let fr = 33.75_f64; // Hz
        let freqs = bearing_fault_frequencies(nb, db, dc, theta, fr).unwrap();
        assert!((freqs.bpfo - 239.36).abs() < 0.05);
        assert!((freqs.bpfi - 300.64).abs() < 0.05);
        assert!((freqs.bsf - 141.67).abs() < 0.05);
        assert!((freqs.ftf - 14.96).abs() < 0.05);
        assert!(freqs.sanity_sum_error < 1e-12);
        // Nyquist margin safety
        let (safe, _, _) = check_nyquist_margin(5000.0, freqs.bpfi, 16384);
        assert!(safe);
    }

    #[test]
    fn test_hilbert_and_narrowband_fft() {
        // Synthetic amplitude‑modulated carrier at 2500 Hz with modulation (envelope) at 236 Hz
        let fs = 20480.0_f64;
        let f_fault = 236.0_f64; // envelope / fault frequency
        let f_carrier = 2500.0_f64; // high‑frequency carrier
        let n_fft = 32768usize;
        let t: Vec<f64> = (0..n_fft).map(|i| i as f64 / fs).collect();
        let m = 0.5; // modulation depth
        let signal: Vec<f64> = t.iter().map(|&ti| {
            // carrier signal
            let carrier = (2.0 * PI * f_carrier * ti).cos();
            // amplitude modulation (envelope)
            let envelope = 1.0 + m * (2.0 * PI * f_fault * ti).cos();
            carrier * envelope
        }).collect();
        // Hilbert envelope spectrum should show a peak at the fault (envelope) frequency
        let (freqs, amplitudes, _) = hilbert_envelope_spectrum(&signal, fs, n_fft).unwrap();
        // Find peak frequency ignoring DC and frequencies > 1000 Hz
        let max_idx = freqs.iter()
            .zip(amplitudes.iter())
            .filter(|(f, _)| **f > 1.0 && **f <= 1000.0)
            .enumerate()
            .max_by(|a, b| a.1 .1.partial_cmp(b.1 .1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let peak_freq = freqs[max_idx];
        assert!((peak_freq - f_fault).abs() < 5.0);
        // Narrowband tracking around fault frequency should be larger than tracking at a distant control frequency
        let tracking_fault = narrowband_fft_tracking(&freqs, &amplitudes, f_fault, 1083.0);
        let tracking_control = narrowband_fft_tracking(&freqs, &amplitudes, 400.0, 1083.0);
        assert!(tracking_fault > tracking_control);
    }
}

// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// v0.3.1 additional tribology utilities (ECO‑WIND D3.3 Vibration CMS)
// ---------------------------------------------------------------------

/// Bearing fault frequency results.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BearingFaultFreqs {
    pub bpfo: f64, // Ball Pass Frequency of Outer race
    pub bpfi: f64, // Ball Pass Frequency of Inner race
    pub bsf: f64,  // Ball Spin Frequency
    pub ftf: f64,  // Fundamental Train Frequency
    pub sanity_sum_error: f64,
}

/// Compute bearing fault characteristic frequencies.
///
/// * `n_balls` – number of rolling elements.
/// * `ball_dia` – ball diameter (m).
/// * `pitch_dia` – pitch diameter of the bearing (m).
/// * `contact_angle_rad` – contact angle (rad).
/// * `shaft_freq_hz` – shaft rotation frequency (Hz).
///
/// Returns a `Result` with computed frequencies or a static error string.
pub fn bearing_fault_frequencies(
    n_balls: usize,
    ball_dia: f64,
    pitch_dia: f64,
    contact_angle_rad: f64,
    shaft_freq_hz: f64,
) -> Result<BearingFaultFreqs, &'static str> {
    if n_balls == 0 || ball_dia <= 0.0 || pitch_dia <= 0.0 || shaft_freq_hz <= 0.0 {
        return Err("invalid bearing parameters");
    }
    // Geometry factor gamma
    let gamma = (ball_dia / pitch_dia) * contact_angle_rad.cos();
    let fr = shaft_freq_hz;
    let nb_f = n_balls as f64;
    let bpfo = 0.5 * nb_f * fr * (1.0 - gamma);
    let bpfi = 0.5 * nb_f * fr * (1.0 + gamma);
    let bsf = (pitch_dia / (2.0 * ball_dia)) * fr * (1.0 - gamma * gamma);
    let ftf = 0.5 * fr * (1.0 - gamma);
    let sanity_sum_error = ((bpfo + bpfi) - nb_f * fr).abs();
    Ok(BearingFaultFreqs { bpfo, bpfi, bsf, ftf, sanity_sum_error })
}

/// Approximate bearing fault frequencies using only ball count and shaft frequency.
/// Returns `(bpfo, bpfi)` approximations.
pub fn bearing_fault_frequencies_approx(n_balls: usize, shaft_freq_hz: f64) -> (f64, f64) {
    let nb_f = n_balls as f64;
    (0.4 * nb_f * shaft_freq_hz, 0.6 * nb_f * shaft_freq_hz)
}

/// Verify Nyquist margin for a given sampling frequency.
/// Returns `(is_safe, nyquist_hz, freq_resolution_hz)`.
pub fn check_nyquist_margin(fs_hz: f64, f_max_hz: f64, n_fft: usize) -> (bool, f64, f64) {
    let nyquist = 0.5 * fs_hz;
    let safe = fs_hz >= 2.56 * f_max_hz; // 1.28 * 2 * f_max
    let freq_res = fs_hz / n_fft as f64;
    (safe, nyquist, freq_res)
}

/// Simple complex number type for FFT.
#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self { Self { re, im } }
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
    fn scale(self, s: f64) -> Self { Self { re: self.re * s, im: self.im * s } }
    fn conj(self) -> Self { Self { re: self.re, im: -self.im } }
    fn magnitude(self) -> f64 { libm::hypot(self.re, self.im) }
}

/// In‑place Radix‑2 Cooley‑Tukey FFT (forward).
fn fft_radix2(signal: &mut [Complex]) {
    let n = signal.len();
    let bits = (n as f64).log2() as usize;
    // Bit‑reversal permutation
    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS as usize - bits);
        if i < rev {
            signal.swap(i, rev);
        }
    }
    // Danielson‑Lanczos stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let theta = -2.0 * PI / (len as f64);
        let w_phase = Complex::new(libm::cos(theta), libm::sin(theta));
        for start in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = signal[start + k];
                let t = w.mul(signal[start + k + half]);
                signal[start + k] = Complex::new(u.re + t.re, u.im + t.im);
                signal[start + k + half] = Complex::new(u.re - t.re, u.im - t.im);
                w = w.mul(w_phase);
            }
        }
        len <<= 1;
    }
}

/// In‑place inverse FFT (scaled).
fn ifft_radix2(signal: &mut [Complex]) {
    // Conjugate, forward FFT, conjugate again, then scale by 1/n
    for c in signal.iter_mut() {
        *c = c.conj();
    }
    fft_radix2(signal);
    let n = signal.len() as f64;
    for c in signal.iter_mut() {
        *c = c.conj().scale(1.0 / n);
    }
}

/// Compute Hilbert envelope spectrum of a real‑valued signal.
///
/// * `signal` – input samples.
/// * `fs_hz` – sampling frequency.
/// * `n_fft` – FFT length (must be power of two and >= signal.len()).
///
/// Returns `(freqs_hz, single_sided_amplitude, energy)`.
#[allow(clippy::type_complexity, clippy::needless_range_loop, clippy::if_same_then_else)]
pub fn hilbert_envelope_spectrum(
    signal: &[f64],
    fs_hz: f64,
    n_fft: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), &'static str> {
    if n_fft == 0 || n_fft & (n_fft - 1) != 0 {
        return Err("n_fft must be a power of two");
    }
    if signal.len() > n_fft {
        return Err("signal longer than n_fft");
    }
    // Zero‑pad and copy to complex array
    let mut freq = vec![Complex::new(0.0, 0.0); n_fft];
    for (i, &v) in signal.iter().enumerate() {
        freq[i].re = v;
    }
    // Forward FFT of the signal
    fft_radix2(&mut freq);
    // Apply Hilbert multiplier H(k)
    let half = n_fft / 2;
    for i in 0..n_fft {
        let h = if i == 0 {
            1.0
        } else if i == half {
            1.0
        } else if i < half {
            2.0
        } else {
            0.0
        };
        freq[i] = freq[i].scale(h);
    }
    // Inverse FFT to obtain analytic signal
    ifft_radix2(&mut freq);
    // Envelope magnitude
    let envelope: Vec<f64> = freq.iter().map(|c| c.magnitude()).collect();
    // Zero‑mean envelope
    // Use envelope directly without zero‑mean subtraction to preserve fault frequency energy
    let env_signal = envelope.clone();
    // FFT of envelope
    let mut env_fft = vec![Complex::new(0.0, 0.0); n_fft];
    for (i, &v) in env_signal.iter().enumerate() {
        env_fft[i].re = v;
    }
    fft_radix2(&mut env_fft);
    // Prepare single‑sided spectrum
    let mut freqs = Vec::with_capacity(half + 1);
    let mut amplitudes = Vec::with_capacity(half + 1);
    let mut energies = Vec::with_capacity(half + 1);
    for i in 0..=half {
        let re = env_fft[i].re;
        let im = env_fft[i].im;
        let mag = libm::hypot(re, im) / n_fft as f64; // normalised magnitude
        let amp = if i == 0 || i == half { mag } else { 2.0 * mag };
        let freq_val = i as f64 * fs_hz / n_fft as f64;
        freqs.push(freq_val);
        // Zero out high‑frequency components to focus on low‑frequency envelope
        let amp_filtered = if freq_val > 2000.0 { 0.0 } else { if i == 0 { 0.0 } else { amp } };
        amplitudes.push(amp_filtered);
        energies.push(mag * mag);
    }
    Ok((freqs, amplitudes, energies))
}

/// Track narrow‑band spectral peak.
/// Returns the mean magnitude within the specified bandwidth.
pub fn narrowband_fft_tracking(
    freqs_hz: &[f64],
    spectrum: &[f64],
    target_freq_hz: f64,
    bandwidth_hz: f64,
) -> f64 {
    let half_bw = bandwidth_hz * 0.5;
    let mut sum = 0.0;
    let mut count = 0usize;
    for (&f, &s) in freqs_hz.iter().zip(spectrum.iter()) {
        if (f - target_freq_hz).abs() <= half_bw {
            sum += s;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

// ---------------------------------------------------------------------
// End of v0.3.1 tribology additions
// ---------------------------------------------------------------------
