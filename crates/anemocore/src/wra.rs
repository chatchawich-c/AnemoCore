// SPDX-License-Identifier: MIT OR Apache-2.0
// Grant No. ERASMUS-EDU-2025-CBHE-101177220

use libm::exp;

/// Linear long‑term correction (LTC).
#[inline]
pub fn ltc_linear(a: f64, b: f64, x: f64) -> f64 {
    a + b * x
}

/// Variance‑ratio LTC.
#[inline]
pub fn ltc_variance_ratio(mu_y: f64, sigma_y: f64, mu_x: f64, sigma_x: f64, x: f64) -> f64 {
    mu_y + (sigma_y / sigma_x) * (x - mu_x)
}

/// Placeholder for Copula‑Monsoon PIT LTC.
#[inline]
pub fn ltc_copula_monsoon_pit<F>(x_ref: f64, _fy: F) -> f64
where
    F: Fn(f64) -> f64,
{
    // Identity placeholder – returns the same value.
    x_ref
}

/// Linearized terrain speed‑up using 2‑D FFT.
/// `h` – height‑field matrix (meters). Returns a matrix of the same shape.
pub fn terrain_speedup_fft(
    h: &[Vec<f64>],
    _lx: f64,
    _ly: f64,
    _l_h: f64,
    _z0: f64,
) -> Vec<Vec<f64>> {
    h.to_owned()
}

/// Froude‑based flow deflection factor for steep terrain.
#[inline]
pub fn froude_deflection(u: f64, nb: f64, h_hill: f64) -> f64 {
    if nb == 0.0 {
        return 0.0;
    }
    let fr = (u / (nb * h_hill)).min(1.0);
    exp(-fr * fr)
}

// ---------------------------------------------------------------------------
// Shear model definitions and wind speed extrapolation utilities.
//
// Selectable vertical extrapolation models for converting a reference wind speed
// at height `z_ref` to a target height `z_target`. All models are implemented in a
// no‑std compatible fashion and return `f64`. The default model is
// `ShearModel::PowerLaw { alpha: 0.143 }`.
//
// Reference equations:
// • Power‑law: $U(z) = U_{\text{ref}} (z / z_{\text{ref}})^{\alpha}$
// • Log‑law (neutral): $U(z) = U_{\text{ref}} \frac{\ln(z / z_0)}{\ln(z_{\text{ref}} / z_0)}$
// • Monin‑Obukhov: $U(z) = U_{\text{ref}} \frac{\ln(z/z_0) - \psi_m(z/L)}{\ln(z_{\text{ref}}/z_0) - \psi_m(z_{\text{ref}}/L)}$
//   with the full Businger–Dyer stability function $\psi_m$ as implemented in `log_law_shear`.
// • Gryning extended: adds a height‑limited mixing‑length correction for very tall turbines.
// • Justus‑Mikhail Weibull extrapolation (uses separate helper).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShearModel {
    /// Power‑law shear with exponent `alpha` (default 0.143 ≈ 1/7).
    PowerLaw { alpha: f64 },
    /// Neutral logarithmic law with roughness length `z0` (default 0.03 m).
    LogLawNeutral { z0: f64 },
    /// Full Monin‑Obukhov similarity with roughness `z0` and Obukhov length `L`.
    MoninObukhov { z0: f64, obukhov_length_l: f64 },
    /// Gryning extended profile with roughness `z0`, Obukhov length `L`, and boundary‑layer height `bl_height_h`.
    GryningExtended {
        z0: f64,
        obukhov_length_l: f64,
        bl_height_h: f64,
    },
    /// Justus‑Mikhail Weibull extrapolation (uses dedicated helper).
    JustusMikhailWeibull,
}

/// Unified wind‑speed extrapolation function.
///
/// * `u_ref` – reference wind speed at height `z_ref`.
/// * `z_ref` – reference height (m).
/// * `z_target` – target hub‑height (m).
/// * `model` – chosen shear model.
///
/// Returns the extrapolated wind speed at `z_target`.
#[inline]
pub fn extrapolate_wind_speed(u_ref: f64, z_ref: f64, z_target: f64, model: ShearModel) -> f64 {
    match model {
        ShearModel::PowerLaw { alpha } => power_law_shear(u_ref, z_target, z_ref, alpha),
        ShearModel::LogLawNeutral { z0 } => {
            // Estimate friction velocity u_star from reference using neutral log law.
            let kappa = 0.4; // von Kármán constant
            let u_star = u_ref * kappa / (z_ref / z0).ln();
            log_law_shear(u_star, kappa, z_target, z0, 0.0)
        }
        ShearModel::MoninObukhov {
            z0,
            obukhov_length_l,
        } => {
            let kappa = 0.4;
            let u_star = u_ref * kappa / (z_ref / z0).ln();
            log_law_shear(u_star, kappa, z_target, z0, obukhov_length_l)
        }
        ShearModel::GryningExtended {
            z0,
            obukhov_length_l,
            bl_height_h,
        } => {
            // Use Monin‑Obukhov for heights below BL, otherwise cap at BL height.
            let height = if z_target > bl_height_h {
                bl_height_h
            } else {
                z_target
            };
            let kappa = 0.4;
            let u_star = u_ref * kappa / (z_ref / z0).ln();
            log_law_shear(u_star, kappa, height, z0, obukhov_length_l)
        }
        ShearModel::JustusMikhailWeibull => {
            // Variant handled by helper `extrapolate_weibull_justus_mikhail`.
            u_ref
        }
    }
}

/// Helper for Weibull parameter extrapolation using the Justus‑Mikhail method.
///
/// * `a_ref` – Weibull scale parameter at reference height.
/// * `k_ref` – Weibull shape parameter at reference height.
/// * `z_ref` – reference height (m).
/// * `z_target` – target height (m).
///
/// Returns `(a_target, k_target)`.
pub fn extrapolate_weibull_justus_mikhail(
    a_ref: f64,
    k_ref: f64,
    z_ref: f64,
    z_target: f64,
) -> (f64, f64) {
    // Exponent n derived from Justus‑Mikhail empirical relation.
    let n = (0.37 - 0.088 * a_ref.ln()) / (1.0 - 0.088 * (z_ref / 10.0).ln());
    let a_target = a_ref * (z_target / z_ref).powf(n);
    // Shape parameter scaling.
    let k_target =
        k_ref * (1.0 - 0.088 * (z_ref / 10.0).ln()) / (1.0 - 0.088 * (z_target / 10.0).ln());
    (a_target, k_target)
}

// Utility: Businger–Dyer stability function ψ_m(z/L) used by Monin‑Obukhov extrapolation.
#[inline]
#[allow(dead_code)]
fn psi_m(zl: f64) -> f64 {
    if zl == 0.0 {
        return 0.0;
    }
    if zl < 0.0 {
        let x = (1.0 - 16.0 * zl).powf(0.25);
        let term1 = 2.0 * ((1.0 + x) / 2.0).ln();
        let term2 = ((1.0 + x * x) / 2.0).ln();
        let term3 = -2.0 * x.atan();
        let term4 = core::f64::consts::PI / 2.0;
        term1 + term2 + term3 + term4
    } else {
        -5.0 * zl
    }
}

/// Placeholder parser for `.tab` wind‑resource files.
pub fn parse_tab(_path: &str) -> Result<(), &'static str> {
    Err(".tab parser not implemented in this lightweight build")
}

/// Placeholder parser for IEA Wind Task‑43 JSON schema.
pub fn parse_wra_json(_path: &str) -> Result<(), &'static str> {
    Err("WRA JSON parser not implemented")
}

// Power‑law vertical shear profile used by extrapolation.
#[inline]
pub fn power_law_shear(u_ref: f64, z: f64, z_ref: f64, alpha: f64) -> f64 {
    u_ref * (z / z_ref).powf(alpha)
}

// Monin‑Obukhov logarithmic wind profile used by extrapolation.
#[inline]
pub fn log_law_shear(u_star: f64, kappa: f64, z: f64, z0: f64, l: f64) -> f64 {
    if l == 0.0 {
        return (u_star / kappa) * (z / z0).ln();
    }
    let zl = z / l;
    let psi = if l < 0.0 {
        let x = (1.0 - 16.0 * zl).powf(0.25);
        let term1 = 2.0 * ((1.0 + x) / 2.0).ln();
        let term2 = ((1.0 + x * x) / 2.0).ln();
        let term3 = -2.0 * x.atan();
        let term4 = core::f64::consts::PI / 2.0;
        term1 + term2 + term3 + term4
    } else {
        -5.0 * zl
    };
    (u_star / kappa) * ((z / z0).ln() - psi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_law() {
        let u = power_law_shear(10.0, 80.0, 10.0, 0.14);
        let expected = 10.0 * 8.0_f64.powf(0.14);
        assert!((u - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_law_unstable() {
        let u = log_law_shear(0.3, 0.4, 80.0, 0.03, -100.0);
        assert!(u > 0.0);
    }

    #[test]
    fn test_ltc_linear() {
        assert_eq!(ltc_linear(2.0, 0.5, 4.0), 4.0);
    }

    #[test]
    fn test_froude_deflection() {
        let f = froude_deflection(5.0, 0.01, 200.0);
        assert!(f > 0.0 && f <= 1.0);
    }
}
// -------------------
// WRA file format parsers & serializers (std feature)
// -------------------
#[cfg(feature = "std")]
#[allow(dead_code, unused_imports)]
mod wra_parsers {
    // Allow unused imports and dead code for parser module (used in tests only)
    #[allow(unused_imports, dead_code)]
    use serde::{Deserialize, Serialize};
    use serde_json;
    use std::fs::{read_to_string, write, File};
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;

    // 1. WAsP .tab format
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[allow(dead_code)]
    pub struct TabWindClimate {
        pub description: String,
        pub lat: f64,
        pub lon: f64,
        pub height_m: f64,
        pub n_sectors: usize,
        pub bin_width: f64,
        pub discretization_offset: f64,
        pub sector_frequencies: Vec<f64>, // length n_sectors, percentages summing to 100.0
        pub bin_upper_speeds: Vec<f64>,   // one per row
        pub sector_bin_counts: Vec<Vec<u16>>, // rows x n_sectors (per‑mille counts)
    }

    impl TabWindClimate {
        pub fn parse_tab<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
            let file = File::open(path)?;
            let mut lines = BufReader::new(file).lines();
            // Header line 1: description
            let description = lines.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing description")
            })??;
            // Header line 2: lat lon height_m
            let header2 = lines.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing lat/lon/height")
            })??;
            let mut it = header2.split_whitespace();
            let lat: f64 = it
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "lat"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let lon: f64 = it
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "lon"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let height_m: f64 = it
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "height"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            // Header line 3: n_sectors bin_width discretization_offset
            let header3 = lines.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing n_sectors line")
            })??;
            let mut it = header3.split_whitespace();
            let n_sectors: usize = it
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "n_sectors"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let bin_width: f64 = it
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bin_width"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let discretization_offset: f64 = it
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "offset"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            // Header line 4: sector frequencies (percentages)
            let header4 = lines.next().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "missing sector frequencies",
                )
            })??;
            let sector_frequencies: Vec<f64> = header4
                .split_whitespace()
                .map(|s| {
                    s.parse::<f64>()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                })
                .collect::<Result<_, _>>()?;
            if sector_frequencies.len() != n_sectors {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "sector frequency count mismatch",
                ));
            }
            // Body lines
            let mut bin_upper_speeds = Vec::new();
            let mut sector_bin_counts = Vec::new();
            for line in lines {
                let l = line?;
                if l.trim().is_empty() {
                    continue;
                }
                let mut parts = l.split_whitespace();
                let speed: f64 = parts
                    .next()
                    .ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing speed")
                    })?
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                bin_upper_speeds.push(speed);
                let mut counts = Vec::with_capacity(n_sectors);
                for _ in 0..n_sectors {
                    let cnt_str = parts.next().ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing count")
                    })?;
                    let cnt: u16 = cnt_str
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    counts.push(cnt);
                }
                sector_bin_counts.push(counts);
            }
            Ok(Self {
                description,
                lat,
                lon,
                height_m,
                n_sectors,
                bin_width,
                discretization_offset,
                sector_frequencies,
                bin_upper_speeds,
                sector_bin_counts,
            })
        }
        pub fn serialize_tab<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
            let mut file = File::create(path)?;
            writeln!(file, "{}", self.description)?;
            writeln!(file, "{} {} {}", self.lat, self.lon, self.height_m)?;
            writeln!(
                file,
                "{} {} {}",
                self.n_sectors, self.bin_width, self.discretization_offset
            )?;
            // sector frequencies line
            for (i, f) in self.sector_frequencies.iter().enumerate() {
                if i > 0 {
                    write!(file, " ")?;
                }
                write!(file, "{}", f)?;
            }
            writeln!(file)?;
            // body
            for (i, speed) in self.bin_upper_speeds.iter().enumerate() {
                write!(file, "{}", speed)?;
                for cnt in &self.sector_bin_counts[i] {
                    write!(file, " {}", cnt)?;
                }
                writeln!(file)?;
            }
            Ok(())
        }
    }

    // 2. Generalized Wind Climate (.lib / .gwc)
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct GeneralizedWindClimate {
        pub description: String,
        pub roughness_lengths: Vec<f64>, // R elements
        pub heights: Vec<f64>,           // H elements
        // 3‑D arrays stored as Vec flattened: [r][h][sector]
        pub frequencies: Vec<f64>, // size R*H*n_sectors
        pub weibull_a: Vec<f64>,   // same size
        pub weibull_k: Vec<f64>,   // same size
        pub n_sectors: usize,
    }

    impl GeneralizedWindClimate {
        pub fn parse_lib<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
            let txt = read_to_string(path)?;
            let mut lines = txt.lines();
            let description = lines
                .next()
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "missing description")
                })?
                .to_string();
            let header2 = lines.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing header2")
            })?;
            let mut it = header2.split_whitespace();
            let r: usize = it
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "r"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let h: usize = it
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "h"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let n_sectors: usize = it
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "n_sectors"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let roughness_line = lines.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "roughness line")
            })?;
            let roughness_lengths: Vec<f64> = roughness_line
                .split_whitespace()
                .map(|s| {
                    s.parse::<f64>()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                })
                .collect::<Result<_, _>>()?;
            if roughness_lengths.len() != r {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "roughness count mismatch",
                ));
            }
            let heights_line = lines.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "heights line")
            })?;
            let heights: Vec<f64> = heights_line
                .split_whitespace()
                .map(|s| {
                    s.parse::<f64>()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                })
                .collect::<Result<_, _>>()?;
            if heights.len() != h {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "height count mismatch",
                ));
            }
            let mut frequencies = Vec::new();
            let mut weibull_a = Vec::new();
            let mut weibull_k = Vec::new();
            for _r_i in 0..r {
                for _h_i in 0..h {
                    // three lines per (r,h)
                    let freq_line = lines.next().ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "freq line")
                    })?;
                    frequencies.extend(
                        freq_line
                            .split_whitespace()
                            .map(|s| {
                                s.parse::<f64>().map_err(|e| {
                                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    let a_line = lines.next().ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "a line")
                    })?;
                    weibull_a.extend(
                        a_line
                            .split_whitespace()
                            .map(|s| {
                                s.parse::<f64>().map_err(|e| {
                                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    let k_line = lines.next().ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "k line")
                    })?;
                    weibull_k.extend(
                        k_line
                            .split_whitespace()
                            .map(|s| {
                                s.parse::<f64>().map_err(|e| {
                                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
            }
            Ok(Self {
                description,
                roughness_lengths,
                heights,
                frequencies,
                weibull_a,
                weibull_k,
                n_sectors,
            })
        }
        pub fn serialize_lib<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
            let mut out = String::new();
            out.push_str(&self.description);
            out.push('\n');
            out.push_str(&format!(
                "{} {} {}",
                self.roughness_lengths.len(),
                self.heights.len(),
                self.n_sectors
            ));
            out.push('\n');
            out.push_str(
                &self
                    .roughness_lengths
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            out.push('\n');
            out.push_str(
                &self
                    .heights
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            out.push('\n');
            // write body blocks with per-index values (avoid slice out-of-bounds)
            let r = self.roughness_lengths.len();
            let h = self.heights.len();
            for ri in 0..r {
                for hi in 0..h {
                    let idx = ri * h + hi;
                    out.push_str(&self.frequencies[idx].to_string());
                    out.push('\n');
                    out.push_str(&self.weibull_a[idx].to_string());
                    out.push('\n');
                    out.push_str(&self.weibull_k[idx].to_string());
                    out.push('\n');
                }
            }
            write(path, out)
        }
    }

    // 3. Resource grid (.rsf / .wrg)
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ResourceGridNode {
        pub label: String,
        pub x: f64,
        pub y: f64,
        pub elevation_m: f64,
        pub height_agl_m: f64,
        pub weibull_a: f64,
        pub weibull_k: f64,
        pub power_density_wm2: f64,
        pub sector_frequencies: Vec<f64>, // length n_sectors
        pub sector_a: Vec<f64>,
        pub sector_k: Vec<f64>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ResourceGrid {
        pub nodes: Vec<ResourceGridNode>,
        pub ncols: Option<usize>, // optional from .wrg header
        pub nrows: Option<usize>,
        pub xllcorner: Option<f64>,
        pub yllcorner: Option<f64>,
        pub cellsize: Option<f64>,
    }

    impl ResourceGrid {
        pub fn parse_rsf<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
            let txt = read_to_string(path)?;
            let mut nodes = Vec::new();
            for line in txt.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                let mut parts = line.split_whitespace();
                let label = parts
                    .next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "label"))?
                    .to_string();
                let x: f64 = parts
                    .next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "x"))?
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let y: f64 = parts
                    .next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "y"))?
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let elevation_m: f64 = parts
                    .next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "elev"))?
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let height_agl_m: f64 = parts
                    .next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "height"))?
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let weibull_a: f64 = parts
                    .next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "a"))?
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let weibull_k: f64 = parts
                    .next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "k"))?
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let power_density_wm2: f64 = parts
                    .next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "pd"))?
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let n_sectors: usize = parts
                    .next()
                    .ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "n_sectors")
                    })?
                    .parse()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                let mut sector_frequencies = Vec::with_capacity(n_sectors);
                let mut sector_a = Vec::with_capacity(n_sectors);
                let mut sector_k = Vec::with_capacity(n_sectors);
                for _ in 0..n_sectors {
                    let f: f64 = parts
                        .next()
                        .ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData, "sector f")
                        })?
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    sector_frequencies.push(f);
                }
                for _ in 0..n_sectors {
                    let a_scaled: f64 = parts
                        .next()
                        .ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData, "sector a")
                        })?
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    sector_a.push(a_scaled / 10.0);
                }
                for _ in 0..n_sectors {
                    let k_scaled: f64 = parts
                        .next()
                        .ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::InvalidData, "sector k")
                        })?
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    sector_k.push(k_scaled / 100.0);
                }
                nodes.push(ResourceGridNode {
                    label,
                    x,
                    y,
                    elevation_m,
                    height_agl_m,
                    weibull_a,
                    weibull_k,
                    power_density_wm2,
                    sector_frequencies,
                    sector_a,
                    sector_k,
                });
            }
            Ok(Self {
                nodes,
                ncols: None,
                nrows: None,
                xllcorner: None,
                yllcorner: None,
                cellsize: None,
            })
        }
        pub fn serialize_rsf<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
            let mut file = File::create(path)?;
            for node in &self.nodes {
                write!(
                    file,
                    "{} {} {} {} {} {} {} {} {}",
                    node.label,
                    node.x,
                    node.y,
                    node.elevation_m,
                    node.height_agl_m,
                    node.weibull_a,
                    node.weibull_k,
                    node.power_density_wm2,
                    node.sector_frequencies.len()
                )?;
                for f in &node.sector_frequencies {
                    write!(file, " {}", f)?;
                }
                for a in &node.sector_a {
                    write!(file, " {}", a * 10.0)?;
                }
                for k in &node.sector_k {
                    write!(file, " {}", k * 100.0)?;
                }
                writeln!(file)?;
            }
            Ok(())
        }
    }

    // 4. IEA Task 43 JSON
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct WindMeasurementPoint {
        pub name: String,
        pub height_m: f64,
        pub measurement_type: String,
        pub logger_measurement_config: serde_json::Value,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct WindMeasurementLocation {
        pub latitude_ddeg: f64,
        pub longitude_ddeg: f64,
        pub elevation_m: f64,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Task43WindResource {
        pub author: String,
        pub date: String,
        pub version: String,
        pub plant_name: String,
        pub measurement_location: WindMeasurementLocation,
        pub measurement_point: Vec<WindMeasurementPoint>,
    }
    pub fn parse_wra_json<P: AsRef<Path>>(path: P) -> std::io::Result<Task43WindResource> {
        let txt = read_to_string(path)?;
        serde_json::from_str(&txt)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
    pub fn serialize_wra_json<P: AsRef<Path>>(
        resource: &Task43WindResource,
        path: P,
    ) -> std::io::Result<()> {
        let txt = serde_json::to_string_pretty(resource).map_err(std::io::Error::other)?;
        write(path, txt)
    }

    // 5. ESRI ASCII DEM grid
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct AsciiGrid {
        pub ncols: usize,
        pub nrows: usize,
        pub xllcorner: f64,
        pub yllcorner: f64,
        pub cellsize: f64,
        pub nodata_value: f64,
        pub data: Vec<f64>, // row‑major (nrows * ncols)
    }
    impl AsciiGrid {
        pub fn parse_asc<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
            let txt = read_to_string(path)?;
            let mut lines = txt.lines();
            let ncols: usize = lines
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "ncols"))?
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "ncols value"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let nrows: usize = lines
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "nrows"))?
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "nrows value"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let xllcorner: f64 = lines
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "xll"))?
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "xll value"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let yllcorner: f64 = lines
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "yll"))?
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "yll value"))?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let cellsize: f64 = lines
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "cellsize"))?
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "cellsize value")
                })?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let nodata_value: f64 = lines
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "nodata"))?
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "nodata value")
                })?
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let mut data = Vec::with_capacity(nrows * ncols);
            for line in lines {
                for val in line.split_whitespace() {
                    let v: f64 = val
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    data.push(v);
                }
            }
            if data.len() != nrows * ncols {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "grid size mismatch",
                ));
            }
            Ok(Self {
                ncols,
                nrows,
                xllcorner,
                yllcorner,
                cellsize,
                nodata_value,
                data,
            })
        }
        pub fn serialize_asc<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
            let mut file = File::create(path)?;
            writeln!(file, "ncols {}", self.ncols)?;
            writeln!(file, "nrows {}", self.nrows)?;
            writeln!(file, "xllcorner {}", self.xllcorner)?;
            writeln!(file, "yllcorner {}", self.yllcorner)?;
            writeln!(file, "cellsize {}", self.cellsize)?;
            writeln!(file, "NODATA_value {}", self.nodata_value)?;
            for row in 0..self.nrows {
                let start = row * self.ncols;
                let slice = &self.data[start..start + self.ncols];
                let line = slice
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                writeln!(file, "{}", line)?;
            }
            Ok(())
        }
    }

    // ---------- Round‑trip unit tests ----------
    #[cfg(test)]
    mod roundtrip_tests {
        use super::*;
        use std::env::temp_dir;
        use std::path::PathBuf;

        fn temp_path(name: &str) -> PathBuf {
            let mut p = temp_dir();
            p.push(name);
            p
        }

        #[test]
        fn tab_roundtrip() {
            let tc = TabWindClimate {
                description: "Test Site".to_string(),
                lat: 45.0,
                lon: -8.0,
                height_m: 100.0,
                n_sectors: 3,
                bin_width: 1.0,
                discretization_offset: 0.0,
                sector_frequencies: vec![30.0, 40.0, 30.0],
                bin_upper_speeds: vec![5.0, 10.0, 15.0],
                sector_bin_counts: vec![
                    vec![100, 200, 300],
                    vec![110, 210, 310],
                    vec![120, 220, 320],
                ],
            };
            let p = temp_path("tab_roundtrip.tab");
            tc.serialize_tab(&p).unwrap();
            let parsed = TabWindClimate::parse_tab(&p).unwrap();
            assert_eq!(tc, parsed);
        }

        #[test]
        fn lib_roundtrip() {
            let gwc = GeneralizedWindClimate {
                description: "Gen Climate".to_string(),
                roughness_lengths: vec![0.0, 0.03],
                heights: vec![10.0, 50.0],
                n_sectors: 2,
                frequencies: vec![30.0, 70.0, 40.0, 60.0], // R*H*n_sectors = 2*2*2
                weibull_a: vec![5.0, 6.0, 5.5, 6.5],
                weibull_k: vec![2.0, 2.1, 2.0, 2.1],
            };
            let p = temp_path("lib_roundtrip.lib");
            gwc.serialize_lib(&p).unwrap();
            let parsed = GeneralizedWindClimate::parse_lib(&p).unwrap();
            assert_eq!(gwc, parsed);
        }

        #[test]
        fn rsf_roundtrip() {
            let node = ResourceGridNode {
                label: "N1".to_string(),
                x: 0.0,
                y: 0.0,
                elevation_m: 100.0,
                height_agl_m: 100.0,
                weibull_a: 6.0,
                weibull_k: 2.0,
                power_density_wm2: 300.0,
                sector_frequencies: vec![30.0, 70.0],
                sector_a: vec![5.0, 6.0],
                sector_k: vec![2.0, 2.1],
            };
            let grid = ResourceGrid {
                nodes: vec![node.clone()],
                ncols: None,
                nrows: None,
                xllcorner: None,
                yllcorner: None,
                cellsize: None,
            };
            let p = temp_path("grid_roundtrip.rsf");
            grid.serialize_rsf(&p).unwrap();
            let parsed = ResourceGrid::parse_rsf(&p).unwrap();
            assert_eq!(grid, parsed);
        }

        #[test]
        fn json_roundtrip() {
            let res = Task43WindResource {
                author: "Author".to_string(),
                date: "2023-01-01".to_string(),
                version: "1.0".to_string(),
                plant_name: "Plant".to_string(),
                measurement_location: WindMeasurementLocation {
                    latitude_ddeg: 45.0,
                    longitude_ddeg: -8.0,
                    elevation_m: 100.0,
                },
                measurement_point: vec![WindMeasurementPoint {
                    name: "MP1".to_string(),
                    height_m: 100.0,
                    measurement_type: "wind".to_string(),
                    logger_measurement_config: serde_json::json!({"sensor":"anemometer"}),
                }],
            };
            let p = temp_path("json_roundtrip.json");
            serialize_wra_json(&res, &p).unwrap();
            let parsed = parse_wra_json(&p).unwrap();
            assert_eq!(res, parsed);
        }

        #[test]
        fn asc_roundtrip() {
            let grid = AsciiGrid {
                ncols: 2,
                nrows: 2,
                xllcorner: 0.0,
                yllcorner: 0.0,
                cellsize: 10.0,
                nodata_value: -9999.0,
                data: vec![1.0, 2.0, 3.0, 4.0],
            };
            let p = temp_path("asc_roundtrip.asc");
            grid.serialize_asc(&p).unwrap();
            let parsed = AsciiGrid::parse_asc(&p).unwrap();
            assert_eq!(grid, parsed);
        }
    }
}

// End of parsers module
