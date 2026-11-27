#![cfg_attr(not(feature = "std"), no_std)]

//! # anemocore
//!
//! Pure-Rust, `#![no_std]`-compatible computational kernels for the **ECO-WIND**
//! multi-scale wind turbine prognostics, grid-forming control, quasicrystal signal
//! processing, market microstructure, and aero-thermal farm wake research program.
//!
//! ## Dual License
//! Licensed under either of Apache License, Version 2.0 (`LICENSE-APACHE`)
//! or MIT license (`LICENSE-MIT`) at your option.

#[cfg(feature = "aero")]
pub mod aero;
#[cfg(feature = "econ")]
pub mod econ_risk;
pub mod edge_governor;
#[cfg(feature = "std")]
pub mod fib_liquidity;
#[cfg(feature = "std")]
pub mod fib_qmc;
pub mod kam_vsg;
pub mod math_no_std;
#[cfg(feature = "std")]
pub mod phyllotaxis_wake;
#[cfg(feature = "std")]
pub mod pi_emeans;
pub mod quasicrystal_wpt;
#[cfg(feature = "std")]
pub mod reg_econ;
#[cfg(feature = "fleet")]
pub mod scada_fleet;
pub mod signal_wpt;
#[cfg(feature = "std")]
pub mod topology_braids;
#[cfg(feature = "std")]
pub mod weibull_est;

/// Golden ratio constant phi = (1 + sqrt(5)) / 2.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
pub const PHI: f64 = 1.6180339887498949;

/// Reciprocal golden ratio phi^-1 = (sqrt(5) - 1) / 2 ~ 0.618034.
pub const PHI_INV: f64 = 0.6180339887498949;

/// Squared reciprocal golden ratio phi^-2 = (3 - sqrt(5)) / 2 ~ 0.381966.
pub const PHI_INV2: f64 = 0.38196601125010515;

/// Vogel Golden Angle theta_g = 2*pi / phi^2 ~ 2.39996323 rad (~137.507764 deg).
pub const GOLDEN_ANGLE_RAD: f64 = 2.3999632297286533;

#[cfg(feature = "std")]
pub mod microsite;
#[cfg(feature = "std")]
pub mod oa;
#[cfg(feature = "std")]
pub mod thermo;
#[cfg(feature = "std")]
pub mod transforms;
#[cfg(feature = "std")]
pub mod tribology;
#[cfg(feature = "std")]
pub mod wake;
#[cfg(feature = "std")]
pub mod wra;
#[cfg(feature = "std")]
pub mod heat_diffusion;
#[cfg(feature = "std")]
pub mod lubricant;

#[cfg(feature = "std")]
pub mod rl_derating;

#[cfg(feature = "std")]
pub mod plugin;

#[cfg(feature = "std")]
pub mod cert;
