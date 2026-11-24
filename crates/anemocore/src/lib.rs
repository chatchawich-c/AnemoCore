#![cfg_attr(not(feature = "std"), no_std)]

//! # ecowind-core
//!
//! Pure-Rust, `#![no_std]`-compatible computational kernels for the **ECO-WIND**
//! multi-scale wind turbine prognostics, grid-forming control, quasicrystal signal
//! processing, market microstructure, and aero-thermal farm wake research program.
//!
//! ## Dual License
//! Licensed under either of Apache License, Version 2.0 (`LICENSE-APACHE`)
//! or MIT license (`LICENSE-MIT`) at your option.

pub mod edge_governor;
pub mod fib_qmc;
pub mod kam_vsg;
pub mod math_no_std;
pub mod pi_emeans;
pub mod quasicrystal_wpt;
pub mod signal_wpt;
pub mod topology_braids;
pub mod weibull_est;

#[cfg(feature = "std")]
pub mod fib_liquidity;
#[cfg(feature = "std")]
pub mod phyllotaxis_wake;

/// Golden ratio constant phi = (1 + sqrt(5)) / 2.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
pub const PHI: f64 = 1.6180339887498949;

/// Reciprocal golden ratio phi^-1 = (sqrt(5) - 1) / 2 ~ 0.618034.
pub const PHI_INV: f64 = 0.6180339887498949;

/// Squared reciprocal golden ratio phi^-2 = (3 - sqrt(5)) / 2 ~ 0.381966.
pub const PHI_INV2: f64 = 0.38196601125010515;

/// Vogel Golden Angle theta_g = 2*pi / phi^2 ~ 2.39996323 rad (~137.507764 deg).
pub const GOLDEN_ANGLE_RAD: f64 = 2.3999632297286533;

pub mod microsite;
pub mod transforms;
