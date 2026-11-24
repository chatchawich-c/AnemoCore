//! #![no_std]
//! Stub module for future thermodynamic models.

#![allow(dead_code)]

/// Placeholder struct representing a thermodynamic model.
pub struct ThermoModel;

impl ThermoModel {
    /// Creates a new instance (does nothing).
    pub const fn new() -> Self {
        Self {}
    }

    /// Example method that returns a constant temperature in Kelvin.
    pub const fn temperature_k(&self) -> f64 {
        298.15
    }
}
