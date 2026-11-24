//! #![no_std]
//! Stub for tribology module (no_std).

#![allow(dead_code)]

/// Placeholder struct for tribology calculations.
pub struct TribologyModel;

impl TribologyModel {
    /// New instance (does nothing).
    pub const fn new() -> Self { Self {} }

    /// Example method returning a constant friction coefficient.
    pub const fn friction_coeff(&self) -> f64 { 0.25 }
}
