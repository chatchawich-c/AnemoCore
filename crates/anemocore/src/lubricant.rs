/// Arrhenius lubricant oxidation life and ZDDP additive depletion models.
///
/// The implementation is deliberately lightweight and uses only the `libm`
/// crate (already a dependency of the project) for the required exponential
/// function.  All functions are `#[cfg(feature = "std")]` compatible, but they
/// also work in a `#![no_std]` environment because they rely only on `core`
/// primitives and `libm` which provides a `no_std` implementation.
///
/// References
/// ----------
/// * ISO 281 – Bearing life based on an Arrhenius temperature acceleration.
/// * ZDDP (zinc‑dialkyl‑dithiophosphate) additive decay is commonly modelled
///   with a first‑order exponential loss proportional to temperature‑accelerated
///   reaction rates.

#[cfg(feature = "std")]
pub mod lubricant {
    use libm::exp;

    const R_GAS: f64 = 8.31446261815324; // J·mol⁻¹·K⁻¹ (universal gas constant)

    /// Compute the remaining service life (hours) of a lubricant based on an
    /// Arrhenius temperature acceleration model.
    ///
    /// * `temp_c` – Operating temperature in °C.
    /// * `ref_temp_c` – Reference temperature in °C at which the reference life
    ///   `ref_life_h` is valid.
    /// * `ref_life_h` – Nominal service life at the reference temperature (hours).
    /// * `activation_energy_j_mol` – Activation energy *Eₐ* in J·mol⁻¹.
    ///
    /// Returns the predicted remaining life in hours.  The formula is
    ///

    /// L = L₀ * exp\( Eₐ / R * (1/T - 1/T₀) \)

    /// where all temperatures are in Kelvin.
    pub fn arrhenius_oil_life(
        temp_c: f64,
        ref_temp_c: f64,
        ref_life_h: f64,
        activation_energy_j_mol: f64,
    ) -> f64 {
        // Convert to Kelvin
        let t_k = temp_c + 273.15;
        let t0_k = ref_temp_c + 273.15;
        if t_k <= 0.0 || t0_k <= 0.0 {
            return f64::INFINITY; // invalid temperature – treat as infinite life
        }
        let exponent = activation_energy_j_mol / R_GAS * (1.0 / t_k - 1.0 / t0_k);
        ref_life_h * exp(exponent)
    }

    /// Simple first‑order ZDDP additive depletion model.
    ///
    /// * `initial_ppm` – Initial additive concentration in ppm.
    /// * `temp_c` – Operating temperature in °C.
    /// * `time_h` – Elapsed time in hours.
    /// * `k_ref` – Reference depletion rate constant at 25 °C (h⁻¹).
    /// * `activation_energy_j_mol` – Activation energy for the depletion reaction.
    ///
    /// The rate constant is temperature‑accelerated using an Arrhenius term:
    /// `k(T) = k_ref * exp(-Eₐ / R * (1/T - 1/T_ref))`.
    /// The remaining concentration is then `C = C₀ * exp(-k(T) * t)`.
    pub fn zddp_depletion(
        initial_ppm: f64,
        temp_c: f64,
        time_h: f64,
        k_ref: f64,
        activation_energy_j_mol: f64,
    ) -> f64 {
        // Reference temperature 25 °C → 298.15 K
        let t_ref_k = 25.0 + 273.15;
        let t_k = temp_c + 273.15;
        if t_k <= 0.0 || time_h < 0.0 {
            return initial_ppm; // invalid inputs – return unchanged
        }
        let k_t = k_ref * exp(-activation_energy_j_mol / R_GAS * (1.0 / t_k - 1.0 / t_ref_k));
        initial_ppm * exp(-k_t * time_h)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        const EPS: f64 = 1e-9;

        #[test]
        fn test_arrhenius_oil_life_basic() {
            // Reference: 200 h at 80 °C, Ea = 150 kJ·mol⁻¹.
            let life = arrhenius_oil_life(100.0, 80.0, 200.0, 150_000.0);
            // Higher temperature should reduce life, ensure it is less than reference.
            assert!(life < 200.0);
        }

        #[test]
        fn test_arrhenius_oil_life_at_reference() {
            let life = arrhenius_oil_life(80.0, 80.0, 200.0, 150_000.0);
            assert!((life - 200.0).abs() < EPS);
        }

        #[test]
        fn test_zddp_depletion_decay() {
            // Simple case: no temperature acceleration (use reference temperature).
            let remaining = zddp_depletion(1000.0, 25.0, 10.0, 0.05, 0.0);
            // Expected: C0 * exp(-k*t) = 1000 * exp(-0.5)
            let expected = 1000.0 * exp(-0.5);
            assert!((remaining - expected).abs() < EPS);
        }

        #[test]
        fn test_zddp_depletion_temperature_acceleration() {
            // Higher temperature should increase depletion rate, resulting in lower remaining.
            let low = zddp_depletion(1000.0, 25.0, 5.0, 0.05, 120_000.0);
            let high = zddp_depletion(1000.0, 75.0, 5.0, 0.05, 120_000.0);
            assert!(high < low);
        }
    }
}
