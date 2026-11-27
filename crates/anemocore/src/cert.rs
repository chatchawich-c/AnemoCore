//! Formal verification certificate emitter for AnemoCore.
//!
//! Emits SMT-LIB2 (`emit_smt2`) and Lean 4 (`emit_lean4`) proof obligations
//! certifying the Betz aerodynamic power coefficient bound ($C_p \le 16/27$)
//! and KAM Golden-Ratio VSG stability margins.

use std::fmt::Write as _;

/// Unified `*Core Standard v1.0` formal certificate emitter trait.
pub trait CertificateEmitter {
    /// Emit an SMT-LIB2 verification script for Z3 / cvc5.
    fn emit_smt2(&self) -> String;

    /// Emit a Lean 4 theorem and proof skeleton.
    fn emit_lean4(&self) -> String;
}

/// Physical and control certificate parameters for AnemoCore.
#[derive(Clone, Copy, Debug)]
pub struct AnemoCertificate {
    /// Rotor power coefficient $C_p$ (Betz limit: $16/27 \approx 0.59259259$).
    pub cp: f64,
    /// Controller detuned angular frequency $\omega_{\mathrm{ctrl}}$ [rad/s].
    pub omega_ctrl: f64,
    /// Drivetrain natural torsional frequency $\omega_n$ [rad/s].
    pub omega_n: f64,
    /// KAM-VSG damping / stability margin $\gamma > 0$.
    pub gamma_margin: f64,
}

impl Default for AnemoCertificate {
    fn default() -> Self {
        let omega_n = 12.5;
        Self {
            cp: 0.48,
            omega_ctrl: omega_n * crate::PHI_INV,
            omega_n,
            gamma_margin: 0.15,
        }
    }
}

impl CertificateEmitter for AnemoCertificate {
    fn emit_smt2(&self) -> String {
        let mut s = String::with_capacity(1024);
        let _ = writeln!(s, "; AnemoCore Formal Certificate (SMT-LIB2)");
        let _ = writeln!(s, "(set-logic QF_NRA)");
        let _ = writeln!(s, "(declare-const cp Real)");
        let _ = writeln!(s, "(declare-const omega_ctrl Real)");
        let _ = writeln!(s, "(declare-const omega_n Real)");
        let _ = writeln!(s, "(declare-const gamma_margin Real)");
        let _ = writeln!(s, "(assert (= cp {:.12}))", self.cp);
        let _ = writeln!(s, "(assert (= omega_ctrl {:.12}))", self.omega_ctrl);
        let _ = writeln!(s, "(assert (= omega_n {:.12}))", self.omega_n);
        let _ = writeln!(s, "(assert (= gamma_margin {:.12}))", self.gamma_margin);
        let _ = writeln!(s, "; Betz bound: 0 <= cp <= 16/27");
        let _ = writeln!(s, "(assert (and (>= cp 0.0) (<= (* 27.0 cp) 16.0)))");
        let _ = writeln!(s, "; KAM-VSG stability: omega_n > 0, omega_ctrl != omega_n, gamma_margin > 0");
        let _ = writeln!(s, "(assert (> omega_n 0.0))");
        let _ = writeln!(s, "(assert (distinct omega_ctrl omega_n))");
        let _ = writeln!(s, "(assert (> gamma_margin 0.0))");
        let _ = writeln!(s, "(check-sat)");
        s
    }

    fn emit_lean4(&self) -> String {
        let mut s = String::with_capacity(1024);
        let _ = writeln!(s, "-- AnemoCore Formal Certificate (Lean 4)");
        let _ = writeln!(s, "namespace AnemoCore.Cert");
        let _ = writeln!(s, "def cp : Float := {:.12}", self.cp);
        let _ = writeln!(s, "def betzLimit : Float := 16.0 / 27.0");
        let _ = writeln!(s, "def omegaCtrl : Float := {:.12}", self.omega_ctrl);
        let _ = writeln!(s, "def omegaN : Float := {:.12}", self.omega_n);
        let _ = writeln!(s, "def gammaMargin : Float := {:.12}", self.gamma_margin);
        let _ = writeln!(
            s,
            "theorem betz_bound_holds : cp >= 0.0 && cp <= betzLimit := by native_decide"
        );
        let _ = writeln!(
            s,
            "theorem kam_vsg_stable : omegaN > 0.0 && omegaCtrl != omegaN && gammaMargin > 0.0 := by native_decide"
        );
        let _ = writeln!(s, "end AnemoCore.Cert");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certificate_emitter_outputs() {
        let cert = AnemoCertificate::default();
        let smt = cert.emit_smt2();
        let lean = cert.emit_lean4();
        assert!(smt.contains("(set-logic QF_NRA)"));
        assert!(smt.contains("(<= (* 27.0 cp) 16.0)"));
        assert!(smt.contains("(check-sat)"));
        assert!(lean.contains("theorem betz_bound_holds"));
        assert!(lean.contains("theorem kam_vsg_stable"));
    }
}
