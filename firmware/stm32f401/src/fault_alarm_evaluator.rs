// SPDX-License-Identifier: MIT OR Apache-2.0
//! Simple fault alarm evaluator for firmware.

/// Evaluates whether a given sample exceeds a predefined threshold.
pub struct FaultAlarmEvaluator {
    /// Threshold value; samples above this trigger an alarm.
    threshold: i32,
}

impl FaultAlarmEvaluator {
    /// Create a new evaluator with the default threshold (e.g., 1000).
    pub const fn new() -> Self {
        Self { threshold: 1000 }
    }

    /// Set a custom threshold.
    pub const fn with_threshold(threshold: i32) -> Self {
        Self { threshold }
    }

    /// Returns `true` if the sample exceeds the threshold.
    pub fn evaluate(&self, sample: i32) -> bool {
        sample > self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluator() {
        let eval = FaultAlarmEvaluator::new();
        assert!(!eval.evaluate(500));
        assert!(eval.evaluate(1500));
    }
}
