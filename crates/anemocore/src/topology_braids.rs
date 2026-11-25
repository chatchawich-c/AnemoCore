// SPDX-License-Identifier: MIT OR Apache-2.0
// Grant No. ERASMUS-EDU-2025-CBHE-101177220

/// Simple representation of a braid word using generator indices.
/// For n strands, generators are 1..(n-1). Positive integer = sigma_i, negative = sigma_i^{-1}.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BraidWord<const N: usize> {
    /// Fixed‑size array storing generator indices.
    generators: [i8; N],
    /// Number of valid entries in `generators`.
    len: usize,
}

/// Custom error type for braid operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BraidError {
    Full,
}

impl<const N: usize> Default for BraidWord<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> BraidWord<N> {
    /// Create an empty braid word.
    pub const fn new() -> Self {
        Self {
            generators: [0; N],
            len: 0,
        }
    }

    /// Append a generator to the word. Returns `Ok(())` or `Err(BraidError::Full)` if full.
    pub fn push(&mut self, gen: i8) -> Result<(), BraidError> {
        if self.len < N {
            self.generators[self.len] = gen;
            self.len += 1;
            Ok(())
        } else {
            Err(BraidError::Full)
        }
    }

    /// Reduce the word by cancelling adjacent inverse pairs and applying far‑commutativity.
    pub fn reduce(&mut self) {
        // Cancellation pass.
        let mut i = 0usize;
        while i + 1 < self.len {
            if self.generators[i] == -self.generators[i + 1] {
                // Remove both entries by shifting left.
                for j in i..self.len - 2 {
                    self.generators[j] = self.generators[j + 2];
                }
                self.len -= 2;
                i = i.saturating_sub(1);
            } else {
                i += 1;
            }
        }
        // Far‑commutativity: sort by absolute value (simple bubble sort).
        let mut swapped = true;
        while swapped {
            swapped = false;
            for j in 0..self.len.saturating_sub(1) {
                if (self.generators[j] as i16).abs() > (self.generators[j + 1] as i16).abs() {
                    self.generators.swap(j, j + 1);
                    swapped = true;
                }
            }
        }
    }

    /// Compute a placeholder reduced Burau matrix trace invariant.
    /// Returns the sum of absolute generator indices.
    pub fn burau_trace(&self) -> i32 {
        self.generators[..self.len]
            .iter()
            .map(|&g| (g as i32).abs())
            .sum()
    }

    /// Access the current slice of generators.
    pub fn as_slice(&self) -> &[i8] {
        &self.generators[..self.len]
    }
}

/// Discrete vortex helicity for a set of velocity (`u`) and vorticity (`ω`) samples.
/// `volumes` supplies the cell volumes ΔV.
pub fn vortex_helicity(u: &[f64], ω: &[f64], volumes: &[f64]) -> f64 {
    let n = u.len().min(ω.len()).min(volumes.len());
    let mut h = 0.0;
    for i in 0..n {
        h += u[i] * ω[i] * volumes[i];
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn braid_reduce_and_trace() {
        let mut w: BraidWord<64> = BraidWord::new();
        w.push(1).unwrap();
        w.push(-1).unwrap(); // cancels
        w.push(2).unwrap();
        w.push(1).unwrap();
        w.push(3).unwrap();
        w.reduce();
        assert_eq!(w.as_slice(), &[1, 2, 3]);
        assert_eq!(w.burau_trace(), 6);
    }

    #[test]
    fn helicity_compute() {
        let u = [1.0, 2.0, 3.0];
        let ω = [0.5, 0.5, 0.5];
        let v = [1.0, 1.0, 1.0];
        let h = vortex_helicity(&u, &ω, &v);
        assert!((h - 3.0).abs() < 1e-12);
    }
}
