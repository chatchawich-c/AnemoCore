// SPDX-License-Identifier: MIT OR Apache-2.0
// Grant No. ERASMUS-EDU-2025-CBHE-101177220

use core::sync::atomic::{AtomicUsize, Ordering};

/// A lock‑free single‑producer single‑consumer ring buffer.
/// Capacity `N` must be a power of two for wrap‑around to work efficiently.
#[repr(align(64))]
pub struct SpscRingBuffer<const N: usize> {
    head: AtomicUsize,
    tail: AtomicUsize,
    buffer: [i32; N],
}

/// Custom error type for ring buffer operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeError {
    Full,
}

impl<const N: usize> Default for SpscRingBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> SpscRingBuffer<N> {
    /// Creates a new empty buffer.
    pub const fn new() -> Self {
        Self {
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            buffer: [0; N],
        }
    }

    /// Attempts to push a value into the buffer. Returns `Ok(())` on success,
    /// or `Err(EdgeError::Full)` if the buffer is full.
    pub fn push(&mut self, val: i32) -> Result<(), EdgeError> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if (head + 1) % N == tail % N {
            return Err(EdgeError::Full);
        }
        self.buffer[head % N] = val;
        self.head.store((head + 1) % N, Ordering::Release);
        Ok(())
    }

    /// Attempts to pop a value from the buffer. Returns `Some(value)` if not empty.
    pub fn pop(&mut self) -> Option<i32> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail % N == head % N {
            return None;
        }
        let val = self.buffer[tail % N];
        self.tail.store((tail + 1) % N, Ordering::Release);
        Some(val)
    }
}

/// Bartlett latent projector – computes a simple moving‑average‑like projection.
/// `input` is the source slice, `output` is a mutable slice that receives the
/// projected values. The projection performs an O(k·p) multiply‑accumulate where
/// `k = output.len()` and `p = input.len()`.
pub fn bartlett_projector(input: &[f64], output: &mut [f64]) {
    let k = output.len();
    if k == 0 || input.is_empty() {
        return;
    }
    let group = input.len() / k;
    for (i, out) in output.iter_mut().enumerate() {
        let start = i * group;
        let end = if i == k - 1 {
            input.len()
        } else {
            start + group
        };
        let sum: f64 = input[start..end].iter().copied().sum();
        let weight = (end - start) as f64;
        *out = if weight > 0.0 { sum / weight } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_basic() {
        const N: usize = 8;
        let mut buf = SpscRingBuffer::<N>::new();
        for i in 0..4i32 {
            assert!(buf.push(i).is_ok());
        }
        for i in 0..4i32 {
            assert_eq!(buf.pop(), Some(i));
        }
        assert_eq!(buf.pop(), None);
    }

    #[test]
    fn bartlett_projection() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 3];
        bartlett_projector(&input, &mut out);
        assert!((out[0] - 1.5).abs() < 1e-12);
        assert!((out[1] - 3.5).abs() < 1e-12);
        assert!((out[2] - 5.5).abs() < 1e-12);
    }
}
