// SPDX-License-Identifier: MIT OR Apache-2.0
//! Zero‑heap ring buffer for ADC samples (i32) used in STM32F401 firmware.

/// Fixed‑size ring buffer without heap allocation.
/// `N` is the capacity (must be > 0).

#[derive(Clone, Copy, Debug)]
pub struct FixedPointRingBuffer<const N: usize> {
    data: [i32; N],
    head: usize, // points to next write position
    len: usize,
}
impl<const N: usize> Default for FixedPointRingBuffer<N> {
    fn default() -> Self { Self::new() }
}

impl<const N: usize> FixedPointRingBuffer<N> {
    /// Create a new empty buffer.
    pub const fn new() -> Self {
        Self {
            data: [0; N],
            head: 0,
            len: 0,
        }
    }

    /// Push a new sample, overwriting the oldest when full.
    pub fn push(&mut self, value: i32) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
    }

    /// Return the number of valid samples currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get an iterator over the samples in chronological order (oldest → newest).
    pub fn iter(&self) -> impl Iterator<Item = i32> + '_ {
        (0..self.len).map(move |i| {
            let idx = (self.head + N - self.len + i) % N;
            self.data[idx]
        })
    }
}
