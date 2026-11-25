// SPDX-License-Identifier: MIT OR Apache-2.0
//! Edge pipeline for processing DMA frames in the STM32F401 firmware.

use super::fixed_point_ring_buffer::FixedPointRingBuffer;
use super::fault_alarm_evaluator::FaultAlarmEvaluator;

/// Number of samples the ring buffer can hold. Adjust as needed for the MCU.
pub const BUFFER_SIZE: usize = 64;

/// EdgePipeline orchestrates the processing of a DMA frame using the core
/// mathematical kernels from `anemocore`. For this minimal implementation we
/// simply store the samples in a zero‑heap ring buffer and provide a fault alarm
/// evaluator. More sophisticated processing (e.g., calling `quasicrystal_wpt`,
/// `signal_wpt`, `kam_vsg`) can be added later without changing the public API.
pub struct EdgePipeline;

impl EdgePipeline {
    /// Create a new pipeline instance.
    pub const fn new() -> Self {
        Self {}
    }

    /// Process a DMA frame of signed 32‑bit samples.
    ///
    /// Returns a tuple containing:
    ///   1. A `FixedPointRingBuffer` populated with the frame samples.
    ///   2. A default `FaultAlarmEvaluator`.
    ///
    /// The generic buffer size is fixed at `BUFFER_SIZE`. If the frame is larger
    /// than this size, the oldest samples will be overwritten (ring‑buffer
    /// semantics).
    pub fn process_dma_frame(
        &self,
        frame: &[i32],
    ) -> (FixedPointRingBuffer<BUFFER_SIZE>, FaultAlarmEvaluator) {
        let mut buffer = FixedPointRingBuffer::<BUFFER_SIZE>::new();
        for &sample in frame {
            buffer.push(sample);
        }
        let evaluator = FaultAlarmEvaluator::new();
        (buffer, evaluator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_dma_frame_basic() {
        let pipeline = EdgePipeline::new();
        let frame = [1, 2, 3, 4, 5];
        let (buf, eval) = pipeline.process_dma_frame(&frame);
        let collected: Vec<i32> = buf.iter().collect();
        assert_eq!(collected, frame);
        assert!(!eval.evaluate(0));
    }
}
