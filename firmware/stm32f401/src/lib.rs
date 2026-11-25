// SPDX-License-Identifier: MIT OR Apache-2.0
//! STM32F401 firmware crate (no_std).

#![cfg_attr(not(test), no_std)]

pub mod fixed_point_ring_buffer;
pub mod fault_alarm_evaluator;
pub mod edge_pipeline;

pub use fixed_point_ring_buffer::FixedPointRingBuffer;
pub use fault_alarm_evaluator::FaultAlarmEvaluator;
pub use edge_pipeline::EdgePipeline;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_pipeline_process() {
        // Create a dummy DMA frame with some sample values.
        let frame: [i32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let pipeline = EdgePipeline::new();
        let (buffer, evaluator) = pipeline.process_dma_frame(&frame);
        // Verify buffer contains the samples (order preserved).
        let collected: Vec<i32> = buffer.iter().collect();
        assert_eq!(collected, frame);
        // Simple alarm check: no sample exceeds threshold 1000.
        assert!(!evaluator.evaluate(500));
        assert!(!evaluator.evaluate(0));
    }
}
