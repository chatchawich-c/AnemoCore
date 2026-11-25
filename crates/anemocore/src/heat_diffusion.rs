#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Graph Laplacian Heat Diffusion (spatial k‑NN)
///
/// The function performs a simple explicit Euler integration of the heat diffusion
/// equation on an undirected graph defined by a neighbour list.  The graph Laplacian
/// is defined as `L = D - A` where `D` is the degree matrix and `A` the adjacency
/// matrix.  The discrete update is
///
/// `v_{t+1} = v_t - dt * L * v_t`
///
/// where `dt` is a non‑negative time step.  The implementation is deterministic
/// and never panics; malformed inputs (e.g. mismatched lengths) result in the
/// original vector being returned unchanged.
///
/// # Arguments
/// * `values` – Initial scalar field on the graph nodes.
/// * `adjacency` – Vector where `adjacency[i]` contains the indices of the
///   neighbours of node `i`.  All neighbour indices must be valid and the graph
///   is assumed undirected (the function does not enforce symmetry).
/// * `dt` – Time step for the explicit Euler integration.  Must be `>= 0.0`.
/// * `steps` – Number of integration steps to perform.
///
/// # Returns
/// A new `Vec<f64>` containing the diffused values after `steps` updates.  The
/// original `values` vector is left untouched.
///
/// # Examples
/// ```ignore
/// let values = vec![1.0, 0.0, 0.0];
/// let adjacency = vec![vec![1], vec![0, 2], vec![1]];
/// let result = heat_diffuse(&values, &adjacency, 0.1, 1);
/// assert!((result[0] - 0.9).abs() < 1e-12);
/// assert!((result[1] - 0.1).abs() < 1e-12);
/// assert!((result[2] - 0.0).abs() < 1e-12);
/// ```
pub fn heat_diffuse(values: &[f64], adjacency: &[Vec<usize>], dt: f64, steps: usize) -> Vec<f64> {
    if dt <= 0.0 || steps == 0 {
        return values.to_vec();
    }
    if values.len() != adjacency.len() {
        return values.to_vec();
    }
    let n = values.len();
    let mut v = values.to_vec();
    for _ in 0..steps {
        // Compute L * v
        let mut lv = vec![0.0_f64; n];
        for i in 0..n {
            let deg = adjacency[i].len() as f64;
            let mut sum_neigh = 0.0_f64;
            for &j in &adjacency[i] {
                if j < n {
                    sum_neigh += v[j];
                }
            }
            lv[i] = deg * v[i] - sum_neigh;
        }
        // Explicit Euler update: v = v - dt * (L * v)
        for i in 0..n {
            v[i] -= dt * lv[i];
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_graph_one_step() {
        let values = vec![1.0_f64, 0.0, 0.0];
        let adjacency = vec![vec![1], vec![0, 2], vec![1]];
        let result = heat_diffuse(&values, &adjacency, 0.1, 1);
        assert!((result[0] - 0.9).abs() < 1e-12);
        assert!((result[1] - 0.1).abs() < 1e-12);
        assert!((result[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_multiple_steps_conservation() {
        let values = vec![1.0, 0.0, 0.0];
        let adjacency = vec![vec![1], vec![0, 2], vec![1]];
        let result = heat_diffuse(&values, &adjacency, 0.1, 10);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mismatched_lengths_returns_original() {
        let values = vec![1.0, 2.0];
        let adjacency = vec![vec![0]]; // length mismatch
        let result = heat_diffuse(&values, &adjacency, 0.1, 1);
        assert_eq!(result, values);
    }
}
