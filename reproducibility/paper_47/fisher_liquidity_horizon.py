#!/usr/bin/env python3
"""Paper 47 Supplement (Gap 6): Fisher Information Liquidity Event Horizon det(J) -> 0."""

import numpy as np


def fisher_liquidity_horizon(score_samples: np.ndarray, eps_det: float = 1e-4) -> tuple[float, bool]:
    """Empirical Fisher Information Matrix determinant det(J) -> 0 signals liquidity freeze."""
    fim = (score_samples.T @ score_samples) / float(len(score_samples))
    det_j = float(np.linalg.det(fim))
    return det_j, det_j <= eps_det


if __name__ == "__main__":
    rng = np.random.default_rng(47)
    healthy = rng.normal(scale=1.0, size=(200, 2))
    collinear = np.column_stack([healthy[:, 0], 1e-4 * healthy[:, 1]])
    print("P47 healthy:", fisher_liquidity_horizon(healthy))
    print("P47 horizon freeze:", fisher_liquidity_horizon(collinear))
