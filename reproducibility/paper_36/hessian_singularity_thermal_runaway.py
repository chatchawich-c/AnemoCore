#!/usr/bin/env python3
"""Paper 36 Supplement (Gap 5): Hessian Singularity det(H_G) <= 0 for Bearing Thermal Runaway."""

import numpy as np


def check_hessian_singularity(hessian_g: np.ndarray) -> tuple[float, bool]:
    """Thermal runaway horizon occurs when det(H_G) <= 0 (loss of convex free-energy stability)."""
    det_h = float(np.linalg.det(hessian_g))
    return det_h, det_h <= 0.0


if __name__ == "__main__":
    H_stable = np.array([[4.2, -0.5], [-0.5, 3.8]])
    H_runaway = np.array([[1.2, 2.5], [2.5, 1.8]])
    print("P36 stable:", check_hessian_singularity(H_stable))
    print("P36 runaway:", check_hessian_singularity(H_runaway))
