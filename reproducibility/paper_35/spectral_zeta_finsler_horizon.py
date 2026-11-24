#!/usr/bin/env python3
"""Paper 35 Supplement (Gap 3): Spectral Zeta Determinant det_zeta(L_F) & Finsler Trapping Horizon."""

import numpy as np


def spectral_zeta_log_det(laplacian: np.ndarray, tol: float = 1e-10) -> float:
    """Computes log(det_zeta L_F) = sum_{lambda_i > tol} ln(lambda_i)."""
    eigs = np.linalg.eigvalsh(laplacian)
    pos = eigs[eigs > tol]
    return float(np.sum(np.log(pos)))


def finsler_trapping_horizon(u_wake: np.ndarray, asym_beta: float, u_thresh: float = 3.2) -> np.ndarray:
    """Finsler Randers-type metric anisotropy triggers aerodynamic wake trapping below u_thresh."""
    effective_speed = u_wake * (1.0 - np.clip(asym_beta, 0.0, 0.6))
    return effective_speed <= u_thresh


if __name__ == "__main__":
    L = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
    print(f"P35 log det_zeta(L_F) = {spectral_zeta_log_det(L):.6f}")
