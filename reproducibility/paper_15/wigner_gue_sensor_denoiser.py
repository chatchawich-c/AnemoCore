#!/usr/bin/env python3
"""Paper 15 Supplement: Wigner-GUE RMT Level-Spacing Eigenvalue Filter for 237 SCADA Sensors."""

import numpy as np


def wigner_gue_pdf(s: np.ndarray) -> np.ndarray:
    """Wigner surmise (beta=2, GUE): p(s) = (32 / pi^2) * s^2 * exp(-4 * s^2 / pi)."""
    return (32.0 / (np.pi**2)) * (s**2) * np.exp(-4.0 * (s**2) / np.pi)


def filter_bulk_eigenvalues_marchenko_pastur(corr: np.ndarray, n_samples: int) -> tuple[np.ndarray, float]:
    """Separates signal modes above lambda_plus from RMT noise bulk."""
    p_dim = corr.shape[0]
    q = p_dim / float(max(n_samples, 1))
    lambda_plus = (1.0 + np.sqrt(q)) ** 2
    eigvals = np.sort(np.linalg.eigvalsh(corr))[::-1]
    signal_modes = eigvals[eigvals > lambda_plus]
    return signal_modes, float(lambda_plus)


if __name__ == "__main__":
    s_grid = np.linspace(0.0, 3.0, 31)
    pdf = wigner_gue_pdf(s_grid)
    rng = np.random.default_rng(15)
    X = rng.normal(size=(2000, 237))
    modes, lp = filter_bulk_eigenvalues_marchenko_pastur(np.corrcoef(X, rowvar=False), 2000)
    print(f"P15 Wigner-GUE peak PDF={np.max(pdf):.4f} | MP bound={lp:.4f} | spikes={len(modes)}")
