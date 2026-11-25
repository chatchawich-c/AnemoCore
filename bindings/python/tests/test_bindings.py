"""End-to-end Python binding parity and error-handling tests for `ecowind_core`."""

import math
import unittest

import ecowind_core as ew
import numpy as np


class TestEcowindPythonBindings(unittest.TestCase):
    def test_exported_constants(self) -> None:
        self.assertAlmostEqual(ew.PHI, (1.0 + math.sqrt(5.0)) / 2.0, places=12)
        self.assertAlmostEqual(ew.PHI_INV, (math.sqrt(5.0) - 1.0) / 2.0, places=12)
        self.assertAlmostEqual(ew.FRAME_BOUND_A, 2.482, places=6)
        self.assertAlmostEqual(ew.FRAME_BOUND_B, 2.492, places=6)

    def test_kam_vsg_step(self) -> None:
        p_wt_mw, freq_ratio, tj_c = ew.kam_vsg_step(0.005, -0.02, -0.01, 0.001)
        self.assertTrue(math.isfinite(p_wt_mw) and p_wt_mw > 0.0)
        # Golden-ratio KAM detuned control enforces omega_ctrl / omega_n = phi^-1
        self.assertAlmostEqual(freq_ratio, ew.PHI_INV, places=10)
        self.assertTrue(math.isfinite(tj_c))

    def test_wpt_db4_indicators_and_validation(self) -> None:
        # Exact Paper 33 signal fixture from signal_wpt::tests::p33_db4_wpt_matches_numpy_parity
        t = np.arange(64, dtype=np.float64) / 64.0
        sig_p33 = np.sin(8.0 * np.pi * t) + 0.3 * np.cos(24.0 * np.pi * t)
        p_low_mid, p_high, kappa_high, h3_renyi = ew.wpt_db4_indicators(sig_p33)
        self.assertAlmostEqual(p_low_mid, 0.2966570345, places=8)
        self.assertAlmostEqual(p_high, 0.0007641162, places=8)
        self.assertAlmostEqual(h3_renyi, 0.7740334671, places=8)
        self.assertGreater(kappa_high, 0.0)

        # 1024-sample mixed signal check
        x = np.sin(np.linspace(0.0, 8.0 * np.pi, 1024, endpoint=False))
        p_lm, p_hi, _, h3 = ew.wpt_db4_indicators(x)
        self.assertTrue(0.0 < p_lm + p_hi <= 1.0)
        self.assertGreater(p_lm, p_hi)
        self.assertGreaterEqual(h3, 0.0)

        # Reject non-contiguous slice cleanly
        with self.assertRaises(ValueError):
            ew.wpt_db4_indicators(x[::2])

        # Reject length not divisible by 8
        with self.assertRaises(ValueError):
            ew.wpt_db4_indicators(np.zeros(10, dtype=np.float64))

    def test_quasicrystal_shift_add(self) -> None:
        alternating = np.array([100 if i % 2 == 0 else -100 for i in range(64)], dtype=np.int32)
        energies = ew.quasicrystal_shift_add_energies(alternating)
        self.assertEqual(len(energies), 8)
        self.assertGreater(energies[7], energies[0])

        lo, hi = ew.quasicrystal_shift_add_split(alternating)
        self.assertEqual(len(lo), 32)
        self.assertEqual(len(hi), 32)

    def test_fib_qmc_and_golden_stopping_parity(self) -> None:
        r_cap, s_load, g_val, dtf = ew.fib_qmc_limit_state([0.1, 0.8, 0.3, 0.6, 0.9, 0.2])
        self.assertAlmostEqual(r_cap, 68.6107392823, places=6)
        self.assertAlmostEqual(s_load, 30.7512613229, places=6)
        self.assertAlmostEqual(g_val, 25.0594779594, places=6)
        self.assertAlmostEqual(dtf, 0.8588422274, places=6)

        s_star, iters = ew.fib_golden_stopping(0.35)
        self.assertAlmostEqual(s_star, 73.3189833311, places=5)
        self.assertEqual(iters, 36)

        pf, mean_dtf = ew.fib_qmc_sweep_o1([0.13, 0.47, 0.29, 0.81, 0.61, 0.07])
        self.assertAlmostEqual(pf, 0.1497249462, places=6)
        self.assertAlmostEqual(mean_dtf, 0.4297877089, places=6)

    def test_market_microstructure(self) -> None:
        rng = np.random.default_rng(42)
        prices = 65.0 + np.cumsum(rng.normal(0.0, 0.6, size=120))
        spread = ew.market_roll_spread(prices, 20)
        self.assertEqual(len(spread), len(prices))
        self.assertTrue(np.all(np.isfinite(spread)))
        self.assertTrue(np.all(spread >= 0.0))

        rho = ew.market_lob_density(0.618)
        self.assertGreater(rho, 0.0)

    def test_phyllotaxis_aerothermal_snapshot(self) -> None:
        mean_p_mw, max_t_c, ctd = ew.phyllotaxis_aerothermal_snapshot(25, 9.5, 250.0, 15.0, 600.0)
        self.assertGreater(mean_p_mw, 0.0)
        self.assertGreater(max_t_c, 15.0)
        self.assertGreater(ctd, 0.0)

    def test_pi_emeans_and_bartlett_projection(self) -> None:
        alpha = ew.pi_emeans_alpha(1.2, 0.5, 8, 1024, 100_000)
        self.assertAlmostEqual(alpha, 1.2 * math.sqrt(8 * 1024 / 100_000), places=12)
        clean_a = ew.pi_emeans_cyber_alpha(0.34, 4.2, 0.05)
        fdia_a = ew.pi_emeans_cyber_alpha(0.34, 42.0, 0.05)
        self.assertGreater(clean_a, 0.34)
        self.assertLess(fdia_a, clean_a * 0.002)

        p_dim, k_dim = 237, 12
        mu = np.full(p_dim, 10.0, dtype=np.float64)
        inv_std = np.full(p_dim, 0.5, dtype=np.float64)
        x = np.array([10.0 + (j % 3) * 2.0 for j in range(p_dim)], dtype=np.float64)
        w_fs = np.array([((k + 1) * 0.01) for k in range(k_dim) for _ in range(p_dim)], dtype=np.float64)
        latent = ew.bartlett_factor_project_vec(x, mu, inv_std, w_fs, k_dim)
        self.assertEqual(len(latent), k_dim)
        self.assertAlmostEqual(float(latent[0]), 2.37, places=10)
        self.assertAlmostEqual(float(latent[11]), 28.44, places=10)

    def test_fdia_jacobi_and_weyl_lattice(self) -> None:
        res_clean, bad_clean = ew.stealth_fdia_wind_check(12.56, 12.31, 1.5)
        self.assertFalse(bad_clean)
        self.assertAlmostEqual(res_clean, 0.25, places=10)
        res_fdia, bad_fdia = ew.stealth_fdia_wind_check(4.20, 10.80, 1.5)
        self.assertTrue(bad_fdia)
        self.assertAlmostEqual(res_fdia, 6.60, places=10)

        breached, t_col, omega2, kappa_eff = ew.jacobi_horizon_guard(0.25, 0.5, 0.1)
        self.assertFalse(breached)
        self.assertAlmostEqual(t_col, 2.0 * math.pi, places=9)
        self.assertAlmostEqual(omega2 / 40.75, ew.PHI_INV, places=10)
        self.assertTrue(math.isfinite(kappa_eff))

        breached_bad, _, _, _ = ew.jacobi_horizon_guard(12.0, 1e-5, 0.2)
        self.assertTrue(breached_bad)

        lags = ew.weyl_convergent_lags(8)
        self.assertEqual(lags, [1, 1, 2, 3, 5, 8, 13, 21])
        pts = ew.rank1_lattice_qmc(8, 5, 0.0)
        self.assertEqual(len(pts), 8)
        self.assertAlmostEqual(float(pts[1]), 0.625, places=12)


if __name__ == "__main__":
    unittest.main()
