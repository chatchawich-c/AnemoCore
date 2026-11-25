import math
from anemocore._core import (
    py_arrhenius_oil_life,
    py_heat_diffuse,
    py_zddp_depletion,
)


def test_py_heat_diffuse_line_graph_and_conservation():
    initial = [1.0, 0.0, 0.0]
    adj = [[1], [0, 2], [1]]
    out_1 = py_heat_diffuse(initial, adj, 0.1, 1)
    assert len(out_1) == 3
    assert math.isclose(out_1[0], 0.9, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(out_1[1], 0.1, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(out_1[2], 0.0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(sum(out_1), sum(initial), rel_tol=1e-12, abs_tol=1e-12)

    out_5 = py_heat_diffuse(initial, adj, 0.1, 5)
    assert math.isclose(sum(out_5), sum(initial), rel_tol=1e-12, abs_tol=1e-12)


def test_py_arrhenius_oil_life_monotonic_temperature():
    ref_life = 10_000.0
    ea = 80_000.0
    at_ref = py_arrhenius_oil_life(70.0, 70.0, ref_life, ea)
    hotter = py_arrhenius_oil_life(90.0, 70.0, ref_life, ea)
    cooler = py_arrhenius_oil_life(55.0, 70.0, ref_life, ea)

    assert math.isclose(at_ref, ref_life, rel_tol=1e-12, abs_tol=1e-12)
    assert hotter < at_ref < cooler


def test_py_zddp_depletion_decay_and_temperature():
    c0 = 1200.0
    k_ref = 1e-4
    ea = 75_000.0
    rem_cold = py_zddp_depletion(c0, 60.0, 500.0, k_ref, ea)
    rem_hot = py_zddp_depletion(c0, 100.0, 500.0, k_ref, ea)

    assert 0.0 < rem_hot < rem_cold < c0
