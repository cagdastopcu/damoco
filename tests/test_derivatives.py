import numpy as np
from scipy.interpolate import RegularGridInterpolator

from domoco import co_phidot1, co_phidot2, co_phidot3, co_prciter, co_resid_decomp


def test_phidot_family():
    fs = 200.0
    t = np.arange(0, 10, 1 / fs)
    p1 = 2.0 * t
    p2 = 1.5 * t + 0.3
    p3 = 0.7 * t - 0.1

    d1, p1c = co_phidot1(p1, fs)
    assert len(d1) == len(p1c)
    assert np.mean(np.abs(d1 - 2.0)) < 1e-2

    d1b, d2b, p1b, p2b = co_phidot2(p1, p2, fs)
    assert len(d1b) == len(d2b) == len(p1b) == len(p2b)
    assert np.mean(np.abs(d1b - 2.0)) < 1e-2
    assert np.mean(np.abs(d2b - 1.5)) < 1e-2

    d1c, d2c, d3c, *_ = co_phidot3(p1, p2, p3, fs)
    assert np.mean(np.abs(d1c - 2.0)) < 1e-2
    assert np.mean(np.abs(d2c - 1.5)) < 1e-2
    assert np.mean(np.abs(d3c - 0.7)) < 1e-2


def test_residual_decomposition_near_zero():
    n = 50
    arg = np.linspace(0, 2 * np.pi, n)
    X, Y = np.meshgrid(arg, arg)
    q = 1.1 + 0.2 * np.sin(X) + 0.1 * np.cos(Y)
    interp = RegularGridInterpolator((arg, arg), q, bounds_error=False, fill_value=None)

    m = 5000
    phi1 = np.random.default_rng(0).uniform(0, 2 * np.pi, m)
    phi2 = np.random.default_rng(1).uniform(0, 2 * np.pi, m)
    dphi = interp(np.column_stack([phi1, phi2]))

    _, _, std_resid, _, resid = co_resid_decomp(dphi, phi1, phi2, q)
    assert std_resid < 1e-6
    assert np.std(resid) < 1e-6


def test_prciter_shapes():
    n = 64
    x = np.linspace(0, 2 * np.pi, n)
    Q = 0.5 + np.outer(np.sin(x), np.cos(x))
    Z, H, om_opt, min_err = co_prciter(Q, n, fignum=0)
    assert Z.shape == (n,)
    assert H.shape == (n,)
    assert np.isfinite(om_opt)
    assert np.isfinite(min_err)
