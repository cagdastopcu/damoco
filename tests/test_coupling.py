import numpy as np

from domoco import (
    co_fcplfct1,
    co_fcplfct2,
    co_fcpltri,
    co_kcplfct1,
    co_kcplfct2,
    co_nettri,
    co_tricplfan,
)


def test_fcplfct1_and_fcplfct2_recover_grid_model():
    n = 40
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p1 = np.repeat(a, n)
    p2 = np.tile(a, n)
    d1 = 1.0 + 0.3 * np.sin(p1) + 0.2 * np.cos(p2) + 0.1 * np.sin(p1 - p2)
    d2 = 1.2 + 0.25 * np.cos(p1) + 0.15 * np.sin(p2)

    _, q1 = co_fcplfct1(p1, p2, d1, N=1, ngrid=n)
    _, _, q1b, q2b = co_fcplfct2(p1, p2, d1, d2, N=1, ngrid=n)

    arg = np.linspace(0, 2 * np.pi, n)
    X, Y = np.meshgrid(arg, arg)
    q1_true = 1.0 + 0.3 * np.sin(X) + 0.2 * np.cos(Y) + 0.1 * np.sin(X - Y)
    q2_true = 1.2 + 0.25 * np.cos(X) + 0.15 * np.sin(Y)
    assert np.mean(np.abs(q1 - q1_true.T)) < 0.05
    assert np.mean(np.abs(q1b - q1_true.T)) < 0.05
    err_q2 = min(np.mean(np.abs(q2b - q2_true)), np.mean(np.abs(q2b - q2_true.T)))
    assert err_q2 < 0.05


def test_fcpltri_tricplfan_and_nettri():
    rng = np.random.default_rng(42)
    m = 3000
    p1 = rng.uniform(0, 10 * np.pi, m)
    p2 = rng.uniform(0, 10 * np.pi, m)
    p3 = rng.uniform(0, 10 * np.pi, m)
    d1 = 1.0 + 0.4 * np.sin(p2) + 0.15 * np.sin(p3) + 0.2 * np.sin(p2 + p3)
    d2 = 1.2 + 0.6 * np.sin(p1)
    d3 = 0.9 + 0.5 * np.sin(p1)

    Q1, Q2, Q3 = co_fcpltri(p1, p2, p3, d1, d2, d3, N=1)
    COUP, NORM, OMEGA = co_tricplfan(Q1, Q2, Q3, meth=1, thresh=0)
    assert COUP.shape == (3, 3)
    assert NORM.shape == (3,)
    assert OMEGA.shape == (3,)
    assert COUP[1, 0] > COUP[1, 2]
    assert COUP[2, 0] > COUP[2, 1]

    PHI = np.vstack([p1, p2, p3])
    PHI_dot = np.vstack([d1, d2, d3])
    Cn = co_nettri(PHI, PHI_dot, N=1, meth=1, thresh=0)
    assert Cn.shape == (3, 3)
    assert Cn[1, 0] > Cn[1, 2]
    assert Cn[2, 0] > Cn[2, 1]


def test_kernel_coupling_functions_periodic():
    rng = np.random.default_rng(7)
    m = 1000
    p1 = rng.uniform(0, 2 * np.pi, m)
    p2 = rng.uniform(0, 2 * np.pi, m)
    d1 = 1.0 + 0.4 * np.sin(p1) + 0.3 * np.cos(p2)
    d2 = 0.7 + 0.2 * np.cos(p1) + 0.1 * np.sin(p2)

    q1 = co_kcplfct1(p1, p2, d1, ngrid=40, fignum=0)
    q1b, q2b = co_kcplfct2(p1, p2, d1, d2, ngrid=40)

    assert q1.shape == (40, 40)
    assert q1b.shape == (40, 40)
    assert q2b.shape == (40, 40)
    assert np.allclose(q1[-1, :], q1[0, :])
    assert np.allclose(q1[:, -1], q1[:, 0])
    assert np.allclose(q1b[-1, :], q1b[0, :])
    assert np.allclose(q2b[-1, :], q2b[0, :])
