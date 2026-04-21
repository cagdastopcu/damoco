import numpy as np

from domoco import (
    co_cor_diff,
    co_dirin,
    co_dirpar,
    co_fcfcormax,
    co_fcfcor,
    co_fnorm,
    co_gcfcor,
    co_gcfcormax,
    co_gnorm,
    co_maxsync,
    co_maxsync3,
    co_sync,
    co_sync3,
)


def test_sync_indices():
    t = np.linspace(0, 40 * np.pi, 5000)
    p1 = t
    p2 = 2 * t
    s = co_sync(p1, p2, 2, 1)
    assert s > 0.99
    M, maxind, n, m = co_maxsync(p1, p2, 3)
    assert M.shape == (3, 3)
    assert maxind > 0.99
    assert (n, m) == (2, 1)


def test_sync3_indices():
    t = np.linspace(0, 30 * np.pi, 6000)
    p1 = t
    p2 = 2 * t
    p3 = -t
    s = co_sync3(p1, p2, p3, 1, 1, 3)
    assert s > 0.99
    M, maxind, n, m, l = co_maxsync3(p1, p2, p3, 2)
    assert M.shape == (3, 5, 5)
    assert maxind > 0.99
    assert n + 2 * m - l == 0


def test_correlations_and_norms():
    n = 64
    x = np.linspace(0, 2 * np.pi, n)
    X, Y = np.meshgrid(x, x)
    q1 = np.sin(X) + 0.3 * np.cos(Y)
    q2 = q1.copy()
    cor, diff = co_cor_diff(q1, q2)
    assert cor > 0.999
    assert diff < 1e-8
    assert co_gcfcor(q1, q2) > 0.999
    nrmq, omega = co_gnorm(q1)
    assert nrmq > 0
    assert np.isfinite(omega)


def test_fourier_correlation_max_shift():
    N = 1
    S = 2 * N + 1
    q1 = np.zeros((S, S), dtype=complex)
    q1[N, N] = 0.7
    q1[N + 1, N] = 0.4 - 0.2j
    q1[N - 1, N] = np.conjugate(q1[N + 1, N])
    q1[N, N + 1] = 0.3 + 0.1j
    q1[N, N - 1] = np.conjugate(q1[N, N + 1])
    d1 = 0.7
    d2 = 1.1
    q2 = np.zeros_like(q1)
    for n in range(-N, N + 1):
        for m in range(-N, N + 1):
            q2[n + N, m + N] = np.exp(-1j * (n * d1 + m * d2)) * q1[n + N, m + N]

    cor, _, _, q2s = co_fcfcormax(q1, q2, ngrid=180)
    assert cor > 0.99
    assert co_fcfcor(q1, q2s) > 0.99
    nrmq, omega = co_fnorm(q1)
    assert nrmq > 0
    assert np.isfinite(omega)


def test_grid_correlation_max_shift():
    n = 40
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x)
    q1 = np.sin(X) + np.cos(Y)
    q1p = np.zeros((n + 1, n + 1))
    q1p[:-1, :-1] = q1
    q1p[-1, :-1] = q1[0, :]
    q1p[:-1, -1] = q1[:, 0]
    q1p[-1, -1] = q1[0, 0]
    q2p = np.roll(np.roll(q1p, 7, axis=0), 9, axis=1)
    cor, _, _, _ = co_gcfcormax(q1p, q2p)
    assert cor > 0.98


def test_directionality_indices():
    d = co_dirin(0.2, 0.4, 2.0, 2.0)
    assert d > 0
    F1 = np.zeros((3, 3), dtype=complex)
    F2 = np.zeros((3, 3), dtype=complex)
    F1[1, 2] = 0.2
    F2[1, 2] = 0.5
    d2 = co_dirpar(F1, F2)
    assert d2 > 0
