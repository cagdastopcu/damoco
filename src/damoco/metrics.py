"""Correlation, norm, and directionality metrics for coupling analysis.

The definitions in this module mirror DAMOCO formulas on Fourier coefficients
and periodic phase grids.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._utils import _trapz2


def co_dirin(N1, N2, omeg1, omeg2):
    r"""Compute directionality index from coupling norms and autonomous frequencies.

    Defines normalized coupling strengths :math:`c_1=N_1/\omega_1`,
    :math:`c_2=N_2/\omega_2` and returns
    :math:`D=(c_2-c_1)/(c_1+c_2)`.
    """
    c1 = N1 / omeg1
    c2 = N2 / omeg2
    if c1 + c2 < 0.02:
        warnings.warn(
            "The coupling is very weak or the systems are not coupled; "
            "directionality index may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float((c2 - c1) / (c1 + c2))


def co_dirpar(Fcoef1, Fcoef2):
    r"""Compute directionality from external-phase derivative norms.

    For each oscillator, the external contribution is quantified by
    :math:`\sqrt{\sum_{n,m} m^2 |F_{n,m}|^2}` and combined into
    :math:`D=(n_2-n_1)/(n_1+n_2)`.
    """
    F1 = np.asarray(Fcoef1)
    F2 = np.asarray(Fcoef2)
    or_ = (F1.shape[0] - 1) // 2
    m = np.arange(-or_, or_ + 1)
    m2 = (m * m)[None, :]
    NP1 = np.sum(m2 * np.abs(F1) ** 2)
    NP2 = np.sum(m2 * np.abs(F2) ** 2)
    nrm1 = np.sqrt(NP1)
    nrm2 = np.sqrt(NP2)
    if nrm1 + nrm2 < 0.02:
        warnings.warn(
            "The coupling is very weak or the systems are not coupled; "
            "directionality index may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float((nrm2 - nrm1) / (nrm1 + nrm2))


def co_cor_diff(Q1, Q2):
    r"""Compute grid-function correlation and normalized difference.

    After removing means and periodic duplicate boundary points, computes
    :math:`\mathrm{COR}=\frac{\langle Q_1,Q_2\rangle}{\|Q_1\|\|Q_2\|}` and
    :math:`\mathrm{DIFF}=\frac{\|Q_1-Q_2\|}{\|Q_1\|+\|Q_2\|}` using trapezoidal
    integration on the 2D grid.
    """
    q1 = np.asarray(Q1, dtype=float)
    q2 = np.asarray(Q2, dtype=float)
    q1 = q1 - np.mean(q1)
    q2 = q2 - np.mean(q2)
    q1 = q1[:-1, :-1]
    q2 = q2[:-1, :-1]
    autCor1 = np.sqrt(_trapz2(q1**2))
    autCor2 = np.sqrt(_trapz2(q2**2))
    COR = _trapz2(q1 * q2) / (autCor1 * autCor2)
    DIFF = np.sqrt(_trapz2((q1 - q2) ** 2)) / (autCor1 + autCor2)
    return float(COR), float(DIFF)


def co_fcfcor(Qcoef1, Qcoef2):
    r"""Compute Fourier-domain coupling-function correlation.

    Removes the autonomous term :math:`Q_{0,0}` and evaluates the DAMOCO
    Fourier-space correlation with conjugate-reversed index alignment.
    """
    q1 = np.asarray(Qcoef1, dtype=complex).copy()
    q2 = np.asarray(Qcoef2, dtype=complex).copy()
    S = q1.shape[0]
    N = (S - 1) // 2
    q1[N, N] = 0.0
    q2[N, N] = 0.0
    autCor1 = np.real(_trapz2(q1 * np.conjugate(q1)))
    autCor2 = np.real(_trapz2(q2 * np.conjugate(q2)))
    q2r = q2[::-1, ::-1]
    COR = _trapz2(q1 * q2r) / np.sqrt(autCor2 * autCor1)
    return float(np.real(COR))


def co_gcfcor(q1, q2):
    r"""Compute correlation of two coupling functions sampled on a periodic grid.

    Means are removed before evaluating the normalized 2D inner product.
    """
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    a = a[:-1, :-1]
    b = b[:-1, :-1]
    a = a - np.mean(a)
    b = b - np.mean(b)
    autCor1 = _trapz2(a * a)
    autCor2 = _trapz2(b * b)
    COR = _trapz2(a * b) / np.sqrt(autCor1 * autCor2)
    return float(COR)


def co_fcfcormax(Qcoef1, Qcoef2, ngrid=100):
    r"""Maximize Fourier-domain correlation over relative phase shifts.

    Evaluates
    :math:`C(\Delta_1,\Delta_2)` on a uniform :math:`ngrid\times ngrid` shift grid
    and returns maximal correlation together with optimal shifts and shifted
    coefficients :math:`Q^{\mathrm{shift}}_{n,m}=e^{i(n\Delta_1+m\Delta_2)}Q_{n,m}`.
    """
    q1 = np.asarray(Qcoef1, dtype=complex).copy()
    q2 = np.asarray(Qcoef2, dtype=complex).copy()
    S = q1.shape[0]
    N = (S - 1) // 2
    q1[N, N] = 0.0
    q2[N, N] = 0.0

    arg = 2.0 * np.pi * np.arange(ngrid) / ngrid
    X, Y = np.meshgrid(arg, arg)
    M = np.zeros_like(X, dtype=complex)
    autCor1 = np.real(np.sum(q1 * np.conjugate(q1)))
    autCor2 = np.real(np.sum(q2 * np.conjugate(q2)))

    for n in range(-N, N + 1):
        for m in range(-N, N + 1):
            M += (
                np.exp(1j * (-n * X - m * Y))
                * q2[-n + N, -m + N]
                * q1[n + N, m + N]
            )
    Cor12 = np.real(M) / np.sqrt(autCor2 * autCor1)
    idx = np.unravel_index(np.argmax(Cor12), Cor12.shape)
    COR = float(Cor12[idx])
    Delta1 = float(X[idx])
    Delta2 = float(Y[idx])

    n = np.arange(-N, N + 1)[:, None]
    m = np.arange(-N, N + 1)[None, :]
    shift = np.exp(1j * (n * Delta1 + m * Delta2))
    Qcoef2_shift = q2 * shift

    return COR, Delta1, Delta2, Qcoef2_shift


def co_gcfcormax(q1, q2):
    r"""Maximize grid-domain correlation over periodic shifts.

    Uses circular shifts of the second function over all grid offsets and returns
    maximal correlation, best self/external phase shifts, and the shifted grid.
    """
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    a = a[:-1, :-1]
    b = b[:-1, :-1]
    a = a - np.mean(a)
    b = b - np.mean(b)
    ngrid = a.shape[0]
    Q2 = np.block([[b, b], [b, b]])

    autCor1 = _trapz2(a * a)
    autCor2 = _trapz2(b * b)

    m1 = 0
    m2 = 0
    COR = -np.inf
    q2_shift = b.copy()
    for n in range(ngrid):
        for m in range(ngrid):
            A = Q2[ngrid - n : 2 * ngrid - n, ngrid - m : 2 * ngrid - m]
            cur = _trapz2(a * A) / np.sqrt(autCor1 * autCor2)
            if cur > COR:
                m1 = n
                m2 = m
                COR = cur
                q2_shift = A

    Delta_ext = float(np.mod(2 * np.pi - 2 * np.pi * m2 / ngrid, 2 * np.pi))
    Delta_self = float(np.mod(2 * np.pi - 2 * np.pi * m1 / ngrid, 2 * np.pi))

    q2s = np.zeros((ngrid + 1, ngrid + 1), dtype=float)
    q2s[:-1, :-1] = q2_shift
    q2s[-1, :-1] = q2_shift[0, :]
    q2s[:-1, -1] = q2_shift[:, 0]
    q2s[-1, -1] = q2_shift[0, 0]
    return float(COR), Delta_self, Delta_ext, q2s


def co_fnorm(Qcoef):
    r"""Return Fourier coupling norm and autonomous frequency estimate.

    The autonomous frequency estimate is :math:`\omega=\Re(Q_{0,0})`.
    The coupling norm is computed after removing :math:`Q_{0,0}`.
    """
    q = np.asarray(Qcoef, dtype=complex).copy()
    N = (q.shape[0] - 1) // 2
    omega = float(np.real(q[N, N]))
    q[N, N] = 0.0
    Nrmq = np.sqrt(_trapz2(np.abs(q) ** 2))
    return float(Nrmq), omega


def co_gnorm(q):
    r"""Return grid coupling norm and mean frequency term.

    Computes :math:`\omega=\langle q\rangle` and
    :math:`\|q-\omega\|` with DAMOCO grid normalization by
    :math:`(ngrid-1)^2`.
    """
    qq = np.asarray(q, dtype=float)
    ngrid = qq.shape[0]
    ng1_2 = (ngrid - 1) * (ngrid - 1)
    omega = float(np.mean(qq))
    qm = qq - omega
    Nrmq = np.sqrt(_trapz2(qm * qm) / ng1_2)
    return float(Nrmq), omega
