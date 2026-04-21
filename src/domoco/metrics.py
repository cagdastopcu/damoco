"""Correlation, norm, and directionality metrics for coupling analysis."""

from __future__ import annotations

import warnings

import numpy as np

from ._utils import _trapz2


def co_dirin(N1, N2, omeg1, omeg2):
    """Compute directionality index from coupling norms and autonomous frequencies."""
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
    """Compute directionality index from Fourier coefficients via partial derivatives."""
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
    """Compute correlation and normalized difference between two grid coupling functions."""
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
    """Compute correlation of two coupling functions represented by Fourier coefficients."""
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
    """Compute correlation of two coupling functions given on a grid."""
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
    """Maximize Fourier-domain coupling-function correlation over phase shifts."""
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
    """Maximize grid-domain coupling-function correlation over phase shifts."""
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
    """Return norm of Fourier-represented coupling function and its constant term."""
    q = np.asarray(Qcoef, dtype=complex).copy()
    N = (q.shape[0] - 1) // 2
    omega = float(np.real(q[N, N]))
    q[N, N] = 0.0
    Nrmq = np.sqrt(_trapz2(np.abs(q) ** 2))
    return float(Nrmq), omega


def co_gnorm(q):
    """Return norm and mean value of a coupling function given on a grid."""
    qq = np.asarray(q, dtype=float)
    ngrid = qq.shape[0]
    ng1_2 = (ngrid - 1) * (ngrid - 1)
    omega = float(np.mean(qq))
    qm = qq - omega
    Nrmq = np.sqrt(_trapz2(qm * qm) / ng1_2)
    return float(Nrmq), omega
