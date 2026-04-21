"""Fourier/kernel coupling-function estimators and network-level analyzers."""

from __future__ import annotations

import warnings

import numpy as np
from matplotlib import pyplot as plt

from ._utils import PI2, _as_1d
from .synchrony import co_maxsync3


def co_fcplfct1(phi1, phi2, dphi1, N, ngrid=100):
    """Estimate one 2D coupling function via Fourier-series regression."""
    p1 = np.unwrap(_as_1d(phi1))
    p2 = np.unwrap(_as_1d(phi2))
    dp1 = _as_1d(dphi1)

    A = np.zeros((4 * N + 1, 4 * N + 1), dtype=complex)
    off = 2 * N
    or21 = 2 * N + 1

    for n in range(-2 * N, 2 * N + 1):
        for m in range(-2 * N, n + 1):
            val = np.mean(np.exp(1j * (n * p1 + m * p2)))
            A[n + off, m + off] = val
            A[-n + off, -m + off] = np.conjugate(val)

    B1 = np.zeros(or21 * or21, dtype=complex)
    C = np.zeros((or21 * or21, or21 * or21), dtype=complex)

    ind = 0
    for n in range(-N, N + 1):
        i1_1 = (n + N) * or21
        for m in range(-N, N + 1):
            i1 = i1_1 + m + N
            i4 = m + off
            tmp = np.exp(-1j * (n * p1 + m * p2))
            B1[ind] = np.mean(dp1 * tmp)
            ind += 1
            for r in range(-N, N + 1):
                i3 = (r + N) * or21 + N
                i2 = (n - r) + off
                for s in range(-N, N + 1):
                    C[i1, i3 + s] = A[i2, i4 - s]

    qc1 = np.linalg.solve(np.conjugate(C), B1)
    Qcoef = qc1.reshape((or21, or21), order="C")

    arg = PI2 * np.arange(ngrid) / (ngrid - 1)
    Y, X = np.meshgrid(arg, arg)
    q = np.zeros((ngrid, ngrid), dtype=complex)
    for n in range(-N, N + 1):
        for m in range(-N, N + 1):
            q += Qcoef[n + N, m + N] * np.exp(1j * (n * X + m * Y))
    return Qcoef, np.real(q)


def co_fcplfct2(phi1, phi2, dphi1, dphi2, N, ngrid=100):
    """Estimate two 2D coupling functions via Fourier-series regression."""
    p1 = np.unwrap(_as_1d(phi1))
    p2 = np.unwrap(_as_1d(phi2))
    dp1 = _as_1d(dphi1)
    dp2 = _as_1d(dphi2)

    A = np.zeros((4 * N + 1, 4 * N + 1), dtype=complex)
    off = 2 * N
    or21 = 2 * N + 1

    for n in range(-2 * N, 2 * N + 1):
        for m in range(-2 * N, n + 1):
            val = np.mean(np.exp(1j * (n * p1 + m * p2)))
            A[n + off, m + off] = val
            A[-n + off, -m + off] = np.conjugate(val)

    B1 = np.zeros(or21 * or21, dtype=complex)
    B2 = np.zeros_like(B1)
    C = np.zeros((or21 * or21, or21 * or21), dtype=complex)

    ind = 0
    for n in range(-N, N + 1):
        i1_1 = (n + N) * or21
        for m in range(-N, N + 1):
            i1 = i1_1 + m + N
            i4 = m + off
            tmp = np.exp(-1j * (n * p1 + m * p2))
            B1[ind] = np.mean(dp1 * tmp)
            B2[ind] = np.mean(dp2 * tmp)
            ind += 1
            for r in range(-N, N + 1):
                i3 = (r + N) * or21 + N
                i2 = (n - r) + off
                for s in range(-N, N + 1):
                    C[i1, i3 + s] = A[i2, i4 - s]

    qc1 = np.linalg.solve(np.conjugate(C), B1)
    qc2 = np.linalg.solve(np.conjugate(C), B2)

    Qcoef1 = qc1.reshape((or21, or21), order="C")
    Qcoef2 = qc2.reshape((or21, or21), order="C").T

    arg = PI2 * np.arange(ngrid) / (ngrid - 1)
    Y, X = np.meshgrid(arg, arg)
    q1 = np.zeros((ngrid, ngrid), dtype=complex)
    q2 = np.zeros((ngrid, ngrid), dtype=complex)
    for n in range(-N, N + 1):
        for m in range(-N, N + 1):
            tmp = np.exp(1j * (n * X + m * Y))
            q1 += Qcoef1[n + N, m + N] * tmp
            q2 += Qcoef2[n + N, m + N] * tmp
    return Qcoef1, Qcoef2, np.real(q1), np.real(q2)


def co_fcpltri(phi1, phi2, phi3, Dphi1, Dphi2, Dphi3, N):
    """Estimate three 3D coupling functions for a triad via Fourier fitting."""
    p1 = np.unwrap(_as_1d(phi1))
    p2 = np.unwrap(_as_1d(phi2))
    p3 = np.unwrap(_as_1d(phi3))
    d1 = _as_1d(Dphi1)
    d2 = _as_1d(Dphi2)
    d3 = _as_1d(Dphi3)

    ncf = 2 * N + 1
    ncf2 = ncf * ncf
    off = 2 * N
    A = np.zeros((4 * N + 1, 4 * N + 1, 4 * N + 1), dtype=complex)

    for n in range(-2 * N, 2 * N + 1):
        for m in range(-2 * N, 2 * N + 1):
            for k in range(-2 * N, m + 1):
                val = np.mean(np.exp(1j * (n * p1 + m * p2 + k * p3)))
                A[n + off, m + off, k + off] = val
                A[-n + off, -m + off, -k + off] = np.conjugate(val)

    n3 = ncf**3
    B1 = np.zeros(n3, dtype=complex)
    B2 = np.zeros_like(B1)
    B3 = np.zeros_like(B1)
    C = np.zeros((n3, n3), dtype=complex)

    ind = 0
    for r in range(-N, N + 1):
        for s in range(-N, N + 1):
            for q in range(-N, N + 1):
                expv = np.exp(-1j * (r * p1 + s * p2 + q * p3))
                B1[ind] = np.mean(d1 * expv)
                B2[ind] = np.mean(d2 * expv)
                B3[ind] = np.mean(d3 * expv)
                row = (r + N) * ncf2 + (s + N) * ncf + (q + N)
                ind += 1
                for n in range(-N, N + 1):
                    for m in range(-N, N + 1):
                        for k in range(-N, N + 1):
                            col = (n + N) * ncf2 + (m + N) * ncf + (k + N)
                            C[row, col] = A[(n - r) + off, (m - s) + off, (k - q) + off]

    coeff1 = np.linalg.solve(C, B1)
    coeff2 = np.linalg.solve(C, B2)
    coeff3 = np.linalg.solve(C, B3)

    Qcoef1 = np.zeros((ncf, ncf, ncf), dtype=complex)
    Qcoef2 = np.zeros_like(Qcoef1)
    Qcoef3 = np.zeros_like(Qcoef1)
    for n in range(ncf):
        for m in range(ncf):
            for k in range(ncf):
                idx = n * ncf2 + m * ncf + k
                Qcoef1[n, m, k] = coeff1[idx]
                Qcoef2[m, k, n] = coeff2[idx]
                Qcoef3[k, n, m] = coeff3[idx]
    return Qcoef1, Qcoef2, Qcoef3


def _co_3to2(Q, N, thresh):
    """Internal helper to compute partial norms/derivatives from triadic coefficients."""
    QQ = np.asarray(Q, dtype=complex).copy()
    thresh_val = np.max(np.abs(QQ)) * thresh / 100.0
    QQ[np.abs(QQ) < thresh_val] = 0.0
    or1 = N
    or21 = 2 * N + 1
    Q12 = np.zeros((or21, or21), dtype=complex)
    Q13 = np.zeros((or21, or21), dtype=complex)
    C123 = 0.0
    PART12 = 0.0
    PART13 = 0.0
    TOT = 0.0
    for n in range(or21):
        for m in range(or21):
            Q12[n, m] = QQ[n, m, or1]
            Q13[n, m] = QQ[n, or1, m]
            for k in range(or21):
                qv = QQ[n, m, k]
                aq2 = np.abs(qv**2)
                if m != or1 and k != or1:
                    C123 += aq2
                PART12 += (m - or1) ** 2 * aq2
                PART13 += (k - or1) ** 2 * aq2
                TOT += aq2
    C12 = np.sqrt(np.sum(np.abs(Q12) ** 2))
    C13 = np.sqrt(np.sum(np.abs(Q13) ** 2))
    return C12, C13, np.sqrt(C123), np.sqrt(PART12), np.sqrt(PART13), np.sqrt(TOT)


def co_tricplfan(Qcoef1, Qcoef2, Qcoef3, meth=1, thresh=2):
    """Analyze 3-oscillator coupling structure from triadic Fourier coefficients."""
    Q1 = np.asarray(Qcoef1, dtype=complex).copy()
    Q2 = np.asarray(Qcoef2, dtype=complex).copy()
    Q3 = np.asarray(Qcoef3, dtype=complex).copy()
    N = (Q1.shape[0] - 1) // 2
    COUP = np.zeros((3, 3), dtype=float)
    NORM = np.zeros(3, dtype=float)

    OMEGA = np.zeros(3, dtype=float)
    OMEGA[0] = np.abs(Q1[N, N, N])
    OMEGA[1] = np.abs(Q2[N, N, N])
    OMEGA[2] = np.abs(Q3[N, N, N])
    Q1[N, N, N] = 0.0
    Q2[N, N, N] = 0.0
    Q3[N, N, N] = 0.0

    if meth < 3:
        COUP[0, 1], COUP[0, 2], COUP[0, 0], _, _, NORM[0] = _co_3to2(Q1, N, thresh)
    else:
        _, _, _, COUP[0, 1], COUP[0, 2], NORM[0] = _co_3to2(Q1, N, thresh)
    if meth in (2, 4) and OMEGA[0] != 0:
        COUP[0, :] /= OMEGA[0]

    if meth < 3:
        COUP[1, 2], COUP[1, 0], COUP[1, 1], _, _, NORM[1] = _co_3to2(Q2, N, thresh)
    else:
        _, _, _, COUP[1, 2], COUP[1, 0], NORM[1] = _co_3to2(Q2, N, thresh)
    if meth in (2, 4) and OMEGA[1] != 0:
        COUP[1, :] /= OMEGA[1]

    if meth < 3:
        COUP[2, 0], COUP[2, 1], COUP[2, 2], _, _, NORM[2] = _co_3to2(Q3, N, thresh)
    else:
        _, _, _, COUP[2, 0], COUP[2, 1], NORM[2] = _co_3to2(Q3, N, thresh)
    if meth in (2, 4) and OMEGA[2] != 0:
        COUP[2, :] /= OMEGA[2]

    return COUP, NORM, OMEGA


def co_nettri(PHI, PHI_dot, N, meth=1, thresh=2):
    """Estimate directed coupling matrix for networks by scanning all triplets."""
    phi = np.asarray(PHI, dtype=float)
    phidot = np.asarray(PHI_dot, dtype=float)
    if phi.shape[1] < phi.shape[0]:
        phi = phi.T
    if phidot.shape[1] < phidot.shape[0]:
        phidot = phidot.T
    M = min(phidot.shape[0], phidot.shape[1])
    C = np.zeros((M, M, max(M - 2, 1)), dtype=float)
    IN = np.zeros((M, M), dtype=int)

    for n in range(M):
        for m in range(n + 1, M):
            for k in range(m + 1, M):
                _, maxind, _, _, _ = co_maxsync3(phi[n, :], phi[m, :], phi[k, :], N)
                if maxind > 0.5:
                    warnings.warn(
                        "Oscillators are too close to synchrony; results may be unreliable.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                Qcoef1, Qcoef2, Qcoef3 = co_fcpltri(
                    phi[n, :],
                    phi[m, :],
                    phi[k, :],
                    phidot[n, :],
                    phidot[m, :],
                    phidot[k, :],
                    N,
                )
                COUP, _, _ = co_tricplfan(Qcoef1, Qcoef2, Qcoef3, meth, thresh)

                C[n, m, IN[n, m]] = COUP[0, 1]
                IN[n, m] += 1
                C[m, n, IN[m, n]] = COUP[1, 0]
                IN[m, n] += 1
                C[n, k, IN[n, k]] = COUP[0, 2]
                IN[n, k] += 1
                C[k, n, IN[k, n]] = COUP[2, 0]
                IN[k, n] += 1
                C[m, k, IN[m, k]] = COUP[1, 2]
                IN[m, k] += 1
                C[k, m, IN[k, m]] = COUP[2, 1]
                IN[k, m] += 1

    COUPf = np.zeros((M, M), dtype=float)
    for n in range(M):
        for m in range(M):
            depth = max(IN[n, m], 1)
            COUPf[n, m] = np.min(C[n, m, :depth])
    return COUPf


def _kernel_cpl(phi1, phi2, ngrid, al_x, al_y):
    """Internal kernel precomputation for kernel-based coupling estimators."""
    ng1 = ngrid - 1
    x = PI2 * np.arange(ng1) / ng1
    p1 = _as_1d(phi1)
    p2 = _as_1d(phi2)
    Kx = np.exp(al_x * (np.cos(p1[:, None] - x[None, :]) - 1.0))
    Ky = np.exp(al_y * (np.cos(p2[:, None] - x[None, :]) - 1.0))
    Nrm = np.einsum("lk,ln->kn", Kx, Ky)
    return Kx, Ky, Nrm


def co_kcplfct1(phi1, phi2, phi1_dot, ngrid, fignum=0, al_x=None, al_y=None):
    """Estimate one coupling function from phases using kernel smoothing."""
    ng1 = ngrid - 1
    if al_x is None:
        al_x = (ng1 / PI2) ** 2
    if al_y is None:
        al_y = al_x
    Kx, Ky, Nrm = _kernel_cpl(phi1, phi2, ngrid, al_x, al_y)
    dp1 = _as_1d(phi1_dot)
    num1 = np.einsum("l,lk,ln->kn", dp1, Kx, Ky)
    q1core = num1 / Nrm

    q1 = np.zeros((ngrid, ngrid), dtype=float)
    q1[:ng1, :ng1] = q1core
    q1[-1, :ng1] = q1[0, :ng1]
    q1[:ng1, -1] = q1[:ng1, 0]
    q1[-1, -1] = q1[0, 0]

    if fignum > 0:
        arg = PI2 * np.arange(ngrid) / ng1
        X, Y = np.meshgrid(arg, arg)
        fig = plt.figure(num=fignum)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.plot_surface(X, Y, q1, cmap="jet")
        ax.set_xlim(0, PI2)
        ax.set_ylim(0, PI2)
        ax.set_xlabel(r"$\phi_1$")
        ax.set_ylabel(r"$\phi_2$")
        ax.set_zlabel(r"$q_1$")
    return q1


def co_kcplfct2(phi1, phi2, phi1_dot, phi2_dot, ngrid, al_x=None, al_y=None):
    """Estimate two coupling functions from phases using kernel smoothing."""
    ng1 = ngrid - 1
    if al_x is None:
        al_x = (ng1 / PI2) ** 2
    if al_y is None:
        al_y = al_x
    Kx, Ky, Nrm = _kernel_cpl(phi1, phi2, ngrid, al_x, al_y)
    dp1 = _as_1d(phi1_dot)
    dp2 = _as_1d(phi2_dot)
    num1 = np.einsum("l,lk,ln->kn", dp1, Kx, Ky)
    num2 = np.einsum("l,lk,ln->kn", dp2, Kx, Ky)
    q1core = num1 / Nrm
    q2core = num2 / Nrm

    q1 = np.zeros((ngrid, ngrid), dtype=float)
    q2 = np.zeros((ngrid, ngrid), dtype=float)
    q1[:ng1, :ng1] = q1core
    q2[:ng1, :ng1] = q2core
    q1[-1, :ng1] = q1[0, :ng1]
    q1[:ng1, -1] = q1[:ng1, 0]
    q1[-1, -1] = q1[0, 0]
    q2[-1, :ng1] = q2[0, :ng1]
    q2[:ng1, -1] = q2[:ng1, 0]
    q2[-1, -1] = q2[0, 0]
    return q1, q2.T
