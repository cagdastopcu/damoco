"""Protophase extraction and protophase-to-phase transformation methods.

Implements DAMOCO protophase estimators and Fourier-based protophase-to-phase
transformations with the same defaults and conventions as MATLAB code.
"""

from __future__ import annotations

import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fminbound
from scipy.signal import hilbert

from ._utils import PI2, _as_1d


def co_fbtransf1(theta, nfft=80, alpha=0.05, ngrid=50):
    r"""Transform protophase to phase with smoothed Fourier correction.

    Uses
    :math:`\phi=\theta+2\sum_{k=1}^{K}e^{-\frac{1}{2}k^2\alpha^2}\Im\left[S_k\frac{e^{ik\theta}-1}{k}\right]`
    where :math:`S_k` are empirical Fourier coefficients of protophase density.

    Parameters
    ----------
    theta : array_like
        Input protophase trajectory.
    nfft : int, default=80
        Number of Fourier harmonics :math:`K` used for correction.
    alpha : float, default=0.05
        Gaussian smoothing width in harmonic domain. Larger values suppress
        high-order harmonics more strongly.
    ngrid : int, default=50
        Number of points for returning sampled phase-density correction
        :math:`\sigma(\theta)` on :math:`[0, 2\pi]`.

    Returns
    -------
    tuple
        ``(phi, arg, sigma)`` where ``phi`` is transformed phase, ``arg`` is
        the angular grid, and ``sigma`` is the estimated phase-density profile.
    """
    th = _as_1d(theta)
    Spl = np.zeros(nfft, dtype=complex)
    al2 = alpha * alpha

    cut = np.where(np.diff(np.mod(th, PI2)) < 0)[0] + 1
    if len(cut) < 2:
        raise ValueError("theta must contain at least two full cycles.")
    seg = th[cut[0] : cut[-1] + 1]
    npt = len(seg)

    for k in range(1, nfft + 1):
        Spl[k - 1] = np.sum(np.exp(-1j * k * seg)) / npt

    phi = th.copy()
    arg = PI2 * np.arange(ngrid) / (ngrid - 1)
    sigma = np.ones(ngrid, dtype=float)
    for k in range(1, nfft + 1):
        kernel = np.exp(-0.5 * k * k * al2)
        sk = Spl[k - 1]
        sigma += kernel * 2.0 * np.real(sk * np.exp(1j * k * arg))
        phi += kernel * 2.0 * np.imag(sk * (np.exp(1j * k * th) - 1.0) / k)
    return phi, arg, sigma


def co_fbtrT(theta, ngrid=50):
    r"""Transform protophase to phase with Tenreiro-optimal harmonic cutoff.

    Computes :math:`S_k` and chooses optimal truncation index by minimizing the
    Tenreiro criterion before applying unsmoothed Fourier correction.

    Parameters
    ----------
    theta : array_like
        Input protophase trajectory.
    ngrid : int, default=50
        Number of samples used for the returned density profile.

    Returns
    -------
    tuple
        ``(phi, arg, sigma)`` with phase estimate ``phi``, grid ``arg``, and
        density correction ``sigma``.
    """
    th = _as_1d(theta)
    nfft = 100
    Spl = np.zeros(nfft, dtype=complex)
    Hl = np.zeros(nfft, dtype=float)

    cut = np.where(np.diff(np.mod(th, PI2)) < 0)[0] + 1
    if len(cut) < 2:
        raise ValueError("theta must contain at least two full cycles.")
    seg = th[cut[0] : cut[-1] + 1]
    npt = len(seg)

    S = 0.0 + 0.0j
    c = float(npt + 1) / float(npt - 1)
    for k in range(1, nfft + 1):
        Spl[k - 1] = np.sum(np.exp(-1j * k * seg)) / npt
        S = S + Spl[k - 1] * np.conjugate(Spl[k - 1]) - 1.0 / float(npt)
        Hl[k - 1] = k / npt - c * np.real(S)
    indopt = int(np.argmin(Hl)) + 1

    phi = th.copy()
    arg = PI2 * np.arange(ngrid) / (ngrid - 1)
    sigma = np.ones(ngrid, dtype=float)
    for k in range(1, indopt + 1):
        sk = Spl[k - 1]
        sigma += 2.0 * np.real(sk * np.exp(1j * k * arg))
        phi += 2.0 * np.imag(sk * (np.exp(1j * k * th) - 1.0) / k)
    return phi, arg, sigma


def co_distproto(x, NV):
    r"""Estimate protophase from 2D embedding by distance normalization.

    Defines cycles by Poincare-section crossings with normal vector ``NV`` and
    maps cumulative arc-length along each cycle to :math:`[0,2\pi)`.

    Parameters
    ----------
    x : array_like
        Complex signal or real 2D embedding with shape ``(2, N)`` (or
        transposed ``(N, 2)``).
    NV : array_like
        Normal vector of the Poincare section in embedding space.

    Returns
    -------
    tuple
        ``(theta, Start, Stop)`` where ``theta`` is unwrapped protophase and
        ``Start``/``Stop`` delimit the valid interval between first and last
        complete section crossings.
    """
    xin = np.asarray(x)
    nv = _as_1d(NV)
    if np.iscomplexobj(xin):
        y = np.vstack([np.real(xin), np.imag(xin)])
    else:
        y = np.asarray(xin, dtype=float)
    if y.ndim == 1:
        raise ValueError("x must be complex or a 2 x N real embedding.")
    if y.shape[0] > y.shape[1]:
        y = y.T
    y = y.astype(float)
    y[0, :] /= np.std(y[0, :])
    y[1, :] /= np.std(y[1, :])

    L = y.shape[1]
    Pro = np.zeros(L, dtype=float)
    Se = np.zeros(L, dtype=int)
    theta = np.zeros(L, dtype=float)

    for n in range(L):
        Pro[n] = nv @ y[:, n]

    V = []
    for n in range(1, L):
        if (Pro[n] > 0) and (Pro[n - 1] < 0):
            Se[n] = 1
            V.append(Pro[n] / (Pro[n] - Pro[n - 1]))

    dy = np.gradient(y, axis=1)
    dd = np.linalg.norm(dy, axis=0)
    Dis = np.cumsum(dd)
    Pmin = np.where(Se == 1)[0]
    if len(Pmin) < 2:
        raise ValueError("Could not detect enough Poincare section crossings.")

    for i in range(len(Pmin) - 1):
        p0 = Pmin[i]
        p1 = Pmin[i + 1]
        R1 = V[i] * (Dis[p0] - Dis[p0 - 1])
        R2 = (1.0 - V[i + 1]) * (Dis[p1] - Dis[p1 - 1])
        den = (Dis[p1 - 1] + R2) - (Dis[p0] - R1)
        for j in range(p0, p1):
            theta[j] = PI2 * (Dis[j] - (Dis[p0] - R1)) / den
    theta = np.unwrap(theta)
    Start = int(Pmin[0])
    Stop = int(Pmin[-1] - 1)
    return theta, Start, Stop


def co_hilbproto(x, fignum=0, x0=0.0, y0=0.0, ntail=1000):
    r"""Estimate protophase from Hilbert embedding angle.

    After edge trimming and optional origin shift, computes
    :math:`\theta=\arg(x+i\mathcal{H}[x]) \mod 2\pi` and reports minimum
    instantaneous amplitude as phase-quality indicator.

    Parameters
    ----------
    x : array_like
        Scalar oscillatory time series.
    fignum : int, default=0
        Figure number for optional embedding plot. Plotting is disabled when
        set to ``0``.
    x0, y0 : float, default=0.0
        Optional shift of embedding origin before angle extraction.
    ntail : int, default=1000
        Number of samples trimmed from both edges to reduce Hilbert transform
        boundary artifacts.

    Returns
    -------
    tuple
        ``(theta, minampl)`` where ``theta`` is protophase and ``minampl`` is
        the minimum embedding amplitude (small values indicate poor phase
        definition).
    """
    xx = _as_1d(x)
    ht = hilbert(xx)
    ht = ht[ntail : len(ht) - ntail]
    ht = ht - np.mean(ht)
    if fignum > 0:
        fig = plt.figure(num=fignum)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.real(ht), np.imag(ht))
        ax.plot([x0], [y0], "ro")
        ax.set_xlabel("signal")
        ax.set_ylabel("HT(signal)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Hilbert embedding")
    ht = ht - x0 - 1j * y0
    theta = np.mod(np.angle(ht), PI2)
    ampl = np.abs(ht)
    minampl = float(np.min(ampl))
    avampl = float(np.mean(ampl))
    if minampl < avampl / 20.0:
        warnings.warn("Phase may be not well-defined.", RuntimeWarning, stacklevel=2)
    return theta, minampl


def co_mmzproto(x, pl=0):
    r"""Estimate protophase via marker-event interpolation.

    Uses four events per cycle (maximum, zero-crossing, minimum, zero-crossing),
    then linearly interpolates phase between marker times.

    Parameters
    ----------
    x : array_like
        Scalar oscillatory signal.
    pl : int, default=0
        If ``1``, produce diagnostic plots of marker consistency and resulting
        phase trajectories.

    Returns
    -------
    tuple
        ``(theta, START, STOP)`` on successful extraction. If marker detection
        is inconsistent, returns ``(array([nan]), nan, nan)`` and emits a
        warning, matching MATLAB behavior.
    """
    xx = _as_1d(x)
    gx = np.diff(xx)
    s = np.sign(gx[:-1]) - np.sign(gx[1:])

    IN1 = np.where(s == 2)[0] + 1
    IN1c = IN1 + gx[IN1 - 1] / (gx[IN1 - 1] - gx[IN1])
    IN3 = np.where(s == -2)[0] + 1
    IN3c = IN3 + gx[IN3 - 1] / (gx[IN3 - 1] - gx[IN3])
    IN2 = np.where((xx[:-1] > 0) & (xx[1:] < 0))[0]
    IN2c = IN2 + xx[IN2] / (xx[IN2] - xx[IN2 + 1])
    IN4 = np.where((xx[:-1] < 0) & (xx[1:] > 0))[0]
    IN4c = IN4 + xx[IN4] / (xx[IN4] - xx[IN4 + 1])

    if len(IN1c) < 2:
        return np.array([np.nan]), np.nan, np.nan

    if len(IN2c) and IN2c[0] < IN1c[0]:
        IN2c = IN2c[1:]
    if len(IN2c) and IN2c[-1] > IN1c[-1]:
        IN2c = IN2c[:-1]
    if len(IN3c) and IN3c[0] < IN1c[0]:
        IN3c = IN3c[1:]
    if len(IN3c) and IN3c[-1] > IN1c[-1]:
        IN3c = IN3c[:-1]
    if len(IN4c) and IN4c[0] < IN1c[0]:
        IN4c = IN4c[1:]
    if len(IN4c) and IN4c[-1] > IN1c[-1]:
        IN4c = IN4c[:-1]

    bad = (
        len(IN2c) != len(IN3c)
        or len(IN3c) != len(IN4c)
        or len(IN2c) != len(IN4c)
        or len(IN1c) - 1 != len(IN2c)
    )
    if bad:
        warnings.warn("Number of marker events inconsistent.", RuntimeWarning, stacklevel=2)
        if pl == 1:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if len(IN1c) > 1:
                ax.plot(np.diff(IN1c))
            if len(IN2c) > 1:
                ax.plot(np.diff(IN2c), "r")
            if len(IN3c) > 1:
                ax.plot(np.diff(IN3c), "g")
            if len(IN4c) > 1:
                ax.plot(np.diff(IN4c), "k")
        return np.array([np.nan]), np.nan, np.nan

    D = IN1c[1:] - IN1c[:-1]
    R2 = np.mean((IN2c - IN1c[:-1]) / D)
    R3 = np.mean((IN3c - IN1c[:-1]) / D)
    R4 = np.mean((IN4c - IN1c[:-1]) / D)

    IN = []
    Pin = []
    for n in range(len(IN1c) - 1):
        IN.extend([IN1c[n], IN2c[n], IN3c[n], IN4c[n]])
        Pin.extend([PI2 * n, PI2 * (n + R2), PI2 * (n + R3), PI2 * (n + R4)])
    IN.append(IN1c[-1])
    Pin.append(PI2 * (len(IN1c) - 1))

    START = int(IN1[0] + 1)
    STOP = int(IN1[-1])
    xq = np.arange(START, STOP + 1)
    theta = np.interp(xq, np.asarray(IN), np.asarray(Pin))

    if pl == 1:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.plot(np.mod(theta, PI2), xx[START : STOP + 1], ".", markersize=4)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.plot(xx)
        ax2.plot(IN1, xx[IN1], ".r")
        ax2.plot(IN2c, np.zeros_like(IN2c), ".g")
        ax2.plot(IN3, xx[IN3], ".k")
        ax2.plot(IN4c, np.zeros_like(IN4c), ".m")

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)
        ax3.plot(np.mod(theta, PI2), np.gradient(theta), ".", markersize=4)
    return theta, START, STOP


def _av_comp(H, theta, N, PL):
    r"""Compute Fourier coefficients of the average cycle.

    Computes :math:`C_n=\frac{\sum H e^{-in\theta}\dot\theta}{\theta_{end}-\theta_{start}}`.

    Parameters
    ----------
    H : array_like
        Complex embedding values.
    theta : array_like
        Unwrapped phase values.
    N : int
        Maximum harmonic index.
    PL : int
        Diagnostic plotting flag (``1`` enables plots).

    Returns
    -------
    numpy.ndarray
        Complex Fourier coefficients ``Cav`` of length ``N + 1``.
    """
    th = _as_1d(theta)
    hh = np.ravel(H)
    Cav = np.zeros(N + 1, dtype=complex)
    d_theta = np.gradient(th)
    denom = th[-1] - th[0]
    for n in range(N + 1):
        Cav[n] = np.sum(hh * np.exp(-1j * n * th) * d_theta) / denom
    if PL == 1:
        p2 = PI2 * np.arange(10000) / 10000.0
        Y = np.zeros_like(p2, dtype=complex)
        for n in range(N + 1):
            Y += Cav[n] * np.exp(1j * n * p2)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.plot(np.real(hh), np.imag(hh), ".", markersize=4)
        ax1.plot(np.real(Y), np.imag(Y), ".r", markersize=4)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.plot(np.abs(Cav))
    return Cav


def _ERav(phi, Cav, H, theta, alpha):
    r"""Error functional for average-cycle projection.

    Objective:
    :math:`|H-Z(\phi)|^2+\alpha|e^{i\phi}-e^{i\theta}|^2`.

    Parameters
    ----------
    phi : float
        Candidate refined phase.
    Cav : array_like
        Average-cycle Fourier coefficients.
    H : complex
        Current complex embedding sample.
    theta : float
        Initial protophase at the same sample.
    alpha : float
        Regularization weight constraining refined phase to remain near
        ``theta`` on the unit circle.

    Returns
    -------
    float
        Scalar objective value minimized in :func:`co_avcyc`.
    """
    N = np.arange(len(Cav))
    Z = np.sum(Cav * np.exp(1j * N * phi))
    return np.abs(H - Z) ** 2 + alpha * np.abs(np.exp(1j * phi) - np.exp(1j * theta)) ** 2


def co_avcyc(x, theta, N, PL=0, alpha=0.0):
    r"""Refine protophase by projection onto an estimated average cycle.

    For each sample, minimizes the average-cycle projection error in a local
    neighborhood around initial protophase estimate ``theta``.

    Parameters
    ----------
    x : array_like
        Scalar signal used to form Hilbert embedding.
    theta : array_like
        Initial protophase estimate aligned with ``x``.
    N : int
        Maximum harmonic for average-cycle reconstruction.
    PL : int, default=0
        Diagnostic plotting flag (``1`` enables plots).
    alpha : float, default=0.0
        Regularization factor in the local optimization objective.

    Returns
    -------
    tuple
        ``(phi, Cav)`` where ``phi`` is refined phase and ``Cav`` are average
        cycle Fourier coefficients.
    """
    xx = _as_1d(x)
    th = _as_1d(theta)
    xx = xx / np.max(xx)
    H = hilbert(xx)
    if len(H) <= 2000 or len(th) <= 2000:
        raise ValueError("x and theta must be longer than 2000 due to tail trimming.")
    H = H[1000:-1000]
    th = th[1000:-1000]
    Cav = _av_comp(H, th, N, PL)

    phi = np.zeros_like(th, dtype=float)
    DIS = np.mean(np.gradient(th))
    for i in range(len(th)):
        phi[i] = fminbound(_ERav, th[i] - DIS, th[i] + DIS, args=(Cav, H[i], th[i], alpha))

    if PL == 1:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.mod(phi, PI2), np.gradient(phi), "g.", markersize=4)
        ax.plot(np.mod(th, PI2), np.gradient(th), "r.", markersize=4)
    return phi, Cav
