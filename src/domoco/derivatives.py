"""Phase-derivative estimation and model quality/decomposition utilities.

Implements Savitzky-Golay differentiation, residual decomposition of phase
models, and iterative PRC-forcing factorization used in DAMOCO workflows.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_coeffs

from ._utils import PI2, _as_1d
from .metrics import co_gnorm


def _sg_diff(phi: np.ndarray, fsample: float) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate an unwrapped phase sequence with Savitzky-Golay filtering."""
    norder = 4
    sl = 4
    wl = 2 * sl + 1
    coeff = savgol_coeffs(window_length=wl, polyorder=norder, deriv=1, use="dot")
    p = np.unwrap(phi)
    windows = sliding_window_view(p, wl)
    d = windows @ coeff
    return d * fsample, p[sl:-sl]


def co_phidot1(phi, fsample):
    r"""Estimate instantaneous frequency from one phase time series.

    Computes :math:`\dot\phi(t)` by Savitzky-Golay local polynomial derivative
    (order 4, window length 9), matching MATLAB defaults.
    """
    p = _as_1d(phi)
    d, p_cut = _sg_diff(p, fsample)
    return d, p_cut


def co_phidot2(phi1, phi2, fsample):
    r"""Estimate instantaneous frequencies for two phase time series.

    Applies the same Savitzky-Golay differentiator to each input phase and
    returns truncated aligned outputs (boundary points removed).
    """
    p1 = _as_1d(phi1)
    p2 = _as_1d(phi2)
    d1, p1c = _sg_diff(p1, fsample)
    d2, p2c = _sg_diff(p2, fsample)
    return d1, d2, p1c, p2c


def co_phidot3(phi1, phi2, phi3, fsample):
    r"""Estimate instantaneous frequencies for three phase time series.

    Uses identical differentiator and truncation for all channels to preserve
    temporal alignment of :math:`\dot\phi_1,\dot\phi_2,\dot\phi_3`.
    """
    p1 = _as_1d(phi1)
    p2 = _as_1d(phi2)
    p3 = _as_1d(phi3)
    d1, p1c = _sg_diff(p1, fsample)
    d2, p2c = _sg_diff(p2, fsample)
    d3, p3c = _sg_diff(p3, fsample)
    return d1, d2, d3, p1c, p2c, p3c


def _dPhi(Q, phi1, phi2):
    r"""Interpolate :math:`Q(\phi_1,\phi_2)` on wrapped phase coordinates."""
    qq = np.asarray(Q, dtype=float)
    s1 = qq.shape[0] - 1
    arg = PI2 * np.arange(s1 + 1) / s1
    interp = RegularGridInterpolator(
        (arg, arg),
        qq,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    p1 = np.mod(_as_1d(phi1), PI2)
    p2 = np.mod(_as_1d(phi2), PI2)
    pts = np.column_stack([p1, p2])
    return interp(pts)


def co_resid_decomp(dPhi1, Phi1, Phi2, q):
    r"""Decompose measured derivative into model and residual parts.

    Reconstructs :math:`\dot\phi_1^{\text{synth}}=q(\phi_1,\phi_2)` by bilinear
    interpolation on the periodic grid and returns residual
    :math:`r=\dot\phi_1-\dot\phi_1^{\text{synth}}` with standard deviations.
    """
    dphi1 = _as_1d(dPhi1)
    phi1 = _as_1d(Phi1)
    phi2 = _as_1d(Phi2)
    dPhi1_synth = _dPhi(q, phi1, phi2)
    Resid = dphi1 - dPhi1_synth
    Std_dPhi1 = float(np.std(dphi1 - np.mean(dphi1), ddof=1))
    Std_dPhi1_synth = float(np.std(dPhi1_synth - np.mean(dPhi1_synth), ddof=1))
    Std_Resid = float(np.std(Resid - np.mean(Resid), ddof=1))
    return Std_dPhi1, Std_dPhi1_synth, Std_Resid, dPhi1_synth, Resid


def _decomp(F: np.ndarray, kmax: int, ngrid: int, niter: int) -> tuple[np.ndarray, np.ndarray]:
    """Iterative rank-1 factorization helper for `co_prciter`."""
    H0 = F[kmax, :].copy()
    Z = np.zeros(ngrid, dtype=float)
    H = np.zeros(ngrid, dtype=float)
    for _ in range(niter):
        Z = np.trapezoid(F * H0[None, :], axis=1)
        Z = Z / np.trapezoid(H0 * H0)
        H = np.trapezoid(F * Z[:, None], axis=0)
        H = H / np.trapezoid(Z * Z)
        H0 = H
    return Z, H


def co_prciter(Q, ngrid, fignum=0):
    r"""Iteratively decompose :math:`Q-\omega` into :math:`Z(\phi_1)H(\phi_2)`.

    Scans candidate :math:`\omega`, minimizes residual standard deviation, then
    returns optimal :math:`Z`, :math:`H`, :math:`\omega_{\mathrm{opt}}`, and the
    minimal decomposition error.
    """
    q = np.asarray(Q, dtype=float)
    niter = 10
    c1 = 1.0
    c2 = 1.0
    om1 = c1 * np.min(q)
    om2 = c2 * np.max(q)
    npt_om = 200
    omega = np.linspace(om1, om2, npt_om)
    decomp_err = np.zeros_like(omega)

    absq = np.abs(q)
    maxq = np.max(absq)
    flat_idx = np.argmax(absq.ravel(order="F"))
    kmax = np.unravel_index(flat_idx, absq.shape, order="F")[0]

    Z = np.zeros(ngrid, dtype=float)
    H = np.zeros(ngrid, dtype=float)
    for i, om in enumerate(omega):
        F = q - om
        Z, H = _decomp(F, kmax, ngrid, niter)
        R = F - np.outer(Z, H)
        decomp_err[i] = np.std(R, ddof=1)

    ind = int(np.argmin(decomp_err))
    min_err = float(decomp_err[ind])
    om_opt = float(omega[ind])
    Z, H = _decomp(q - om_opt, kmax, ngrid, niter)

    if fignum > 0:
        Qn = co_gnorm(q)[0]
        fig = plt.figure(num=fignum)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(omega, decomp_err / Qn, "ro")
        ax.set_xlabel("omega")
        ax.set_ylabel("decomposition error / ||Q||")
    return Z, H, om_opt, min_err
