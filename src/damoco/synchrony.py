"""Synchronization index utilities.

All synchronization indices are computed as magnitudes of circular means of
complex exponentials, exactly as in the DAMOCO MATLAB reference implementation.
"""

from __future__ import annotations

import numpy as np

from ._utils import _as_1d


def co_sync(P1, P2, n, m):
    r"""Compute the :math:`n:m` synchronization index.

    The estimator is
    :math:`\rho_{n:m}=\left|\langle e^{i(n\phi_1-m\phi_2)}\rangle\right|`.
    Values near 1 indicate strong phase locking of order :math:`n:m`, while
    values near 0 indicate weak or absent locking.

    Parameters
    ----------
    P1, P2 : array_like
        Phase or protophase time series of equal length.
    n, m : int
        Integer synchronization orders.

    Returns
    -------
    float
        Estimated synchronization index in ``[0, 1]``.
    """
    p1 = _as_1d(P1)
    p2 = _as_1d(P2)
    return float(np.abs(np.mean(np.exp(1j * (n * p1 - m * p2)))))


def co_sync3(P1, P2, P3, n, m, l):
    r"""Compute the :math:`n:m:l` synchronization index for three phases.

    The estimator is
    :math:`\rho_{n:m:l}=\left|\langle e^{i(n\phi_1+m\phi_2+l\phi_3)}\rangle\right|`.

    Parameters
    ----------
    P1, P2, P3 : array_like
        Phase or protophase time series of equal length.
    n, m, l : int
        Integer synchronization orders.

    Returns
    -------
    float
        Estimated synchronization index in ``[0, 1]``.
    """
    p1 = _as_1d(P1)
    p2 = _as_1d(P2)
    p3 = _as_1d(P3)
    return float(np.abs(np.mean(np.exp(1j * (n * p1 + m * p2 + l * p3)))))


def co_maxsync(theta1, theta2, or_):
    r"""Scan orders :math:`n,m\in\{1,\dots,\mathrm{or}\}` and locate maximum.

    Returns the full matrix :math:`M[n,m]=\rho_{n:m}` plus the maximum value and
    corresponding integer orders.

    Parameters
    ----------
    theta1, theta2 : array_like
        Phase/protophase time series.
    or_ : int
        Maximum order to scan.

    Returns
    -------
    tuple
        ``(M_SyncIn, maxind, n_theta1, m_theta2)`` where ``M_SyncIn`` has shape
        ``(or_, or_)``.
    """
    t1 = _as_1d(theta1)
    t2 = _as_1d(theta2)
    M_SyncIn = np.zeros((or_, or_), dtype=float)
    maxind = 0.0
    n_theta1 = 1
    m_theta2 = 1
    for n in range(1, or_ + 1):
        for m in range(1, or_ + 1):
            index = co_sync(t1, t2, n, m)
            if index > maxind:
                maxind = index
                n_theta1 = n
                m_theta2 = m
            M_SyncIn[n - 1, m - 1] = index
    return M_SyncIn, maxind, n_theta1, m_theta2


def co_maxsync3(theta1, theta2, theta3, or_):
    r"""Scan 3-way integer orders and locate maximal triadic synchronization.

    Scanned domain follows DAMOCO convention:
    :math:`n\in[0,\mathrm{or}]`, :math:`m,l\in[-\mathrm{or},\mathrm{or}]`.
    The all-zero tuple is excluded from the maximization criterion.

    Parameters
    ----------
    theta1, theta2, theta3 : array_like
        Phase/protophase time series.
    or_ : int
        Maximum absolute order to scan.

    Returns
    -------
    tuple
        ``(M_SyncIn, maxind, n_theta1, m_theta2, l_theta3)`` where
        ``M_SyncIn`` has shape ``(or_+1, 2*or_+1, 2*or_+1)``.
    """
    t1 = _as_1d(theta1)
    t2 = _as_1d(theta2)
    t3 = _as_1d(theta3)
    M_SyncIn = np.zeros((or_ + 1, 2 * or_ + 1, 2 * or_ + 1), dtype=float)
    maxind = 0.0
    n_theta1 = 0
    m_theta2 = 0
    l_theta3 = 0
    for n in range(0, or_ + 1):
        for m in range(-or_, or_ + 1):
            for l in range(-or_, or_ + 1):
                index = co_sync3(t1, t2, t3, n, m, l)
                if index > maxind and (n != 0 or m != 0 or l != 0):
                    maxind = index
                    n_theta1 = n
                    m_theta2 = m
                    l_theta3 = l
                M_SyncIn[n, m + or_, l + or_] = index
    return M_SyncIn, maxind, n_theta1, m_theta2, l_theta3
