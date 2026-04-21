"""Synchronization index utilities (2-oscillator and 3-oscillator)."""

from __future__ import annotations

import numpy as np

from ._utils import _as_1d


def co_sync(P1, P2, n, m):
    """Compute n:m synchronization index from two phase/protophase signals."""
    p1 = _as_1d(P1)
    p2 = _as_1d(P2)
    return float(np.abs(np.mean(np.exp(1j * (n * p1 - m * p2)))))


def co_sync3(P1, P2, P3, n, m, l):
    """Compute n:m:l synchronization index from three phase/protophase signals."""
    p1 = _as_1d(P1)
    p2 = _as_1d(P2)
    p3 = _as_1d(P3)
    return float(np.abs(np.mean(np.exp(1j * (n * p1 + m * p2 + l * p3)))))


def co_maxsync(theta1, theta2, or_):
    """Scan all n,m in [1..or_] and return synchronization matrix and maximum."""
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
    """Scan 3-way synchronization orders and return max index and corresponding tuple."""
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
