"""Internal numeric helpers.

This module contains small utility routines shared by multiple DAMOCO
submodules. They are intentionally minimal and not part of the public API.
"""

from __future__ import annotations

import numpy as np


PI2 = 2.0 * np.pi


def _as_1d(x: np.ndarray | list[float]) -> np.ndarray:
    """Convert input to a 1D float NumPy array."""
    arr = np.asarray(x)
    return np.ravel(arr).astype(float)


def _trapz2(a: np.ndarray):
    """Compute 2D trapezoidal integral by nested 1D trapezoid calls."""
    return np.trapezoid(np.trapezoid(a, axis=0), axis=0)


def _trapz1(a: np.ndarray) -> float:
    """Compute 1D trapezoidal integral."""
    return float(np.trapezoid(a))
