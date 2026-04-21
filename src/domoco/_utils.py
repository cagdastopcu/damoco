"""Internal numeric helpers."""

from __future__ import annotations

import numpy as np


PI2 = 2.0 * np.pi


def _as_1d(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x)
    return np.ravel(arr).astype(float)


def _trapz2(a: np.ndarray):
    return np.trapezoid(np.trapezoid(a, axis=0), axis=0)


def _trapz1(a: np.ndarray) -> float:
    return float(np.trapezoid(a))
