"""Matplotlib plotting helpers mirroring DAMOCO MATLAB plot utilities.

These are visualization wrappers; they do not alter estimated coefficients or
coupling functions.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

from ._utils import PI2


def co_plotcoef(F, pltype, fignum=None, own_lab=None, ext_lab=None, tit=None):
    r"""Plot :math:`|F_{n,m}|` or :math:`|Q_{n,m}|` for one oscillator model.

    Parameters
    ----------
    F : array_like
        Fourier coefficient matrix.
    pltype : int
        Plot mode: 0 custom labels, 1 phase defaults, >1 protophase defaults.
    fignum : int | None, optional
        Matplotlib figure number.

    Returns
    -------
    tuple
        ``(figure, axes)`` handle tuple.
    """
    FF = np.asarray(F, dtype=complex).copy()
    or_ = (max(FF.shape) - 1) // 2
    n = np.arange(-or_, or_ + 1)
    FF[or_, or_] = 0
    fig = plt.figure(num=fignum) if fignum is not None else plt.figure()
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        np.abs(FF),
        extent=[-or_, or_, -or_, or_],
        origin="lower",
        cmap="jet",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax)
    if pltype == 0:
        ax.set_ylabel(own_lab or "")
        ax.set_xlabel(ext_lab or "")
        ax.set_title(tit or "")
    elif pltype == 1:
        ax.set_ylabel("n (order_own)")
        ax.set_xlabel("m (order_ext)")
        ax.set_title("Coefficients of phase dynamics: Qcoef_n,m")
    else:
        ax.set_ylabel("n (order_own)")
        ax.set_xlabel("m (order_ext)")
        ax.set_title("Coefficients of protophase dynamics: Fcoef_n,m")
    return fig, ax


def co_plotcplf(f, pltype, fignum=None, own_lab=None, ext_lab=None, zlab=None, tit=None):
    r"""Plot one function :math:`q(\phi_{own},\phi_{ext})` (or protophase analog).

    Parameters
    ----------
    f : array_like
        2D coupling/protophase function sampled on a periodic grid.
    pltype : int
        Plot mode: 0 custom labels, 1 phase defaults, >1 protophase defaults.

    Returns
    -------
    tuple
        ``(figure, axes3d)`` handle tuple.
    """
    ff = np.asarray(f, dtype=float)
    ngrid = max(ff.shape)
    arg = PI2 * np.arange(ngrid) / (ngrid - 1)
    X, Y = np.meshgrid(arg, arg)
    fig = plt.figure(num=fignum) if fignum is not None else plt.figure()
    fig.clf()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(X, Y, ff, cmap="jet")
    ax.set_xlim(0, PI2)
    ax.set_ylim(0, PI2)
    if pltype == 0:
        ax.set_ylabel(own_lab or "")
        ax.set_xlabel(ext_lab or "")
        ax.set_zlabel(zlab or "")
        ax.set_title(tit or "")
    elif pltype == 1:
        ax.set_ylabel(r"$\phi_{own}$")
        ax.set_xlabel(r"$\phi_{ext}$")
        ax.set_zlabel(r"$q(\phi_{own}, \phi_{ext})$")
        ax.set_title("phase dynamics")
    else:
        ax.set_ylabel(r"$\theta_{own}$")
        ax.set_xlabel(r"$\theta_{ext}$")
        ax.set_zlabel(r"$f(\theta_{own}, \theta_{ext})$")
        ax.set_title("protophase dynamics")
    return fig, ax


def co_plot2coef(
    F1,
    F2,
    pltype,
    fignum=None,
    own_lab_1=None,
    ext_lab_1=None,
    tit_1=None,
    own_lab_2=None,
    ext_lab_2=None,
    tit_2=None,
):
    r"""Plot two coefficient matrices side by side with shared color scale.

    Parameters
    ----------
    F1, F2 : array_like
        Fourier coefficient matrices.
    pltype : int
        Plot mode: 0 custom labels, 1 phase defaults, >1 protophase defaults.

    Returns
    -------
    tuple
        ``(figure, (ax_left, ax_right))`` handle tuple.
    """
    A = np.asarray(F1, dtype=complex).copy()
    B = np.asarray(F2, dtype=complex).copy()
    or_ = (max(A.shape) - 1) // 2
    n = np.arange(-or_, or_ + 1)
    A[or_, or_] = 0
    B[or_, or_] = 0
    AF1 = np.abs(A)
    AF2 = np.abs(B)
    vmax = np.max(np.concatenate([AF1.ravel(), AF2.ravel()]))

    fig = plt.figure(num=fignum) if fignum is not None else plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    im1 = ax1.imshow(
        AF1,
        extent=[-or_, or_, -or_, or_],
        origin="lower",
        cmap="jet",
        aspect="equal",
        vmin=0,
        vmax=vmax,
    )
    ax2.imshow(
        AF2,
        extent=[-or_, or_, -or_, or_],
        origin="lower",
        cmap="jet",
        aspect="equal",
        vmin=0,
        vmax=vmax,
    )
    fig.colorbar(im1, ax=[ax1, ax2], location="bottom", fraction=0.08, pad=0.15)

    if pltype == 0:
        ax1.set_ylabel(own_lab_1 or "")
        ax1.set_xlabel(ext_lab_1 or "")
        ax1.set_title(tit_1 or "")
        ax2.set_ylabel(own_lab_2 or "")
        ax2.set_xlabel(ext_lab_2 or "")
        ax2.set_title(tit_2 or "")
    elif pltype == 1:
        ax1.set_ylabel("n (order_own)")
        ax1.set_xlabel("m (order_ext)")
        ax1.set_title("Phase dynamics: Qcoef-1_n,m")
        ax2.set_ylabel("n (order_own)")
        ax2.set_xlabel("m (order_ext)")
        ax2.set_title("Phase dynamics: Qcoef-2_n,m")
    else:
        ax1.set_ylabel("n (order_own)")
        ax1.set_xlabel("m (order_ext)")
        ax1.set_title("Protophase dynamics: Fcoef-1_n,m")
        ax2.set_ylabel("n (order_own)")
        ax2.set_xlabel("m (order_ext)")
        ax2.set_title("Protophase dynamics: Fcoef-2_n,m")
    return fig, (ax1, ax2)


def co_plot2cplf(f1, f2, pltype, fignum=None, xl=None, yl=None, zl1=None, zl2=None, tit=None):
    r"""Plot two coupling/protophase surfaces for comparative inspection.

    Parameters
    ----------
    f1, f2 : array_like
        2D coupling/protophase functions on periodic grids.
    pltype : int
        Plot mode: 0 custom labels, 1 phase defaults, >1 protophase defaults.

    Returns
    -------
    tuple
        ``(figure, (ax_left, ax_right))`` handle tuple.
    """
    a = np.asarray(f1, dtype=float)
    b = np.asarray(f2, dtype=float)
    ngrid = max(a.shape)
    arg = PI2 * np.arange(ngrid) / (ngrid - 1)
    X, Y = np.meshgrid(arg, arg)

    if pltype == 1:
        xl = r"$\phi_1$"
        yl = r"$\phi_2$"
        zl1 = r"$q_1(\phi_1,\phi_2)$"
        zl2 = r"$q_2(\phi_2,\phi_1)$"
        tit = "phase dynamics"
    elif pltype > 1:
        xl = r"$\theta_1$"
        yl = r"$\theta_2$"
        zl1 = r"$f_1(\theta_1,\theta_2)$"
        zl2 = r"$f_2(\theta_2,\theta_1)$"
        tit = "protophase dynamics"

    fig = plt.figure(num=fignum) if fignum is not None else plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.plot_surface(X, Y, a.T, cmap="jet")
    ax2.plot_surface(X, Y, b, cmap="jet")
    for ax in (ax1, ax2):
        ax.set_xlim(0, PI2)
        ax.set_ylim(0, PI2)
        ax.set_xlabel(xl or "")
        ax.set_ylabel(yl or "")
    ax1.set_zlabel(zl1 or "")
    ax2.set_zlabel(zl2 or "")
    ax2.set_title(tit or "")
    return fig, (ax1, ax2)


def co_plottri(fignum, cpar, thresh=0.07):
    r"""Plot triadic directed coupling graph from matrix ``cpar``.

    Entries ``cpar(i,j)`` represent effect of node ``j`` on node ``i``.
    Edges weaker than ``thresh * max(cpar)`` are omitted.

    Returns
    -------
    tuple
        ``(figure, axes)`` handle tuple.
    """
    c = np.asarray(cpar, dtype=float).T
    cm = np.max(c)
    fig = plt.figure(num=fignum)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    x1, y1 = 0.0, 1.0
    x2, y2 = np.cos(np.pi / 6), -np.sin(np.pi / 6)
    x3, y3 = -x2, y2
    xs = np.array([x1, x2, x3])
    ys = np.array([y1, y2, y3])

    ax.scatter(xs, ys, s=1200, facecolors="none", edgecolors="k", linewidths=3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    for j in range(3):
        ax.text(xs[j], ys[j], str(j + 1), fontsize=18, ha="center", va="center")

    wm = 6.0
    if cm <= 0:
        return fig, ax
    c = c * wm / cm
    wt = thresh * wm

    def draw(start, end, width):
        arr = FancyArrowPatch(
            posA=start,
            posB=end,
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=width,
            color="k",
        )
        ax.add_patch(arr)

    a31s, a31e = (x3 + 0.25, y3 + 0.15), (x1 - 0.05, y1 - 0.26)
    a13s, a13e = (x1 - 0.25, y1 - 0.15), (x3 + 0.05, y3 + 0.25)
    a12s, a12e = (x1 + 0.05, y1 - 0.26), (x2 - 0.25, y2 + 0.15)
    a21s, a21e = (x2 - 0.03, y2 + 0.26), (x1 + 0.25, y1 - 0.13)
    a23s, a23e = (x2 - 0.25, y2 - 0.09), (x3 + 0.25, y3 - 0.09)
    a32s, a32e = (x3 + 0.25, y3 + 0.09), (x2 - 0.25, y3 + 0.09)
    a01e = (0.0, y1 - 0.29)
    a02e = (x2 - 0.25, y2 + 0.11)
    a03e = (x3 + 0.25, y2 + 0.11)

    if c[2, 0] > wt:
        draw(a31s, a31e, c[2, 0])
    if c[0, 2] > wt:
        draw(a13s, a13e, c[0, 2])
    if c[0, 1] > wt:
        draw(a12s, a12e, c[0, 1])
    if c[1, 0] > wt:
        draw(a21s, a21e, c[1, 0])
    if c[1, 2] > wt:
        draw(a23s, a23e, c[1, 2])
    if c[2, 1] > wt:
        draw(a32s, a32e, c[2, 1])
    if c[0, 0] > wt:
        draw((0.0, 0.0), a01e, c[0, 0])
    if c[1, 1] > wt:
        draw((0.0, 0.0), a02e, c[1, 1])
    if c[2, 2] > wt:
        draw((0.0, 0.0), a03e, c[2, 2])

    return fig, ax
