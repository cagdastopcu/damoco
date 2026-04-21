import numpy as np

from domoco import co_plot2coef, co_plot2cplf, co_plotcoef, co_plotcplf, co_plottri


def test_plotting_smoke():
    n = 21
    F = np.random.default_rng(0).normal(size=(n, n)) + 1j * np.random.default_rng(1).normal(
        size=(n, n)
    )
    f = np.random.default_rng(2).normal(size=(41, 41))
    cpar = np.abs(np.random.default_rng(3).normal(size=(3, 3)))

    fig1, _ = co_plotcoef(F, pltype=1, fignum=101)
    fig2, _ = co_plotcplf(f, pltype=1, fignum=102)
    fig3, _ = co_plot2coef(F, F * 0.7, pltype=1, fignum=103)
    fig4, _ = co_plot2cplf(f, f * 0.5, pltype=1, fignum=104)
    fig5, _ = co_plottri(105, cpar, thresh=0.05)

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    assert fig4 is not None
    assert fig5 is not None
