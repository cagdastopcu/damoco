import numpy as np

from damoco import (
    co_avcyc,
    co_distproto,
    co_fbtrT,
    co_fbtransf1,
    co_hilbproto,
    co_mmzproto,
)


def test_fourier_transforms_on_uniform_phase():
    theta = np.linspace(0, 80 * np.pi, 12000)
    p1, arg1, sigma1 = co_fbtransf1(theta, nfft=30, alpha=0.05, ngrid=80)
    p2, arg2, sigma2 = co_fbtrT(theta, ngrid=80)
    assert p1.shape == theta.shape
    assert p2.shape == theta.shape
    assert arg1.shape == sigma1.shape
    assert arg2.shape == sigma2.shape
    assert np.mean(np.abs(p1 - theta)) < 1e-2
    assert np.mean(np.abs(p2 - theta)) < 1e-2


def test_hilb_mmz_dist_proto():
    t = np.linspace(0, 60 * np.pi, 10000)
    x = np.sin(t)
    theta_h, minamp = co_hilbproto(x, fignum=0, ntail=200)
    assert len(theta_h) == len(x) - 400
    assert minamp > 0

    theta_m, start, stop = co_mmzproto(x, pl=0)
    assert np.isfinite(start)
    assert np.isfinite(stop)
    assert len(theta_m) > 0
    assert stop > start

    z = np.cos(t) + 1j * np.sin(t)
    theta_d, s0, s1 = co_distproto(z, np.array([1.0, 0.0]))
    assert s1 > s0
    assert np.nanmax(theta_d) > np.nanmin(theta_d)


def test_avcyc_basic_shape():
    t = np.linspace(0, 40 * np.pi, 2200)
    x = np.sin(t) + 0.2 * np.sin(2 * t)
    theta0 = t.copy()
    phi, Cav = co_avcyc(x, theta0, N=3, PL=0, alpha=0.1)
    assert len(phi) == len(theta0) - 2000
    assert Cav.shape == (4,)
