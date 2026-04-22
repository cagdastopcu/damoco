# domoco Documentation

## Overview

`domoco` is a Python implementation of the DAMOCO MATLAB toolbox functions for dynamical coupling analysis.

DAMOCO: Data Analysis with Models Of Coupled Oscillators

The project keeps the original `co_*` function names to make migration from MATLAB scripts straightforward.

## MATLAB Reference

- DAMOCO MATLAB toolbox page: http://www.stat.physik.uni-potsdam.de/%7Emros/damoco2.html

## Credits

- MATLAB toolbox creators: Björn Kralemann, Michael Rosenblum, Arkady Pikovsky (DAMOCO toolbox), and collaborators.
- Ported to Python by: Çağdaş Topçu.

## Installation

```bash
pip install -e .[dev]
```

## Rigorous Background

- Mathematical theory: [THEORY.md](THEORY.md)
- Validation methodology: [VALIDATION.md](VALIDATION.md)

## Package Structure

- `domoco.synchrony`: synchronization indices and max-scan utilities.
- `domoco.metrics`: correlations, norms, and directionality measures.
- `domoco.derivatives`: phase derivatives, residual decomposition, PRC iteration.
- `domoco.coupling`: Fourier/kernel coupling function estimation and triad/network analysis.
- `domoco.protophase`: protophase extraction and protophase-to-phase transforms.
- `domoco.plotting`: plotting helpers compatible with the original workflow.

## API (Public Functions)

### Synchrony

- `co_sync(P1, P2, n, m)`
- `co_sync3(P1, P2, P3, n, m, l)`
- `co_maxsync(theta1, theta2, or_)`
- `co_maxsync3(theta1, theta2, theta3, or_)`

### Metrics

- `co_dirin(N1, N2, omeg1, omeg2)`
- `co_dirpar(Fcoef1, Fcoef2)`
- `co_cor_diff(Q1, Q2)`
- `co_fcfcor(Qcoef1, Qcoef2)`
- `co_gcfcor(q1, q2)`
- `co_fcfcormax(Qcoef1, Qcoef2, ngrid=100)`
- `co_gcfcormax(q1, q2)`
- `co_fnorm(Qcoef)`
- `co_gnorm(q)`

### Derivatives and Decomposition

- `co_phidot1(phi, fsample)`
- `co_phidot2(phi1, phi2, fsample)`
- `co_phidot3(phi1, phi2, phi3, fsample)`
- `co_resid_decomp(dPhi1, Phi1, Phi2, q)`
- `co_prciter(Q, ngrid, fignum=0)`

### Coupling Estimation and Analysis

- `co_fcplfct1(phi1, phi2, dphi1, N, ngrid=100)`
- `co_fcplfct2(phi1, phi2, dphi1, dphi2, N, ngrid=100)`
- `co_fcpltri(phi1, phi2, phi3, Dphi1, Dphi2, Dphi3, N)`
- `co_tricplfan(Qcoef1, Qcoef2, Qcoef3, meth=1, thresh=2)`
- `co_nettri(PHI, PHI_dot, N, meth=1, thresh=2)`
- `co_kcplfct1(phi1, phi2, phi1_dot, ngrid, fignum=0, al_x=None, al_y=None)`
- `co_kcplfct2(phi1, phi2, phi1_dot, phi2_dot, ngrid, al_x=None, al_y=None)`

### Protophase and Transformations

- `co_fbtransf1(theta, nfft=80, alpha=0.05, ngrid=50)`
- `co_fbtrT(theta, ngrid=50)`
- `co_distproto(x, NV)`
- `co_hilbproto(x, fignum=0, x0=0.0, y0=0.0, ntail=1000)`
- `co_mmzproto(x, pl=0)`
- `co_avcyc(x, theta, N, PL=0, alpha=0.0)`

### Plotting

- `co_plotcoef(F, pltype, fignum=None, own_lab=None, ext_lab=None, tit=None)`
- `co_plotcplf(f, pltype, fignum=None, own_lab=None, ext_lab=None, zlab=None, tit=None)`
- `co_plot2coef(...)`
- `co_plot2cplf(...)`
- `co_plottri(fignum, cpar, thresh=0.07)`

## Development

Run tests:

```bash
pytest -q
```
