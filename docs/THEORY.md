# Mathematical Theory (DAMOCO Port)

## Scope

This document summarizes the mathematical objects implemented in `domoco`, with notation aligned to the original DAMOCO toolbox.

Reference MATLAB toolbox:
- http://www.stat.physik.uni-potsdam.de/%7Emros/damoco2.html

## 1. Phase and Synchronization Indices

For phase/protophase signals `phi_1(t), phi_2(t), phi_3(t)`, synchronization indices are:

- Two-oscillator:
  - `rho_{n:m} = | <exp(i (n*phi_1 - m*phi_2))> |`
- Three-oscillator:
  - `rho_{n:m:l} = | <exp(i (n*phi_1 + m*phi_2 + l*phi_3))> |`

where `<...>` is the time average.

Implemented by:
- `co_sync`, `co_sync3`, `co_maxsync`, `co_maxsync3`

## 2. Fourier Representation of Coupling Functions

Pairwise coupling functions are represented as:

- `q(phi_1, phi_2) = sum_{n=-N..N} sum_{m=-N..N} Q_{n,m} exp(i (n phi_1 + m phi_2))`

Triadic coupling functions are represented as:

- `q(phi_1, phi_2, phi_3) = sum_{n,m,k=-N..N} Q_{n,m,k} exp(i (n phi_1 + m phi_2 + k phi_3))`

Coefficients are estimated by linear regression in the complex Fourier basis:

- Solve `C * q_vec = B` where matrix `C` is built from empirical Fourier moments of phases.

Implemented by:
- `co_fcplfct1`, `co_fcplfct2`, `co_fcpltri`

## 3. Correlation Metrics for Coupling Functions

### 3.1 Grid-based correlation

For grid functions `q1`, `q2` (after mean removal):

- `COR = <q1, q2> / (||q1|| ||q2||)`

where inner product and norm are computed with 2D trapezoidal integration.

Implemented by:
- `co_gcfcor`, `co_cor_diff`

### 3.2 Fourier-based correlation

Correlation is computed directly from Fourier coefficients with DAMOCO index reversal convention.

Implemented by:
- `co_fcfcor`

### 3.3 Max-shift correlation

Relative phase shifts are optimized:

- `COR_max = max_{Delta_1, Delta_2} COR(q1, q2 shifted by Delta_1, Delta_2)`

Implemented by:
- `co_fcfcormax`, `co_gcfcormax`

## 4. Coupling Norms and Directionality

### 4.1 Norms

- Grid norm:
  - `omega = mean(q)`
  - `Nrm = ||q - omega||`
- Fourier norm:
  - `omega = Re(Q_{0,0})`
  - `Nrm = ||Q with Q_{0,0}=0||`

Implemented by:
- `co_gnorm`, `co_fnorm`

### 4.2 Directionality index

Basic form (normalized by autonomous frequencies):

- `c1 = N1 / omega1`
- `c2 = N2 / omega2`
- `D = (c2 - c1) / (c1 + c2)`

Fourier-partial-derivative variant uses:

- `nrm = sqrt(sum_{n,m} m^2 |F_{n,m}|^2)`

Implemented by:
- `co_dirin`, `co_dirpar`

## 5. Triadic/Network Coupling Structure

Given triadic Fourier coefficients, pair effects are summarized by:

- Partial norms (methods 1/2)
- Partial derivative norms (methods 3/4)
- Optional normalization by autonomous frequencies `omega` (methods 2/4)

For network inference (`M > 3`), all triplets are analyzed and pairwise coupling is aggregated by the minimum over all triplets containing that ordered pair.

Implemented by:
- `co_tricplfan`, `co_nettri`

## 6. Phase Derivatives and Residual Decomposition

Phase derivatives are estimated with Savitzky-Golay differentiation (order 4, window length 9), matching MATLAB defaults.

Residual decomposition:

- `dphi_synth = q(phi_1, phi_2)` (interpolated on periodic grid)
- `resid = dphi_measured - dphi_synth`

Implemented by:
- `co_phidot1`, `co_phidot2`, `co_phidot3`, `co_resid_decomp`

## 7. PRC/Forcing Factorization

`co_prciter` approximates:

- `Q(phi_1, phi_2) - omega ≈ Z(phi_1) * H(phi_2)`

by scanning `omega`, minimizing residual standard deviation, and iterating alternating updates for `Z` and `H`.

## 8. Protophase Estimation and Protophase-to-Phase Transform

### 8.1 Hilbert protophase

- `theta = arg( x + i H[x] ) mod 2pi`

Implemented by:
- `co_hilbproto`

### 8.2 Marker-based protophase

Uses 4 markers per cycle (max, zero crossing, min, zero crossing), then linear interpolation.

Implemented by:
- `co_mmzproto`

### 8.3 Distance-based protophase

Uses cumulative arc-length between Poincare section crossings.

Implemented by:
- `co_distproto`

### 8.4 Fourier transforms to true phase

Protophase density Fourier coefficients are used to correct `theta` into `phi`.

Implemented by:
- `co_fbtransf1`, `co_fbtrT`

### 8.5 Average-cycle projection

Computes complex average cycle and refines phase by minimizing projection error pointwise.

Implemented by:
- `co_avcyc`

## 9. Numerical Conventions

- Phases are wrapped with period `2*pi` where required.
- Grid functions keep periodic end points (last index duplicates first).
- Means/autonomous frequencies are explicitly removed in correlation/norm metrics where DAMOCO does so.
- Many estimators are sensitive to near-synchrony, weak coupling, and noisy derivatives; use thresholds and residual checks.
