# Validation and Reproducibility Guide

## Goal

Establish parity between the original DAMOCO MATLAB implementation and this Python port for:

DAMOCO: Data Analysis with Models Of Coupled Oscillators

- API behavior (function names, defaults, outputs)
- numerical equivalence (within floating-point tolerance)
- expected qualitative behavior on synthetic systems

Reference MATLAB toolbox:
- http://www.stat.physik.uni-potsdam.de/%7Emros/damoco2.html

## 1. Validation Layers

### Layer A: Unit tests on mathematical invariants

Examples:
- synchronization indices are near 1 on constructed resonances
- correlation of identical functions is near 1
- phase derivative estimators recover known constant derivatives
- periodic boundary duplication is preserved for grid outputs

Current automated tests are in:
- `tests/test_sync_metrics.py`
- `tests/test_derivatives.py`
- `tests/test_coupling.py`
- `tests/test_protophase.py`
- `tests/test_plotting.py`

Run:

```bash
pytest -q
```

### Layer B: Cross-language parity tests (recommended)

For strict MATLAB-to-Python parity, use the same inputs in both environments:

1. Generate test data in MATLAB and save (`.mat`) including:
   - input phases/time series
   - outputs from selected `co_*` functions
2. Load same inputs in Python, run corresponding `domoco.co_*` calls.
3. Compare with metrics:
   - max absolute error (`L_inf`)
   - relative `L2` error
   - correlation

Suggested tolerance bands:
- `1e-12` to `1e-9` for simple algebraic quantities
- `1e-8` to `1e-5` for interpolative/integrative quantities
- case-specific tolerance for ill-conditioned regressions

### Layer C: End-to-end scientific validation

On representative datasets:
- compare inferred directionality signs
- compare coupling matrix sparsity/rank ordering
- compare triad/network conclusions under same threshold settings

## 2. Suggested Benchmark Cases

1. **Exact phase locking synthetic signals**
   - verify `co_sync`, `co_sync3`, `co_maxsync`, `co_maxsync3`
2. **Known Fourier coupling model**
   - verify `co_fcplfct1/2`, `co_fcfcor`, `co_fcfcormax`
3. **Known grid coupling model with shifts**
   - verify `co_gcfcor`, `co_gcfcormax`, `co_cor_diff`
4. **Noisy phase derivatives**
   - verify `co_phidot*`, `co_resid_decomp`
5. **Triad with designed coupling graph**
   - verify `co_fcpltri`, `co_tricplfan`, `co_nettri`
6. **Protophase methods**
   - verify monotonicity and cycle consistency of `co_hilbproto`, `co_mmzproto`, `co_distproto`, `co_fbtransf1`, `co_fbtrT`, `co_avcyc`

## 3. Known Sources of MATLAB vs Python Differences

- default linear algebra backend and conditioning effects
- interpolation engine differences
- floating-point summation order
- endpoint handling in numerical integration
- phase unwrap boundary behavior on noisy signals

These usually produce small numeric differences that should not change qualitative scientific interpretation when tolerances are chosen properly.

## 4. Reporting Validation

For publication-quality reproducibility, include:

- exact `domoco` version (PyPI tag or commit hash)
- Python, NumPy, SciPy versions
- MATLAB version/toolbox state
- dataset/preprocessing details
- tolerance definitions and pass/fail criteria

## 5. Optional Extension

Add a dedicated `tests/test_matlab_parity.py` that loads frozen MATLAB reference fixtures (`.mat`) and performs direct parity checks in CI.
