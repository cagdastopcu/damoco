# domoco

`domoco` is a Python port of the DAMOCO MATLAB toolbox functions in this repository.

DAMOCO: Data Analysis with Models Of Coupled Oscillators

## Reference MATLAB Toolbox

- Official DAMOCO MATLAB page: http://www.stat.physik.uni-potsdam.de/%7Emros/damoco2.html

## Attribution

- MATLAB toolbox creators: Björn Kralemann, Michael Rosenblum, Arkady Pikovsky (DAMOCO toolbox), and collaborators.
- Ported to Python by: Çağdaş Topçu.

## Scope

- Function names are preserved (`co_fcplfct2`, `co_tricplfan`, `co_hilbproto`, etc.).
- Default arguments and conventions were ported to match MATLAB behavior.
- Plotting utilities were reimplemented with Matplotlib.

## Install

Published package:

```bash
pip install domoco
```

Directly from GitHub:

```bash
pip install git+https://github.com/cagdastopcu/domoco.git
```

Local editable development install:

```bash
pip install -e .[dev]
```

## Test

```bash
pytest -q
```

## Documentation

- Full user/developer docs: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)
- Mathematical theory: [docs/THEORY.md](docs/THEORY.md)
- Validation guide: [docs/VALIDATION.md](docs/VALIDATION.md)
- PyPI release guide: [docs/PUBLISH.md](docs/PUBLISH.md)
