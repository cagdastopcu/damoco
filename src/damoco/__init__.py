"""damoco: Python port of DAMOCO MATLAB functions.

DAMOCO: Data Analysis with Models Of Coupled Oscillators

Reference MATLAB toolbox:
http://www.stat.physik.uni-potsdam.de/%7Emros/damoco2.html

Attribution:
- MATLAB toolbox creators: Björn Kralemann, Michael Rosenblum, Arkady Pikovsky, and collaborators.
- Ported to Python by: Çağdaş Topçu.
"""

from .coupling import (
    co_fcplfct1,
    co_fcplfct2,
    co_fcpltri,
    co_kcplfct1,
    co_kcplfct2,
    co_nettri,
    co_tricplfan,
)
from .derivatives import (
    co_phidot1,
    co_phidot2,
    co_phidot3,
    co_prciter,
    co_resid_decomp,
)
from .metrics import (
    co_cor_diff,
    co_dirin,
    co_dirpar,
    co_fcfcormax,
    co_fcfcor,
    co_fnorm,
    co_gcfcor,
    co_gcfcormax,
    co_gnorm,
)
from .plotting import (
    co_plot2coef,
    co_plot2cplf,
    co_plotcoef,
    co_plotcplf,
    co_plottri,
)
from .protophase import (
    co_avcyc,
    co_distproto,
    co_fbtrT,
    co_fbtransf1,
    co_hilbproto,
    co_mmzproto,
)
from .synchrony import co_maxsync, co_maxsync3, co_sync, co_sync3

__all__ = [
    "co_avcyc",
    "co_cor_diff",
    "co_dirin",
    "co_dirpar",
    "co_distproto",
    "co_fbtrT",
    "co_fbtransf1",
    "co_fcfcormax",
    "co_fcfcor",
    "co_fcplfct1",
    "co_fcplfct2",
    "co_fcpltri",
    "co_fnorm",
    "co_gcfcor",
    "co_gcfcormax",
    "co_gnorm",
    "co_hilbproto",
    "co_kcplfct1",
    "co_kcplfct2",
    "co_maxsync",
    "co_maxsync3",
    "co_mmzproto",
    "co_nettri",
    "co_phidot1",
    "co_phidot2",
    "co_phidot3",
    "co_plot2coef",
    "co_plot2cplf",
    "co_plotcoef",
    "co_plotcplf",
    "co_plottri",
    "co_prciter",
    "co_resid_decomp",
    "co_sync",
    "co_sync3",
    "co_tricplfan",
]
