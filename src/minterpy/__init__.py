"""
This is the minterpy package init.

isort:skip_file
"""

from .version import version as __version__

__all__ = [
    "__version__",
]

# --- Core sub-package
from . import core  # noqa
from .core import *  # noqa

__all__ += core.__all__

# --- Polynomial sub-package
from . import polynomials  # noqa
from .polynomials import *  # noqa

__all__ += polynomials.__all__

# --- Transformation sub-package
from . import transformations  # noqa
from .transformations import *  # noqa

__all__ += transformations.__all__

# --- Interpolation module
from . import interpolation  # noqa
from .interpolation import *  # noqa

__all__ += interpolation.__all__

# --- Extras (sub-packages)
from . import extras  # noqa
from .extras import regression  # noqa
from .extras.regression import *  # noqa

__all__ += regression.__all__

# High-level utility functions
from . import services  # noqa
from .services import *  # noqa

__all__ += services.__all__
