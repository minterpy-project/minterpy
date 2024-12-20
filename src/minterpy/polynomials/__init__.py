"""
This sub-package contains the concrete implementations of polynomial bases.

Each concrete polynomial basis class is defined in its own module.
All the concrete polynomial bases implement (i.e., inherit from) the abstract
base class :py:class:`MultivariatePolynomialSingleABC
<.core.ABC.multivariate_polynomial_abstract.MultivariatePolynomialSingleABC>`.

+---------------------------------+-----------------------------------------------------+
| Module                          | Description                                         |
+=================================+=====================================================+
| :py:mod:`.lagrange_polynomial`  | The Lagrange polynomial basis                       |
+---------------------------------+-----------------------------------------------------+
| :py:mod:`.newton_polynomial`    | The Newton polynomial basis                         |
+---------------------------------+-----------------------------------------------------+
| :py:mod:`.canonical_polynomial` | The canonical polynomial basis                      |
+---------------------------------+-----------------------------------------------------+
| :py:mod:`.chebyshev_polynomial` | The Chebyshev polynomial (of the first kind) basis  |
+---------------------------------+-----------------------------------------------------+
"""

__all__ = []

from . import canonical_polynomial  # noqa
from .canonical_polynomial import *  # noqa

__all__ += canonical_polynomial.__all__

from . import newton_polynomial  # noqa
from .newton_polynomial import *  # noqa

__all__ += newton_polynomial.__all__

from . import lagrange_polynomial  # noqa
from .lagrange_polynomial import *  # noqa

__all__ += lagrange_polynomial.__all__

from . import chebyshev_polynomial  # noqa
from .chebyshev_polynomial import * # noqa

__all__ += chebyshev_polynomial.__all__
