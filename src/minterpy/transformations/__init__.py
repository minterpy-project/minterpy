"""
This sub-package contains the concrete implementations of transformations.

Each concrete basis transformation class is defined in a separate module, with
each class handling the the transformation from one polynomial basis
(the origin) to another basis (the target).
All the concrete classes implement the abstract base class
:py:class:`TransformationABC
<.core.ABC.transformation_abstract.TransformationABC>`.

+------------------------+---------------------------------------------------------------------+
| Module                 | Description                                                         |
+========================+=====================================================================+
| :py:mod:`.lagrange`    | Concrete transformation classes **from** the Lagrange basis         |
+------------------------+---------------------------------------------------------------------+
| :py:mod:`.newton`      | Concrete transformation classes **from** the Newton basis           |
+------------------------+---------------------------------------------------------------------+
| :py:mod:`.canonical`   | Concrete transformation classes **from** the canonical basis        |
+------------------------+---------------------------------------------------------------------+
| :py:mod:`.chebyshev`   | Concrete transformation classes **from** the Chebyshev basis        |
+------------------------+---------------------------------------------------------------------+
| :py:mod:`.identity`    | Concrete transformation class **from** one basis to the same basis  |
+------------------------+---------------------------------------------------------------------+
| :py:mod:`.interface`   | High-level helper functions related to basis transformations        |
+------------------------+---------------------------------------------------------------------+
| :py:mod:`.utils`       | Low-level utility functions related to basis transformations        |
+------------------------+---------------------------------------------------------------------+
"""


__all__ = []

from . import canonical  # noqa
from .canonical import *  # noqa

__all__ += canonical.__all__

from . import newton  # noqa
from .newton import *  # noqa

__all__ += newton.__all__

from . import lagrange  # noqa
from .lagrange import *  # noqa

__all__ += lagrange.__all__

from . import chebyshev  # noqa
from .chebyshev import *  # noqa

__all__ += chebyshev.__all__

from . import identity  # noqa
from .identity import *  # noqa

__all__ += identity.__all__

from . import interface  # noqa
from .interface import *  # noqa

__all__ += interface.__all__

from . import utils  # noqa # utils are not exposed to toplevel!
