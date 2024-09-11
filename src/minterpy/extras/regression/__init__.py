"""
This sub-package provides polynomial regressions as an extra to Minterpy.

+---------------------------------+--------------------------------------------------------------------+
| Module                          | Description                                                        |
+=================================+====================================================================+
| :py:mod:`.regression_abc`       | The abstract base class for all polynomial regression classes      |
+---------------------------------+--------------------------------------------------------------------+
| :py:mod:`.ordinary_regression`  | The concrete implementation of the ordinary polynomial regression  |
+---------------------------------+--------------------------------------------------------------------+
"""

__all__ = []

from . import regression_abc  # noqa
from .regression_abc import *

from . import ordinary_regression
from .ordinary_regression import *

__all__ += regression_abc.__all__
__all__ += ordinary_regression.__all__

