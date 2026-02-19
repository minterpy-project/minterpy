"""
The core sub-package of Minterpy.

The sub-package contains several top domain-specific classes
(e.g., :py:mod:`.multi_index`, :py:mod:`.grid`)
and the abstract base classes.

.. important::

   This sub-package forms the computational core of Minterpy.
   Its components are tightly coupled, and any significant changes may have
   far-reaching downstream effects on correctness, accuracy, and performance.
   We recommend discussing any proposed modifications
   with the project maintainers before proceeding.

+-------------------------+--------------------------------------------------------------------------------------------+
| Module / Sub-package    | Description                                                                                |
+=========================+============================================================================================+
| :py:mod:`.multi_index`  | The set of multi-indices representing the exponents of multidimensional polynomials        |
+-------------------------+--------------------------------------------------------------------------------------------+
| :py:mod:`.domain`       | The user-defined domain and its transformation to and from the internal domain             |
+-------------------------+--------------------------------------------------------------------------------------------+
| :py:mod:`.grid`         | The interpolation grid of unisolvent nodes defined by multi-indices and generating points  |
+-------------------------+--------------------------------------------------------------------------------------------+
| :py:mod:`.tree`         | The data to carry out the multidimensional divided difference scheme (DDS)                 |
+-------------------------+--------------------------------------------------------------------------------------------+
| :py:mod:`.ABC`          | The abstract base classes for multivariate polynomial representations                      |
+-------------------------+--------------------------------------------------------------------------------------------+
"""

__all__ = []

from . import multi_index  # noqa
from .multi_index import *  # noqa

__all__ += multi_index.__all__

from . import domain  # noqa
from .domain import *  # noqa

__all__ += domain.__all__

from . import grid  # noqa
from .grid import *  # noqa

__all__ += grid.__all__

from . import ABC  # noqa # ABCs are not exposed to the top level!
from . import tree  # noqa
