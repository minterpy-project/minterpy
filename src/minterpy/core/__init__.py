"""
The core sub-package of Minterpy.

The sub-package contains several top domain-specific classes
(e.g., :py:mod:`.multi_index`)
and the abstract base classes.

.. warning::

   Modifications to this sub-package should only be made with a thorough
   understanding of the overall Minterpy code base.
   We strongly advise holding discussions with the project maintainers
   prior to any modifications.

+-------------------------+--------------------------------------------------------------------------------------+
| Module / Sub-package    | Description                                                                          |
+=========================+======================================================================================+
| :py:mod:`.multi_index`  | The set of multi-indices representing the exponents of multidimensional polynomials  |
+-------------------------+--------------------------------------------------------------------------------------+
| :py:mod:`.grid`         | The grid on which interpolating polynomials live                                     |
+-------------------------+--------------------------------------------------------------------------------------+
| :py:mod:`.tree`         | The data to carry out the multidimensional divided difference scheme (DDS)           |
+-------------------------+--------------------------------------------------------------------------------------+
| :py:mod:`.ABC`          | The core abstract base classes                                                       |
+-------------------------+--------------------------------------------------------------------------------------+
"""

__all__ = []

from . import multi_index  # noqa
from .multi_index import *  # noqa

__all__ += multi_index.__all__

from . import grid  # noqa
from .grid import *  # noqa

__all__ += grid.__all__

from . import ABC  # noqa # ABCs are not exposed to the top level!
from . import tree  # noqa
