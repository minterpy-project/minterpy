"""
The ABC sub-package of Minterpy.

The sub-package contains all the important abstract base classes of Minterpy.

+----------------------------------------------+-----------------------------------------------------------------+
| Module                                       | Description                                                     |
+==============================================+=================================================================+
| :py:mod:`.multivariate_polynomial_abstract`  | Define all polynomial bases                                     |
+----------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`.transformation_abstract`           | Define all transformations from polynomial basis to another     |
+----------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`.operator_abstract`                 | Define all transformation operators between polynomial bases    |
+----------------------------------------------+-----------------------------------------------------------------+
"""

__all__ = []


from . import multivariate_polynomial_abstract  # noqa
from .multivariate_polynomial_abstract import *  # noqa

__all__ += multivariate_polynomial_abstract.__all__


from . import transformation_abstract  # noqa
from .transformation_abstract import *  # noqa

__all__ += transformation_abstract.__all__


from . import operator_abstract  # noqa
from .operator_abstract import *  # noqa

__all__ += operator_abstract.__all__
