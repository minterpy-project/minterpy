"""
This sub-package contains the concrete implementation of transformation operators.

Each concrete transformation operator class is defined in a separate module.
All the concrete classes implement (i.e., inherit from) the abstract
base class :py:class:`OperatorABC
<.core.ABC.transformation_operator.OperatorABC>`.

Polynomial basis transformations involve the transformation the polynomial
coefficients from one basis to another.
This is achieved via matrix operations where the transformation
is represented by a matrix.

Some concrete classes implemented in this sub-package exploits the special
structure of certain transformations to enable faster computations and reduce
storage requirements.

+-----------------------------+-----------------------------------------------------------------------+
| Module / Sub-Package        | Description                                                           |
+=============================+=======================================================================+
| :py:mod:`.matrix_operator`  | Transformation operator with a global matrix (no structure)           |
+-----------------------------+-----------------------------------------------------------------------+
| :py:mod:`.barycentric`      | Transformation operator with a recursive triangular sparse structure  |
+-----------------------------+-----------------------------------------------------------------------+
"""
from . import barycentric, matrix_operator  # noqa
