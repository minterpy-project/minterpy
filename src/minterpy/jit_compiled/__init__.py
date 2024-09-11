"""
This sub-package collects all just-in-time compiled codes for Minterpy.

Minterpy accelerates numerous performance-critical internal functions through
just-in-time (JIT) compilation using the `Numba <http://numba.pydata.org>`_
package.

Most JIT-compiled code is decorated with ``njit`` instead of ``jit``.
This means that the input and output requirements for the compiled functions
are explicityly define in advance.
While ``njit`` is less flexible than ``jit``, it is typically faster and
more memory efficient.

.. warning::

  Input validation and error checking are generally not implemented within
  JIT-compiled functions. Therefore, it is the caller's responsibility
  to ensure sure that the input and output requirements are met.

+----------------------------+-------------------------------------------------------------------+
| Module                     | Description                                                       |
+============================+===================================================================+
| :py:mod:`.common`          | Common performance-sensitivity numerical routines                 |
+----------------------------+-------------------------------------------------------------------+
| :py:mod:`.multi_index`     | Compiled numerical routines related to multi-indices              |
+----------------------------+-------------------------------------------------------------------+
| :py:mod:`.newton.diff`     | Compiled numerical routines to differentiate Newton polynomials   |
+----------------------------+-------------------------------------------------------------------+
| :py:mod:`.newton.eval`     | Compiled numerical routines to evaluate Newton polynomials        |
+----------------------------+-------------------------------------------------------------------+
| :py:mod:`.canonical`       | Compiled numerical routines related to the canonical basis        |
+----------------------------+-------------------------------------------------------------------+
| :py:mod:`.transformations` | Compiled and common numerical routines for basis transformations  |
+----------------------------+-------------------------------------------------------------------+
"""