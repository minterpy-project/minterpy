============
Internal API
============

This section contains the documentation for all low-level functions
and numerical routines that support the high-level constructs of Minterpy
(e.g., :py:class:`Grid <.core.grid.Grid>`,
:py:class:`MultiIndexSet <.core.multi_index.MultiIndexSet>`,
:py:class:`LagrangePolynomial <.polynomials.lagrange_polynomial.LagrangePolynomial>`,
:py:class:`NewtonPolynomial <.polynomials.newton_polynomial.NewtonPolynomial>`).

These functions typically operate on low-level data structures,
such as NumPy arrays or lists.

The internal utility functions in the Minterpy codebase are categorized into
two main types:

- Non-compiled internal functions
- Compiled (accelerated) internal functions

For several performance-critical numerical routines,
Minterpy leverages `Numba <http://numba.pydata.org>`_
to compile the code just-in-time,
significantly accelerating runtime performance.

.. warning::

   While these functions are publicly documented, due to their lower-level
   nature they may subject to sudden changes with each Minterpy release.
   The development team does not guarantee the stability of these function
   interfaces and reserves the right to modify them as needed.

+----------------------------------------------------------------------+-----------------------------------------------------------------+
| Module / Sub-package                                                 | Description                                                     |
+======================================================================+=================================================================+
| :py:mod:`Non-Compiled <minterpy.utils>`                              | Utility sub-package used across Minterpy (without compilation)  |
+----------------------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`JIT-Compiled <minterpy.jit_compiled>`                       | Just-in-time compiled numerical routines                        |
+----------------------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`DDS <minterpy.dds>`                                         | Implementation of the multivariate divided difference scheme    |
+----------------------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`Transformation Utilities <minterpy.transformations.utils>`  | Utility module with routines to compute transformation matrices |
+----------------------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`Generating Points <minterpy.gen_points>`                    | Utility module with routines to generate interpolation points   |
+----------------------------------------------------------------------+-----------------------------------------------------------------+

.. toctree::
   :maxdepth: 1
   :hidden:

   Non-Compiled <utils/index>
   JIT-Compiled <jit_compiled/index>
   DDS <dds>
   Transformation utility <trafoutils>
   Generating points <gen-points>







