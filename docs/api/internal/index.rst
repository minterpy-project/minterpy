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
| :py:mod:`Non-Compiled <minterpy.utils>`                              | A high-level interface to conveniently create interpolants      |
+----------------------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`JIT-Compiled <minterpy.jit_compiled>`                       | Top domain-specific classes and abstract classes                |
+----------------------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`DDS <minterpy.dds>`                                         | Concrete classes representing polynomial bases                  |
+----------------------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`Transformation Utilities <minterpy.transformations.utils>`  | Concrete classes for polynomial basis transformations           |
+----------------------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`Generating Points <minterpy.gen_points>`                    | Concrete classes for polynomial basis transformation operators  |
+----------------------------------------------------------------------+-----------------------------------------------------------------+

.. toctree::
   :maxdepth: 1
   :hidden:

   Non-Compiled <utils/index>
   JIT-Compiled <jit_compiled/index>
   DDS <dds>
   Transformation utility <trafoutils>
   Generating points <gen-points>







