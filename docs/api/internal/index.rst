############
Internal API
############

This section contains the documentation of all low-level functions
and numerical routines that support high-level constructs of Minterpy
(e.g., :py:class:`Grid <.core.grid.Grid>`,
:py:class:`MultiIndexSet <.core.multi_index.MultiIndexSet>`,
:py:class:`LagrangePolynomial <.polynomials.lagrange_polynomial.LagrangePolynomial>`,
:py:class:`NewtonPolynomial <.polynomials.newton_polynomial.NewtonPolynomial>`).

These functions typically operates on low-level data structures like
NumPy arrays or lists.

Internal utility functions in the Minterpy codebase are classified into two
broad categories:

1. Non-compiled internal functions
2. Compiled (accelerated) internal functions

For several performance critical numerical routines, Minterpy uses
`Numba <http://numba.pydata.org>`_ to compile the code just-in-time
and accelerate the runtime.

.. toctree::
   :maxdepth: 1

   Non-compiled <utils/index>
   Compiled <jit_compiled/index>
   Transformation utility <trafoutils>
   Generating points <gen-points>
