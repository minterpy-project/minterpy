"""
The utility sub-package used across Minterpy.

The utility functions are organized into modules based on the context
in which they are used.
For example, utilities related to the Newton polynomial are located
in the ``minterpy.utils.polynomials.newton`` module.

This organization prevents utility modules from being scattered across various
Minterpy sub-packages.
Although a utility module may suggest that the functions within are generic,
they are usually specific to a particular context.

Moreover, these functions are organized not only because they may be used
in multiple places, but also to represent a distinct layer of abstraction.
These functions operate on lower-level data structures,
predominantly NumPy arrays, and do not assume any knowledge of higher-level
constructs from the Minterpy abstraction,
such as instances of :py:mod:`concrete polynomials <.polynomials>`,
:py:class:`.MultiIndexSet`, or :py:class:`.Grid`.

+-----------------------------------+---------------------------------------------------------------+
| Module                            | Description                                                   |
+===================================+===============================================================+
| :py:mod:`.arrays`                 | Common and relevant numerical routines that operate on arrays |
+-----------------------------------+---------------------------------------------------------------+
| :py:mod:`.multi_index`            | Numerical routines relevant to multi-indices of exponents     |
+-----------------------------------+---------------------------------------------------------------+
| :py:mod:`.polynomials.lagrange`   | Numerical routines relevant to the Lagrange basis             |
+-----------------------------------+---------------------------------------------------------------+
| :py:mod:`.polynomials.newton`     | Numerical routines relevant to the Newton basis               |
+-----------------------------------+---------------------------------------------------------------+
| :py:mod:`.polynomials.canonical`  | Numerical routines relevant to the canonical basis            |
+-----------------------------------+---------------------------------------------------------------+
| :py:mod:`.polynomials.chebyshev`  | Numerical routines relevant to the Chebyshev basis            |
+-----------------------------------+---------------------------------------------------------------+
| :py:mod:`.quad`                   | Numerical routines relevant to quadrature                     |
+-----------------------------------+---------------------------------------------------------------+
| :py:mod:`.verification`           | Utility functions to verify a given value                     |
+-----------------------------------+---------------------------------------------------------------+
"""
