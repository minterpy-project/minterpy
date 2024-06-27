"""
The utility sub-package used across Minterpy.

The utility functions are organized into modules according to the context
in which they appear. For instance, utilities related to the Newton polynomial
is in the `newton.py` module.

The reason to organize them here is to avoid having utility modules scattered
all across Minterpy sub-packages. A utility module gives the impression that
the functions within are generic but they are most probably not. The functions
would appear in a very specific context.
"""
