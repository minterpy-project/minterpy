"""
The utility sub-package used across Minterpy.

The utility functions are organized into modules according to the context
in which they appear. For instance, utilities related to the Newton polynomial
is in the `newton.py` module.

The reason to organize them here is to avoid having utility modules scattered
all across Minterpy sub-packages. A utility module gives the impression that
the functions within are generic, but they are most probably not.
The functions would appear in a very specific context.

Furthermore, the organization of such functions are not limited to
them being used in multiple places, but also as a distinct layer of
abstractions. The functions included here are functions that operates on
a lower-level data structure predominantly NumPy arrays.

The functions assume no knowledge about constructs coming from the upper layer
of Minterpy abstraction such as instances of the concrete polynomial classes,
``MultiIndexSet``, ``Grid``, etc.
"""
