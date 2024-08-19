"""
Common public high-level utility functions of the Minterpy package.
"""
import numpy as np

import minterpy as mp
from typing import TYPE_CHECKING, Union

from minterpy.global_settings import INT_DTYPE
from minterpy.core.grid import Grid
from minterpy.core.multi_index import MultiIndexSet

# To avoid circular import in the type hints
if TYPE_CHECKING:
    from minterpy.core.ABC import MultivariatePolynomialSingleABC

__all__ = ["is_scalar"]


def is_scalar(
    obj: Union["MultivariatePolynomialSingleABC", Grid, MultiIndexSet],
) -> bool:
    """Check if a Minterpy object is a scalar.

    This check applies to both polynomial and grid objects.
    A scalar multidimensional polynomial (resp. grid) consists of a single
    multi-index set element of :math:`(0, \ldots, 0)`.

    Parameters
    ----------
    obj : Union[MultivariatePolynomialSingleABC, Grid, MultiIndexSet]
        A given polynomial, Grid, or MultiIndexSet to check.

    Returns
    -------
    bool
        ``True`` if the polynomial (resp. grid) is a constant scalar
        polynomial (resp. grid); ``False`` otherwise.

    Notes
    -----
    - A constant scalar polynomial is more specific than simply a constant
      polynomial. A constant polynomial may have a large multi-index set but
      with the coefficients that corresponds to the non-constant terms have
      zero value (for non-Lagrange polynomial). In the case of a Lagrange
      polynomial, a constant polynomial means that all the coefficients have
      a single unique value.
    """
    if isinstance(obj, MultiIndexSet):
        return _is_scalar_multi_index(obj)

    if isinstance(obj, Grid):
        return _is_scalar_grid(obj)

    if isinstance(obj, mp.core.ABC.MultivariatePolynomialSingleABC):
        return _is_scalar_poly(obj)

    raise TypeError(f"Not recognized type of object ({type(obj)}).")


def _is_scalar_multi_index(multi_index: MultiIndexSet) -> bool:
    """Check if a MultiIndexSet instance is a scalar."""
    exp_zero = np.zeros(multi_index.spatial_dimension, dtype=INT_DTYPE)
    has_zero = exp_zero in multi_index
    if not has_zero:
        return False
    if len(multi_index) != 1:
        return False

    return True


def _is_scalar_grid(grid: Grid) -> bool:
    """Check if a grid instance is a scalar."""
    mi_grid = grid.multi_index
    exp_zero = np.zeros(mi_grid.spatial_dimension, dtype=INT_DTYPE)
    has_zero = exp_zero in mi_grid
    if not has_zero:
        return False
    if len(mi_grid) != 1:
        return False

    return True


def _is_scalar_poly(poly: "MultivariatePolynomialSingleABC") -> bool:
    """Check if a polynomial instance is a scalar polynomial."""
    # Check if the polynomial is initialized
    try:
        _ = poly.coeffs
    except ValueError:
        return False

    # Check the multi-index set with early exit strategy
    mi = poly.multi_index
    # ...with zeros
    exp_zero = np.zeros(mi.spatial_dimension, dtype=INT_DTYPE)
    has_zero = exp_zero in mi
    if not has_zero:
        return False
    # only a single element
    if len(mi) != 1:
        return False

    if poly.indices_are_separate:
        return _is_scalar_grid(poly.grid)

    return True
