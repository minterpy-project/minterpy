"""
Common public high-level utility functions of the Minterpy package.
"""
import numpy as np

from typing import TYPE_CHECKING

from minterpy.global_settings import INT_DTYPE

# To avoid circular import in the type hints
if TYPE_CHECKING:
    from minterpy.core.ABC import MultivariatePolynomialSingleABC

__all__ = ["is_scalar"]


def is_scalar(poly: "MultivariatePolynomialSingleABC") -> bool:
    """Check if a polynomial instance is a constant scalar polynomial.

    A constant scalar multidimensional polynomial consists of a single
    multi-index set element of :math:`(0, \ldots, 0)`.

    Parameters
    ----------
    poly : MultivariatePolynomialSingleABC
        A given polynomial to check.

    Returns
    -------
    bool
        ``True`` if the polynomial is a constant scalar polynomial,
        ``False`` otherwise.

    Notes
    -----
    - A constant scalar polynomial is more specific than simply a constant
      polynomial. A constant polynomial may have a large multi-index set but
      with the coefficients that corresponds to the non-constant terms have
      zero value (for non-Lagrange polynomial). In the case of a Lagrange
      polynomial, a constant polynomial means that all the coefficients have
      a single unique value.
    """
    # Check if the polynomial is initialized
    try:
        _ = poly.coeffs
    except ValueError:
        return False

    # Check the multi-index set with early exit strategy
    mi = poly.multi_index
    # ...with zeros
    has_zero = np.zeros(mi.spatial_dimension, dtype=INT_DTYPE) in mi
    if not has_zero:
        return False
    # only a single element
    if len(mi) != 1:
        return False

    return True
