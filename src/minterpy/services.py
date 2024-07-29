"""
Common public high-level utility functions of the Minterpy package.
"""
import numpy as np

from typing import TYPE_CHECKING

from minterpy.global_settings import INT_DTYPE

# To avoid circular import in the type hints
if TYPE_CHECKING:
    from minterpy.core.ABC import MultivariatePolynomialSingleABC

__all__ = ["is_constant"]


def is_constant(poly: "MultivariatePolynomialSingleABC") -> bool:
    """Check if a polynomial instance is a constant.

    Constant multidimensional polynomial consists of a single multi-index set
    element of :math:`(0, \ldots, 0)`.

    Parameters
    ----------
    poly : MultivariatePolynomialSingleABC
        A given polynomial to check.

    Returns
    -------
    bool
        ``True`` if the polynomial is a constant polynomial,
        ``False`` otherwise.
    """
    # Check the multi-index set with early exit strategy
    mi = poly.multi_index
    # ...it has only one element
    has_one_element = len(mi) == 1
    if not has_one_element:
        return False
    # ...with zeros
    has_zero = np.zeros(mi.spatial_dimension, dtype=INT_DTYPE) in mi
    if not has_zero:
        return False

    # Check if the polynomial is initialized
    try:
        _ = poly.coeffs
    except ValueError:
        return False

    return True
