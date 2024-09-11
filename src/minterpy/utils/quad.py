"""
This module contains utility functions related to quadrature.
"""

from typing import Callable

import numpy as np
from scipy.special import roots_legendre


def gauss_leg(
    fun: Callable, num_points: int, bounds: np.ndarray
) -> np.ndarray:
    """Integrate a one-dimensional function using Gauss-Legendre quadrature.

    Parameters
    ----------
    fun : Callable
        The function to integrate, the output may be a vector.
    num_points : int
        The number of points used in the quadrature scheme.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds of integration, an 1-by-2 array (lower and upper bounds).

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integral of the function over the given bounds.
    """
    quad_nodes, quad_weights = roots_legendre(num_points)

    bound_diff = np.diff(bounds)
    bound_sum = np.sum(bounds)

    fun_vals = fun(bound_diff / 2.0 * quad_nodes + bound_sum / 2)

    integrals = bound_diff / 2 * quad_weights @ fun_vals

    return integrals
