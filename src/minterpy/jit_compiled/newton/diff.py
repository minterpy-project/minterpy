"""
A module for compiled code for polynomial differentiation in the Newton basis.
"""
import math

import numpy as np
from numba import njit, prange

from minterpy.global_settings import F_2D, F_1D, I_1D, FLOAT, INT, I_2D
from minterpy.jit_compiled.common import combinations_iter, get_max_columnwise

__all__ = []


@njit(F_2D(F_1D, I_1D, F_2D, I_1D), cache=True)
def create_lut(
    x: np.ndarray,
    max_exponents: np.ndarray,
    generating_points: np.ndarray,
    derivative_order_along: np.ndarray,
):
    """Create a look-up table of one-dimensional Newton monomials derivatives
    evaluated on a single query point.

    The following symbols are used in the following as shortcuts:

    - ``m``: spatial dimension
    - ``n``: polynomial degree
    - ``n_max``: maximum exponent in all dimension

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        A single query point at which the derivative is evaluated;
        the values are given in a one-dimensional array of length ``m``.
    max_exponents : :class:`numpy:numpy.ndarray`
        The maximum exponents in the multi-index set for each dimension given
        as one-dimensional non-negative integer array of length ``m``.
    generating_points : :class:`numpy:numpy.ndarray`
        Interpolation points for each dimension given as a two-dimensional
        array of shape ``(m, n + 1)``.
    derivative_order_along : :class:`numpy:numpy.ndarray`
        Specification of the orders of derivative along each dimension given
        as a one-dimensional non-negative integer array of length ``m``.
        For example, the array ``np.array([2, 3, 1])`` specifies the 2nd-order,
        3rd-order, and 1st-order derivatives along the 1st, 2nd, and 3rd
        dimension, respectively.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The look-up table (LUT) of shape ``(n_max, m)`` where each column
        consists of the derivative value of one-dimensional Newton monomials
        at the (single) query point.

    Notes
    -----
    - This function is compiled (NJIT-ted) with the help of Numba.

    See Also
    --------
    minterpy.jit_compiled.newton.diff.compute_from_lut
        Based on the look-up table (LUT) created by the current function,
        compute the multi-dimensional differentiated Newton monomials
        on a single query point for all elements in the multi-index set.
    """
    # Get the spatial dimension of the polynomial
    m = len(x)

    # Create an output array, differentiated 1D monomials in each dimension
    num_prods = np.max(max_exponents) + 1
    products = np.empty((num_prods, m), dtype=FLOAT)

    # Loop over each dimension
    for j in range(m):

        # Get dimension-dependent data
        max_exponent_in_dim = max_exponents[j]
        x_j = x[j]
        order = derivative_order_along[j]

        # Differentiate the monomials
        if order == 0:
            # No differentation along this dimension
            prod = 1.0  # multiplicative identity
            for i in range(max_exponent_in_dim):
                prod *= (x_j - generating_points[i, j])
                exponent = i + 1
                products[exponent, j] = prod

        else:
            # Take partial derivative of `order` along this dimension

            # Degree of monomial < order of derivative...
            products[:order, j] = 0.0  # then the derivatives are zero

            # Order of derivative > the degree of monomial...
            if order >= num_prods:
                continue  # all zeros, move on to the next dimension

            # The `order`-th derivative of the `order`-th monomial...
            fact = math.gamma(order + 1)
            products[order, j] = fact  # then it is the factorial
            # NOTE: use `math.gamma(n + 1)` instead of `math.factorial(n)`
            # as the latter is not supported by Numba.

            # Use chain rule to compute the derivative of products
            # for the higher-degree monomials
            for k in range(order + 1, max_exponent_in_dim + 1):
                elements = np.arange(k, dtype=INT)
                combs = combinations_iter(elements, int(k - order))

                res = 0.0  # additive identity
                for comb in combs:
                    prod = 1.0  # multiplicative identity
                    for idx in comb:
                        prod *= (x_j - generating_points[idx, j])
                    res += prod

                res *= fact
                products[k, j] = res

    return products


@njit(F_1D(I_2D, F_2D, I_1D), cache=True)
def compute_from_lut(
    exponents: np.ndarray,
    lut_diff: np.ndarray,
    derivative_order_along: np.ndarray,
) -> np.ndarray:
    """Evaluate the derivatives of multi-dimensional Newton monomials evaluated
    on a single query point.

    The following symbols are used in the following as shortcuts:

    - ``m``: spatial dimension
    - ``N``: number of coefficients or the cardinality of the multi-index set
    - ``n_max``: maximum exponent in all dimension

    Parameters
    ----------
    exponents : :class:`numpy:numpy.ndarray`
        Set of exponents given as a two-dimensional non-negative integer array
        of shape ``(N, m)``.
    lut_diff : :class:`numpy:numpy.ndarray`
        A look-up table that consists of the one-dimensional differentiated
        Newton monomials evaluated on a single query point.
        The table is a two-dimensional array of shape ``(n_max, m)``.
    derivative_order_along : :class:`numpy:numpy.ndarray`
        Specification of the orders of derivative along each dimension given
        as a one-dimensional non-negative integer array of length ``m``.
        For example, the array ``np.array([2, 3, 1])`` specifies the 2nd-order,
        3rd-order, and 1st-order derivatives along the 1st, 2nd, and 3rd
        dimension, respectively.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The values of all differentiated multi-dimensional Newton monomials
        in a multi-index set of exponents evaluated on a single query point.
        The array is one-dimensional of length ``N``.

    Notes
    -----
    - This function is compiled (NJIT-ted) with the help of Numba.

    See Also
    --------
    minterpy.jit_compiled.newton.diff.create_lut
        Construct a look-up table of 1-D Newton monomials derivative evaluated
        on a single query point; used by this function to compute
        the multi-dimensional monomials of all elements in the multi-index set.
    """
    num_monomials, num_dim = exponents.shape

    # Create an output array
    monomials = np.empty(num_monomials, dtype=FLOAT)

    # Loop over all monomials in the multi-index set
    for i in range(num_monomials):
        newt_mon_val = 1.0  # multiplicative identity
        for j in range(num_dim):
            exponent = exponents[i, j]
            if exponent > 0:
                newt_mon_val *= lut_diff[exponent, j]
            else:
                order = derivative_order_along[j]
                if order > 0:
                    # monomial degree < order of derivative
                    newt_mon_val = 0.0
                # Otherwise no need to multiply with exponent 0

        monomials[i] = newt_mon_val

    return monomials


@njit(F_1D(F_1D, F_2D, I_2D, F_2D, I_1D), cache=True)
def eval_single_query(
    x: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    derivative_order_along: np.ndarray,
) -> np.ndarray:
    """Evaluate the derivative of a Newton poly(s) on a single query point.

    The following symbols are used in the subsequent description as shortcuts:

    - ``m``: spatial dimension
    - ``n``: polynomial degree
    - ``N``: number of coefficients or the cardinality of the multi-index set
    - ``np``: number of polynomials (i.e., number of coefficient sets)

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        A single query point at which the derivative is evaluated;
        the values are given in a one-dimensional array of length ``m``.
    coefficients : :class:`numpy:numpy.ndarray`
        The coefficients of the Newton polynomial;
        the values are given in a two-dimensional array of shape ``(N, np)``.
    exponents : :class:`numpy:numpy.ndarray`
        Set of exponents given as a two-dimensional non-negative integer array
        of shape ``(N, m)``.
    generating_points : :class:`numpy:numpy.ndarray`
        Interpolation points for each dimension given as a two-dimensional
        array of shape ``(m, n + 1)``.
    derivative_order_along : :class:`numpy:numpy.ndarray`
        Specification of the orders of derivative along each dimension given
        as a one-dimensional non-negative integer array of length ``m``.
        For example, the array ``np.array([2, 3, 1])`` specifies the 2nd-order,
        3rd-order, and 1st-order derivatives along the 1st, 2nd, and 3rd
        dimension, respectively.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The value of the derivative at the query point given
        as a one-dimensional array of length ``np``.

    Notes
    -----
    - This is a direct differentiation and evaluation of polynomial(s) in
      the Newton basis on a single query point via chain rule
      without any transformation to another (e.g., canonical) basis.
    - This function is compiled (NJIT-ted) with the help of Numba.

    See Also
    --------
    minterpy.jit_compiled.newton.diff.eval_multiple_query
        Evaluation of the derivative of polynomial(s) in the Newton basis
        for multiple query points.
    """
    # Compute the column-wise maximum of the exponents
    max_exponents = get_max_columnwise(exponents)

    # Compute differentiated 1D Newton monomials look-up table (LUT)
    lut_diff = create_lut(
        x,
        max_exponents,
        generating_points,
        derivative_order_along,
    )

    # Compute the differentiated Newton monomials values from the LUT
    monomials = compute_from_lut(
        exponents,
        lut_diff,
        derivative_order_along,
    )

    # Make the coefficients contiguous so Numba does not complain
    coefficients = np.ascontiguousarray(coefficients)
    monomials = np.ascontiguousarray(monomials)

    return np.dot(monomials, coefficients)


@njit(F_2D(F_2D, F_2D, I_2D, F_2D, I_1D), cache=True)
def eval_multiple_query(
    xx: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    derivative_order_along: np.ndarray,
) -> np.ndarray:
    """Evaluate the derivative of Newton poly(s) on multiple query points.

    The following symbols are used in the subsequent description as shortcuts:

    - ``k``: number of query points
    - ``m``: spatial dimension
    - ``n``: polynomial degree
    - ``N``: number of coefficients or the cardinality of the multi-index set
    - ``np``: number of polynomials (i.e., number of coefficient sets)

    Parameters
    ----------
    xx : :class:`numpy:numpy.ndarray`
        The set of query points at which the derivative is evaluated;
        the values are given in a two-dimensional array of shape ``(k, m)``.
    coefficients : :class:`numpy:numpy.ndarray`
        The coefficients of the Newton polynomial;
        the values are given in a two-dimensional array of shape ``(N, np)``.
    exponents : :class:`numpy:numpy.ndarray`
        Set of exponents given as a two-dimensional non-negative integer array
        of shape ``(N, m)``.
    generating_points : :class:`numpy:numpy.ndarray`
        Interpolation points for each dimension given as a two-dimensional
        array of shape ``(m, n + 1)``.
    derivative_order_along : :class:`numpy:numpy.ndarray`
        Specification of the orders of derivative along each dimension given
        as a one-dimensional non-negative integer array of length ``m``.
        For example, the array ``np.array([2, 3, 1])`` specifies the 2nd-order,
        3rd-order, and 1st-order derivatives along the 1st, 2nd, and 3rd
        dimension, respectively.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The value of the derivative at the query point given
        as a two-dimensional array of shape ``(k, np)``.

    Notes
    -----
    - This is a direct differentiation and evaluation of polynomial(s) in
      the Newton basis on multiple query points via chain rule
      without any transformation to another (e.g., canonical) basis.
    - This function is compiled (NJIT-ted) with the help of Numba.

    See Also
    --------
    minterpy.jit_compiled.newton.diff.eval_single_query
        Evaluation of the derivative of polynomial(s) in the Newton form
        for a single query point.
    """
    num_points = len(xx)
    num_polys = coefficients.shape[1]

    # Create the output array
    output = np.empty(shape=(num_points, num_polys), dtype=FLOAT)

    # Loop over query points
    for i in range(num_points):
        x_i = xx[i, :]

        # Compute the evaluation at a single point
        res = eval_single_query(
            x_i,
            coefficients,
            exponents,
            generating_points,
            derivative_order_along,
        )

        output[i, :] = res

    return output


@njit(F_2D(F_2D, F_2D, I_2D, F_2D, I_1D), parallel=True, nogil=True)
def eval_multiple_query_par(
    xx: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    derivative_order_along: np.ndarray,
) -> np.ndarray:
    """Evaluate the derivative of Newton polynomial(s) on multiple query points
    in parallel.

    The following symbols are used in the subsequent description as shortcuts:

    - ``k``: number of query points
    - ``m``: spatial dimension
    - ``n``: polynomial degree
    - ``N``: number of coefficients or the cardinality of the multi-index set
    - ``np``: number of polynomials (i.e., number of coefficient sets)

    Parameters
    ----------
    xx : :class:`numpy:numpy.ndarray`
        The set of query points at which the derivative is evaluated;
        the values are given in a two-dimensional array of shape ``(k, m)``.
    coefficients : :class:`numpy:numpy.ndarray`
        The coefficients of the Newton polynomial;
        the values are given in a two-dimensional array of shape ``(N, np)``.
    exponents : :class:`numpy:numpy.ndarray`
        Set of exponents given as a two-dimensional non-negative integer array
        of shape ``(N, m)``.
    generating_points : :class:`numpy:numpy.ndarray`
        Interpolation points for each dimension given as a two-dimensional
        array of shape ``(m, n + 1)``.
    derivative_order_along : :class:`numpy:numpy.ndarray`
        Specification of the orders of derivative along each dimension given
        as a one-dimensional non-negative integer array of length ``m``.
        For example, the array ``np.array([2, 3, 1])`` specifies the 2nd-order,
        3rd-order, and 1st-order derivatives along the 1st, 2nd, and 3rd
        dimension, respectively.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The value of the derivative at the query point given
        as a two-dimensional array of shape ``(k, np)``.

    Notes
    -----
    - This is a direct differentiation and evaluation of polynomial in
      the Newton basis via chain rule without any transformation to another
      (e.g., canonical) basis.
    - This function is compiled (NJIT-ted) and executed in parallel
      with the help of Numba.

    See Also
    --------
    minterpy.jit_compiled.newton.diff.eval_single_query
        Evaluation of the derivative of polynomial(s) in the Newton basis
        for a single query point.
    minterpy.jit_compiled.newton.diff.eval_multiple_query
        Evaluation of the derivative of polynomial(s) in the Newton basis
        for multiple query points on a single CPU.
    """
    num_points = len(xx)
    num_polys = coefficients.shape[1]

    # Create the output array
    output = np.empty(shape=(num_points, num_polys), dtype=FLOAT)

    # Loop over query points
    for i in prange(num_points):
        x = xx[i, :]

        # Compute the evaluation at a single point
        res = eval_single_query(
            x,
            coefficients,
            exponents,
            generating_points,
            derivative_order_along,
        )

        output[i, :] = res

    return output
