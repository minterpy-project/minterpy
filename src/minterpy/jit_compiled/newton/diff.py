"""
A module for compiled code for polynomial differentiation in the Newton basis.

Notes
-----
- The most "fine-grained" functions must be defined first in order for Numba
  to properly infer the function types.
"""
import math

import numpy as np
from numba import njit, prange, void

from minterpy.global_settings import (
    F_2D,
    F_1D,
    I_1D,
    FLOAT,
    FLOAT_DTYPE,
    INT_DTYPE,
    I_2D,
)

from minterpy.jit_compiled.common import (
    combinations_iter,
    dot,
    get_max_columnwise,
    n_choose_r,
)

__all__ = []


@njit(F_2D(F_1D, F_2D), cache=True)
def create_lut_difference(
    x: np.ndarray,
    generating_points: np.ndarray
) -> np.ndarray:
    """Create a look-up table containing the differences between a query point
    and the generating points in each dimension.

    The table (matrix) contains :math:`(x_j - p_{i, j})` where :math:`p_{i, j}`
    is the :math:`i`-th generating point of the :math:`j`-th dimension.

    Parameters
    ----------
    x : :class:`numpy:numpy.ndarray`
        The query point, a one-dimensional array of length ``m``, where ``m``
        is the spatial dimension of the polynomial.
    generating_points : :class:`numpy:numpy.ndarray`
        Interpolation points for each dimension given as a two-dimensional
        array of shape ``(n + 1, m)``, where ``n`` is the maximum polynomial
        degree in all dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        A look-up table (LUT) of shape ``(n, m)`` containing the difference
        between the query point and the generating points (size ``n``)
        in each dimension (size ``m``).

    Notes
    -----
    - This function is compiled (NJIT-ted) with the help of Numba.
    """
    # Create an output array
    n, m = generating_points.shape
    # NOTE: Last row of the generating points is not used
    lut = np.empty(shape=(n - 1, m), dtype=FLOAT_DTYPE)

    # Construct the table
    for i in range(n - 1):
        for j in range(m):
            lut[i, j] = x[j] - generating_points[i, j]

    return lut


@njit(FLOAT(I_2D, F_1D, F_1D, F_1D), cache=True)
def create_lua_differentiated(
    combinations: np.ndarray,
    lua_difference: np.ndarray,
    prev_lua_prod: np.ndarray,
    lua_prod: np.ndarray,
) -> float:
    """Create a differentiated product look-up array (LUA) in one-dimension
    for a single query point, a given monomial degree, and order of derivative.

    Due to chain rule, the differentiated one-dimensional Newton monomials
    evaluated at a query point consists of sum of products of difference
    between the query point and the generating points.

    The products that appears in the sum of a given degree may be recycled
    in the computation of the products of the next higher degree to avoid
    the re-computations from scratch.

    Therefore, this function both returns the sum of products of a given degree
    and modifies in-place the product array to be re-used.

    Parameters
    ----------
    combinations : :class:`numpy:numpy.ndarray`
        The combination of terms to multiply, the information regarding
        the degree of monomials and the order of derivative is encoded here
        and does not appear explicitly in this function.
    lua_difference : :class:`numpy:numpy.ndarray`
        The one-dimensional array containing the difference between
        a query point and the generating points in one dimension.
    prev_lua_prod : :class:`numpy:numpy.ndarray`
        The products of difference from previous computation (degree).
    lua_prod : :class:`numpy:numpy.ndarray`
        The products of difference for the current computation; it is passed
        to avoid reinitialization.

    Returns
    -------
    float
        The sum of products of difference between the query point and
        the generating points for a differentiated one-dimensional Newton
        monomial of a given degree and a given order of derivative.

    Notes
    -----
    - At the end of the computation, the current products are stored in
      the previous products array to be used in the next call.
    - The parameter (array) ``lua_prod`` is not actually used in the subsequent
      computation but is passed to avoid re-initialization of the array.
    - This function is compiled (NJIT-ted) with the help of Numba.
    """
    # Create outputs
    res_sum = 0.0  # Additive identity

    # ---- Previous product look-up array is specified
    idx_prev_lua_prod = -1
    prev_last_idx = np.inf

    for idx, comb in enumerate(combinations):
        if comb[-1] <= prev_last_idx:
            # NOTE: When the last element of combination array remains or
            # it becomes smaller, then it is a sign that all the previous
            # values also change; it's a sign to use different cached value
            # Example: 2-element combinations from [0, 1, 2, 3]
            # is [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            idx_prev_lua_prod += 1
        tmp = prev_lua_prod[idx_prev_lua_prod] * lua_difference[comb[-1]]
        lua_prod[idx] = tmp
        res_sum += tmp
        prev_last_idx = comb[-1]

    # NOTE: Update the previous product array to be used in the subsequent call
    for idx in range(len(combinations)):
        prev_lua_prod[idx] = lua_prod[idx]

    return res_sum


@njit(void(F_1D, I_1D, F_2D, I_1D, F_2D))
def create_lut_differentiated(
    x: np.ndarray,
    max_exponents: np.ndarray,
    generating_points: np.ndarray,
    derivative_order_along: np.ndarray,
    products_placeholder: np.ndarray,
):
    """Create a look-up table of one-dimensional Newton monomials derivatives
    evaluated on a single query point for all dimensions.

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
    products_placeholder : :class:`numpy:numpy.ndarray`
        A placeholder for a look-up table containing differentiated Newton
        monomials per dimension; the differentiated monomials are products
        of difference terms.

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
    # Get the relevant problem sizes
    m = len(x)  # spatial dimension
    num_prods = np.max(max_exponents) + 1  # max. number of diff. products

    # Compute the LUT of difference between query point and gen. points
    lut_difference = create_lut_difference(x, generating_points)

    # Loop over each dimension
    for j in range(m):

        # Get dimension-dependent data
        max_exponent_in_dim = int(max_exponents[j])
        order = int(derivative_order_along[j])
        lua_1d = lut_difference[:, j]

        # Differentiate the monomials
        if order == 0:
            # No differentiation along this dimension
            prod = 1.0  # multiplicative identity
            for i in range(max_exponent_in_dim):
                prod *= lua_1d[i]
                exponent = i + 1
                products_placeholder[exponent, j] = prod

        else:
            # Take partial derivative of `order` along this dimension

            # Degree of monomial < order of derivative...
            # then the derivatives are outright zero
            products_placeholder[:order, j] = 0.0

            # Order of derivative > the degree of monomial...
            if order >= num_prods:
                continue  # all zeros, move on to the next dimension

            # The `order`-th derivative of the `order`-th monomial...
            fact = math.gamma(order + 1)
            products_placeholder[order, j] = fact  # then it is the factorial
            # NOTE: use `math.gamma(n + 1)` instead of `math.factorial(n)`
            # as the latter is not supported by Numba.

            # Order of derivative == the degree of monomial
            if order == (num_prods - 1):
                continue  # move on to the next dimension now

            # Use chain rule to compute the derivative of products
            # for the higher-degree monomials
            max_num_combinations = n_choose_r(
                max_exponent_in_dim,
                max_exponent_in_dim - order,
            )

            # Create placeholders for this dimension
            lua_k = np.zeros(max_num_combinations, dtype=FLOAT_DTYPE)
            lua_prod = np.zeros(max_num_combinations, dtype=FLOAT_DTYPE)
            # Initialize
            res_sum = 0.0
            for i in range(order + 1):
                lua_k[i] = lua_1d[i]
                res_sum += lua_1d[i]
            res_sum *= fact
            products_placeholder[order + 1, j] = res_sum

            for k in range(order + 2, max_exponent_in_dim + 1):
                elements = np.arange(k, dtype=INT_DTYPE)
                combs = combinations_iter(elements, int(k - order))

                # Create iterative look-up array
                res = create_lua_differentiated(combs, lua_1d, lua_k, lua_prod)
                res *= fact

                # Update the products placeholder
                products_placeholder[k, j] = res


@njit(void(I_2D, F_2D, I_1D, F_1D))
def compute_monomials_from_lut(
    exponents: np.ndarray,
    lut_diff: np.ndarray,
    derivative_order_along: np.ndarray,
    monomials_placeholder: np.ndarray,
):
    """Evaluate the derivatives of multi-dimensional Newton monomials evaluated
    on a query point from a look-up table of differentiated monomial terms.

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
    monomials_placeholder : :class:`numpy:numpy.ndarray`
        The placeholder for all the differentiated monomials values evaluated
        at a single query point. The placeholder is a one-dimensional array of
        length ``N``.

    Notes
    -----
    - This function is compiled (NJIT-ted) with the help of Numba.
    """
    num_monomials, num_dim = exponents.shape

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

        # Update the placeholder
        monomials_placeholder[i] = newt_mon_val


@njit(void(F_1D, I_2D, I_1D, F_2D, I_1D, F_2D, F_1D))
def eval_monomials_single_query(
    x: np.ndarray,
    exponents: np.ndarray,
    max_exponents: np.ndarray,
    generating_points: np.ndarray,
    derivative_order_along: np.ndarray,
    products_placeholder: np.ndarray,
    monomials_placeholder: np.ndarray,
):
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
    exponents : :class:`numpy:numpy.ndarray`
        Set of exponents given as a two-dimensional non-negative integer array
        of shape ``(N, m)``.
    max_exponents : :class:`numpy:numpy.ndarray`
        The maximum exponent values in each dimension as a one-dimensional
        array of length ``m``; this is to avoid re-computation.
    generating_points : :class:`numpy:numpy.ndarray`
        Interpolation points for each dimension given as a two-dimensional
        array of shape ``(m, n + 1)``.
    derivative_order_along : :class:`numpy:numpy.ndarray`
        Specification of the orders of derivative along each dimension given
        as a one-dimensional non-negative integer array of length ``m``.
        For example, the array ``np.array([2, 3, 1])`` specifies the 2nd-order,
        3rd-order, and 1st-order derivatives along the 1st, 2nd, and 3rd
        dimension, respectively.
    products_placeholder : :class:`numpy:numpy.ndarray`
        A placeholder for a look-up table containing differentiated Newton
        monomials per dimension; the differentiated monomials are products
        of difference terms.
    monomials_placeholder : :class:`numpy:numpy.ndarray`
        The differentiated Newton monomials evaluated on a single query point.
        The array is of shape ``N``.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The value of the derivative at the query point given
        as a one-dimensional array of length ``np``.

    Notes
    -----
    - This is a direct differentiation and evaluation of Newton monomials
      on a single query point via chain rule without any transformation
      to another (e.g., canonical) basis.
    - This function is compiled (NJIT-ted) with the help of Numba.

    See Also
    --------
    minterpy.jit_compiled.newton.diff.eval_multiple_query
        Evaluation of the derivative of polynomial(s) in the Newton basis
        for multiple query points.
    """
    # Compute differentiated 1D Newton monomials look-up table (LUT)
    create_lut_differentiated(
        x,
        max_exponents,
        generating_points,
        derivative_order_along,
        products_placeholder,
    )

    # Compute the differentiated Newton monomials values from the LUT
    compute_monomials_from_lut(
        exponents,
        products_placeholder,
        derivative_order_along,
        monomials_placeholder,
    )


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
    minterpy.jit_compiled.newton.diff.eval_monomials_single_query
        Evaluation of the derivative of monomials in the Newton form
        for a single query point.
    """
    # Get relevant problem sizes
    num_points = len(xx)
    num_polys = coefficients.shape[1]
    m = xx.shape[1]
    num_monomials = len(exponents)

    # Compute the column-wise maximum of the exponents to avoid re-computation
    max_exponents = get_max_columnwise(exponents)
    num_prods = np.max(max_exponents) + 1  # Maximum number of product terms

    # Create the output array
    output = np.empty(shape=(num_points, num_polys), dtype=FLOAT_DTYPE)

    # Loop over query points
    for i in range(num_points):
        x_i = xx[i, :]

        # Construct placeholders per query point
        products_placeholder = np.ones(shape=(num_prods, m), dtype=FLOAT_DTYPE)
        monomials_placeholder = np.ones(num_monomials, dtype=FLOAT_DTYPE)

        # Compute the evaluation of monomials at a single query point
        eval_monomials_single_query(
            x_i,
            exponents,
            max_exponents,
            generating_points,
            derivative_order_along,
            products_placeholder,
            monomials_placeholder
        )

        # Compute the differentiated Newton polynomial values
        output[i, :] = dot(monomials_placeholder, coefficients)

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
    minterpy.jit_compiled.newton.diff.eval_monomials_single_query
        Evaluation of the derivative of monomials in the Newton form
        for a single query point.
    minterpy.jit_compiled.newton.diff.eval_multiple_query
        Evaluation of the derivative of polynomial(s) in the Newton basis
        for multiple query points on a single CPU.
    """
    # Get relevant problem sizes
    num_points = len(xx)
    num_polys = coefficients.shape[1]
    m = xx.shape[1]
    num_monomials = len(exponents)

    # Compute the column-wise maximum of the exponents to avoid re-computation
    max_exponents = get_max_columnwise(exponents)
    num_prods = np.max(max_exponents) + 1  # Maximum number of product terms

    # Create the output array
    output = np.empty(shape=(num_points, num_polys), dtype=FLOAT_DTYPE)

    # Loop over query points
    for i in prange(num_points):
        x_i = xx[i, :]

        # Construct placeholders per query point
        # NOTE: using placeholders are faster for some problem sizes
        products_placeholder = np.ones(shape=(num_prods, m), dtype=FLOAT_DTYPE)
        monomials_placeholder = np.ones(num_monomials, dtype=FLOAT_DTYPE)

        # Compute the evaluation of monomials at a single query point
        eval_monomials_single_query(
            x_i,
            exponents,
            max_exponents,
            generating_points,
            derivative_order_along,
            products_placeholder,
            monomials_placeholder
        )

        # Compute the differentiated Newton polynomial values
        output[i, :] = dot(monomials_placeholder, coefficients)

    return output
