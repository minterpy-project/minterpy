"""
This module provides computational routines relevant to polynomials
in the Newton basis.
"""
import itertools
import math
import numpy as np

from minterpy.utils.quad import gauss_leg
from minterpy.utils.verification import check_dtype
from minterpy.global_settings import FLOAT_DTYPE, INT_DTYPE, DEBUG
from minterpy.jit_compiled.newton.eval import eval_newton_monomials_multiple


def eval_newton_monomials(
    x: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    verify_input: bool = False,
    triangular: bool = False,
) -> np.ndarray:
    """Newton evaluation function.

    Compute the value of each Newton monomial on each given point. Internally it uses ``numba`` accelerated evaluation function.

    :param x: The points to evaluate the polynomials on.
    :type x: np.ndarray
    :param exponents: the multi indices "alpha" for every Newton polynomial corresponding to the exponents of this "monomial"
    :type exponents: np.ndarray
    :param generating_points: Nodes where the Newton polynomial lives on. (Or the points which generate these nodes?!)
    :type generating_points: np.ndarray
    :param verify_input: weather the data types of the input should be checked. Turned off by default for performance.
    :type verify_input: bool
    :param triangular: weather or not the output will be of lower triangular form. This will skip the evaluation of some values. Defaults to :class:`False`.
    :type triangular: bool

    :return: the value of each Newton polynomial on each point. The output shape is ``(k, N)``, where ``k`` is the number of points and ``N`` is the number of coeffitions of the Newton polyomial.
    :rtype: np.ndarray

    .. todo::
        - rename ``generation_points`` according to :class:`Grid`.
        - use instances of :class:`MultiIndex` and/or :class:`Grid` instead of the array representations of them.
        - ship this to the submodule ``newton_polynomials``.
        - Refactor the "triangular" parameter, the Newton monomials
          only becomes triangular if evaluated at the unisolvent nodes.
          So it needs a special function instead of parametrizing this function
          that can give a misleading result.

    See Also
    --------
    eval_all_newt_polys : concrete ``numba`` accelerated implementation of polynomial evaluation in Newton base.
    """
    # Assumption: query points array has been verified
    num_monomials, m = exponents.shape
    num_points = len(x)
    if verify_input:
        check_dtype(x, FLOAT_DTYPE)
        check_dtype(exponents, INT_DTYPE)

    # NOTE: the downstream numba-accelerated function does not support kwargs,
    # so the maximum exponent per dimension must be computed here
    max_exponents = np.max(exponents, axis=0)

    # Create placeholders for the final and intermediate results
    result_placeholder = np.empty(
        shape=(num_points, num_monomials),
        dtype=FLOAT_DTYPE,
    )
    prod_placeholder = np.empty(
        shape=(np.max(max_exponents) + 1, m),
        dtype=FLOAT_DTYPE,
    )

    # Compute the Newton monomials on all the query points
    eval_newton_monomials_multiple(
        x,
        exponents,
        generating_points,
        max_exponents,
        prod_placeholder,
        result_placeholder,
        triangular,
    )

    return result_placeholder


def eval_newton_polynomials(
    xx: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    verify_input: bool = False,
    batch_size: int = None,
):
    """Evaluate the polynomial(s) in Newton form at multiple query points.

    Iterative implementation of polynomial evaluation in Newton form

    This version able to handle both:
        - list of input points x (2D input)
        - list of input coefficients (2D input)

    Here we use the notations:
        - ``n`` = polynomial degree
        - ``N`` = amount of coefficients
        - ``k`` = amount of points
        - ``p`` = amount of polynomials


    .. todo::
        - idea for improvement: make use of the sparsity of the exponent matrix and avoid iterating over the zero entries!
        - refac the explanation and documentation of this function.
        - use instances of :class:`MultiIndex` and/or :class:`Grid` instead of the array representations of them.
        - ship this to the submodule ``newton_polynomials``.

    :param xx: Arguemnt array with shape ``(m, k)`` the ``k`` points to evaluate on with dimensionality ``m``.
    :type xx: np.ndarray
    :param coefficients: The coefficients of the Newton polynomials.
        NOTE: format fixed such that 'lagrange2newton' conversion matrices can be passed
        as the Newton coefficients of all Lagrange monomials of a polynomial without prior transponation
    :type coefficients: np.ndarray, shape = (N, p)
    :param exponents: a multi index ``alpha`` for every Newton polynomial corresponding to the exponents of this ``monomial``
    :type exponents: np.ndarray, shape = (m, N)
    :param generating_points: Grid values for every dimension (e.g. Leja ordered Chebychev values).
        the values determining the locations of the hyperplanes of the interpolation grid.
        the ordering of the values determine the spacial distribution of interpolation nodes.
        (relevant for the approximation properties and the numerical stability).
    :type generating_points: np.ndarray, shape = (m, n+1)
    :param verify_input: weather the data types of the input should be checked. turned off by default for speed.
    :type verify_input: bool, optional
    :param batch_size: batch size of query points
    :type batch_size: int, optional

    :raise TypeError: If the input ``generating_points`` do not have ``dtype = float``.

    :return: (k, p) the value of each input polynomial at each point. TODO squeezed into the expected shape (1D if possible). Notice, format fixed such that the regression can use the result as transformation matrix without transponation

    Notes
    -----
    - This method is faster than the recursive implementation of ``tree.eval_lp(...)`` for a single point and a single polynomial (1 set of coeffs):
        - time complexity: :math:`O(mn+mN) = O(m(n+N)) = ...`
        - pre-computations: :math:`O(mn)`
        - evaluation: :math:`O(mN)`
        - space complexity: :math:`O(mn)` (precomputing and storing the products)
        - evaluation: :math:`O(0)`
    - advantage:
        - just operating on numpy arrays, can be just-in-time (jit) compiled
        - can evaluate multiple polynomials without recomputing all intermediary results

    See Also
    --------
    evaluate_multiple : ``numba`` accelerated implementation which is called internally by this function.
    convert_eval_output: ``numba`` accelerated implementation of the output converter.
    """
    # Get the relevant data
    verify_input = verify_input or DEBUG
    num_points = len(xx)

    # Get batch size
    # TODO: Verify the batch size
    if batch_size is None or batch_size >= num_points:
        newton_monomials = eval_newton_monomials(
            xx,
            exponents,
            generating_points,
            verify_input,
            False
        )
        results = newton_monomials @ coefficients
    else:
        # Evaluate the Newton polynomials in batches
        results = eval_newton_polynomials_batch(
            xx,
            coefficients,
            exponents,
            generating_points,
            batch_size
        )

    return results


def eval_newton_polynomials_batch(
    xx: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    batch_size: int
):
    """Evaluate the polynomial in Newton form in batches of query points.

    Notes
    -----
    - It is assumed that the inputs have all been verified and rectified.
    - It would be more expensive to evaluate smaller batch sizes
      but with a smaller memory footprint in any given iteration.
    - If memory does not permit whole evaluation of query points,
      consider using smaller but not the smallest batch size (i.e., not 1).
    """

    # Get some important numbers
    num_points = xx.shape[0]
    if coefficients.ndim == 1:
        num_polynomials = 1
    else:
        num_polynomials = coefficients.shape[1]
    # Create an output placeholder
    results_placeholder = np.empty(
        shape=(num_points, num_polynomials),
        dtype=FLOAT_DTYPE,
    )

    # Batch processing: evaluate the polynomials at a batch of query points
    n_batches, remainder = divmod(num_points, batch_size)
    if remainder != 0:
        n_batches += 1
    for idx in range(n_batches):
        start_idx = idx * batch_size
        if idx == n_batches - 1 and remainder != 0:
            end_idx = idx * batch_size + remainder
        else:
            end_idx = (idx + 1) * batch_size

        # Get the current batch of query points
        xx_batch = xx[start_idx:end_idx, :]
        # Compute the Newton monomials for the batch
        newton_monomials = eval_newton_monomials(
            xx_batch,
            exponents,
            generating_points,
            False,
            False
        )

        # Compute the polynomial values for the batch
        results_placeholder[start_idx:end_idx] = \
            newton_monomials @ coefficients

    return results_placeholder


def deriv_newt_eval(x: np.ndarray, coefficients: np.ndarray, exponents: np.ndarray,
                    generating_points: np.ndarray, derivative_order_along: np.ndarray) -> np.ndarray:
    """Evaluate the derivative of a polynomial in the Newton form.

     m = spatial dimension
     n = polynomial degree
     N = number of coefficients
     p = number of polynomials
     k = number of evaluation points

    Parameters
    ----------
    x: (k, m) the k points to evaluate on with dimensionality m.
    coefficients: (N, p) the coefficients of the Newton polynomial(s).
    exponents: (m, N) a multi index "alpha" for every Newton polynomial
        corresponding to the exponents of this "monomial"
    generating_points: (m, n+1) grid values for every dimension (e.g. Leja ordered Chebychev values).
    derivative_order_along: (m) specifying the order along each dimension to compute the derivative
    eg. [2,3,1] will compute respectively 2nd order, 3rd order, and 1st order along spatial dimensions
    0, 1, and 2.

    Returns
    -------
    (k) the value of derivative of the polynomial evaluated at each point.

    Notes
    -----
    - Can compute derivative polynomials without transforming to canonical basis.
    - This derivative evaluation is done by taking derivatives of the Newton monomials.
    - JIT compilation using Numba was not used here as ``combinations()``
      from the ``itertools`` module does not work with Numba.
    - Due to multiple nested loops this implementation is very slow except
      for a small problem (small dimension and small polynomial degree).

    TODO
    ----
    - Refactor this initial implementation of polynomial differentiation
      in the Newton basis.
    """
    # Get relevant data
    # N, coefficients, m, nr_points, nr_polynomials, x = \
    #     rectify_eval_input(x, coefficients, exponents, False)
    num_monomials, m = exponents.shape
    num_points = x.shape[0]
    if coefficients.ndim == 1:
        num_polynomials = 1
    else:
        num_polynomials = coefficients.shape[1]
    max_exponents = np.max(exponents, axis=0)

    # Result of the derivative evaluation
    results = np.empty((num_points, num_polynomials), dtype=FLOAT_DTYPE)

    # Array to store individual basis monomial evaluations
    monomial_vals= np.empty(num_monomials, dtype=FLOAT_DTYPE)

    num_prods = np.max(max_exponents) + 1
    # Array to store products in basis monomial along each dimension
    products = np.empty((num_prods, m), dtype=FLOAT_DTYPE)

    # Newton monomials have to be evaluated at each input point separately
    for point_idx in range(num_points):
        x_single = x[point_idx, :]

        # Constructing the products array
        for i in range(m):
            max_exp_in_dim = max_exponents[i]
            x_i = x_single[i]
            order = derivative_order_along[i]
            if order == 0: # no partial derivative along this dimension
                prod = 1.0
                for j in range(max_exp_in_dim):  # O(n)
                    p_ij = generating_points[j, i]
                    prod *= (x_i - p_ij)
                    # NOTE: shift index by one
                    exponent = j + 1  # NOTE: otherwise the result type is float
                    products[exponent, i] = prod
            else: # take partial derivative of 'order' along this dimension

                # derivative of first 'order' newt monomials will be 0 as their degree < order
                products[:order, i] = 0.0

                # if order of derivative larger than the degree
                if order >= num_prods:
                    continue

                # derivative of newt monomial 'order' will be just factorial of order
                fact = math.factorial(order)
                products[order, i] = fact

                # for all bigger monomials, use chain rule of differentiation to compute derivative of products
                for q in range(order + 1, max_exp_in_dim + 1):
                    combs = itertools.combinations(range(q), q-order)
                    res = 0.0
                    for comb in combs: # combs is a generator for combinations
                        prod = np.prod(x_i - generating_points[list(comb), i])
                        res += prod

                    res *= fact
                    products[q, i] = res

        # evaluate all Newton polynomials. O(Nm)
        for j in range(num_monomials):
            # the exponents of each monomial ("alpha")
            # are the indices of the products which need to be multiplied
            newt_mon_val = 1.0  # required as multiplicative identity
            for i in range(m):
                exp = exponents[j, i]
                # NOTE: an exponent of 0 should not cause a multiplication
                if exp > 0:
                    newt_mon_val *= products[exp, i]
                else:
                    order = derivative_order_along[i]
                    if order > 0:
                        newt_mon_val = 0.0
            monomial_vals[j] = newt_mon_val

        results[point_idx] = np.sum(monomial_vals[:,None] * coefficients, axis=0)

    return results


def integrate_monomials_newton(
    exponents: np.ndarray, generating_points: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    """Integrate the monomials in the Newton basis given a set of exponents.

    Parameters
    ----------
    exponents : :class:`numpy:numpy.ndarray`
        A set of exponents from a multi-index set that defines the polynomial,
        an ``(N, M)`` array, where ``N`` is the number of exponents
        (multi-indices) and ``M`` is the number of spatial dimensions.
        The number of exponents corresponds to the number of monomials.
    generating_points : :class:`numpy:numpy.ndarray`
        A set of generating points of the interpolating polynomial,
        a ``(P + 1, M)`` array, where ``P`` is the maximum degree of
        the polynomial in any dimensions and ``M`` is the number
        of spatial dimensions.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integrated Newton monomials, an ``(N,)`` array, where N is
        the number of monomials (exponents).
    """
    # --- Get some basic data
    num_monomials, num_dim = exponents.shape
    max_exp = np.max(exponents)
    max_exps_in_dim = np.max(exponents, axis=0)

    # --- Compute the integrals of one-dimensional bases
    one_dim_integrals = np.empty((max_exp + 1, num_dim))  # A lookup table
    for j in range(num_dim):
        max_exp_in_dim = max_exps_in_dim[j]
        exponents_1d = np.arange(max_exp_in_dim + 1)[:, np.newaxis]
        generating_points_in_dim = generating_points[:, j][:, np.newaxis]
        # NOTE: Newton monomials are polynomials, Gauss-Legendre quadrature
        #       will be exact for degree == 2*num_points - 1.
        quad_num_points = np.ceil((max_exp_in_dim + 1) / 2)

        # Compute the integrals
        # NOTE: 'eval_newton_monomials()' expects a two-dimensional array
        one_dim_integrals[: max_exp_in_dim + 1, j] = gauss_leg(
            lambda x: eval_newton_monomials(
                x[:, np.newaxis], exponents_1d, generating_points_in_dim
            ),
            num_points=quad_num_points,
            bounds=bounds[j],
        )

    # --- Compute integrals of the monomials (multi-dimensional basis)
    monomials_integrals = np.zeros(num_monomials)
    for i in range(num_monomials):
        out = 1.0
        for j in range(num_dim):
            exp = exponents[i, j]
            out *= one_dim_integrals[exp, j]
        monomials_integrals[i] = out

    # TODO: The whole integration domain is assumed to be :math:`[-1, 1]^M`
    #       where :math:`M` is the number of spatial dimensions because
    #       the current interpolating polynomial itself is defined
    #       in that domain. This condition may be relaxed in the future
    #       and the implementation.

    return monomials_integrals
