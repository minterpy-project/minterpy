import numpy as np
from numba import njit, void

from minterpy.global_settings import F_1D, I_2D, F_2D, I_1D, B_TYPE


@njit(void(F_1D, I_2D, F_2D, I_1D, F_2D, F_1D), cache=True)  # O(Nm)
def eval_newton_monomials_single(
    x_single,
    exponents,
    generating_points,
    max_exponents,
    products_placeholder,
    monomials_placeholder,
) -> None:
    """Precomputes the value of all given Newton basis polynomials at a point.

    Core of the fast polynomial evaluation algorithm.
    - ``m`` spatial dimension
    - ``N`` number of monomials
    - ``n`` maximum exponent in each dimension

    :param x_single: coordinates of the point. The shape has to be ``m``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param generating_points: generating points used to generate the grid. The shape is ``(n x m)``.
    :param max_exponents: array with maximum exponent in each dimension. The shape has to be ``m``.
    :param products_placeholder: a numpy array for storing the (chained) products.
    :param monomials_placeholder: a numpy array of length N for storing the values of all Newton basis polynomials.

    Notes
    -----
    - This is a Numba-accelerated function.
    - The function precompute all the (chained) products required during Newton evaluation for a single query point
      with complexity of ``O(mN)``.
    - The (pre-)computation of Newton monomials is coefficient agnostic.
    - Results are stored in the placeholder arrays. The function returns None.
    """

    # NOTE: the maximal exponent might be different in every dimension,
    #    in this case the matrix becomes sparse (towards the end)
    # NOTE: avoid index shifting during evaluation (has larger complexity than pre-computation!)
    #    by just adding one empty row in front. ATTENTION: these values must not be accessed!
    #    -> the exponents of each monomial ("alpha") then match the indices of the required products

    # Create the products matrix
    m = exponents.shape[1]
    for i in range(m):
        max_exp_in_dim = max_exponents[i]
        x_i = x_single[i]
        prod = 1.0
        for j in range(max_exp_in_dim):  # O(n)
            # TODO there are n+1 1D grid values, the last one will never be used!?
            p_ij = generating_points[j, i]
            prod *= x_i - p_ij
            # NOTE: shift index by one
            exponent = j + 1  # NOTE: otherwise the result type is float
            products_placeholder[exponent, i] = prod

    # evaluate all Newton polynomials. O(Nm)
    N = exponents.shape[0]
    for j in range(N):
        # the exponents of each monomial ("alpha")
        # are the indices of the products which need to be multiplied
        newt_mon_val = 1.0  # required as multiplicative identity
        for i in range(m):
            exp = exponents[j, i]
            # NOTE: an exponent of 0 should not cause a multiplication
            # (inefficient, numerical instabilities)
            if exp > 0:
                newt_mon_val *= products_placeholder[exp, i]
        monomials_placeholder[j] = newt_mon_val
    #NOTE: results have been stored in the numpy arrays. no need to return anything.


@njit(void(F_2D, I_2D, F_2D, I_1D, F_2D, F_2D, B_TYPE), cache=True)
def eval_newton_monomials_multiple(
    xx: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    max_exponents: np.ndarray,
    products_placeholder: np.ndarray,
    monomials_placeholder: np.ndarray,
    triangular: bool
) -> None:
    """Evaluate the Newton monomials at multiple query points.

    The following notations are used below:

    - :math:`m`: the spatial dimension of the polynomial
    - :math:`p`: the (maximum) degree of the polynomial in any dimension
    - :math:`n`: the number of elements in the multi-index set (i.e., monomials)
    - :math:`\mathrm{nr_{points}}`: the number of query (evaluation) points
    - :math:`\mathrm{nr_polynomials}`: the number of polynomials with different
      coefficient sets of the same multi-index set

    :param xx: numpy array with coordinates of points where polynomial is to be evaluated.
              The shape has to be ``(k x m)``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param generating_points: generating points used to generate the grid. The shape is ``(n x m)``.
    :param max_exponents: array with maximum exponent in each dimension. The shape has to be ``m``.
    :param products_placeholder: a numpy array for storing the (chained) products.
    :param monomials_placeholder: placeholder numpy array where the results of evaluation are stored.
                               The shape has to be ``(k x p)``.
    :param triangular: whether the output will be of lower triangular form or not.
                       -> will skip the evaluation of some values
    :return: the value of each Newton polynomial at each point. The shape will be ``(k x N)``.

    Notes
    -----
    - This is a Numba-accelerated function.
    - The memory footprint for evaluating the Newton monomials iteratively
      with a single query point at a time is smaller than evaluating all
      the Newton monomials on all query points.
      However, when multiplied with multiple coefficient sets,
      this approach will be faster.
    - Results are stored in the placeholder arrays. The function returns None.
    """

    n_points = xx.shape[0]

    # By default, all exponents are "active" unless xx are unisolvent nodes
    active_exponents = exponents
    # Iterate each query points and evaluate the Newton monomials
    for idx in range(n_points):

        x_single = xx[idx, :]

        # Get the row view of the monomials placeholder;
        # this would be the evaluation results of a single query point
        monomials_placeholder_single = monomials_placeholder[idx]

        if triangular:
            # TODO: Refactor this, this is triangular because the monomials
            #       are evaluated at the unisolvent nodes, otherwise it won't
            #       be and the results would be misleading.
            # When evaluated on unisolvent nodes, some values will be a priori 0
            n_active_polys = idx + 1
            # Only some exponents are active
            active_exponents = exponents[:n_active_polys, :]
            # IMPORTANT: initialised empty. set all others to 0!
            monomials_placeholder_single[n_active_polys:] = 0.0
            # Only modify the non-zero entries
            monomials_placeholder_single = \
                monomials_placeholder_single[:n_active_polys]

        # Evaluate the Newton monomials on a single query point
        # NOTE: Due to "view" access,
        # the whole 'monomials_placeholder' will be modified
        eval_newton_monomials_single(
            x_single,
            active_exponents,
            generating_points,
            max_exponents,
            products_placeholder,
            monomials_placeholder_single
        )
