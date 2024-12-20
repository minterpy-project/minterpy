"""
A module for compiled code for polynomial manipulation in the canonical basis.

Notes
-----
- The most "fine-grained" functions must be defined first in order for Numba
  to properly infer the function types.
"""
import numpy as np
from numba import njit, void

from minterpy.global_settings import F_2D, I_2D
from minterpy.jit_compiled.multi_index import search_lex_sorted


@njit(void(F_2D, F_2D, I_2D, F_2D), cache=True)
def can_eval_mult(x_multiple, coeffs, exponents, result_placeholder):
    """Naive evaluation of polynomials in canonical basis.

    - ``m`` spatial dimension
    - ``k`` number of points
    - ``N`` number of monomials
    - ``p`` number of polynomials

    :param x_multiple: numpy array with coordinates of points where polynomial is to be evaluated.
                       The shape has to be ``(k x m)``.
    :param coeffs: numpy array of polynomial coefficients in canonical basis. The shape has to be ``(N x p)``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param result_placeholder: placeholder numpy array where the results of evaluation are stored.
                               The shape has to be ``(k x p)``.

    Notes
    -----
    This is a naive evaluation; a more numerically accurate approach would be to transform to Newton basis and
    using the newton evaluation scheme.

    Multiple polynomials in the canonical basis can be evaluated at once by having a 2D coeffs array. It is assumed
    that they all have the same set of exponents.

    """
    nr_coeffs, nr_polys = coeffs.shape
    r = result_placeholder
    nr_points, _ = x_multiple.shape
    for i in range(nr_coeffs):  # each monomial
        exp = exponents[i, :]
        for j in range(nr_points):  # evaluated on each point
            x = x_multiple[j, :]
            monomial_value = np.prod(np.power(x, exp))
            for k in range(nr_polys):  # reuse intermediary results
                c = coeffs[i, k]
                r[j, k] += c * monomial_value


@njit(void(I_2D, F_2D, I_2D, F_2D, I_2D, F_2D), cache=True)
def compute_coeffs_poly_prod(
    exponents_1: np.ndarray,
    coeffs_1: np.ndarray,
    exponents_2: np.ndarray,
    coeffs_2: np.ndarray,
    exponents_prod: np.ndarray,
    coeffs_prod: np.ndarray,
):
    r"""Compute the coefficients of polynomial product in the canonical basis.

    For example, suppose: :math:`A = \{ (0, 0) , (1, 0), (0, 1) \}` with
    coefficients :math:`c_A = (1.0 , 2.0, 3.0)` is multiplied with
    :math:`B = \{ (0, 0) , (1, 0) \}` with coefficients
    :math:`c_B = (1.0 , 5.0)`. The product multi-index set is
    :math:`A \times B = \{ (0, 0) , (1, 0), (2, 0), (0, 1), (1, 1) \}`.

    The corresponding coefficients of the product are:

    - :math:`(0, 0)` is coming from :math:`(0, 0) + (0, 0)`, the coefficient
      is :math:`1.0 \times 1.0 = 1.0`
    - :math:`(1, 0)` is coming from :math:`(0, 0) + (1, 0)` and
      :math:`(1, 0) + (0, 0)`, the coefficient is
      :math:`1.0 \times 5.0 + 2.0 \times 1.0 = 7.0`
    - :math:`(2, 0)` is coming from :math:`(1, 0) + (1, 0)`, the coefficient
      is :math:`2.0 \times 5.0 = 10.0`
    - :math:`(0, 1)` is coming from :math:`(0, 1) + (0, 0)`, the coefficient
      is :math:`3.0 \times 1.0 = 3.0`
    - :math:`(1, 1)` is coming from :math:`(0, 1) + (1, 0)`, the coefficient
      is :math:`3.0 \times 5.0 = 15.0`

    or :math:`c_{A \times B} = (1.0, 7.0, 10.0, 3.0, 15.0)`.

    Parameters
    ----------
    exponents_1 : :class:`numpy:numpy.ndarray`
        The multi-indices exponents of the first multidimensional polynomial
        operand in the multiplication expression.
    coeffs_1 : :class:`numpy:numpy.ndarray`
        The coefficients of the first multidimensional polynomial operand.
    exponents_2 : :class:`numpy:numpy.ndarray`
        The multi-indices exponents of the second multidimensional polynomial.
    coeffs_2 : :class:`numpy:numpy.ndarray`
        The coefficients of the second multidimensional polynomial operand.
    exponents_prod : :class:`numpy:numpy.ndarray`
        The multi-indices exponents that are the product between
        ``exponents_1`` and ``exponents_2`` (i.e., the sum of cartesian
        products).
    coeffs_prod : :class:`numpy:numpy.ndarray`
        The placeholder for the corresponding product coefficients.

    Notes
    -----
    - ``exponents_1``, ``exponents_2``, ``exponents_prod`` are assumed to be
      two-dimensional integer arrays that are sorted lexicographically.
    - ``exponents_prod`` is assumed to be the result of multiplying
      ``exponents_1`` and ``exponents_2`` as multi-indices.
    - ``coeffs_1``, ``coeffs_2``, and ``coeffs_prod`` are assumed to be
      two-dimensional float arrays. Their number of columns must be the same.
    - ``coeffs_prod`` is a placeholder array to store the results; it must
      be initialized with zeros.
    - The function does not check whether the above assumptions are fulfilled;
      the caller is responsible to make sure of that. If the assumptions are
      not fulfilled, the function may not raise any exception but produce
      the wrong results.
    """
    for idx_1 in range(len(exponents_1)):
        for idx_2 in range(len(exponents_2)):
            exponent_prod = exponents_1[idx_1] + exponents_2[idx_2]
            idx = search_lex_sorted(exponents_prod, exponent_prod)
            # NOTE: The output placeholder must be initialized with zeros!
            coeffs_prod[idx] += coeffs_1[idx_1] * coeffs_2[idx_2]
