import numpy as np
from numba import njit, void

from minterpy.global_settings import F_2D, I_2D


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
