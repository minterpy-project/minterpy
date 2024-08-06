"""
This module provides computational routines relevant to polynomials
in the canonical basis.
"""
from __future__ import annotations

import numpy as np

from minterpy.utils.multi_index import find_match_between


def integrate_monomials(
    exponents: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Integrate the monomials in the canonical basis given a set of exponents.

    Parameters
    ----------
    exponents : :class:`numpy:numpy.ndarray`
        A set of exponents from a multi-index set that defines the polynomial,
        an ``(N, M)`` array, where ``N`` is the number of exponents
        (multi-indices) and ``M`` is the number of spatial dimensions.
        The number of exponents corresponds to the number of monomials.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integrated Canonical monomials, an ``(N,)`` array, where ``N`` is
        the number of monomials (exponents).
    """
    bounds_diff = np.diff(bounds)

    if np.allclose(bounds_diff, 2):
        # NOTE: Over the whole canonical domain [-1, 1]^M, no need to compute
        #       the odd-degree terms.
        case = np.all(np.mod(exponents, 2) == 0, axis=1)  # All even + 0
        even_terms = exponents[case]

        monomials_integrals_even_terms = bounds_diff.T / (even_terms + 1)

        monomials_integrals = np.zeros(exponents.shape)
        monomials_integrals[case] = monomials_integrals_even_terms

        return monomials_integrals.prod(axis=1)

    # NOTE: Bump the exponent by 1 (polynomial integration)
    bounds_power = np.power(bounds.T[:, None, :], (exponents + 1)[None, :, :])
    bounds_diff = bounds_power[1, :] - bounds_power[0, :]

    monomials_integrals = np.prod(bounds_diff / (exponents + 1), axis=1)

    # TODO: The whole integration domain is assumed to be :math:`[-1, 1]^M`
    #       where :math:`M` is the number of spatial dimensions because
    #       the polynomial itself is defined in that domain. Polynomials in
    #       the canonical basis, however, are defined on the reals.
    #       The restriction may be relaxed in the future
    #       and the implementation should be modified.

    return monomials_integrals


def compute_coeffs_poly_sum(
    exponents_1: np.ndarray,
    coeffs_1: np.ndarray,
    exponents_2: np.ndarray,
    coeffs_2: np.ndarray,
    exponents_sum: np.ndarray,
):
    r"""Compute the coefficients of polynomial sum in the canonical basis.

    For example, suppose: :math:`A = \{ (0, 0) , (1, 0), (0, 1) \}` with
    coefficients :math:`c_A = (1.0 , 2.0, 3.0)` is summed with
    :math:`B = \{ (0, 0), (1, 0), (2, 0) \}` with coefficients
    :math:`c_B = (1.0, 5.0, 3.0)`. The union/sum multi-index set is
    :math:`A \times B = \{ (0, 0), (1, 0), (2, 0), (0, 1) \}`.

    The corresponding coefficients of the sum are:

    - :math:`(0, 0)` appears in both operands, so the coefficient
      is :math:`1.0 + 1.0 = 2.0`
    - :math:`(1, 0)` appears in both operands, so the coefficient is
      :math:`2.0 + 5.0 = 7.0`
    - :math:`(2, 0)` only appears in the second operand, so the coefficient
      is :math:`3.0`
    - :math:`(0, 1)` only appears in the first operand, so the coefficient
      is :math:`3.0`

    or :math:`c_{A | B} = (2.0, 7.0, 3.0, 3.0)`.

    Parameters
    ----------
    exponents_1 : :class:`numpy:numpy.ndarray`
        The multi-indices exponents of the first multidimensional polynomial
        operand in the addition expression.
    coeffs_1 : :class:`numpy:numpy.ndarray`
        The coefficients of the first multidimensional polynomial operand.
    exponents_2 : :class:`numpy:numpy.ndarray`
        The multi-indices exponents of the second multidimensional polynomial.
    coeffs_2 : :class:`numpy:numpy.ndarray`
        The coefficients of the second multidimensional polynomial operand.
    exponents_sum : :class:`numpy:numpy.ndarray`
        The multi-indices exponents that are the sum between
        ``exponents_1`` and ``exponents_2`` (i.e., the union of both).

    Notes
    -----
    - ``exponents_1``, ``exponents_2``, ``exponents_prod`` are assumed to be
      two-dimensional integer arrays that are sorted lexicographically.
    - ``exponents_sum`` is assumed to be the result of unionizing
      ``exponents_1`` and ``exponents_2`` as multi-indices.
    - ``coeffs_1`` and ``coeffs_2`` are assumed to be two-dimensional float
      arrays. Their number of columns must be the same.
    - ``coeffs_sum`` is a placeholder array to store the results; it must
      be initialized with zeros.
    - The function does not check whether the above assumptions are fulfilled;
      the caller is responsible to make sure of that. If the assumptions are
      not fulfilled, the function may not raise any exception but produce
      the wrong results.
    """
    # Create the output array
    num_monomials = exponents_sum.shape[0]
    num_polynomials = coeffs_1.shape[1]
    coeffs_sum = np.zeros((num_monomials, num_polynomials))

    # Get the matching indices
    idx_1 = find_match_between(exponents_1, exponents_sum)
    idx_2 = find_match_between(exponents_2, exponents_sum)

    coeffs_sum[idx_1, :] += coeffs_1[:, :]
    coeffs_sum[idx_2, :] += coeffs_2[:, :]

    return coeffs_sum


def eval_polynomials(
    xx: np.ndarray,
    coeffs: np.ndarray,
    exponents: np.ndarray,
) -> np.ndarray:
    """Evaluate polynomial in the canonical basis.

    Parameters
    ----------
    xx : :class:`numpy:numpy.ndarray`
        Array of query points in the at which the polynomial(s) is evaluated.
        The array is of shape ``(N, m)`` where ``N`` is the number of points
        and ``m`` is the spatial dimension of the polynomial.
    coeffs : :class:`numpy:numpy.ndarray`
        The coefficients of the polynomial in the canonical basis. A single set
        of coefficients is given as a one-dimensional array while multiple sets
        are given as a two-dimensional array.
    exponents : :class:`numpy:numpy.ndarray`
        The exponents of the polynomial as multi-indices, a two-dimensional
        positive integer array.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The output of the polynomial evaluation. If the polynomial consists
        of a single coefficient set, the output array is one-dimensional with
        a length of ``N``. If the polynomial consists of multiple coefficients
        sets, the output array is two-dimensional with a shape of
        ``(N, n_poly)`` where ``n_poly`` is the number of coefficient sets.

    Notes
    -----
    - This implementation is considered unsafe and may fail spectacularly
      for polynomials of moderate degrees. Consider a more advanced
      implementation in the future.

    """
    monomials = np.prod(
        np.power(xx[:, None, :], exponents[None, :, :]),
        axis=-1,
    )
    yy = np.dot(monomials, coeffs)

    return yy
