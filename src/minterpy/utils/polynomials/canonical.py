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
