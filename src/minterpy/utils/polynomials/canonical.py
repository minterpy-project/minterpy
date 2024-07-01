"""
This module provides computational routines relevant to polynomials
in the canonical basis.
"""

import numpy as np


def integrate_monomials_canonical(
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

    TODO
    ----
    - The whole integration domain is assumed to be :math:`[-1, 1]^M` where
      :math:`M` is the number of spatial dimensions because the polynomial
      itself is defined in that domain. This condition may be relaxed in
      the future and the implementation below should be modified.
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

    return monomials_integrals
