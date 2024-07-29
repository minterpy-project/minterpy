"""
This module defines ``ChebyshevPolynomial`` class for Chebyshev polynomials
of the first kind.

Some common notations/symbols used below:

- ``m``: the number of spatial dimensions
- ``N``: the number of monomials and coefficients
- ``k``: the number of evaluation/query points
- ``Np``: the number of polynomials (i.e., set of coefficients)

Chebyshev polynomials are defined on :math:`[-1, 1]^m`.
"""
import numpy as np

from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.utils.polynomials.chebyshev import evaluate_chebyshev_polynomials
from minterpy.utils.polynomials.interface import (
    compute_poly_sum_data_chebyshev,
)
from minterpy.utils.verification import dummy, verify_domain

__all__ = ["ChebyshevPolynomial"]


def _chebyshev_add(
    poly_1: "ChebyshevPolynomial",
    poly_2: "ChebyshevPolynomial",
) -> "ChebyshevPolynomial":
    """Add two polynomial instances in the Chebyshev basis.

    This is the concrete implementation of ``_add()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract base class specifically for
    polynomials in the Chebyshev basis.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the addition expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the addition expression.

    Returns
    -------
    ChebyshevPolynomial
        The sum of two polynomials in the Chebyshev basis; a new instance
        of polynomial.

    Notes
    -----
    - This function assumes: both polynomials must be in the Chebyshev basis,
      they must be initialized, have the same dimension, their domains are
      matching, and their length must be the same. These conditions are not
      checked explicitly in this function.
    """
    # --- Get the ingredients of a summed polynomial in the Chebyshev basis
    grd_sum, mi_sum, coeffs_sum = compute_poly_sum_data_chebyshev(
        poly_1,
        poly_2,
    )
    # NOTE: Because it is assumed that 'poly_1' and 'poly_2' have
    # a matching domain, it does not matter which one to use
    user_domain_sum = poly_1.user_domain
    internal_domain_sum = poly_1.internal_domain

    # --- Return a new instance
    return ChebyshevPolynomial(
        mi_sum,
        coeffs=coeffs_sum,
        user_domain=user_domain_sum,
        internal_domain=internal_domain_sum,
        grid=grd_sum,
    )


def chebyshev_eval(
    chebyshev_polynomials: "ChebyshevPolynomial",
    xx: np.ndarray,
) -> np.ndarray:
    """Wrapper for the evaluation function in the Chebyshev bases.

    Parameters
    ----------
    chebyshev_polynomials : ChebyshevPolynomial
        The Chebyshev polynomial(s) to be evaluated.
    xx : np.ndarray
        The array of query points of shape ``(k, m)`` at which the monomials
        are evaluated. The values must be in :math:`[-1, 1]^m`.

    Notes
    -----
    - This function must have the specific signature to conform with the
      requirement of the abstract base class.
    - Multiple Chebyshev polynomials having the same set of exponents living
      on the same grid are defined by a multiple set of coefficients.

    .. todo::
        - Allows batch evaluations somewhere upstream.
        - make sure the input is in the domain [-1, 1]^m somewhere upstream.
    """
    # Get required data from the object
    exponents = chebyshev_polynomials.multi_index.exponents
    coefficients = chebyshev_polynomials.coeffs

    results = evaluate_chebyshev_polynomials(xx, exponents, coefficients)

    return results


class ChebyshevPolynomial(MultivariatePolynomialSingleABC):
    """Datatype to describe polynomials in Chebyshev bases."""

    # Virtual Functions
    _add = staticmethod(_chebyshev_add)
    _sub = staticmethod(dummy)
    _mul = staticmethod(dummy)
    _div = staticmethod(dummy)  # type: ignore
    _pow = staticmethod(dummy)  # type: ignore
    _eval = staticmethod(chebyshev_eval)
    _iadd = staticmethod(dummy)

    _partial_diff = staticmethod(dummy)
    _diff = staticmethod(dummy)

    _integrate_over = staticmethod(dummy)

    generate_internal_domain = staticmethod(verify_domain)
    generate_user_domain = staticmethod(verify_domain)
