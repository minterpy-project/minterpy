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
from minterpy.utils.polynomials.chebyshev import (
    evaluate_polynomials,
    compute_poly_sum_coeffs,
)
from minterpy.utils.polynomials.interface import (
    get_grid_and_multi_index_poly_sum,
    PolyData,
    shape_coeffs,
)
from minterpy.utils.verification import dummy, verify_domain

__all__ = ["ChebyshevPolynomial"]


def add_chebyshev(
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
    poly_data = _compute_poly_sum_data_chebyshev(poly_1, poly_2)

    # --- Return a new instance
    return ChebyshevPolynomial(**poly_data._asdict())


def eval_chebyshev(
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

    results = evaluate_polynomials(xx, exponents, coefficients)

    return results


class ChebyshevPolynomial(MultivariatePolynomialSingleABC):
    """Datatype to describe polynomials in Chebyshev bases."""

    # Virtual Functions
    _add = staticmethod(add_chebyshev)
    _sub = staticmethod(dummy)
    _mul = staticmethod(dummy)
    _div = staticmethod(dummy)  # type: ignore
    _pow = staticmethod(dummy)  # type: ignore
    _eval = staticmethod(eval_chebyshev)
    _iadd = staticmethod(dummy)

    _partial_diff = staticmethod(dummy)
    _diff = staticmethod(dummy)

    _integrate_over = staticmethod(dummy)

    generate_internal_domain = staticmethod(verify_domain)
    generate_user_domain = staticmethod(verify_domain)


# --- Internal utility functions
def _compute_poly_sum_data_chebyshev(
    poly_1: "ChebyshevPolynomial",
    poly_2: "ChebyshevPolynomial",
) -> PolyData:
    """Compute the data to create a summed polynomial in the Chebyshev basis.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the addition expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the addition expression.

    Returns
    -------
    PolyData
        A tuple with the main ingredients to construct a summed polynomial
        in the Chebyshev basis.

    Notes
    -----
    - Both polynomials are assumed to have the same dimension
      and matching domains.
    """
    # --- Get the grid and multi-index set of the summed polynomial
    grd_sum, mi_sum = get_grid_and_multi_index_poly_sum(poly_1, poly_2)

    # --- Process the coefficients
    # Shape the coefficients; ensure they have the same dimension
    coeffs_1, coeffs_2 = shape_coeffs(poly_1, poly_2)

    # Compute the coefficients of the summed polynomial
    coeffs_sum = compute_poly_sum_coeffs(
        poly_1.multi_index.exponents,
        coeffs_1,
        poly_2.multi_index.exponents,
        coeffs_2,
        mi_sum.exponents,
    )

    # --- Process the domains
    # NOTE: Because it is assumed that 'poly_1' and 'poly_2' have
    # matching domains, it does not matter which one to use
    internal_domain_sum = poly_1.internal_domain
    user_domain_sum = poly_1.user_domain

    return PolyData(
        mi_sum,
        coeffs_sum,
        internal_domain_sum,
        user_domain_sum,
        grd_sum,
    )
