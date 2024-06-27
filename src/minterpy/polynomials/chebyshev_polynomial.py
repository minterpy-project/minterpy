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

from scipy.special import eval_chebyt

from .utils import dummy
from ..core.ABC import MultivariatePolynomialSingleABC
from minterpy.utils.verification import verify_domain

__all__ = ["ChebyshevPolynomial", "evaluate_chebyshev_monomials"]


def evaluate_chebyshev_monomials(
    xx: np.ndarray,
    exponents: np.ndarray,
) -> np.ndarray:
    """Evaluate the Chebyshev monomials at all query points.

    Parameters
    ----------
    xx : np.ndarray
        The array of query points of shape ``(k, m)`` at which the monomials
        are evaluated. The values must be in :math:`[-1, 1]^m`.
    exponents : np.ndarray
        The non-negative integer array of polynomial exponents (i.e., as
        multi-indices) of shape ``(N, m)``.

    Returns
    -------
    np.ndarray
        The value of each Chebyshev basis evaluated at each given point.
        The array is of shape ``(k, N)``.
    """
    # One-dimensional monomials in each dimension
    monomials = eval_chebyt(exponents[None, :, :], xx[:, None, :])

    # Multi-dimensional monomials by tensor product
    monomials = np.prod(monomials, axis=-1)

    return monomials


def evaluate_chebyshev_polynomials(
    xx: np.ndarray,
    exponents: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate polynomial(s) in the Chebyshev bases.

    Parameters
    ----------
    xx : np.ndarray
        The array of query points of shape ``(k, m)`` at which the monomials
        are evaluated. The values must be in :math:`[-1, 1]^m`.
    exponents : np.ndarray
        The non-negative integer array of polynomial exponents (i.e., as
        multi-indices) of shape ``(N, m)``.
    coefficients : np.ndarray
        The array of coefficients of the polynomials of shape ``(N, Np)``.
        Multiple sets of coefficients (``Np > 1``) indicate multiple Chebyshev
        polynomials evaluated at the same time at the same query points.
    Notes
    -----
    The Chebyshev Polynomial has domain [-1,1]
    """
    # Evaluate the monomials
    monomials = evaluate_chebyshev_monomials(xx, exponents)

    # Multiply with the coefficients
    results = monomials @ coefficients

    return results


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
    _add = staticmethod(dummy)
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
