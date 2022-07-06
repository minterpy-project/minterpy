"""
ChebyshevPolynomial class

Defines the Chebyshev polynomials of the first kind.
"""
from typing import Any, Optional

from .utils import evaluate_chebyshev_monomials
from ..core.ABC import MultivariatePolynomialSingleABC
from ..core.verification import verify_domain

__all__ = ["ChebyshevPolynomial"]


def dummy(x: Optional[Any] = None) -> None:
    """Placeholder function.

    .. warning::
      This feature is not implemented yet!
    """
    raise NotImplementedError("This feature is not implemented yet.")


def chebyshev_eval(chebyshev_poly, x):
    """Evaluation function in Chebyshev base.

    Parameters
    ----------
    chebyshev_poly: The :class:`ChebyshevPolynomial` which is evaluated.
    x: Points where the Chebyshev polynomial is to be evaluated. The input shape needs to be ``(N,dim)``, where
    ``N`` refers to the number of points and ``dim`` refers to the dimension of the domain space.

    Notes
    -----
    The Chebyshev Polynomial has domain [-1,1]
    """

    monomials_eval = evaluate_chebyshev_monomials(x, chebyshev_poly.multi_index.exponents)

    ## Multiply with coeffs
    res = monomials_eval @ chebyshev_poly.coeffs

    return res

# TODO redundant
chebyshev_generate_internal_domain = verify_domain
chebyshev_generate_user_domain = verify_domain


class ChebyshevPolynomial(MultivariatePolynomialSingleABC):
    """Datatype to describe polynomials in Chebyshev base.


    Attributes
    ----------
    coeffs
    nr_active_monomials
    spatial_dimension
    unisolvent_nodes

    """

    # Virtual Functions
    _add = staticmethod(dummy)
    _sub = staticmethod(dummy)
    _mul = staticmethod(dummy)
    _div = staticmethod(dummy)  # type: ignore
    _pow = staticmethod(dummy)  # type: ignore
    _eval = chebyshev_eval

    _partial_diff = staticmethod(dummy)
    _diff = staticmethod(dummy)

    generate_internal_domain = staticmethod(chebyshev_generate_internal_domain)
    generate_user_domain = staticmethod(chebyshev_generate_user_domain)
