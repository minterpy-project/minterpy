"""
Concrete implementations of the Transformation classes for the ChebyshevPolynomial.

Transformations from Chebyshev basis to Lagrange basis is provided.
"""

from minterpy.core.ABC import TransformationABC
from minterpy.polynomials import ChebyshevPolynomial, LagrangePolynomial

from .utils import (
    _build_chebyshev_to_lagrange_operator
)

__all__ = ["ChebyshevToLagrange"]


class ChebyshevToLagrange(TransformationABC):
    """Transformation from ChebyshevPolynomial to LagrangePolynomial"""

    origin_type = ChebyshevPolynomial
    target_type = LagrangePolynomial
    _get_transformation_operator = _build_chebyshev_to_lagrange_operator
