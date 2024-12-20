"""
Implementations of the transformation classes **from**
:py:class:`.ChebyshevPolynomial` (polynomials in the
:ref:`Chebyshev basis <fundamentals/polynomial-bases:Chebyshev basis>`
of the first kind) **to**:

- :py:class:`.LagrangePolynomial` (polynomials in the
  :ref:`Lagrange basis <fundamentals/polynomial-bases:Lagrange basis>`)
- :py:class:`.NewtonPolynomial` (polynomials in the
  :ref:`Newton basis <fundamentals/polynomial-bases:Newton basis>`)
- :py:class:`.CanonicalPolynomial` (polynomials in the
  :ref:`canonical basis <fundamentals/polynomial-bases:Canonical basis>`)
"""
from minterpy.core.ABC import TransformationABC
from minterpy.polynomials import (
    ChebyshevPolynomial,
    LagrangePolynomial,
    NewtonPolynomial,
    CanonicalPolynomial,
)
from minterpy.transformations.utils import (
    build_chebyshev_to_lagrange_operator,
    build_chebyshev_to_newton_operator,
    build_chebyshev_to_canonical_operator,
)

__all__ = ["ChebyshevToLagrange", "ChebyshevToNewton", "ChebyshevToCanonical"]


class ChebyshevToLagrange(TransformationABC):
    """Transformation from the Chebyshev basis to the Lagrange basis."""

    origin_type = ChebyshevPolynomial
    target_type = LagrangePolynomial
    _get_transformation_operator = build_chebyshev_to_lagrange_operator


class ChebyshevToNewton(TransformationABC):
    """Transformation from the Chebyshev basis to the Newton basis."""

    origin_type = ChebyshevPolynomial
    target_type = NewtonPolynomial
    _get_transformation_operator = build_chebyshev_to_newton_operator


class ChebyshevToCanonical(TransformationABC):
    """Transformation from the Chebyshev basis to the Canonical basis."""

    origin_type = ChebyshevPolynomial
    target_type = CanonicalPolynomial
    _get_transformation_operator = build_chebyshev_to_canonical_operator
