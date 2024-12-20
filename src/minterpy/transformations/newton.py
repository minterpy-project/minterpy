"""
Implementations of the transformation classes **from**
:py:class:`.NewtonPolynomial` (polynomials in the
:ref:`Newton basis <fundamentals/polynomial-bases:Newton basis>`)
**to**:

- :py:class:`.LagrangePolynomial` (polynomials in the
  :ref:`Lagrange basis <fundamentals/polynomial-bases:Lagrange basis>`)
- :py:class:`.CanonicalPolynomial` (polynomials in the
  :ref:`canonical basis <fundamentals/polynomial-bases:Canonical basis>`)
- :py:class:`.ChebyshevPolynomial` (polynomials in the
  :ref:`Chebyshev basis <fundamentals/polynomial-bases:Chebyshev basis>`
  of the first kind)
"""
from minterpy.core.ABC import TransformationABC
from minterpy.polynomials import (
    NewtonPolynomial,
    LagrangePolynomial,
    CanonicalPolynomial,
    ChebyshevPolynomial,
)
from minterpy.transformations.utils import (
    build_newton_to_lagrange_operator,
    build_newton_to_canonical_operator,
    build_newton_to_chebyshev_operator,
)

__all__ = ["NewtonToLagrange", "NewtonToCanonical", "NewtonToChebyshev"]


class NewtonToLagrange(TransformationABC):
    """Transformation from the Newton basis to the Lagrange basis."""

    origin_type = NewtonPolynomial
    target_type = LagrangePolynomial
    _get_transformation_operator = build_newton_to_lagrange_operator


class NewtonToCanonical(TransformationABC):
    """Transformation from the Newton basis to the canonical basis."""

    origin_type = NewtonPolynomial
    target_type = CanonicalPolynomial
    _get_transformation_operator = build_newton_to_canonical_operator


class NewtonToChebyshev(TransformationABC):
    """Transformation from the Newton basis to the Chebyshev basis."""

    origin_type = NewtonPolynomial
    target_type = ChebyshevPolynomial
    _get_transformation_operator = build_newton_to_chebyshev_operator
