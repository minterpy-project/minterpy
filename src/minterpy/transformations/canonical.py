"""
Implementations of the transformation classes **from**
:py:class:`.CanonicalPolynomial` (polynomials in the
:ref:`canonical basis <fundamentals/polynomial-bases:Canonical basis>`)
**to**:

- :py:class:`.LagrangePolynomial` (polynomials in the
  :ref:`Lagrange basis <fundamentals/polynomial-bases:Lagrange basis>`)
- :py:class:`.NewtonPolynomial` (polynomials in the
  :ref:`Newton basis <fundamentals/polynomial-bases:Newton basis>`)
- :py:class:`.ChebyshevPolynomial` (polynomials in the
  :ref:`Chebyshev basis <fundamentals/polynomial-bases:Chebyshev basis>`
  of the first kind)
"""
from minterpy.core.ABC import TransformationABC
from minterpy.polynomials import (
    CanonicalPolynomial,
    LagrangePolynomial,
    NewtonPolynomial,
    ChebyshevPolynomial,
)
from minterpy.transformations.utils import (
    build_canonical_to_lagrange_operator,
    build_canonical_to_newton_operator,
    build_canonical_to_chebyshev_operator,
)

__all__ = ["CanonicalToNewton", "CanonicalToLagrange", "CanonicalToChebyshev"]


class CanonicalToNewton(TransformationABC):
    """Transformation from the canonical basis to the Newton basis."""

    origin_type = CanonicalPolynomial
    target_type = NewtonPolynomial
    _get_transformation_operator = build_canonical_to_newton_operator


class CanonicalToLagrange(TransformationABC):
    """Transformation from the canonical basis to the Lagrange basis."""

    origin_type = CanonicalPolynomial
    target_type = LagrangePolynomial
    _get_transformation_operator = build_canonical_to_lagrange_operator


class CanonicalToChebyshev(TransformationABC):
    """Transformation from the canonical basis to the Chebyshev basis."""

    origin_type = CanonicalPolynomial
    target_type = ChebyshevPolynomial
    _get_transformation_operator = build_canonical_to_chebyshev_operator
