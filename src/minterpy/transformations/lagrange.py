"""
Implementations of the transformation classes of **from**
:py:class:`.LagrangePolynomial` (polynomials in the
:ref:`Lagrange basis <fundamentals/polynomial-bases:Lagrange basis>`)
**to**:

- :py:class:`.NewtonPolynomial` (polynomials in the
  :ref:`Newton basis <fundamentals/polynomial-bases:Newton basis>`)
- :py:class:`.CanonicalPolynomial` (polynomials in the
  :ref:`canonical basis <fundamentals/polynomial-bases:Canonical basis>`)
- :py:class:`.ChebyshevPolynomial` (polynomials in the
  :ref:`Chebyshev basis <fundamentals/polynomial-bases:Chebyshev basis>`
  of the first kind)
"""
from minterpy.core.ABC import TransformationABC
from minterpy.polynomials import (
    LagrangePolynomial,
    NewtonPolynomial,
    CanonicalPolynomial,
    ChebyshevPolynomial,
)
from minterpy.transformations.utils import (
    build_lagrange_to_canonical_operator,
    build_lagrange_to_newton_operator,
    build_lagrange_to_chebyshev_operator,
)

__all__ = ["LagrangeToNewton", "LagrangeToCanonical", "LagrangeToChebyshev"]


class LagrangeToNewton(TransformationABC):
    """Transformation from the Lagrange basis to the Newton basis."""

    origin_type = LagrangePolynomial
    target_type = NewtonPolynomial
    _get_transformation_operator = build_lagrange_to_newton_operator


class LagrangeToCanonical(TransformationABC):
    """Transformation from the Lagrange basis to the canonical basis."""

    origin_type = LagrangePolynomial
    target_type = CanonicalPolynomial
    _get_transformation_operator = build_lagrange_to_canonical_operator


class LagrangeToChebyshev(TransformationABC):
    """Transformation from the Lagrange basis to the Chebyshev basis."""

    origin_type = LagrangePolynomial
    target_type = ChebyshevPolynomial
    _get_transformation_operator = build_lagrange_to_chebyshev_operator
