"""
This module contains the `LagrangePolynomial` class.

The `LagrangePolynomial` class is a concrete implementation of the abstract
base class :py:class:`MultivariatePolynomialSingleABC
<.core.ABC.multivariate_polynomial_abstract.MultivariatePolynomialSingleABC>`
for polynomials in the Lagrange basis.

Background information
----------------------

The relevant section of the
:ref:`fundamentals/polynomial-bases:Lagrange polynomials` contains a more
detailed explanation regarding the polynomials in the Lagrange form.

Implementation details
----------------------

`LagrangePolynomial` is currently designed to be a bare concrete
implementation of the abstract base class
:py:class:`MultivariatePolynomialSingleABC
<.core.ABC.multivariate_polynomial_abstract.MultivariatePolynomialSingleABC>`.
In other words, most (if not all) concrete implementation of the abstract
methods are left undefined and will raise an exception when called or invoked.

`LagrangePolynomial` serves as an entry point to Minterpy polynomials
especially in the context of function approximations because the intuitiveness
of the corresponding coefficients (i.e., they are the function values at
the grid points). However, the polynomial itself is not fully featured
(e.g., no addition, multiplication, etc.) as compared to polynomials in
the other basis.

----

"""
from __future__ import annotations

import copy
import numpy as np

from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.services import is_scalar
from minterpy.utils.polynomials.interface import PolyData, shape_coeffs
from minterpy.utils.polynomials.lagrange import integrate_monomials_lagrange
from minterpy.utils.verification import dummy, verify_domain

__all__ = ["LagrangePolynomial"]


def add_lagrange(
    poly_1: "LagrangePolynomial",
    poly_2: "LagrangePolynomial",
) -> "LagrangePolynomial":
    """Add two instances of polynomials in the Lagrange basis.

    This is the concrete implementation of ``_add()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    handling polynomials in the Lagrange basis.

    Parameters
    ----------
    poly_1 : LagrangePolynomial
        Left operand of the addition/subtraction expression.
    poly_2 : LagrangePolynomial
        Right operand of the addition/subtraction expression.

    Returns
    -------
    LagrangePolynomial
        The sum of two polynomials in the Lagrange basis as a new instance
        of polynomial, also in the Lagrange basis.

    Notes
    -----
    - This function assumes: both polynomials must be in the Lagrange basis,
      they must be initialized (coefficients are not ``None``),
      have the same dimension and their domains are matching,
      and the number of polynomials per instance are the same.
      These conditions are not explicitly checked in this function.
    """
    # --- Get the ingredients of the summed polynomial in the Lagrange basis
    poly_sum_data = _compute_poly_sum_data_lagrange(poly_1, poly_2)

    # --- Return a new instance
    return LagrangePolynomial(**poly_sum_data._asdict())


def integrate_over_lagrange(
    poly: "LagrangePolynomial",
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the definite integral of a polynomial in the Lagrange basis.

    Parameters
    ----------
    poly : LagrangePolynomial
        The polynomial of which the integration is carried out.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integral value of the polynomial over the given domain.
    """
    quad_weights = _compute_quad_weights_lagrange(poly, bounds)

    return quad_weights @ poly.coeffs


# TODO redundant
lagrange_generate_internal_domain = verify_domain
lagrange_generate_user_domain = verify_domain


class LagrangePolynomial(MultivariatePolynomialSingleABC):
    """Concrete implementation of polynomials in the Lagrange basis.

    A polynomial in the Lagrange basis is the sum of so-called Lagrange
    polynomials, each of which is multiplied with a coefficient.

    The value a *single* Lagrange monomial is per definition :math:`1`
    at one of the grid points and :math:`0` on all the other points.

    Notes
    -----
    - The Lagrange polynomials commonly appear in the Wikipedia article is
      in Minterpy considered the "monomial". In other words, a polynomial in
      the Lagrange basis is the sum of Lagrange monomials each of which is
      multiplied with a coefficient.
    - A polynomial in the Lagrange basis may also be defined also
      for multi-indices of exponents which are not downward-closed.
      In such cases, the corresponding Lagrange monomials also form a basis.
      These mononomials still possess their special property of being :math:`1`
      at a single grid point and :math:`0` at all the other points,
      with respect to the given grid.
    """
    # Virtual Functions
    _add = staticmethod(add_lagrange)
    _sub = staticmethod(dummy)  # type: ignore
    _mul = staticmethod(dummy)  # type: ignore
    _div = staticmethod(dummy)  # type: ignore
    _pow = staticmethod(dummy)  # type: ignore
    _eval = staticmethod(dummy)  # type: ignore
    _iadd = staticmethod(dummy)

    _partial_diff = staticmethod(dummy)
    _diff = staticmethod(dummy)

    _integrate_over = staticmethod(integrate_over_lagrange)

    generate_internal_domain = staticmethod(lagrange_generate_internal_domain)
    generate_user_domain = staticmethod(lagrange_generate_user_domain)


def _compute_poly_sum_data_lagrange(
    poly_1: LagrangePolynomial,
    poly_2: LagrangePolynomial,
) -> PolyData:
    """Compute the data to create a summed polynomial in the Lagrange basis.

    Parameters
    ----------
    poly_1 : LagrangePolynomial
        Left operand of the addition/subtraction expression.
    poly_2 : LagrangePolynomial
        Right operand of the addition/subtraction expression.

    Returns
    -------
    PolyData
        The ingredients to construct a summed polynomial in the Lagrange basis.

    Notes
    -----
    - Only addition with or between constant scalar polynomials is supported.
      In other words, one of the polynomials must be a scalar polynomial.
    - Both polynomials are assumed to have the same type, spatial dimension,
      and matching domains. This has been made sure by the abstract base class.
    """
    # --- Only if one of the operands is a constant scalar polynomial
    if not is_scalar(poly_1) and not is_scalar(poly_2):
        raise NotImplementedError(
            "General polynomial-polynomial addition/subtraction "
            f"for {type(poly_1)} is not supported."
    )

    # --- Get the grid and multi-index set of the summed polynomial
    # NOTE: Simply take the one with the largest multi-index set
    # (i.e., the non-scalar polynomial)
    non_scalar_poly = poly_1 if not is_scalar(poly_1) else poly_2
    grd_sum = copy.copy(non_scalar_poly.grid)
    if non_scalar_poly.indices_are_separate:
        mi_sum = copy.copy(non_scalar_poly.multi_index)
    else:
        mi_sum = grd_sum.multi_index

    # --- Process the coefficients
    # NOTE: In the Lagrange basis, adding a scalar applies to all coefficients
    # because there is no "constant" term in the multi-index set.
    # Shape the coefficients; ensure they have the same dimension
    coeffs_1, coeffs_2 = shape_coeffs(poly_1, poly_2)
    coeffs_sum = coeffs_1 + coeffs_2

    # --- Process the domains
    # NOTE: Because it is assumed that 'poly_1' and 'poly_2' have
    # matching domains, it does not matter which one to use
    internal_domain_sum = poly_1.internal_domain
    user_domain_sum = poly_1.user_domain

    return PolyData(
        multi_index=mi_sum,
        coeffs=coeffs_sum,
        internal_domain=internal_domain_sum,
        user_domain=user_domain_sum,
        grid=grd_sum,
    )


def _compute_quad_weights_lagrange(
    poly: LagrangePolynomial,
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the quadrature weights of a polynomial in the Lagrange basis.
    """
    # Get the relevant data from the polynomial instance
    exponents = poly.multi_index.exponents
    generating_points = poly.grid.generating_points
    # ...from the MultiIndexTree
    tree = poly.grid.tree
    split_positions = tree.split_positions
    subtree_sizes = tree.subtree_sizes
    masks = tree.stored_masks

    quad_weights = integrate_monomials_lagrange(
        exponents,
        generating_points,
        split_positions,
        subtree_sizes,
        masks,
        bounds,
    )

    return quad_weights
