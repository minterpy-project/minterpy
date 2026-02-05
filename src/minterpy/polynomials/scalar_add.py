"""
Scalar addition for polynomials.

This module implements scalar addition for polynomial bases where adding
a constant affects only the coefficient of the (0, ..., 0) monomial
(Newton, Canonical, Chebyshev). It excludes Lagrange polynomials, where
scalar addition affects all coefficients uniformly.
"""
import numpy as np

from minterpy.global_settings import SCALAR
from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.core import Grid, MultiIndexSet


def scalar_add_via_monomials(
    poly: MultivariatePolynomialSingleABC,
    scalar: SCALAR,
) -> MultivariatePolynomialSingleABC:
    r"""Add an instance of polynomial with a real scalar based on the monomial.

    Monomial-based scalar addition add the scalar to the polynomial coefficient
    that corresponds to the multi-index set element of :math:`(0, \ldots, 0)`
    (if exists). If the element does not exist, meaning that the polynomial
    does not have a constant term, then the multi-index set is extended.

    Parameters
    ----------
    poly : MultivariatePolynomialSingleABC
        A polynomial instance to be added with a scalar.
    scalar : SCALAR
        The real scalar number to be added to the polynomial instance.

    Returns
    -------
    MultivariatePolynomialSingleABC
        The summed polynomial; the polynomial is a new instance.

    Notes
    -----
    - Currently ``NewtonPolynomial``, ``CanonicalPolynomial``, and
      ``ChebyshevPolynomial`` follow monomial-based scalar addition, while
      ``LagrangePolynomial`` does not. For the latter, because the coefficients
      are function values, adding scalar will add to all coefficients.
    """
    # Create a constant polynomial
    poly_scalar = _create_scalar_poly(poly, scalar)

    # Rely on the `__add__()` method implemented upstream
    return poly + poly_scalar


def _create_scalar_poly(
    poly: MultivariatePolynomialSingleABC,
    scalar: SCALAR,
) -> MultivariatePolynomialSingleABC:
    """Create a constant scalar polynomial from a given polynomial.

    Parameters
    ----------
    poly : MultivariatePolynomialSingleABC
        An instance of polynomial from which a constant polynomial will be
        created.
    scalar : SCALAR
        Real numbers for the coefficient value of the constant polynomial.

    Returns
    -------
    MultivariatePolynomialSingleABC
        A polynomial of the same instance as ``poly`` having the same grid and
        domains but with a single element multi-index set. If the grid does
        not include the element (0, ..., 0) then the grid will be extended.

    Notes
    -----
    - Assumes ``poly`` is initialized (``len(poly) >= 1``). The caller
      verifies this precondition.
    """
    # Create a single-element multi-index set of (0, ..., 0)
    dim = poly.spatial_dimension
    lp_degree = poly.multi_index.lp_degree
    mi_0 = MultiIndexSet.from_degree(dim, poly_degree=0, lp_degree=lp_degree)

    # Create a Grid containing the constant term
    # The input polynomial's grid may not include (0, ..., 0) if its
    # multi-index set is non-downward-closed, so we construct a minimal
    # grid containing only this element.
    grd_0 = Grid(
        mi_0,
        poly.grid.generating_function,
        poly.grid.generating_points,
        domain=poly.grid.domain,
    )

    # Create the coefficient
    if len(poly) == 1:
        coeffs_0 = np.array([scalar], dtype=poly.coeffs.dtype)
    else:
        coeffs_0 = scalar * np.ones(
            shape=(1, len(poly)),
            dtype=poly.coeffs.dtype,
        )

    # Return a polynomial instance of the same class as input
    return poly.__class__(multi_index=mi_0, coeffs=coeffs_0, grid=grd_0)
