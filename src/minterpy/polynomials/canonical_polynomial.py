"""
Base class for polynomials in the canonical base.

"""
from __future__ import annotations

import numpy as np

from scipy.special import factorial

from minterpy.global_settings import INT_DTYPE
from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.utils.polynomials.canonical import (
    eval_polynomials,
    integrate_monomials,
)
from minterpy.utils.polynomials.interface import (
    compute_coeffs_poly_prod_via_monomials,
    compute_coeffs_poly_sum_via_monomials,
    get_grid_and_multi_index_poly_prod,
    get_grid_and_multi_index_poly_sum,
    PolyData,
    scalar_add_via_monomials,
)
from minterpy.utils.verification import (
    dummy,
    verify_domain,
)
from minterpy.utils.arrays import make_coeffs_2d
from minterpy.utils.multi_index import find_match_between
from minterpy.jit_compiled.multi_index import all_indices_are_contained

__all__ = ["CanonicalPolynomial"]


# --- Evaluation
def eval_canonical(poly: "CanonicalPolynomial", xx: np.ndarray) -> np.ndarray:
    """Evaluate polynomial(s) in the canonical basis on a set of query points.

    Parameters
    ----------
    poly : CanonicalPolynomial
        The instance of polynomial in the canonical basis to be evaluated.
    xx : :class:`numpy:numpy.ndarray`
        Array of query points in the at which the polynomial(s) is evaluated.
        The array is of shape ``(N, m)`` where ``N`` is the number of points
        and ``m`` is the spatial dimension of the polynomial.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The output of the polynomial evaluation. If the polynomial consists
        of a single coefficient set, the output array is one-dimensional with
        a length of ``N``. If the polynomial consists of multiple coefficients
        sets, the output array is two-dimensional with a shape of
        ``(N, n_poly)`` where ``n_poly`` is the number of coefficient sets.

    See Also
    --------
    minterpy.utils.polynomials.canonical.eval_polynomials
        The actual implementation of the evaluation of polynomials in
        the canonical basis.
    """
    coeffs = poly.coeffs
    exponents = poly.multi_index.exponents

    return eval_polynomials(xx, coeffs, exponents)


# --- Arithmetics (Addition, Multiplication)
def add_canonical(
    poly_1: "CanonicalPolynomial",
    poly_2: "CanonicalPolynomial",
) -> "CanonicalPolynomial":
    """Add two polynomial instances in the canonical basis.

    This is the concrete implementation of ``_add()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    polynomial in the canonical basis.

    Parameters
    ----------
    poly_1 : CanonicalPolynomial
        Left operand of the addition expression.
    poly_2 : CanonicalPolynomial
        Right operand of the addition expression.

    Returns
    -------
    CanonicalPolynomial
        The sum of two polynomials in the canonical basis as a new instance
        of polynomial.

    Notes
    -----
    - This function assumes: both polynomials must be in canonical basis,
      they must be initialized, have the same dimension and their domains
      are matching, and the number of polynomials per instance are the same.
      These conditions are not explicitly checked in this function; the caller
      is responsible for the verification.
    """
    # --- Get the ingredients of the summed polynomial in the Canonical basis
    poly_sum_data = _compute_data_poly_sum(poly_1, poly_2)

    # --- Return a new instance
    return CanonicalPolynomial(**poly_sum_data._asdict())


def mul_canonical(
    poly_1: "CanonicalPolynomial",
    poly_2: "CanonicalPolynomial",
) -> "CanonicalPolynomial":
    """Multiply two polynomial instances in the canonical basis.

    This is the concrete implementation of ``_mul()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    polynomials in the canonical basis.

    Parameters
    ----------
    poly_1 : CanonicalPolynomial
        Left operand of the multiplication expression.
    poly_2 : CanonicalPolynomial
        Right operand of the multiplication expression.

    Returns
    -------
    CanonicalPolynomial
        The product of two polynomials in the canonical basis; a new instance
        of polynomial.

    Notes
    -----
    - This function assumes: both polynomials must be in the canonical basis,
      they must be initialized, have the same dimension and their domains
      are matching, and the number of polynomials per instance are the same.
      These conditions are not explicitly checked in this function; the caller
      is responsible for the verification.
    """
    # --- Get the ingredients of the product polynomial in the canonical basis
    poly_prod_data = _compute_data_poly_prod(poly_1, poly_2)

    # --- Return a new instance
    return CanonicalPolynomial(**poly_prod_data._asdict())


# TODO redundant
canonical_generate_internal_domain = verify_domain
canonical_generate_user_domain = verify_domain


def _canonical_partial_diff(poly: "CanonicalPolynomial", dim: int, order: int) -> "CanonicalPolynomial":
    """ Partial differentiation in Canonical basis.
    """
    spatial_dim = poly.multi_index.spatial_dimension
    deriv_order_along = np.zeros(spatial_dim, dtype=INT_DTYPE)
    deriv_order_along[dim] = order
    return _canonical_diff(poly, deriv_order_along)


def _canonical_diff(poly: "CanonicalPolynomial", order: np.ndarray) -> "CanonicalPolynomial":
    """ Partial differentiation in Canonical basis.
    """

    coeffs = make_coeffs_2d(poly.coeffs)
    exponents = poly.multi_index.exponents

    # Guard rails in ABC ensures that the len(order) == poly.spatial_dimension
    subtracted_exponents = exponents - order

    # compute mask for non-negative multi index entries
    diff_exp_mask = np.all(exponents >= order, axis = 1)

    # multi index entries in the differentiated polynomial
    diff_exponents = subtracted_exponents[diff_exp_mask]

    # Checking if the necessary multi index entries are present
    # Zero size check is needed here as all_indices_are_contained throws error otherwise
    if diff_exponents.size != 0 and not all_indices_are_contained(diff_exponents, exponents):
        raise ValueError(f"Cannot differentiate as some of the required multi indices are not present.")

    # coefficients of the differentiated polynomial
    diff_coeffs = coeffs[diff_exp_mask] * np.prod(factorial(exponents[diff_exp_mask]) / factorial(diff_exponents),
                                                  axis=1)[:,None]

    # The differentiated polynomial being expressed wrt multi indices of the given poly
    # NOTE: 'find_match_between' assumes 'exponents' is lexicographically ordered
    map_pos = find_match_between(diff_exponents, exponents)
    new_coeffs = np.zeros_like(coeffs)
    new_coeffs[map_pos] = diff_coeffs

    # Squeezing the last dimension to handle single polynomial
    return CanonicalPolynomial.from_poly(poly, new_coeffs.reshape(poly.coeffs.shape))


def _canonical_integrate_over(
    poly: "CanonicalPolynomial",
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the definite integral of a polynomial in the canonical basis.

    Parameters
    ----------
    poly : CanonicalPolynomial
        The polynomial of which the integration is carried out.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integral value of the polynomial over the given domain.
    """
    # --- Compute the integral of the canonical monomials (quadrature weights)
    quad_weights = _compute_quad_weights(poly, bounds)

    return quad_weights @ poly.coeffs


class CanonicalPolynomial(MultivariatePolynomialSingleABC):
    """Concrete implementation of polynomials in the canonical basis."""
    # --- Virtual Functions

    # Evaluation
    _eval = staticmethod(eval_canonical)

    # Arithmetics (polynomial-polynomial)
    _add = staticmethod(add_canonical)
    _sub = staticmethod(dummy)  # type: ignore
    _mul = staticmethod(mul_canonical)
    _div = staticmethod(dummy)  # type: ignore
    _pow = staticmethod(dummy)  # type: ignore

    # Arithmetics (polynomial-scalar)
    _scalar_add = staticmethod(scalar_add_via_monomials)

    # Calculus
    _partial_diff = staticmethod(_canonical_partial_diff)
    _diff = staticmethod(_canonical_diff)
    _integrate_over = staticmethod(_canonical_integrate_over)

    # Domain generation
    generate_internal_domain = staticmethod(canonical_generate_internal_domain)
    generate_user_domain = staticmethod(canonical_generate_user_domain)


# --- Internal utility functions
def _compute_data_poly_sum(
    poly_1: "CanonicalPolynomial",
    poly_2: "CanonicalPolynomial",
) -> PolyData:
    """Compute the data to create a summed polynomial in the canonical basis.

    Addition or subtraction of polynomials in the canonical basis is based
    on the adding (resp. subtracting) the coefficients of the matching
    monomials of the two polynomial operands (i.e., the matching elements of
    the two multi-index sets).

    Parameters
    ----------
    poly_1 : CanonicalPolynomial
        Left operand of the addition expression.
    poly_2 : CanonicalPolynomial
        Right operand of the addition expression.

    Returns
    -------
    PolyData
        The ingredients to construct a summed polynomial in the canonical
        basis.

    Notes
    -----
    - Both polynomials are assumed to have the same dimension
      and matching domains.
    """
    # --- Get the grid and multi-index set of the summed polynomial
    grd_sum, mi_sum = get_grid_and_multi_index_poly_sum(poly_1, poly_2)

    # --- Process the coefficients
    # NOTE: indices may or may not be separate, use the summed multi-index set
    #       instead of the one attached to grid
    coeffs_sum = compute_coeffs_poly_sum_via_monomials(poly_1, poly_2, mi_sum)

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


def _compute_data_poly_prod(
    poly_1: "CanonicalPolynomial",
    poly_2: "CanonicalPolynomial",
) -> PolyData:
    """Compute the data to create a product polynomial in the canonical basis.

    Multiplication of polynomials in the canonical basis is based on
    multiplying the coefficients of the matching monomials of the two
    polynomial operands (i.e., the matching elements of the two multi-index
    sets). In the canonical basis, multiplying two monomials results in a
    a monomial of a higher degree (the degree sum of the two monomials).

    Parameters
    ----------
    poly_1 : CanonicalPolynomial
        Left operand of the multiplication expression.
    poly_2 : CanonicalPolynomial
        Right operand of the multiplication expression.

    Returns
    -------
    PolyData
        The ingredients to construct a product polynomial in the canonical
        basis.

    Notes
    -----
    - Both polynomials are assumed to have the same dimension
      and matching domains.
    """
    # --- Get the grid and multi-index set of the summed polynomial
    grd_prod, mi_prod = get_grid_and_multi_index_poly_prod(poly_1, poly_2)

    # --- Process the coefficients
    # NOTE: indices may or may not be separate, use the summed multi-index set
    #       instead of the one attached to grid
    coeffs_prod = compute_coeffs_poly_prod_via_monomials(
        poly_1,
        poly_2,
        mi_prod,
    )

    # --- Process the domains
    # NOTE: Because it is assumed that 'poly_1' and 'poly_2' have
    # matching domains, it does not matter which one to use
    internal_domain_prod = poly_1.internal_domain
    user_domain_prod = poly_1.user_domain

    return PolyData(
        multi_index=mi_prod,
        coeffs=coeffs_prod,
        internal_domain=internal_domain_prod,
        user_domain=user_domain_prod,
        grid=grd_prod,
    )


def _compute_quad_weights(
    poly: CanonicalPolynomial,
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the quadrature weights of a polynomial in the Canonical basis.
    """
    # Get the relevant data from the polynomial instance
    exponents = poly.multi_index.exponents

    quad_weights = integrate_monomials(exponents, bounds)

    return quad_weights
