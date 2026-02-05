"""
Base class for polynomials in the canonical base.

"""
from __future__ import annotations

import numpy as np

from scipy.special import factorial

from minterpy import MultiIndexSet, Grid
from minterpy.global_settings import INT_DTYPE
from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.polynomials.arithmetic import (
    get_compute_coeffs_add,
    get_compute_coeffs_mul,
)
from minterpy.polynomials.scalar_add import scalar_add_via_monomials
from minterpy.utils.polynomials.canonical import (
    eval_polynomials,
    integrate_monomials,
)
from minterpy.utils.verification import dummy
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
    multi_index: MultiIndexSet,
    grid: Grid,
) -> "CanonicalPolynomial":
    """Add two polynomial instances in the canonical basis.

    This is the concrete implementation of ``_add()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    polynomial in the canonical basis.

    Parameters
    ----------
    poly_1 : CanonicalPolynomial
        Left operand of the addition.
    poly_2 : CanonicalPolynomial
        Right operand of the addition.
    multi_index : MultiIndexSet
        The multi-index set of the resulting polynomial, i.e., the union
        of the multi-index sets of the operands.
    grid : Grid
        The grid of the resulting polynomial, i.e., the union of the grids
        of the two operands.

    Returns
    -------
    CanonicalPolynomial
        The sum of two polynomials in the canonical basis as a new instance
        of polynomial.

    Notes
    -----
    - This function assumes the caller has verified: both polynomials are in
      the canonical basis, both are initialized, their domains match, and they
      have the same number of polynomial instances.
    - The provided ``multi_index`` may differ from ``grid.multi_index`` when
      the polynomial multi-index set is a subset of the grid multi-index set
      (i.e., separated indices). The coefficients are computed
      with respect to the given ``multi_index``.
    """
    # Get the function to compute the sum coefficients via monomials
    compute_coeffs = get_compute_coeffs_add("monomials")

    # Compute the coefficients
    coeffs = compute_coeffs(poly_1, poly_2, multi_index)

    # Create and return a new instance of polynomial
    return CanonicalPolynomial(multi_index, coeffs, grid)


def mul_canonical(
    poly_1: "CanonicalPolynomial",
    poly_2: "CanonicalPolynomial",
    multi_index: MultiIndexSet,
    grid: Grid,
) -> "CanonicalPolynomial":
    """Multiply two polynomial instances in the canonical basis.

    This is the concrete implementation of ``_mul()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    polynomials in the canonical basis.

    Parameters
    ----------
    poly_1 : CanonicalPolynomial
        Left operand of the multiplication.
    poly_2 : CanonicalPolynomial
        Right operand of the multiplication.
    multi_index : MultiIndexSet
        The multi-index set of the resulting polynomial, i.e., the product
        of the multi-index sets of the two operands.
    grid : Grid
        The grid of the resulting polynomial, i.e., the product of the grids
        of the two operands.

    Returns
    -------
    CanonicalPolynomial
        The product of the two polynomials in the canonical basis as a new
        polynomial instance.

    Notes
    -----
    - This function assumes the caller has verified: both polynomials are in
      the canonical basis, both are initialized, their domains match, and they
      have the same number of polynomial instances.
    - The provided ``multi_index`` may differ from ``grid.multi_index`` when
      the polynomial multi-index set is a subset of the grid multi-index set
      (i.e., separated indices). The coefficients are computed
      with respect to the given ``multi_index``.
    """
    # Get the function to compute the product coefficients via monomials
    compute_coeffs = get_compute_coeffs_mul("monomials")

    # Compute the coefficients
    coeffs = compute_coeffs(poly_1, poly_2, multi_index)

    # Create and return a new instance of polynomial
    return CanonicalPolynomial(multi_index, coeffs, grid)


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


# --- Internal utility functions
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
