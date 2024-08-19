"""
This module defines ``ChebyshevPolynomial`` class for Chebyshev polynomials
of the first kind.

Some common notations/symbols used below:

- ``m``: the number of spatial dimensions
- ``N``: the number of monomials and coefficients
- ``k``: the number of evaluation/query points
- ``Np``: the number of polynomials (i.e., set of coefficients)

Chebyshev polynomials are defined on :math:`[-1, 1]^m`.
"""
import numpy as np

from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.core import Grid, MultiIndexSet
from minterpy.utils.polynomials.chebyshev import (
    evaluate_monomials,
    evaluate_polynomials,
)
from minterpy.utils.polynomials.interface import (
    compute_coeffs_poly_sum_via_monomials,
    compute_coeffs_poly_prod_via_monomials,
    get_grid_and_multi_index_poly_prod,
    get_grid_and_multi_index_poly_sum,
    PolyData,
    scalar_add_via_monomials,
    select_active_monomials,
)
from minterpy.utils.verification import dummy, verify_domain
from minterpy.services import is_scalar


__all__ = ["ChebyshevPolynomial"]


# --- Evaluation
def eval_chebyshev(
    chebyshev_polynomials: "ChebyshevPolynomial",
    xx: np.ndarray,
) -> np.ndarray:
    """Wrapper for the evaluation function in the Chebyshev bases.

    Parameters
    ----------
    chebyshev_polynomials : ChebyshevPolynomial
        The Chebyshev polynomial(s) to be evaluated.
    xx : np.ndarray
        The array of query points of shape ``(k, m)`` at which the monomials
        are evaluated. The values must be in :math:`[-1, 1]^m`.

    Notes
    -----
    - This function must have the specific signature to conform with the
      requirement of the abstract base class.
    - Multiple Chebyshev polynomials having the same set of exponents living
      on the same grid are defined by a multiple set of coefficients.

    .. todo::
        - Allows batch evaluations somewhere upstream.
        - make sure the input is in the domain [-1, 1]^m somewhere upstream.
    """
    # Get required data from the object
    exponents = chebyshev_polynomials.multi_index.exponents
    coefficients = chebyshev_polynomials.coeffs

    results = evaluate_polynomials(xx, exponents, coefficients)

    return results


# --- Arithmetics (Addition, Multiplication)
def add_chebyshev(
    poly_1: "ChebyshevPolynomial",
    poly_2: "ChebyshevPolynomial",
) -> "ChebyshevPolynomial":
    """Add two polynomial instances in the Chebyshev basis.

    This is the concrete implementation of ``_add()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract base class specifically for
    polynomials in the Chebyshev basis.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the addition expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the addition expression.

    Returns
    -------
    ChebyshevPolynomial
        The product of two polynomials in the Chebyshev basis as a new instance
        of polynomial in the Chebyshev basis.

    Notes
    -----
    - This function assumes: both polynomials must be in the Chebyshev basis,
      they must be initialized (coefficients are not ``None``),
      have the same dimension and their domains are matching,
      and the number of polynomials per instance are the same.
      These conditions are not explicitly checked in this function; the caller
      is responsible for the verification.
    """
    # --- Get the ingredients of a summed polynomial in the Chebyshev basis
    poly_data = _compute_data_poly_sum(poly_1, poly_2)

    # --- Return a new instance
    return ChebyshevPolynomial(**poly_data._asdict())


def mul_chebyshev(
    poly_1: "ChebyshevPolynomial",
    poly_2: "ChebyshevPolynomial",
) -> "ChebyshevPolynomial":
    """Multiply two polynomial instances in the Chebyshev basis.

    This is the concrete implementation of ``_mul()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    polynomials in the Chebyshev basis.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the multiplication expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the multiplication expression.

    Returns
    -------
    ChebyshevPolynomial
        The product of two polynomials in the Chebyshev basis as a new instance
        of polynomial in the Chebyshev basis.

    Notes
    -----
    - This function assumes: both polynomials must be in the Chebyshev basis,
      they must be initialized (coefficients are not ``None``),
      have the same dimension and their domains are matching,
      and the number of polynomials per instance are the same.
      These conditions are not explicitly checked in this function; the caller
      is responsible for the verification.
    """
    # --- Get the ingredients of the product polynomial in the Chebyshev basis
    poly_prod_data = _compute_data_poly_prod(poly_1, poly_2)

    # --- Return a new instance
    return ChebyshevPolynomial(**poly_prod_data._asdict())


class ChebyshevPolynomial(MultivariatePolynomialSingleABC):
    """Concrete implementation of polynomials in the Chebyshev bases."""
    # --- Virtual Functions

    # Evaluation
    _eval = staticmethod(eval_chebyshev)

    # Arithmetics (polynomial-polynomial)
    _add = staticmethod(add_chebyshev)
    _sub = staticmethod(dummy)
    _mul = staticmethod(mul_chebyshev)
    _div = staticmethod(dummy)  # type: ignore
    _pow = staticmethod(dummy)  # type: ignore

    # Arithmetics (polynomial-scalar)
    _scalar_add = staticmethod(scalar_add_via_monomials)

    # Calculus
    _partial_diff = staticmethod(dummy)  # type: ignore
    _diff = staticmethod(dummy)  # type: ignore
    _integrate_over = staticmethod(dummy)  # type: ignore

    # Domain generation
    generate_internal_domain = staticmethod(verify_domain)
    generate_user_domain = staticmethod(verify_domain)


# --- Internal utility functions
def _compute_data_poly_sum(
    poly_1: "ChebyshevPolynomial",
    poly_2: "ChebyshevPolynomial",
) -> PolyData:
    """Compute the data to create a summed polynomial in the Chebyshev basis.

    Addition or subtraction of polynomials in the Chebyshev basis is based
    on adding (resp. subtracting) the coefficients of the matching monomials
    of the two polynomial operands (i.e., the matching elements of the two
    multi-index sets). This procedure is the same as that of the canonical
    polynomial.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the addition/subtraction expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the addition/subtraction expression.

    Returns
    -------
    PolyData
        The ingredients to construct a summed polynomial in the Chebyshev
        basis.

    Notes
    -----
    - Both polynomials are assumed to have the same type, the same spatial
      dimension, and matching domains. These conditions have been made sure
      upstream.
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
        mi_sum,
        coeffs_sum,
        internal_domain_sum,
        user_domain_sum,
        grd_sum,
    )


def _compute_data_poly_prod(
    poly_1: ChebyshevPolynomial,
    poly_2: ChebyshevPolynomial,
) -> PolyData:
    """Compute the data to create a product polynomial in the Chebyshev basis.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the multiplication expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the multiplication expression.

    Returns
    -------
    PolyData
        A tuple with all the ingredients to construct a product polynomial
        in the Newton basis.

    Notes
    -----
    - Both polynomials are assumed to have the same spatial dimension and
      matching domains. These conditions have been made sure upstream.
    """
    # --- Get the grid and multi-index set of the summed polynomial
    grd_prod, mi_prod = get_grid_and_multi_index_poly_prod(poly_1, poly_2)

    # --- Process the coefficients
    # NOTE: indices may or may not be separate, use the summed multi-index set
    #       instead of the one attached to grid
    coeffs_prod = _compute_coeffs_poly_prod(poly_1, poly_2, grd_prod, mi_prod)

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


def _compute_coeffs_poly_prod(
    poly_1: ChebyshevPolynomial,
    poly_2: ChebyshevPolynomial,
    grid_prod: Grid,
    multi_index_prod: MultiIndexSet,
) -> np.ndarray:
    """Compute the coefficients of the product polynomial in Chebyshev basis.

    In general, the coefficients of a product polynomial in the Chebyshev basis
    are obtained by going through the Lagrange basis first.
    Specifically, the Lagrange coefficients are computed by multiplying
    the evaluation results of the Chebyshev polynomial operands on the product
    Grid. Afterward, these coefficients are transformed to the Chebyshev
    coefficients.

    This is because the multiplication of two Chebyshev monomial does not
    return the monomial of a higher degree. For instance, the Chebyshev
    monomial of degree :math:`3` is not the result of multiplying Chebyshev
    monomials of degree :math:`1` and :math:`2`.

    However, if one of the polynomial operands has a scalar multi-index set
    regardless of the grid, then the coefficients is obtained by multiplying
    the coefficients of the non-scalar polynomial with the coefficient of
    the scalar polynomial.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the multiplication expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the multiplication expression.
    grid_prod : Grid
        The grid of the product polynomial.
    multi_index_prod : MultiIndexSet
        The multi-index of the product polynomial.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The coefficients of the product between two polynomials.
    """
    # --- Handle the case where no transformation is required
    # If one of the operands has a scalar multi-index set
    if is_scalar(poly_1.multi_index) or is_scalar(poly_2.multi_index):
        return compute_coeffs_poly_prod_via_monomials(
            poly_1,
            poly_2,
            multi_index_prod,
        )

    return _compute_coeffs_poly_prod_via_lagrange(
        poly_1,
        poly_2,
        grid_prod,
        multi_index_prod,
    )


def _compute_coeffs_poly_prod_via_lagrange(
    poly_1: ChebyshevPolynomial,
    poly_2: ChebyshevPolynomial,
    grid_prod: Grid,
    multi_index_prod: MultiIndexSet,
) -> np.ndarray:
    """Compute the coefficients of a product Chebyshev polynomial via Lagrange.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the multiplication expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the multiplication expression.
    grid_prod : Grid
        The Grid associated with the product polynomial.
    multi_index_prod : MultiIndexSet
        The multi-index set of the product polynomial.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The coefficients of the product polynomial in the Chebyshev basis.

    Notes
    -----
    - Both polynomials are assumed to have the same spatial dimension and
      matching domains. These conditions have been made sure upstream.
    """
    # Compute the values of the operands at the unisolvent nodes
    lag_coeffs_1 = grid_prod(poly_1)
    lag_coeffs_2 = grid_prod(poly_2)
    lag_coeffs_prod = lag_coeffs_1 * lag_coeffs_2

    # Compute the Chebyshev monomials at the unisolvent nodes
    cheb2lag = evaluate_monomials(
        grid_prod.unisolvent_nodes,
        grid_prod.multi_index.exponents,
    )

    # Compute the inverse transformation
    cheb_coeffs_prod = np.linalg.solve(cheb2lag, lag_coeffs_prod)

    # Deal with separate indices, select only w.r.t the active monomials
    if poly_1.indices_are_separate or poly_2.indices_are_separate:
        cheb_coeffs_prod = select_active_monomials(
            cheb_coeffs_prod,
            grid_prod,
            multi_index_prod,
        )

    return cheb_coeffs_prod
