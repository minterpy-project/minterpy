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
    compute_poly_sum_coeffs,
)
from minterpy.utils.polynomials.interface import (
    get_grid_and_multi_index_poly_prod,
    get_grid_and_multi_index_poly_sum,
    PolyData,
    scalar_add_monomial_based,
    select_active_monomials,
    shape_coeffs,
)
from minterpy.jit_compiled.canonical import (
    compute_coeffs_poly_prod as compute_coeffs_poly_prod_canonical,
)
from minterpy.utils.verification import dummy, verify_domain
from minterpy.services import is_scalar


__all__ = ["ChebyshevPolynomial"]


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
        The sum of two polynomials in the Chebyshev basis; a new instance
        of polynomial.

    Notes
    -----
    - This function assumes: both polynomials must be in the Chebyshev basis,
      they must be initialized, have the same dimension, their domains are
      matching, and their length must be the same. These conditions are not
      checked explicitly in this function.
    """
    # --- Get the ingredients of a summed polynomial in the Chebyshev basis
    poly_data = _compute_poly_sum_data_chebyshev(poly_1, poly_2)

    # --- Return a new instance
    return ChebyshevPolynomial(**poly_data._asdict())


def mul_chebyshev(
    poly_1: "ChebyshevPolynomial",
    poly_2: "ChebyshevPolynomial",
) -> "ChebyshevPolynomial":
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
    - This function assumes: both polynomials must be in canonical basis,
      they must be initialized, have the same dimension and their domains
      are matching, and the number of polynomials per instance are the same.
      These conditions are not explicitly checked in this function.
    """
    # --- Get the ingredients of the product polynomial in the canonical basis
    poly_prod_data = _compute_poly_prod_data_chebyshev(poly_1, poly_2)

    # --- Return a new instance
    return ChebyshevPolynomial(**poly_prod_data._asdict())


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
    _scalar_add = staticmethod(scalar_add_monomial_based)

    # Calculus
    _partial_diff = staticmethod(dummy)  # type: ignore
    _diff = staticmethod(dummy)  # type: ignore
    _integrate_over = staticmethod(dummy)  # type: ignore

    # Domain generation
    generate_internal_domain = staticmethod(verify_domain)
    generate_user_domain = staticmethod(verify_domain)


# --- Internal utility functions
def _compute_poly_sum_data_chebyshev(
    poly_1: "ChebyshevPolynomial",
    poly_2: "ChebyshevPolynomial",
) -> PolyData:
    """Compute the data to create a summed polynomial in the Chebyshev basis.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the addition expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the addition expression.

    Returns
    -------
    PolyData
        A tuple with the main ingredients to construct a summed polynomial
        in the Chebyshev basis.

    Notes
    -----
    - Both polynomials are assumed to have the same dimension
      and matching domains.
    """
    # --- Get the grid and multi-index set of the summed polynomial
    grd_sum, mi_sum = get_grid_and_multi_index_poly_sum(poly_1, poly_2)

    # --- Process the coefficients
    # Shape the coefficients; ensure they have the same dimension
    coeffs_1, coeffs_2 = shape_coeffs(poly_1, poly_2)

    # Compute the coefficients of the summed polynomial
    coeffs_sum = compute_poly_sum_coeffs(
        poly_1.multi_index.exponents,
        coeffs_1,
        poly_2.multi_index.exponents,
        coeffs_2,
        mi_sum.exponents,
    )

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


def _compute_poly_prod_data_chebyshev(
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
    - Both polynomials are assumed to have the same dimension
      and matching domains.
    """
    # --- Get the grid and multi-index set of the summed polynomial
    grd_prod, mi_prod = get_grid_and_multi_index_poly_prod(poly_1, poly_2)

    # --- Process the coefficients
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
        The coefficients of the product polynomial between two polynomials.
    """
    # --- Handle case of scalar polynomial (no transformation required)
    # NOTE: Whether the underlying grids are scalar is irrelevant
    is_scalar_poly = is_scalar(poly_1) or is_scalar(poly_2)
    if is_scalar_poly:
        return _compute_coeffs_scalar_poly_prod(
            poly_1,
            poly_2,
            multi_index_prod,
        )

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


def _compute_coeffs_scalar_poly_prod(
    poly_1: ChebyshevPolynomial,
    poly_2: ChebyshevPolynomial,
    multi_index_prod: MultiIndexSet,
) -> np.ndarray:
    """Compute the coefficients of a summed constant polynomial.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the addition/subtraction expression.
    poly_2 : ChebyshevPolynomial
        Right operand of the addition/subtraction expression.
    multi_index_prod : MultiIndexSet
        The multi-index set of the product polynomial.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The coefficients of the summed polynomial in the Newton basis.

    Notes
    -----
    - The function is used when at least one of ``poly_1`` and ``poly_2`` is
      a constant polynomial.
    - For addition/subtraction involving a constant polynomial, the procedure
      to compute the coefficients of a polynomial sum in the canonical basis
      is used, i.e., find matching index term and add the coefficients and
      therefore, avoid the use of transformation (a special case).
    """
    # Shape the coefficients; ensure they have the same dimension
    coeffs_1, coeffs_2 = shape_coeffs(poly_1, poly_2)

    # Pre-allocate output array placeholder
    num_monomials = len(multi_index_prod)
    num_polys = len(poly_1)
    coeffs_prod = np.zeros((num_monomials, num_polys))

    # Compute the coefficients (use pre-allocated placeholder as output)
    # NOTE: indices may or may not be separate,
    # use the multi-index instead of the one attached to grid
    exponents_1 = poly_1.multi_index.exponents
    exponents_2 = poly_2.multi_index.exponents
    exponents_prod = multi_index_prod.exponents
    compute_coeffs_poly_prod_canonical(
        exponents_1,
        coeffs_1,
        exponents_2,
        coeffs_2,
        exponents_prod,
        coeffs_prod,
    )

    return coeffs_prod
