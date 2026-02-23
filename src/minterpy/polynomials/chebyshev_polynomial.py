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
from minterpy.polynomials.arithmetic import (
    get_compute_coeffs_add,
    get_compute_coeffs_mul,
    select_active_monomials,
)
from minterpy.polynomials.scalar_add import scalar_add_via_monomials
from minterpy.utils.polynomials.chebyshev import (
    evaluate_monomials,
    evaluate_polynomials,
)
from minterpy.utils.verification import dummy
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
    multi_index: MultiIndexSet,
    grid: Grid,
) -> "ChebyshevPolynomial":
    """Add two polynomial instances in the Chebyshev basis.

    This is the concrete implementation of ``_add()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    polynomial in the Chebyshev basis.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the addition.
    poly_2 : ChebyshevPolynomial
        Right operand of the addition.
    multi_index : MultiIndexSet
        The multi-index set of the resulting polynomial, i.e., the union
        of the multi-index sets of the operands.
    grid : Grid
        The grid of the resulting polynomial, i.e., the union of the grids
        of the two operands.

    Returns
    -------
    ChebyshevPolynomial
        The sum of two polynomials in the Chebyshev basis as a new instance
        of polynomial.

    Notes
    -----
    - The Chebyshev basis is closed under addition so the coefficients of
      the resulting polynomial can be summed up via the monomials rule.
    - This function assumes the caller has verified: both polynomials are in
      the Chebyshev basis, both are initialized, their domains match, and they
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
    return ChebyshevPolynomial(multi_index, coeffs, grid)


def mul_chebyshev(
    poly_1: "ChebyshevPolynomial",
    poly_2: "ChebyshevPolynomial",
    multi_index: MultiIndexSet,
    grid: Grid,
) -> "ChebyshevPolynomial":
    """Multiply two polynomial instances in the Chebyshev basis.

    This is the concrete implementation of ``_mul()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    polynomials in the Chebyshev basis.

    Parameters
    ----------
    poly_1 : ChebyshevPolynomial
        Left operand of the multiplication.
    poly_2 : ChebyshevPolynomial
        Right operand of the multiplication.
    multi_index : MultiIndexSet
        The multi-index set of the resulting polynomial, i.e., the product
        of the multi-index sets of the operands.
    grid : Grid
        The grid of the resulting polynomial, i.e., the product of the grids
        of the two operands.

    Returns
    -------
    ChebyshevPolynomial
        The product of two polynomials in the Chebyshev basis as a new instance
        of polynomial.

    Notes
    -----
    - The Chebyshev basis is in general not closed under multiplication,
      so the coefficients of the resulting polynomial must be computed via
      the transformation from the Lagrange coefficients.
    - If one operand has a scalar multi-index, monomials multiplication
      is used instead.
    - This function assumes the caller has verified: both polynomials are in
      the Chebyshev basis, both are initialized, their domains match, and they
      have the same number of polynomial instances.
    - The provided ``multi_index`` may differ from ``grid.multi_index`` when
      the polynomial multi-index set is a subset of the grid multi-index set
      (i.e., separated indices). The coefficients are computed
      with respect to the given ``multi_index``.
    """
    # Handle the case where no transformation is required
    if is_scalar(poly_1.multi_index) or is_scalar(poly_2.multi_index):
        # If one of the operands has a scalar multi-index set
        # (regardless of the grid), compute the coefficients via monomials
        compute_coeffs = get_compute_coeffs_mul("monomials")
        coeffs = compute_coeffs(poly_1, poly_2, multi_index)
    else:
        # Compute the coefficients via a transformation from Lagrange coeffs.
        compute_coeffs = get_compute_coeffs_mul("lagrange")
        coeffs_lag = compute_coeffs(poly_1, poly_2, grid)
        coeffs = _transform_lag2cheb(coeffs_lag, multi_index, grid)

    # Create and return a new instance of polynomial
    return ChebyshevPolynomial(multi_index, coeffs, grid)


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
    _diff = staticmethod(dummy)  # type: ignore
    _integrate_over = staticmethod(dummy)  # type: ignore


# --- Internal utility functions
def _transform_lag2cheb(
    coeffs_lag: np.ndarray,
    multi_index: MultiIndexSet,
    grid: Grid,
) -> np.ndarray:
    """Transform Lagrange coefficients to Chebyshev coefficients.

    Given polynomial coefficients in the Lagrange basis, compute the equivalent
    coefficients in the Chebyshev basis by solving a linear system based on
    evaluating Chebyshev monomials at the unisolvent nodes.

    Parameters
    ----------
    coeffs_lag : np.ndarray
        Coefficients in the Lagrange basis. Shape is ``(N, K)`` where ``N`` is
        the number of unisolvent nodes (equals ``len(grid.multi_index)``) and
        ``K`` is the number of polynomial instances.
    multi_index : MultiIndexSet
        The multi-index set of the target polynomial. In case of separated
        indices cases, it may be a subset of ``grid.multi_index``.
    grid : Grid
        The grid on which the Lagrange polynomial lives.

    Returns
    -------
    np.ndarray
        Coefficients in the Chebyshev basis. Shape is ``(N, K)`` where ``N``
        equals ``len(multi_index)`` and ``K`` is the number of polynomial
        instances.

    Notes
    -----
    - The transformation solves the linear system ``A @ c_cheb = c_lag`` where
      ``A`` is the Chebyshev-to-Lagrange transformation matrix formed by
      evaluating Chebyshev monomials at the unisolvent nodes.
    """
    # Compute the Chebyshev-to-Lagrange transformation matrix
    # (Chebyshev monomials evaluated at unisolvent nodes)
    cheb2lag = evaluate_monomials(
        grid.unisolvent_nodes,
        grid.multi_index.exponents,
    )

    # Solve for Chebyshev coefficients: cheb2lag @ cheb_coeffs = lag_coeffs
    coeffs_cheb = np.linalg.solve(cheb2lag, coeffs_lag)

    # Handle separated indices: select only coefficients for active monomials
    if multi_index != grid.multi_index:
        coeffs_cheb = select_active_monomials(coeffs_cheb, grid, multi_index)

    return coeffs_cheb
