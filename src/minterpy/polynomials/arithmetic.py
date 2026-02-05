"""
Implementations of arithmetic operations on polynomials at coefficients level.
"""
import numpy as np

from typing import Callable, Literal, Tuple

from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.core.multi_index import MultiIndexSet
from minterpy.core.grid import Grid
from minterpy.jit_compiled.canonical import compute_coeffs_poly_prod
from minterpy.utils.multi_index import find_match_between


def select_active_monomials(
    coeffs: np.ndarray,
    grid: Grid,
    active_multi_index: MultiIndexSet,
) -> np.ndarray:
    """Get the coefficients that corresponds to the active monomials.

    Parameters
    ----------
    coeffs : :class:`numpy:numpy.ndarray`
        The coefficients of a polynomial associated with the multi-index set
        of the grid on which the polynomial lives. They are stored in an array
        whose length is the same as the length of ``grid.multi_index``.
    grid : Grid
        The grid on which the polynomial lives.
    active_multi_index : MultiIndexSet
        The multi-index set of active monomials; the coefficients will be
        picked according to this multi-index set.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The coefficients of a polynomial associated with the active monomials
        as specified by ``multi_index``.

    Notes
    -----
    - ``active_multi_index`` must be a subset of ``grid.multi_index``. This
       precondition is assumed to be fulfilled by construction upstream.
    """
    exponents_multi_index = active_multi_index.exponents
    exponents_grid = grid.multi_index.exponents
    active_idx = find_match_between(exponents_multi_index, exponents_grid)

    return coeffs[active_idx]

def get_compute_coeffs_add(mode: Literal["monomials", "lagrange"]) -> Callable:
    """Get the function to compute the coefficients of a polynomial sum.

    Parameters
    ----------
    mode : str
        The mode of the coefficients computation.

    Returns
    -------
    Callable
        The function to compute the coefficients of a polynomial sum.
    """
    if mode == "monomials":
        return _compute_coeffs_poly_add_via_monomials

    return _compute_coeffs_poly_add_via_lagrange


def get_compute_coeffs_mul(mode: Literal["monomials", "lagrange"]) -> Callable:
    """Get the function to compute the coefficients of a polynomial product.

    Parameters
    ----------
    mode : str
        The mode of the coefficients computation.

    Returns
    -------
    Callable
        The function to compute the coefficients of a polynomial  product.
    """
    if mode == "monomials":
        return _compute_coeffs_poly_mul_via_monomials

    return _compute_coeffs_poly_mul_via_lagrange


def _compute_coeffs_poly_add_via_monomials(
    poly_1: MultivariatePolynomialSingleABC,
    poly_2: MultivariatePolynomialSingleABC,
    multi_index_add: MultiIndexSet,
) -> np.ndarray:
    r"""Compute the coefficients of a summed polynomial via the monomials.

    For example, suppose: :math:`A = \{ (0, 0) , (1, 0), (0, 1) \}` with
    coefficients :math:`c_A = (1.0 , 2.0, 3.0)` is summed with
    :math:`B = \{ (0, 0), (1, 0), (2, 0) \}` with coefficients
    :math:`c_B = (1.0, 5.0, 3.0)`. The union/sum multi-index set is
    :math:`A \times B = \{ (0, 0), (1, 0), (2, 0), (0, 1) \}`.

    The corresponding coefficients of the sum are:

    - :math:`(0, 0)` appears in both operands, so the coefficient
      is :math:`1.0 + 1.0 = 2.0`
    - :math:`(1, 0)` appears in both operands, so the coefficient is
      :math:`2.0 + 5.0 = 7.0`
    - :math:`(2, 0)` only appears in the second operand, so the coefficient
      is :math:`3.0`
    - :math:`(0, 1)` only appears in the first operand, so the coefficient
      is :math:`3.0`

    or :math:`c_{A | B} = (2.0, 7.0, 3.0, 3.0)`.

    Parameters
    ----------
    poly_1 : MultivariatePolynomialSingleABC
        Left operand of the polynomial-polynomial addition.
    poly_2 : MultivariatePolynomialSingleABC
        Right operand of the polynomial-polynomial addition.
    multi_index_add : MultiIndexSet
        The multi-index set of the summed polynomial, i.e., the union
        of the multi-index sets of the two operands.

    Notes
    -----
    - ``multi_index_sum`` is assumed to be the result of unionizing
      ``poly_1.multi_index`` and ``poly_2.multi_index``.
    - The lengths of ``poly_1`` and ``poly_2`` are assumed to be the same.
    - The function does not check whether the above assumptions are fulfilled;
      the caller is responsible to make sure of that. If the assumptions are
      not fulfilled, the function may not raise any exception but produce
      the wrong results.
    """
    # Shape the coefficients; ensure they have the same dimension
    coeffs_1, coeffs_2 = _shape_coeffs(poly_1.coeffs, poly_2.coeffs)

    # Get the exponents
    mi_1 = _match_mi_dim(poly_1.multi_index, multi_index_add)
    mi_2 = _match_mi_dim(poly_2.multi_index, multi_index_add)

    exponents_1 = mi_1.exponents
    exponents_2 = mi_2.exponents
    exponents_sum = multi_index_add.exponents

    # Create the output array
    num_monomials = len(multi_index_add)
    num_polynomials = len(poly_1)
    coeffs_poly_sum = np.zeros((num_monomials, num_polynomials))

    # Get the matching indices
    idx_1 = find_match_between(exponents_1, exponents_sum)
    idx_2 = find_match_between(exponents_2, exponents_sum)

    coeffs_poly_sum[idx_1, :] += coeffs_1[:, :]
    coeffs_poly_sum[idx_2, :] += coeffs_2[:, :]

    return coeffs_poly_sum


def _compute_coeffs_poly_add_via_lagrange(
    poly_1: MultivariatePolynomialSingleABC,
    poly_2: MultivariatePolynomialSingleABC,
    grid_add: Grid,
) -> np.ndarray:
    """Compute the coefficients of a summed polynomial via Lagrange polynomial.

    Parameters
    ----------
    poly_1 : MultivariatePolynomialSingleABC
        Left operand of the addition/subtraction expression.
    poly_2 : MultivariatePolynomialSingleABC
        Right operand of the addition/subtraction expression.
    grid_add : Grid
        The Grid associated with the summed polynomial.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The coefficients of the summed polynomial in the Newton basis.
    """
    # Compute the values of the operands at the unisolvent nodes
    # NOTE: The grid may be of higher dimension than one of the polynomials;
    #       evaluation must ignore the extra dimensions
    lag_coeffs_1 = grid_add(poly_1, truncate_cols=True)
    lag_coeffs_2 = grid_add(poly_2, truncate_cols=True)
    lag_coeffs_add = lag_coeffs_1 + lag_coeffs_2

    return lag_coeffs_add


def _compute_coeffs_poly_mul_via_monomials(
    poly_1: MultivariatePolynomialSingleABC,
    poly_2: MultivariatePolynomialSingleABC,
    multi_index_mul: MultiIndexSet,
) -> np.ndarray:
    r"""Compute the coefficients of a product polynomial via the monomials.

    For example, suppose: :math:`A = \{ (0, 0) , (1, 0), (0, 1) \}` with
    coefficients :math:`c_A = (1.0 , 2.0, 3.0)` is multiplied with
    :math:`B = \{ (0, 0) , (1, 0) \}` with coefficients
    :math:`c_B = (1.0 , 5.0)`. The product multi-index set is
    :math:`A \times B = \{ (0, 0) , (1, 0), (2, 0), (0, 1), (1, 1) \}`.

    The corresponding coefficients of the product are:

    - :math:`(0, 0)` is coming from :math:`(0, 0) + (0, 0)`, the coefficient
      is :math:`1.0 \times 1.0 = 1.0`
    - :math:`(1, 0)` is coming from :math:`(0, 0) + (1, 0)` and
      :math:`(1, 0) + (0, 0)`, the coefficient is
      :math:`1.0 \times 5.0 + 2.0 \times 1.0 = 7.0`
    - :math:`(2, 0)` is coming from :math:`(1, 0) + (1, 0)`, the coefficient
      is :math:`2.0 \times 5.0 = 10.0`
    - :math:`(0, 1)` is coming from :math:`(0, 1) + (0, 0)`, the coefficient
      is :math:`3.0 \times 1.0 = 3.0`
    - :math:`(1, 1)` is coming from :math:`(0, 1) + (1, 0)`, the coefficient
      is :math:`3.0 \times 5.0 = 15.0`

    or :math:`c_{A \times B} = (1.0, 7.0, 10.0, 3.0, 15.0)`.

    Parameters
    ----------
    poly_1 : MultivariatePolynomialSingleABC
        Left operand of the polynomial-polynomial multiplication expression.
    poly_2 : MultivariatePolynomialSingleABC
        Right operand of the polynomial-polynomial multiplication expression.
    multi_index_mul : MultiIndexSet
        The multi-index set of the product polynomial, i.e., the product of the
        multi-index sets of the two operands.

    Notes
    -----
    - ``multi_index_mul`` is assumed to be the result of multiplying
      ``poly_1.multi_index`` and ``poly_2.multi_index``.
    - The lengths of ``poly_1`` and ``poly_2`` are assumed to be the same.
    - The function does not check whether the above assumptions are fulfilled;
      the caller is responsible to make sure of that. If the assumptions are
      not fulfilled, the function may not raise any exception but produce
      the wrong results.
    """
    # Shape the coefficients; ensure they have the same dimension
    coeffs_1, coeffs_2 = _shape_coeffs(poly_1.coeffs, poly_2.coeffs)

    # Pre-allocate output array placeholder
    num_monomials = len(multi_index_mul)
    num_polys = len(poly_1)
    coeffs_mul = np.zeros((num_monomials, num_polys))

    # Compute the coefficients (use pre-allocated placeholder as output)
    # NOTE: Handle separated indices case, i.e., use the provided multi_index
    #       rather than grid.multi_index
    mi_1 = _match_mi_dim(poly_1.multi_index, multi_index_mul)
    mi_2 = _match_mi_dim(poly_2.multi_index, multi_index_mul)

    exponents_1 = mi_1.exponents
    exponents_2 = mi_2.exponents
    exponents_mul = multi_index_mul.exponents
    compute_coeffs_poly_prod(
        exponents_1,
        coeffs_1,
        exponents_2,
        coeffs_2,
        exponents_mul,
        coeffs_mul,
    )

    return coeffs_mul


def _compute_coeffs_poly_mul_via_lagrange(
    poly_1: MultivariatePolynomialSingleABC,
    poly_2: MultivariatePolynomialSingleABC,
    grid_mul: Grid,
) -> np.ndarray:
    """Compute the coefficients of a product Newton polynomial via Lagrange.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the multiplication.
    poly_2 : NewtonPolynomial
        Right operand of the multiplication.
    grid_mul : Grid
        The Grid associated with the product polynomial, i.e., the product of
        the grids of the two operands.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The coefficients of the product polynomial in the Newton basis.

    Notes
    -----
    - Both polynomials are assumed to have the same spatial dimension and
      matching domains. These conditions have been made sure upstream.
    """
    # Compute the values of the operands at the unisolvent nodes
    # NOTE: The grid may be of higher dimension than one of the polynomials;
    #       evaluation must ignore the extra dimensions
    lag_coeffs_1 = grid_mul(poly_1, truncate_cols=True)
    lag_coeffs_2 = grid_mul(poly_2, truncate_cols=True)
    lag_coeffs_prod = lag_coeffs_1 * lag_coeffs_2

    return lag_coeffs_prod


def _shape_coeffs(
    coeffs_1: np.ndarray,
    coeffs_2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shape the coefficients of two polynomials to be two-dimensional.

    Parameters
    ----------
    coeffs_1 : np.ndarray
        The coefficients of the first polynomial.
    coeffs_2 : np.ndarray
        The coefficients of the second polynomial.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The coefficients of the two polynomials, shaped to be two-dimensional.

    Notes
    -----
    - A polynomial with a single set of coefficients is assumed to be
      one-dimensional, but certain operations must be performed on
      two-dimensional arrays. For example, a 1D array of shape ``(N,)``
      is reshaped to ``(N, 1)`` where ``N`` is the number of coefficients.
    """
    if coeffs_1.ndim == 1:
        coeffs_1 = coeffs_1[:, np.newaxis]

    if coeffs_2.ndim == 1:
        coeffs_2 = coeffs_2[:, np.newaxis]

    return coeffs_1, coeffs_2


def _match_mi_dim(mi_1: MultiIndexSet, mi_2: MultiIndexSet) -> MultiIndexSet:
    """Expand the first multi-index set to match the dimension of the second.

    Parameters
    ----------
    mi_1 : MultiIndexSet
        The multi-index set to expand.
    mi_2 : MultiIndexSet
        The target multi-index set whose dimension to match.

    Returns
    -------
    MultiIndexSet
        ``mi_1`` expanded to match ``mi_2.spatial_dimension``, or ``mi_1``
        unchanged if dimensions already match.

    Notes
    -----
    - Assumes ``mi_1.spatial_dimension <= mi_2.spatial_dimension``.
    """
    if mi_1.spatial_dimension < mi_2.spatial_dimension:
        return mi_1.expand_dim(mi_2.spatial_dimension)

    return mi_1
