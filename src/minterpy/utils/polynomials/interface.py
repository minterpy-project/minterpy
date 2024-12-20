"""
This module contains functions that bridge between the upper layer of
abstraction (``NewtonPolynomial``, ``LagrangePolynomial``, etc.) to the
lower layer of abstraction (numerical routines that operates on arrays) that
typically resides in the ``minterpy.utils`` or ``minterpy.jit_compiled``.

The idea behind this module is to minimize the detail of computations
inside the concrete polynomial modules.
"""
import numpy as np

from typing import NamedTuple, Tuple, Union

from minterpy.global_settings import SCALAR
from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.core import Grid, MultiIndexSet
from minterpy.utils.multi_index import find_match_between
from minterpy.jit_compiled.canonical import compute_coeffs_poly_prod


class PolyData(NamedTuple):
    """Container for complete inputs to create a polynomial in any basis."""
    multi_index: MultiIndexSet
    coeffs: np.ndarray
    internal_domain: np.ndarray
    user_domain: np.ndarray
    grid: Grid


def shape_coeffs(
    poly_1: MultivariatePolynomialSingleABC,
    poly_2: MultivariatePolynomialSingleABC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shape the polynomial coefficients before carrying out binary operations.

    Parameters
    ----------
    poly_1 : MultivariatePolynomialSingleABC
        The first operand in a binary polynomial expression.
    poly_2 : MultivariatePolynomialSingleABC
        The second operand in a binary polynomial expression.

    Returns
    -------
    Tuple[:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`]
        A tuple of polynomial coefficients, the first and second operands,
        respectively. Both are two-dimensional arrays with the length of
        the polynomials as the number of columns.

    Notes
    -----
    - Relevant binary expressions include subtraction, addition,
      and multiplication with polynomials as both operands.
    """
    assert len(poly_1) == len(poly_2)

    num_poly = len(poly_1)
    if num_poly > 1:
        return poly_1.coeffs, poly_2.coeffs

    coeffs_1 = poly_1.coeffs[:, np.newaxis]
    coeffs_2 = poly_2.coeffs[:, np.newaxis]

    return coeffs_1, coeffs_2


def get_grid_and_multi_index_poly_sum(
    poly_1: MultivariatePolynomialSingleABC,
    poly_2: MultivariatePolynomialSingleABC,
) -> Tuple[Grid, MultiIndexSet]:
    """Get the grid and multi-index set of a summed polynomial.

    Parameters
    ----------
    poly_1 : MultivariatePolynomialSingleABC
        The first operand in the addition expression.
    poly_2 : MultivariatePolynomialSingleABC
        The second operand in the addition expression.

    Returns
    -------
    Tuple[Grid, MultiIndexSet]
        The instances of `Grid` and `MultiIndexSet` of the summed polynomial.
    """
    # --- Compute the union of the grid instances
    grd_sum = poly_1.grid | poly_2.grid

    # --- Compute union of the multi-index sets if they are separate
    if poly_1.indices_are_separate or poly_2.indices_are_separate:
        mi_sum = poly_1.multi_index | poly_2.multi_index
    else:
        # Otherwise use the one attached to the grid instance
        mi_sum = grd_sum.multi_index

    return grd_sum, mi_sum


def get_grid_and_multi_index_poly_prod(
    poly_1: MultivariatePolynomialSingleABC,
    poly_2: MultivariatePolynomialSingleABC,
) -> Tuple[Grid, MultiIndexSet]:
    """Get the grid and multi-index set of a product polynomial.

    Parameters
    ----------
    poly_1 : MultivariatePolynomialSingleABC
        The first operand in the addition expression.
    poly_2 : MultivariatePolynomialSingleABC
        The second operand in the addition expression.

    Returns
    -------
    Tuple[Grid, MultiIndexSet]
        The instances of `Grid` and `MultiIndexSet` of the product polynomial.
    """
    # --- Compute the union of the grid instances
    grd_prod = poly_1.grid * poly_2.grid

    # --- Compute union of the multi-index sets if they are separate
    if poly_1.indices_are_separate or poly_2.indices_are_separate:
        mi_prod = poly_1.multi_index * poly_2.multi_index
    else:
        # Otherwise use the one attached to the grid instance
        mi_prod = grd_prod.multi_index

    return grd_prod, mi_prod


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
    - ``active_multi_index`` must be a subset of ``grid.multi_index``.
    """
    exponents_multi_index = active_multi_index.exponents
    exponents_grid = grid.multi_index.exponents
    active_idx = find_match_between(exponents_multi_index, exponents_grid)

    return coeffs[active_idx]


def scalar_add_via_monomials(
    poly: MultivariatePolynomialSingleABC,
    scalar: SCALAR,
) -> MultivariatePolynomialSingleABC:
    """Add an instance of polynomial with a real scalar based on the monomial.

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
      ``LagrangePolynomial`` does not.
    """
    # Create a constant polynomial
    poly_scalar = _create_scalar_poly(poly, scalar)

    # Rely on the `__add__()` method implemented upstream
    return poly + poly_scalar


def compute_coeffs_poly_sum_via_monomials(
    poly_1: MultivariatePolynomialSingleABC,
    poly_2: MultivariatePolynomialSingleABC,
    multi_index_sum: MultiIndexSet,
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
        Left operand of the polynomial-polynomial addition expression.
    poly_2 : MultivariatePolynomialSingleABC
        Right operand of the polynomial-polynomial addition expression.
    multi_index_sum : MultiIndexSet
        The multi-index set of the summed polynomial.

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
    coeffs_1, coeffs_2 = shape_coeffs(poly_1, poly_2)

    # Get the exponents
    exponents_1 = poly_1.multi_index.exponents
    exponents_2 = poly_2.multi_index.exponents
    exponents_sum = multi_index_sum.exponents

    # Create the output array
    num_monomials = len(multi_index_sum)
    num_polynomials = len(poly_1)
    coeffs_poly_sum = np.zeros((num_monomials, num_polynomials))

    # Get the matching indices
    idx_1 = find_match_between(exponents_1, exponents_sum)
    idx_2 = find_match_between(exponents_2, exponents_sum)

    coeffs_poly_sum[idx_1, :] += coeffs_1[:, :]
    coeffs_poly_sum[idx_2, :] += coeffs_2[:, :]

    return coeffs_poly_sum


def compute_coeffs_poly_prod_via_monomials(
    poly_1: MultivariatePolynomialSingleABC,
    poly_2: MultivariatePolynomialSingleABC,
    multi_index_prod: MultiIndexSet,
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
    multi_index_prod : MultiIndexSet
        The multi-index set of the product polynomial.

    Notes
    -----
    - ``multi_index_prod`` is assumed to be the result of multiplying
      ``poly_1.multi_index`` and ``poly_2.multi_index``.
    - The lengths of ``poly_1`` and ``poly_2`` are assumed to be the same.
    - The function does not check whether the above assumptions are fulfilled;
      the caller is responsible to make sure of that. If the assumptions are
      not fulfilled, the function may not raise any exception but produce
      the wrong results.
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
    compute_coeffs_poly_prod(
        exponents_1,
        coeffs_1,
        exponents_2,
        coeffs_2,
        exponents_prod,
        coeffs_prod,
    )

    return coeffs_prod


def _create_scalar_poly(
    poly: MultivariatePolynomialSingleABC,
    scalar: Union[SCALAR, np.ndarray],
) -> MultivariatePolynomialSingleABC:
    """Create a constant scalar polynomial from a given polynomial.

    Parameters
    ----------
    poly : MultivariatePolynomialSingleABC
        An instance of polynomial from which a constant polynomial will be
        created.
    scalar : Union[SCALAR, np.ndarray]
        Real numbers for the coefficient value of the constant polynomial.
        Multiple real numbers (as an array) indicates multiple set of
        coefficients.

    Returns
    -------
    MultivariatePolynomialSingleABC
        A polynomial of the same instance as ``poly`` having the same grid and
        domains but with a single element multi-index set. If the grid does
        not include the element (0, ..., 0) then the grid will be extended.
    """
    # Create a single-element multi-index set of (0, ..., 0)
    dim = poly.spatial_dimension
    lp_degree = poly.multi_index.lp_degree
    mi = MultiIndexSet.from_degree(dim, poly_degree=0, lp_degree=lp_degree)

    # Create a Grid
    # The grid of the polynomial may not include the multi-index set element
    # (0, ..., 0) (i.e., it's non-downward-closed) so create a new one.
    grd = Grid(mi, poly.grid.generating_function, poly.grid.generating_points)

    # Create the coefficient
    if len(poly) == 1:
        coeffs = np.array([scalar])
    else:
        coeffs = scalar * np.ones(
            shape=(1, len(poly)),
            dtype=poly.coeffs.dtype,
        )

    # Return a polynomial instance of the same class as input
    return poly.__class__(
        multi_index=mi,
        coeffs=coeffs,
        internal_domain=poly.internal_domain,
        user_domain=poly.user_domain,
        grid=grd,
    )
