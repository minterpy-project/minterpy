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


def scalar_add_monomial_based(
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
