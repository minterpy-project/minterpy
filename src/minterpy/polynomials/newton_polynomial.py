"""
Module that contains the implementation of Newton polynomial basis.
"""
from __future__ import annotations

import numpy as np

from minterpy.global_settings import DEBUG
from minterpy.core.ABC.multivariate_polynomial_abstract import (
    MultivariatePolynomialSingleABC,
)
from minterpy.core import Grid, MultiIndexSet
from minterpy.dds import dds
from minterpy.utils.verification import dummy, verify_domain
from minterpy.utils.polynomials.newton import (
    eval_newton_polynomials,
    deriv_newt_eval as eval_diff_numpy,
    integrate_monomials_newton,
)
from minterpy.jit_compiled.newton.diff import (
    eval_multiple_query as eval_diff_numba,
    eval_multiple_query_par as eval_diff_numba_par,
)
from minterpy.utils.polynomials.interface import (
    get_grid_and_multi_index_poly_prod,
    get_grid_and_multi_index_poly_sum,
    PolyData,
    scalar_add_monomial_based,
    select_active_monomials,
    shape_coeffs,
)
from minterpy.utils.polynomials.canonical import (
    compute_coeffs_poly_sum as compute_coeffs_poly_sum_canonical,
)
from minterpy.jit_compiled.canonical import (
    compute_coeffs_poly_prod as compute_coeffs_poly_prod_canonical,
)
from minterpy.services import is_scalar

__all__ = ["NewtonPolynomial"]

SUPPORTED_BACKENDS = {
    "numpy": eval_diff_numpy,
    "numba": eval_diff_numba,
    "numba-par": eval_diff_numba_par,
}


# --- Evaluation
def eval_newton(poly: "NewtonPolynomial", xx: np.ndarray) -> np.ndarray:
    """Evaluate polynomial(s) in the Newton basis on a set of query points.

    This is a wrapper for the evaluation function in the Newton basis.

    Parameters
    ----------
    poly : NewtonPolynomial
        The instance of polynomial in the Newton basis to evaluate.
    xx : :class:`numpy:numpy.ndarray`
        Array of query points in the at which the polynomial(s) is evaluated.
        The array is of shape ``(N, m)`` where ``N`` is the number of points
        and ``m`` is the spatial dimension of the polynomial.

    See Also
    --------
    minterpy.utils.polynomials.newton.eval_newton_polynomials
        The actual implementation of the evaluation of polynomials in
        the Newton basis.
    """
    # Get relevant data
    coeffs = poly.coeffs
    exponents = poly.multi_index.exponents
    gen_points = poly.grid.generating_points

    return eval_newton_polynomials(
        xx,
        coeffs,
        exponents,
        gen_points,
        verify_input=DEBUG,
    )


def add_newton(
    poly_1: "NewtonPolynomial",
    poly_2: "NewtonPolynomial",
) -> "NewtonPolynomial":
    """Add two instances of polynomials in the Newton basis.

    This is the concrete implementation of ``_add()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    handling polynomials in the Newton basis.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the addition/subtraction expression.
    poly_2 : NewtonPolynomial
        Right operand of the addition/subtraction expression.

    Returns
    -------
    NewtonPolynomial
        The sum of two polynomials in the Newton basis as a new instance
        of polynomial also in the Newton basis.

    Notes
    -----
    - This function assumes: both polynomials must be in the Newton basis,
      they must be initialized (coefficients are not ``None``),
      have the same dimension and their domains are matching,
      and the number of polynomials per instance are the same.
      These conditions are not explicitly checked in this function.
    """
    # --- Get the ingredients of the summed polynomial in the Newton basis
    poly_sum_data = _compute_poly_sum_data_newton(poly_1, poly_2)

    # --- Return a new instance
    return NewtonPolynomial(**poly_sum_data._asdict())


def mul_newton(
    poly_1: "NewtonPolynomial",
    poly_2: "NewtonPolynomial",
) -> "NewtonPolynomial":
    """Multiply instances of polynomials in the Newton basis.

    This is the concrete implementation of ``_mul()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    handling polynomials in the Newton basis.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the multiplication expression.
    poly_2 : NewtonPolynomial
        Right operand of the multiplication expression.

    Returns
    -------
    NewtonPolynomial
        The product of two polynomials in the Newton basis as a new instance
        of polynomial also in the Newton basis.

    Notes
    -----
    - This function assumes: both polynomials must be in the Newton basis,
      they must be initialized (coefficients are not ``None``),
      have the same dimension and their domains are matching,
      and the number of polynomials per instance are the same.
      These conditions are not explicitly checked in this function.
    """
    # --- Get the ingredients of the product polynomial in the Newton basis
    poly_prod_data = _compute_poly_prod_data_newton(poly_1, poly_2)

    # --- Return a new instance
    return NewtonPolynomial(**poly_prod_data._asdict())


def newton_diff(
    poly: "NewtonPolynomial",
    order: np.ndarray,
    *,
    backend: str = "numba",
) -> "NewtonPolynomial":
    """Differentiate polynomial(s) in the Newton basis of specified orders
    of derivatives along each dimension.

    The orders must be specified for each dimension.
    This is a wrapper for the differentiation function in the Newton basis.

    Parameters
    ----------
    poly : NewtonPolynomial
        The instance of polynomial in Newton form to differentiate.
    order : :class:`numpy:numpy.ndarray`
        A one-dimensional integer array specifying the orders of derivative
        along each dimension. The length of the array must be ``m`` where
        ``m`` is the spatial dimension of the polynomial.
    backend : str
        Computational backend to carry out the differentiation.
        Supported values are:

        - ``"numpy"``: implementation based on NumPy; not performant, only
          applicable for a very small problem size (small degree,
          low dimension).
        - ``"numba"`` (default): implementation based on compiled code with
          the help of Numba; for up to moderate problem size.
        - ``"numba-par"``: parallelized (CPU) implementation based compiled
          code with the help of Numba for relatively large problem sizes.

    Returns
    -------
    NewtonPolynomial
        A new instance of `NewtonPolynomial` that represents the partial
        derivative of the original polynomial of the given order of derivative
        with respect to the specified dimension.
    Notes
    -----
    - The abstract class is responsible to validate ``order``; no additional
      validation regarding that parameter is required here.
    - The transformation of computed Lagrange coefficients of
      the differentiated polynomial to the Newton coefficients is carried out
      using multivariate divided-difference scheme (DDS).

    See Also
    --------
    NewtonPolynomial.diff
        The public method to differentiate the polynomial instance of
        the given orders of derivative along each dimension.
    NewtonPolynomial.partial_diff
        The public method to differentiate the polynomial instance of
        a specified order of derivative with respect to a given dimension.
    """
    # Process the selected backend
    backend = backend.lower()
    if backend not in SUPPORTED_BACKENDS:
        raise NotImplementedError(f"Backend <{backend}> is not supported")
    differentiator = SUPPORTED_BACKENDS[backend]

    # Get relevant data from the polynomial
    grid = poly.grid
    unisolvent_nodes = grid.unisolvent_nodes
    generating_points = grid.generating_points
    tree = grid.tree
    multi_index = poly.multi_index
    exponents = multi_index.exponents
    nwt_coeffs = poly.coeffs

    # Make sure the coefficients are in two-dimension to conform with
    # the differentiator (esp. numba-based) requirement
    if nwt_coeffs.ndim == 1:
        nwt_coeffs = nwt_coeffs[:, np.newaxis]

    # Evaluation of the differentiated Newton polynomial at the unisolvent
    # nodes yields the Lagrange coefficients of the differentiated polynomial.
    lag_diff_coeffs = differentiator(
        unisolvent_nodes,
        nwt_coeffs,
        exponents,
        generating_points,
        order,
    )

    # DDS returns a 2D array, reshaping it according to input coefficient array
    nwt_diff_coeffs = dds(lag_diff_coeffs, tree).reshape(poly.coeffs.shape)

    return NewtonPolynomial(
        coeffs=nwt_diff_coeffs,
        multi_index=multi_index,
        grid=grid,
    )


def newton_partial_diff(
    poly: "NewtonPolynomial",
    dim: int,
    order: int,
    *,
    backend: str = "numba",
) -> "NewtonPolynomial":
    """Differentiate polynomial(s) in the Newton basis with respect to a given
    dimension and order of derivative.

    This is a wrapper for the partial differentiation function in
    the Newton basis.

    Parameters
    ----------
    poly : NewtonPolynomial
        The instance of polynomial in Newton form to differentiate.
    dim : int
        Spatial dimension with respect to which the differentiation
        is taken. The dimension starts at 0 (i.e., the first dimension).
    order : int
        Order of partial derivative.
    backend : str
        Computational backend to carry out the differentiation.
        Supported values are:

        - ``"numpy"``: implementation based on NumPy; not performant, only
          applicable for a very small problem size (small degree,
          low dimension).
        - ``"numba"`` (default): implementation based on compiled code with
          the help of Numba; applicable up to moderate problem size.
        - ``"numba-par"``: parallelized (CPU) implementation based on compiled
          code with the help of Numba for relatively large problem sizes.

    Returns
    -------
    NewtonPolynomial
        A new instance of `NewtonPolynomial` that represents the partial
        derivative of the original polynomial of the given order of derivative
        with respect to the specified dimension.

    Notes
    -----
    - The abstract class is responsible to validate ``dim`` and ``order``; no
      additional validation regarding those two parameters are required here.

    See Also
    --------
    NewtonPolynomial.partial_diff
        The public method to differentiate the polynomial instance of
        a specified order of derivative with respect to a given dimension.
    NewtonPolynomial.diff
        The public method to differentiate the polynomial instance of
        the given orders of derivative along each dimension.
    """
    # Create a specification for differentiation
    spatial_dim = poly.multi_index.spatial_dimension
    deriv_order_along = np.zeros(spatial_dim, dtype=int)
    deriv_order_along[dim] = order

    return newton_diff(poly, deriv_order_along, backend=backend)


def newton_integrate_over(
    poly: "NewtonPolynomial", bounds: np.ndarray
) -> np.ndarray:
    """Compute the definite integral of polynomial(s) in the Newton basis.

    Parameters
    ----------
    poly : NewtonPolynomial
        The polynomial of which the integration is carried out.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper, respectively) of the definite integration,
        specified as an ``(M,2)`` array, where ``M`` is the spatial dimension
        of the polynomial.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integral value of the polynomial over the given domain.
    """
    quad_weights = _compute_quad_weights_newton(poly, bounds)

    return quad_weights @ poly.coeffs


# TODO redundant
newton_generate_internal_domain = verify_domain
newton_generate_user_domain = verify_domain


class NewtonPolynomial(MultivariatePolynomialSingleABC):
    """Concrete implementations of polynomials in the Newton basis.

    For a definition of the Newton basis, see
    :ref:`fundamentals/polynomial-bases:Newton polynomials`.
    """
    # --- Virtual Functions

    # Evaluation
    _eval = staticmethod(eval_newton)

    # Arithmetics (polynomial-polynomial)
    _add = staticmethod(add_newton)
    _sub = staticmethod(dummy)  # type: ignore
    _mul = staticmethod(mul_newton)
    _div = staticmethod(dummy)  # type: ignore
    _pow = staticmethod(dummy)  # type: ignore

    # Arithmetics (polynomial-scalar)
    _scalar_add = staticmethod(scalar_add_monomial_based)

    # Calculus
    _partial_diff = staticmethod(newton_partial_diff)
    _diff = staticmethod(newton_diff)
    _integrate_over = staticmethod(newton_integrate_over)

    # Domain generation
    generate_internal_domain = staticmethod(newton_generate_internal_domain)
    generate_user_domain = staticmethod(newton_generate_user_domain)


# --- Internal utility functions
def _compute_poly_sum_data_newton(
    poly_1: NewtonPolynomial,
    poly_2: NewtonPolynomial,
) -> PolyData:
    """Compute the data to create a summed polynomial in the Newton basis.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the addition/subtraction expression.
    poly_2 : NewtonPolynomial
        Right operand of the addition/subtraction expression.

    Returns
    -------
    PolyData
        The ingredients to construct a summed polynomial in the Newton basis.

    Notes
    -----
    - Both polynomials are assumed to have the same type, spatial dimension,
      and matching domains. This has been made sure by the abstract base class.
    """
    # --- Get the grid and multi-index set of the summed polynomial
    grd_sum, mi_sum = get_grid_and_multi_index_poly_sum(poly_1, poly_2)

    # --- Process the coefficients
    coeffs_sum = _compute_coeffs_poly_sum(poly_1, poly_2, grd_sum, mi_sum)

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


def _compute_coeffs_poly_sum(
    poly_1: NewtonPolynomial,
    poly_2: NewtonPolynomial,
    grid_sum: Grid,
    multi_index_sum: MultiIndexSet,
) -> np.ndarray:
    """Compute the coefficients of a summed polynomial in the Newton basis.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the addition/subtraction expression.
    poly_2 : CanonicalPolynomial
        Right operand of the addition/subtraction expression.
    grid_sum : Grid
        The Grid associated with the summed polynomial.
    multi_index_sum : MultiIndexSet
        The multi-index set of the summed polynomial.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The coefficients of the summed polynomial in the Newton basis.

    Notes
    -----
    - In the case of separate indices, the multi-index set in ``grid_sum``
      differs from ``multi_index_sum``. The latter, however, must be
      a subset of the former (no need to check for that here).
    """
    # Handle the case of compatible grid where no transformation is required
    is_common_grid_1 = poly_1.grid.is_compatible(grid_sum)
    is_common_grid_2 = poly_2.grid.is_compatible(grid_sum)
    is_common_grid = is_common_grid_1 and is_common_grid_2
    is_scalar_poly_grid_1 = is_scalar(poly_1) and is_scalar(poly_1.grid)
    is_scalar_poly_grid_2 = is_scalar(poly_2) and is_scalar(poly_2.grid)
    is_scalar_poly_grid = is_scalar_poly_grid_1 or is_scalar_poly_grid_2
    if is_scalar_poly_grid or is_common_grid:
        return _compute_coeffs_scalar_poly_sum(poly_1, poly_2, multi_index_sum)

    # Compute the values of the operands at the unisolvent nodes
    lag_coeffs_1 = grid_sum(poly_1)
    lag_coeffs_2 = grid_sum(poly_2)
    lag_coeffs_sum = lag_coeffs_1 + lag_coeffs_2

    # Transform the Lagrange coefficients into Newton coefficients
    nwt_coeffs_sum = _transform_lag2nwt(
        lag_coeffs_sum,
        grid_sum,
        poly_1.indices_are_separate or poly_2.indices_are_separate,
        multi_index_sum,
    )

    return nwt_coeffs_sum


def _compute_coeffs_scalar_poly_sum(
    poly_1: NewtonPolynomial,
    poly_2: NewtonPolynomial,
    multi_index_sum: MultiIndexSet,
) -> np.ndarray:
    """Compute the coefficients of a summed constant polynomial.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the addition/subtraction expression.
    poly_2 : NewtonPolynomial
        Right operand of the addition/subtraction expression.
    multi_index_sum : MultiIndexSet
        The multi-index set of the summed polynomial.

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

    exponents_1 = poly_1.multi_index.exponents
    exponents_2 = poly_2.multi_index.exponents
    exponents_sum = multi_index_sum.exponents

    # Compute the coefficients of the summed polynomial
    nwt_coeffs_sum = compute_coeffs_poly_sum_canonical(
        exponents_1,
        coeffs_1,
        exponents_2,
        coeffs_2,
        exponents_sum,
    )

    return nwt_coeffs_sum


def _compute_poly_prod_data_newton(
    poly_1: NewtonPolynomial,
    poly_2: NewtonPolynomial,
) -> PolyData:
    """Compute the data to create a product polynomial in the Newton basis.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the multiplication expression.
    poly_2 : CanonicalPolynomial
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
    # --- Get the grid and multi-index set of the product polynomial
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
    poly_1: NewtonPolynomial,
    poly_2: NewtonPolynomial,
    grid_prod: Grid,
    multi_index_prod: MultiIndexSet,
) -> np.ndarray:
    """Compute the coefficients of polynomial product in the Newton basis.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the multiplication expression.
    poly_2 : CanonicalPolynomial
        Right operand of the multiplication expression.
    grid_prod : Grid
        The Grid associated with the product polynomial.
    multi_index_prod : MultiIndexSet
        The multi-index set of the product polynomial.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The coefficients of the product polynomial in the Newton basis.

    Notes
    -----
    - In the case of separate indices, the multi-index set in ``grid_sum``
      differs from ``multi_index_sum``. The latter, however, must be
      a subset of the former (no need to check for that here).
    """
    # --- Handle case of scalar polynomial (no transformation required)
    # Both polynomials and the underlying grids are scalar
    is_scalar_poly_grid_1 = is_scalar(poly_1) and is_scalar(poly_1.grid)
    is_scalar_poly_grid_2 = is_scalar(poly_2) and is_scalar(poly_2.grid)
    is_scalar_poly_grid = is_scalar_poly_grid_1 or is_scalar_poly_grid_2
    # Some non-scalar grids may still be compatible with the product grid
    is_common_grid_1 = poly_1.grid.is_compatible(grid_prod)
    is_common_grid_2 = poly_2.grid.is_compatible(grid_prod)
    is_common_grid = is_common_grid_1 and is_common_grid_2
    # the grid may not be scalar but the polynomial must still be
    is_scalar_poly = is_scalar(poly_1) or is_scalar(poly_2)
    is_scalar_poly_common_grid = is_scalar_poly and is_common_grid

    if is_scalar_poly_grid or is_scalar_poly_common_grid:
        return _compute_coeffs_scalar_poly_prod(
            poly_1,
            poly_2,
            multi_index_prod,
        )

    # Compute the values of the operands at the unisolvent nodes
    lag_coeffs_1 = grid_prod(poly_1)
    lag_coeffs_2 = grid_prod(poly_2)
    lag_coeffs_prod = lag_coeffs_1 * lag_coeffs_2

    # Transform the Lagrange coefficients into Newton coefficients
    nwt_coeffs_prod = _transform_lag2nwt(
        lag_coeffs_prod,
        grid_prod,
        poly_1.indices_are_separate or poly_2.indices_are_separate,
        multi_index_prod,
    )

    return nwt_coeffs_prod


def _compute_coeffs_scalar_poly_prod(
    poly_1: NewtonPolynomial,
    poly_2: NewtonPolynomial,
    multi_index_prod: MultiIndexSet,
) -> np.ndarray:
    """Compute the coefficients of a summed constant polynomial.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the addition/subtraction expression.
    poly_2 : NewtonPolynomial
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


def _transform_lag2nwt(
    lag_coeffs: np.ndarray,
    grid: Grid,
    indices_are_separate: bool,
    multi_index: MultiIndexSet,
) -> np.ndarray:
    """Transform the (active) Lagrange coefficients to the Newton coefficients.

    Parameters
    ----------
    lag_coeffs : :class:`numpy:numpy.ndarray`
        The coefficients of the polynomial in the Lagrange basis.
    grid : Grid
        The underlying interpolation grid of the polynomial.
    indices_are_separate : bool
        A flag that indicates whether the multi-index set of the grid
        and the given multi-index set are not the same.
    multi_index : MultiIndexSet
        The multi-index set of the polynomial

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The Newton coefficients that correspond to the active monomials.

    Notes
    -----
    - DDS is used in this function to circumvent the problem of circular
      import in Minterpy (transformation class needs polynomial class).
      This is a temporary solution; try to solve the circular import issues
      by better organization and/or using interface functions.
    """
    # Transform the Lagrange coefficients into Newton coefficients
    nwt_coeffs = dds(lag_coeffs, grid.tree)

    # Deal with separate indices, select only w.r.t the active monomials
    if indices_are_separate:
        nwt_coeffs = select_active_monomials(nwt_coeffs, grid, multi_index)

    return nwt_coeffs


def _compute_quad_weights_newton(
    poly: NewtonPolynomial,
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the quadrature weights of a polynomial in the Newton basis.

    The quadrature weights are the integrated monomials in the Newton basis.

    Parameters
    ----------
    poly : NewtonPolynomial
        The polynomial in the Newton basis to be integrated.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds of integration.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The quadrature weights of the polynomial in the Newton basis.
    """
    # Get the relevant data from the polynomial instance
    exponents = poly.multi_index.exponents
    generating_points = poly.grid.generating_points

    quad_weights = integrate_monomials_newton(
        exponents, generating_points, bounds
    )

    return quad_weights
