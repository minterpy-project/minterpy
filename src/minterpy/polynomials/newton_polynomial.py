"""
This module contains the `NewtonPolynomial` class.

The `NewtonPolynomial` class is a concrete implementation of the abstract
base class :py:class:`MultivariatePolynomialSingleABC
<.core.ABC.multivariate_polynomial_abstract.MultivariatePolynomialSingleABC>`
for polynomials in the Newton basis.

Background information
----------------------

The relevant section of the documentation on
:ref:`fundamentals/polynomial-bases:Newton basis` contains a more
detailed explanation regarding the polynomials in the Newton basis.

Implementation details
----------------------

`NewtonPolynomial` is currently the preferred polynomial basis in Minterpy
for various operations. An instance of `NewtonPolynomial` can be evaluated
on a set of query points (unlike :py:class:`LagrangePolynomial
<.polynomials.lagrange_polynomial.LagrangePolynomial>`) and the evaluation
is numerically stable (unlike :py:class:`CanonicalPolynomial
<.polynomials.canonical_polynomial.CanonicalPolynomial>`).

A wide range of arithmetic operations are supported for polynomials in
the Lagrange basis, namely:

- addition and subtraction by a scalar number
- addition and subtraction by another Newton polynomial
- multiplication by a scalar number
- multiplication by another Newton polynomial

Moreover, basic calculus operations are also supported, namely:
differentiation and definite integration.
"""
from __future__ import annotations

import numpy as np

from minterpy.global_settings import DEBUG
from minterpy.core.ABC.multivariate_polynomial_abstract import (
    MultivariatePolynomialSingleABC,
)
from minterpy.core import Grid, MultiIndexSet
from minterpy.dds import dds
from minterpy.utils.verification import dummy
from minterpy.utils.polynomials.newton import (
    eval_newton_polynomials,
    deriv_newt_eval as eval_diff_numpy,
    integrate_monomials_newton,
)
from minterpy.polynomials.arithmetic import (
    get_compute_coeffs_add,
    get_compute_coeffs_mul,
    select_active_monomials,
)
from minterpy.polynomials.scalar_add import scalar_add_via_monomials
from minterpy.jit_compiled.newton.diff import (
    eval_multiple_query as eval_diff_numba,
    eval_multiple_query_par as eval_diff_numba_par,
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


# --- Arithmetics (Addition, Multiplication)
def add_newton(
    poly_1: "NewtonPolynomial",
    poly_2: "NewtonPolynomial",
    multi_index: MultiIndexSet,
    grid: Grid,
) -> "NewtonPolynomial":
    r"""Add two instances of polynomials in the Newton basis.

    This is the concrete implementation of ``_add()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    handling polynomials in the Newton basis.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the addition.
    poly_2 : NewtonPolynomial
        Right operand of the addition.
    multi_index : MultiIndexSet
        The multi-index set of the resulting polynomial, i.e., the union
        of the multi-index sets of the operands.
    grid : Grid
        The grid of the resulting polynomial, i.e., the union of the grids
        of the two operands.

    Returns
    -------
    NewtonPolynomial
        The sum of two polynomials in the Newton basis as a new instance
        of polynomial.

    Notes
    -----
    **Algorithm:**

    The Newton basis is in general not closed under addition because Newton
    monomials depend on the underlying grid's generating points. When grids
    differ, the monomials differ, requiring transformation through the Lagrange
    basis:

    1. Evaluate both Newton polynomials at the union grid's unisolvent nodes
       to obtain Lagrange coefficients
    2. Sum the Lagrange coefficients
    3. Transform back to Newton coefficients on the union grid

    **Special cases (monomials addition used instead):**

    - **Compatible grids**: If both operand grids are compatible with the
      result grid, their Newton monomials are identical, allowing direct
      coefficient addition.
    - **Scalar operand**: If one operand is a constant (multi-index set
      contains only :math:`(0, \ldots, 0)`), it can be added directly
      regardless of grid compatibility.

    **Preconditions:**

    - This function assumes the caller has verified: both polynomials are in
      the Newton basis, both are initialized, their domains match, and they
      have the same number of polynomial instances.
    - The provided ``multi_index`` may differ from ``grid.multi_index`` when
      the polynomial multi-index set is a subset of the grid multi-index set
      (i.e., separated indices). The coefficients are computed with respect
      to the provided ``multi_index``.
    """
    # Handle the case where no transformation is required
    if _is_compute_coeffs_poly_add_via_monomials(poly_1, poly_2, grid):
        compute_coeffs = get_compute_coeffs_add("monomials")
        coeffs = compute_coeffs(poly_1, poly_2, multi_index)
    else:
        # Compute the coefficients via a transformation from Lagrange coeffs.
        compute_coeffs = get_compute_coeffs_add("lagrange")
        coeffs_lag = compute_coeffs(poly_1, poly_2, grid)
        coeffs = _transform_lag2nwt(coeffs_lag, multi_index, grid)

    # Create and return a new instance of polynomial
    return NewtonPolynomial(multi_index, coeffs, grid)


def mul_newton(
    poly_1: "NewtonPolynomial",
    poly_2: "NewtonPolynomial",
    multi_index: MultiIndexSet,
    grid: Grid,
) -> "NewtonPolynomial":
    r"""Multiply instances of polynomials in the Newton basis.

    This is the concrete implementation of ``_mul()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class specifically for
    polynomials in the Newton basis.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the multiplication.
    poly_2 : NewtonPolynomial
        Right operand of the multiplication.
    multi_index : MultiIndexSet
        The multi-index set of the resulting polynomial, i.e., the product
        of the multi-index sets of the operands.
    grid : Grid
        The grid of the resulting polynomial, i.e., the product of the grids
        of the two operands.

    Returns
    -------
    NewtonPolynomial
        The product of two polynomials in the Newton basis as a new instance
        of polynomial.

    Notes
    -----
    **Algorithm:**

    The Newton basis is in general not closed under multiplication because
    the product of Newton monomials does not result in a Newton monomial
    of a higher degree (unlike canonical basis). The coefficients
    of a polynomial product requires transformation through the Lagrange basis:

    1. Evaluate both Newton polynomials at the product grid's unisolvent nodes
       to obtain Lagrange coefficients
    2. Multiply the Lagrange coefficients
    3. Transform back to Newton coefficients on the product grid

    **Special cases (monomials addition used instead):**

    - **Scalar polynomial with scalar grid**: If one operand is a constant
      polynomial (multi-index contains only :math:`(0, \ldots, 0)`)
      and its grid's multi-index is also scalar,
      direct coefficient multiplication is used.
    - **Scalar polynomial with compatible non-scalar grid**: If one operand is
      a constant polynomial but its grid's multi-index is non-scalar, direct
      coefficient multiplication is used only if the grid is compatible with
      the product grid.

    **Preconditions:**

    - This function assumes the caller has verified: both polynomials are in
      the Newton basis, both are initialized, their domains match, and they
      have the same number of polynomial instances.
    - The provided ``multi_index`` may differ from ``grid.multi_index`` when
      the polynomial multi-index set is a subset of the grid multi-index set
      (i.e., separated indices). The coefficients are computed with respect
      to the provided ``multi_index``.
    """
    # Handle the case where no transformation is required
    if _is_compute_coeffs_poly_mul_via_monomials(poly_1, poly_2, grid):
        compute_coeffs = get_compute_coeffs_mul("monomials")
        coeffs = compute_coeffs(poly_1, poly_2, multi_index)
    else:
        # Compute the coefficients via a transformation from Lagrange coeffs.
        compute_coeffs = get_compute_coeffs_mul("lagrange")
        coeffs_lag = compute_coeffs(poly_1, poly_2, grid)
        coeffs = _transform_lag2nwt(coeffs_lag, multi_index, grid)

    # Create and return a new instance of polynomial
    return NewtonPolynomial(multi_index, coeffs, grid)


# --- Calculus
def diff_newton(
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


def partial_diff_newton(
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

    return diff_newton(poly, deriv_order_along, backend=backend)


def integrate_over_newton(
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
    quad_weights = _compute_quad_weights(poly, bounds)

    return quad_weights @ poly.coeffs


class NewtonPolynomial(MultivariatePolynomialSingleABC):
    """Concrete implementations of polynomials in the Newton basis.

    For a definition of the Newton basis, see
    :ref:`fundamentals/polynomial-bases:Newton basis`.
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
    _scalar_add = staticmethod(scalar_add_via_monomials)

    # Calculus
    _partial_diff = staticmethod(partial_diff_newton)
    _diff = staticmethod(diff_newton)
    _integrate_over = staticmethod(integrate_over_newton)


# --- Internal utility functions
def _is_compute_coeffs_poly_add_via_monomials(
    poly_1: NewtonPolynomial,
    poly_2: NewtonPolynomial,
    grid_sum: Grid
) -> bool:
    """Check if the polynomials may be summed up via the monomials.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the multiplication expression.
    poly_2 : NewtonPolynomial
        Right operand of the multiplication expression.
    grid_prod : Grid
        The Grid associated with the summed polynomial.

    Returns
    -------
    bool
        ``True`` if one of the operands is a scalar polynomial, or if both
        the underlying grids are compatible with the given product grid;
        ``False`` otherwise.
    """
    # If one of the operands is a scalar polynomial
    is_scalar_poly = is_scalar(poly_1) or is_scalar(poly_2)
    # ...or if the grids are compatible
    is_compatible_grid_1 = poly_1.grid.has_compatible_gen_points(grid_sum)
    is_compatible_grid_2 = poly_2.grid.has_compatible_gen_points(grid_sum)
    is_compatible_grids = is_compatible_grid_1 and is_compatible_grid_2

    return is_scalar_poly or is_compatible_grids


def _is_compute_coeffs_poly_mul_via_monomials(
    poly_1: NewtonPolynomial,
    poly_2: NewtonPolynomial,
    grid_prod: Grid
) -> bool:
    """Check if the polynomials may be multiplied via the monomials.

    The multiplication can use the monomial approach when the resulting Newton
    polynomial lives on the same grid (with the same generating points) as one
    of the operands. This allows reusing the existing Newton basis without
    recomputation, which is crucial since changing the degree typically alters
    the generating points for unnested grids (e.g., Chebyshev-Lobatto).

    Because Newton polynomials are not closed under multiplication (i.e., the
    product of two Newton basis polynomials is not itself a Newton basis
    polynomial of higher degree), at least one operand must be a scalar
    polynomial to ensure that the product can be represented
    in the same Newton basis as the non-scalar operand.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        Left operand of the multiplication expression.
    poly_2 : NewtonPolynomial
        Right operand of the multiplication expression.
    grid_prod : Grid
        The Grid associated with the product polynomial.

    Returns
    -------
    bool
        ``True`` if one of the operands is a scalar polynomial, or if
        one of them has a scalar monomial and both of the underlying grids are
        compatible with the given product grid; ``False`` otherwise.
    """
    # Check if either operand is a strictly scalar polynomial
    is_scalar_poly = is_scalar(poly_1) or is_scalar(poly_2)

    # Check grid compatibility based on scalar multi_index
    poly_1_scalar_idx = is_scalar(poly_1.multi_index)
    poly_2_scalar_idx = is_scalar(poly_2.multi_index)

    if poly_1_scalar_idx and poly_2_scalar_idx:
        # Both have scalar multi_index: always compatible
        grid_compatible = True
    elif poly_1_scalar_idx:
        # Only poly_1 has scalar multi_index: check poly_2's grid
        grid_compatible = poly_2.grid.has_compatible_gen_points(grid_prod)
    elif poly_2_scalar_idx:
        # Only poly_2 has scalar multi_index: check poly_1's grid
        grid_compatible = poly_1.grid.has_compatible_gen_points(grid_prod)
    else:
        # Neither has scalar multi_index: not compatible
        grid_compatible = False

    # Compatible if either operand is scalar OR grids are compatible
    return is_scalar_poly or grid_compatible


def _transform_lag2nwt(
    coeffs_lag: np.ndarray,
    multi_index: MultiIndexSet,
    grid: Grid,
) -> np.ndarray:
    """Transform the (active) Lagrange coefficients to the Newton coefficients.

    Parameters
    ----------
    lag_coeffs : :class:`numpy:numpy.ndarray`
        The coefficients of the polynomial in the Lagrange basis.
    grid : Grid
        The underlying interpolation grid of the polynomial.
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
    coeffs_nwt = dds(coeffs_lag, grid.tree)

    # Deal with separate indices, select only w.r.t the active monomials
    if multi_index != grid.multi_index:
        coeffs_nwt = select_active_monomials(coeffs_nwt, grid, multi_index)

    return coeffs_nwt


def _compute_quad_weights(
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
