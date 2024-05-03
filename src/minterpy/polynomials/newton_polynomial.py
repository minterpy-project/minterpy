"""
Module of the NewtonPolynomial class

.. todo::
    - implement staticmethods for Newton polynomials (or at least transform them to another base).
"""
import numpy as np

from minterpy.core.ABC.multivariate_polynomial_abstract import (
    MultivariatePolynomialSingleABC,
)
from minterpy.core.verification import verify_domain
from minterpy.dds import dds
from minterpy.global_settings import DEBUG
from minterpy.utils import eval_newton_polynomials
from minterpy.polynomials.utils import (
    deriv_newt_eval as eval_diff_numpy,
    integrate_monomials_newton,
    dummy,
)

from minterpy.jit_compiled.newton.diff import (
    eval_multiple_query as eval_diff_numba,
    eval_multiple_query_par as eval_diff_numba_par,
)

__all__ = ["NewtonPolynomial"]

SUPPORTED_BACKENDS = {
    "numpy": eval_diff_numpy,
    "numba": eval_diff_numba,
    "numba-par": eval_diff_numba_par,
}


def newton_eval(poly: "NewtonPolynomial", xx: np.ndarray) -> np.ndarray:
    """Evaluate polynomial(s) in the Newton basis on a set of query points.

    This is a wrapper for the evaluation function in the Newton basis.

    Parameters
    ----------
    poly : NewtonPolynomial
        The instance of polynomial in Newton form to evaluate.
    xx : :class:`numpy:numpy.ndarray`
        Array of query points in the at which the polynomial(s) is evaluated.
        The array is of shape ``(n,m)`` where ``n`` is the number of points
        and ``m`` is the spatial dimension of the polynomial.

    See Also
    --------
    minterpy.utils.eval_newton_polynomials
        The actual implementation of the evaluation of polynomials in
        the Newton basis.

    TODO
    ----
    - check right order of axes for ``x``.
    """
    return eval_newton_polynomials(
        xx,
        poly.coeffs,
        poly.multi_index.exponents,
        poly.grid.generating_points,
        verify_input=DEBUG,
    )


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
    NewtonPolynomial._diff
        The underlying (lower-level) implementation (possibly, a wrapper) of
        `NewtonPolynomial.diff`.
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
    NewtonPolynomial._partial_diff
        The underlying (lower-level) implementation (possibly, a wrapper) of
        `NewtonPolynomial.partial_diff`.
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

    # --- Compute the integrals of the Newton monomials (quadrature weights)
    exponents = poly.multi_index.exponents
    generating_points = poly.grid.generating_points

    quad_weights = integrate_monomials_newton(
        exponents, generating_points, bounds
    )

    return quad_weights @ poly.coeffs


# TODO redundant
newton_generate_internal_domain = verify_domain
newton_generate_user_domain = verify_domain


class NewtonPolynomial(MultivariatePolynomialSingleABC):
    """Concrete implementations of polynomials in the Newton basis.

    For a definition of the Newton base, see
    :ref:`fundamentals/polynomial-bases:Newton polynomials`.
    """

    # Virtual Functions
    _add = staticmethod(dummy)
    _sub = staticmethod(dummy)
    _mul = staticmethod(dummy)
    _div = staticmethod(dummy)
    _pow = staticmethod(dummy)

    _eval = staticmethod(newton_eval)
    _partial_diff = staticmethod(newton_partial_diff)
    _diff = staticmethod(newton_diff)
    _integrate_over = staticmethod(newton_integrate_over)

    generate_internal_domain = staticmethod(newton_generate_internal_domain)
    generate_user_domain = staticmethod(newton_generate_user_domain)
