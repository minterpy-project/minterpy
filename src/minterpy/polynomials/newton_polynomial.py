"""
Module of the NewtonPolynomial class

.. todo::
    - implement staticmethods for Newton polynomials (or at least transform them to another base).
"""
import numpy as np

from minterpy.core import Grid
from minterpy.core.ABC.multivariate_polynomial_abstract import (
    MultivariatePolynomialSingleABC,
)
from minterpy.core.verification import verify_domain
from minterpy.dds import dds
from minterpy.global_settings import DEBUG
from minterpy.utils import eval_newton_polynomials
from minterpy.polynomials.utils import (
    deriv_newt_eval,
    integrate_monomials_newton,
)

__all__ = ["NewtonPolynomial"]


def dummy():
    """Placeholder function.

    .. warning::
      This feature is not implemented yet!
    """
    # move dummy to util?
    raise NotImplementedError("This feature is not implemented yet.")


def _is_constant_poly(poly: MultivariatePolynomialSingleABC) -> bool:
    """Check if an instance polynomial is constant.

    Parameters
    ----------
    poly : MultivariatePolynomialSingleABC
        A given polynomial to check.

    Returns
    -------
    bool
        ``True`` if the polynomial is a constant polynomial,
        ``False`` otherwise.

    TODO
    ----
    - Refactor this as a common utility function especially when it is
      required by other modules.
    """
    # Check the multi-index set
    mi = poly.multi_index
    has_one_element = len(mi) == 1
    has_zero = np.zeros(mi.spatial_dimension, dtype=np.int_) in mi

    # Check the coefficient values (will raise an exception if there is none
    coeffs = poly.coeffs
    has_one_coeff = len(coeffs) == 1

    return has_one_element and has_zero and has_one_coeff


def newton_eval(newton_poly, x):
    """Evaluation function in Newton base.

    This is a wrapper for the evaluation function in Newton base.

    :param newton_poly: The :class:`NewtonPolynomial` which is evaluated.
    :type newton_poly: NewtonPolynomial
    :param x: The argument(s) the Newton polynomial shall be evaluated. The input shape needs to be ``(N,dim)``, where ``N`` refered to the number of points and ``dim`` refers to the dimension of the domain space, i.e. the coordinates of the argument vector.


    .. todo::
        - check right order of axes for ``x``.

    See Also
    --------
    newt_eval : comcrete implementation of the evaluation in Newton base.

    """
    return eval_newton_polynomials(
        x,
        newton_poly.coeffs,
        newton_poly.multi_index.exponents,
        newton_poly.grid.generating_points,
        verify_input=DEBUG,
    )


def _newton_mul(
    poly_1: "NewtonPolynomial",
    poly_2: "NewtonPolynomial",
) -> "NewtonPolynomial":
    """Multiply two Newton polynomials.

    Parameters
    ----------
    poly_1 : NewtonPolynomial
        The first (left operand) polynomial in the Newton basis to multiply.
    poly_2 : NewtonPolynomial
        The second (right operand) polynomial in the Newton basis to multiply.

    Returns
    -------
    NewtonPolynomial
        The product polynomial in the Newton basis.

    Raises
    ------
    ValueError
        If the resulting multi-index product is not downward-closed.

    Notes
    -----
    - The multi-index set of the product polynomial must be downward-closed
      because DDS is used to transform the Lagrange coefficients to the
      Newton coefficients.
    """
    # MultiIndexSet operation
    mi_1 = poly_1.multi_index
    mi_2 = poly_2.multi_index
    mi_prod = mi_1 * mi_2
    if not mi_prod.is_downward_closed:
        raise ValueError(
            "The resulting multi-index product must be downward-closed."
        )

    # Grid operation
    # TODO: both polynomial must have the same generating function
    #       Otherwise the unisolvent nodes are inconsistent
    grd_prod = Grid(mi_prod)

    if _is_constant_poly(poly_1):
        nwt_coeffs_prod = poly_1.coeffs[0] * poly_2.coeffs
    elif _is_constant_poly(poly_2):
        nwt_coeffs_prod = poly_2.coeffs[0] * poly_1.coeffs
    else:
        unisolvent_nodes = grd_prod.unisolvent_nodes

        # Compute the values at the unisolvent nodes
        dim_1 = poly_1.spatial_dimension
        dim_2 = poly_2.spatial_dimension
        lag_coeffs_1 = poly_1(unisolvent_nodes[:, :dim_1])
        lag_coeffs_2 = poly_2(unisolvent_nodes[:, :dim_2])
        lag_coeffs_prod = lag_coeffs_1 * lag_coeffs_2

        # Transform the coefficients
        # TODO: DDS returns at least 2D array even if there's only one set of
        #       coefficients
        nwt_coeffs_prod = dds(lag_coeffs_prod, grd_prod.tree)

    nwt_poly_prod = NewtonPolynomial(mi_prod, nwt_coeffs_prod, grid=grd_prod)

    return nwt_poly_prod


def _newton_partial_diff(poly: "NewtonPolynomial", dim: int, order: int) -> "NewtonPolynomial":
    """ Partial differentiation in Newton basis.

    Notes
    -----
    Performs a transformation from Lagrange to Newton using DDS.
    """
    spatial_dim = poly.multi_index.spatial_dimension
    deriv_order_along = [0]*spatial_dim
    deriv_order_along[dim] = order
    return _newton_diff(poly, deriv_order_along)


def _newton_diff(poly: "NewtonPolynomial", order: np.ndarray) -> "NewtonPolynomial":
    """ Partial differentiation in Newton basis.

    Notes
    -----
    Performs a transformation from Lagrange to Newton using DDS.
    """

    # When you evaluate the derivatives on the unisolvent nodes, you get the coefficients for
    # the derivative polynomial in Lagrange basis.
    lag_coeffs = deriv_newt_eval(poly.grid.unisolvent_nodes, poly.coeffs,
                                 poly.grid.multi_index.exponents,
                                 poly.grid.generating_points, order)

    # DDS returns a 2D array, reshaping it according to input coefficient array
    newt_coeffs = dds(lag_coeffs, poly.grid.tree).reshape(poly.coeffs.shape)

    return NewtonPolynomial(coeffs=newt_coeffs, multi_index=poly.multi_index,
                              grid=poly.grid)


def _newton_integrate_over(
    poly: "NewtonPolynomial", bounds: np.ndarray
) -> np.ndarray:
    """Compute the definite integral of a polynomial in the Newton basis.

    Parameters
    ----------
    poly : NewtonPolynomial
        The polynomial of which the integration is carried out.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

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
    """Datatype to describe polynomials in Newton base.

    For a definition of the Newton base, see the mathematical introduction.

    .. todo::
        - provide a short definition of this base here.
    """

    # Virtual Functions
    _add = staticmethod(dummy)
    _sub = staticmethod(dummy)
    _mul = staticmethod(_newton_mul)
    _div = staticmethod(dummy)
    _pow = staticmethod(dummy)
    _eval = staticmethod(newton_eval)

    _partial_diff = staticmethod(_newton_partial_diff)
    _diff = staticmethod(_newton_diff)

    _integrate_over = staticmethod(_newton_integrate_over)

    generate_internal_domain = staticmethod(newton_generate_internal_domain)
    generate_user_domain = staticmethod(newton_generate_user_domain)
