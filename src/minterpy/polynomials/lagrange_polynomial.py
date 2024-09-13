"""
This module contains the `LagrangePolynomial` class.

The `LagrangePolynomial` class is a concrete implementation of the abstract
base class :py:class:`MultivariatePolynomialSingleABC
<.core.ABC.multivariate_polynomial_abstract.MultivariatePolynomialSingleABC>`
for polynomials in the Lagrange basis.

Background information
----------------------

The relevant section of the documentation on
:ref:`fundamentals/polynomial-bases:Lagrange basis` contains a more
detailed explanation regarding the polynomials in the Lagrange form.

Implementation details
----------------------

`LagrangePolynomial` is currently designed to be a bare concrete
implementation of the abstract base class
:py:class:`MultivariatePolynomialSingleABC
<.core.ABC.multivariate_polynomial_abstract.MultivariatePolynomialSingleABC>`.
In other words, most (if not all) concrete implementation of the abstract
methods are left undefined and will raise an exception when called or invoked.

`LagrangePolynomial` serves as an entry point to Minterpy polynomials
especially in the context of function approximations because the intuitiveness
of the corresponding coefficients (i.e., they are the function values at
the grid points). However, the polynomial itself is not fully featured
(e.g., no addition, multiplication, etc.) as compared to polynomials in
the other basis.

----

"""
from __future__ import annotations

import copy
import numpy as np

from minterpy.global_settings import SCALAR
from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.utils.polynomials.lagrange import integrate_monomials_lagrange
from minterpy.utils.verification import dummy, verify_domain

__all__ = ["LagrangePolynomial"]


def scalar_add_lagrange(
    poly: "LagrangePolynomial",
    scalar: SCALAR,
) -> "LagrangePolynomial":
    """Add an instance of polynomial in the Lagrange basis with a real scalar.

    This is the concrete implementation of ``_scalar_add`` method in the
    ``MultivariatePolynomialSingleABC`` for handling expressions like
    ``poly + x``, where ``x`` is a real scalar number and ``poly`` is
    an instance of ``LagrangePolynomial``.

    Parameters
    ----------
    poly : LagrangePolynomial
        A polynomial instance in the Lagrange basis to be added with a scalar.
    scalar : SCALAR
        The real scalar number to be added to the polynomial instance.

    Returns
    -------
    LagrangePolynomial
        The summed polynomial in the Lagrange basis; the polynomial is a new
        instance.
    """
    # Create a copy of the polynomial
    poly_sum = copy.deepcopy(poly)

    # Add **all** the coefficients with the real scalar number
    poly_sum.coeffs += scalar

    return poly_sum


def integrate_over_lagrange(
    poly: "LagrangePolynomial",
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the definite integral of a polynomial in the Lagrange basis.

    Parameters
    ----------
    poly : LagrangePolynomial
        The polynomial of which the integration is carried out.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integral value of the polynomial over the given domain.
    """
    quad_weights = _compute_quad_weights(poly, bounds)

    return quad_weights @ poly.coeffs


# TODO redundant
lagrange_generate_internal_domain = verify_domain
lagrange_generate_user_domain = verify_domain


class LagrangePolynomial(MultivariatePolynomialSingleABC):
    """Concrete implementation of polynomials in the Lagrange basis.

    A polynomial in the Lagrange basis is the sum of so-called Lagrange
    polynomials, each of which is multiplied with a coefficient.

    The value a *single* Lagrange monomial is per definition :math:`1`
    at one of the grid points and :math:`0` on all the other points.

    Notes
    -----
    - The Lagrange polynomials commonly appear in the Wikipedia article is
      in Minterpy considered the "monomial". In other words, a polynomial in
      the Lagrange basis is the sum of Lagrange monomials each of which is
      multiplied with a coefficient.
    - A polynomial in the Lagrange basis may also be defined also
      for multi-indices of exponents which are not downward-closed.
      In such cases, the corresponding Lagrange monomials also form a basis.
      These mononomials still possess their special property of being :math:`1`
      at a single grid point and :math:`0` at all the other points,
      with respect to the given grid.
    """
    # --- Virtual Functions

    # Evaluation
    _eval = staticmethod(dummy)  # type: ignore

    # Arithmetics (polynomial-polynomial)
    _add = staticmethod(dummy)  # type: ignore
    _sub = staticmethod(dummy)  # type: ignore
    _mul = staticmethod(dummy)  # type: ignore
    _div = staticmethod(dummy)  # type: ignore
    _pow = staticmethod(dummy)  # type: ignore

    # Arithmetics (polynomial-scalar)
    _scalar_add = staticmethod(scalar_add_lagrange)  # type: ignore

    # Calculus
    _partial_diff = staticmethod(dummy)  # type: ignore
    _diff = staticmethod(dummy)  # type: ignore
    _integrate_over = staticmethod(integrate_over_lagrange)

    # Domain generation
    generate_internal_domain = staticmethod(lagrange_generate_internal_domain)
    generate_user_domain = staticmethod(lagrange_generate_user_domain)


# --- Internal utility functions
def _compute_quad_weights(
    poly: LagrangePolynomial,
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the quadrature weights of a polynomial in the Lagrange basis.
    """
    # Get the relevant data from the polynomial instance
    exponents = poly.multi_index.exponents
    generating_points = poly.grid.generating_points
    # ...from the MultiIndexTree
    tree = poly.grid.tree
    split_positions = tree.split_positions
    subtree_sizes = tree.subtree_sizes
    masks = tree.stored_masks

    quad_weights = integrate_monomials_lagrange(
        exponents,
        generating_points,
        split_positions,
        subtree_sizes,
        masks,
        bounds,
    )

    return quad_weights
