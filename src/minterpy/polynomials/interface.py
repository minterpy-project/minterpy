"""
This module contains functions that bridge between the upper layer of
abstraction (``NewtonPolynomial``, ``LagrangePolynomial``, etc.) to the
lower layer of abstraction (numerical routines that operates on arrays) that
typically resides in the ``minterpy.utils`` or ``minterpy.jit_compiled``.

The idea behind this module is to minimize the detail of computations
inside the concrete polynomial modules.
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from minterpy.utils.polynomials.lagrange import integrate_monomials_lagrange
from minterpy.utils.polynomials.newton import integrate_monomials_newton
from minterpy.utils.polynomials.canonical import integrate_monomials_canonical

# To avoid circular import in the type hints
if TYPE_CHECKING:
    from minterpy.polynomials.canonical_polynomial import CanonicalPolynomial
    from minterpy.polynomials.lagrange_polynomial import LagrangePolynomial
    from minterpy.polynomials.newton_polynomial import NewtonPolynomial


def compute_quad_weights_lagrange(
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


def compute_quad_weights_newton(
    poly: NewtonPolynomial,
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the quadrature weights of a polynomial in the Newton basis.
    """
    # Get the relevant data from the polynomial instance
    exponents = poly.multi_index.exponents
    generating_points = poly.grid.generating_points

    quad_weights = integrate_monomials_newton(
        exponents, generating_points, bounds
    )

    return quad_weights


def compute_quad_weights_canonical(
    poly: CanonicalPolynomial,
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the quadrature weights of a polynomial in the Canonical basis.
    """
    # Get the relevant data from the polynomial instance
    exponents = poly.multi_index.exponents

    quad_weights = integrate_monomials_canonical(exponents, bounds)

    return quad_weights
