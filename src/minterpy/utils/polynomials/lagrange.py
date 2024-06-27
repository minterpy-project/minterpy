"""
This module provides computational routines relevant to polynomials
in the Lagrange basis.
"""
import numpy as np

from minterpy.core.tree import MultiIndexTree
from minterpy.dds import dds
from minterpy.utils.polynomials.newton import integrate_monomials_newton


def integrate_monomials_lagrange(
    exponents: np.ndarray,
    generating_points: np.ndarray,
    tree: MultiIndexTree,
    bounds: np.ndarray,
) -> np.ndarray:
    """Integrate the monomials in the Lagrange basis given a set of exponents.

    Parameters
    ----------
    exponents : :class:`numpy:numpy.ndarray`
        A set of exponents from a multi-index set that defines the polynomial,
        an ``(N, M)`` array, where ``N`` is the number of exponents
        (multi-indices) and ``M`` is the number of spatial dimensions.
        The number of exponents corresponds to the number of monomials.
    generating_points : :class:`numpy:numpy.ndarray`
        A set of generating points of the interpolating polynomial,
        a ``(P + 1, M)`` array, where ``P`` is the maximum degree of
        the polynomial in any dimensions and ``M`` is the number
        of spatial dimensions.
    tree : MultiIndexTree
        The MultiIndexTree to perform multi-dimensional divided-difference
        scheme (DDS) for transforming the Newton basis to Lagrange basis.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integrated Lagrange monomials, an ``(N,)`` array, where ``N``
        is the number of monomials (exponents).

    Notes
    -----
    - The Lagrange monomials are represented in the Newton basis.
      For integration, first integrate the Newton monomials and then transform
      the results back to the Lagrange basis. This is why the `MultiIndexTree`
      instance is needed.

    TODO
    ----
    - The whole integration domain is assumed to be :math:`[-1, 1]^M` where
      :math:`M` is the number of spatial dimensions because the polynomial
      itself is defined in that domain. This condition may be relaxed in
      the future and the implementation below should be modified.
    - Possibly reorganize this function in another module. This function is
      currently only used by the concrete implementation of
      the Lagrange basis.
    """
    monomials_integrals_newton = integrate_monomials_newton(
        exponents, generating_points, bounds
    )
    l2n = dds(np.eye(exponents.shape[0]), tree)

    # --- Carry out the transformation from Newton to Lagrange
    monomials_integrals = l2n.T @ monomials_integrals_newton

    return monomials_integrals
