"""
Module with JIT-compiled routines related polynomial transformations.
"""
from numba import njit, void

from minterpy.global_settings import F_2D, I_2D


@njit(void(F_2D, F_2D, I_2D), cache=True)
def compute_vandermonde_n2c(V_n2c, nodes, exponents):
    """Computes the Vandermonde matrix.

    - ``m`` spatial dimension
    - ``N`` number of monomials

    :param V_n2c: the placeholder array to store the Vandermonde matrix. The shape has to be ``(N x N)``.
    :param nodes: the unisolvent nodes
    :param exponents:  numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.

    """
    num_monomials, spatial_dimension = exponents.shape
    for i in range(num_monomials):
        for j in range(1, num_monomials):
            for d in range(spatial_dimension):
                V_n2c[i, j] *= nodes[i, d] ** exponents[j, d]
