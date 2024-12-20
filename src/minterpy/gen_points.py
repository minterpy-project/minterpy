"""
Module with routines to create generating points for polynomial interpolants.

Generating points are the main ingredient of constructing a set of unisolvent
nodes (i.e., interpolation nodes) on which a polynomial interpolant is uniquely
determined.

In one dimension, a set of generating points is the same as the unisolvent
nodes. In higher dimension, a set of unisolvent nodes are constructed based on
the generating points in each dimension and the multi-index set of polynomial
exponents.
"""
import numpy as np

from minterpy.global_settings import FLOAT_DTYPE


def gen_points_chebyshev(
    poly_degree: int,
    spatial_dimension: int,
) -> np.ndarray:
    """Create generating points from Chebyshev points.


    """
    xx = gen_chebychev_2nd_order_leja_ordered(poly_degree)

    gen_points = gen_points_from_values(xx, spatial_dimension)

    return gen_points


def gen_points_from_values(
    generating_values: np.ndarray,
    spatial_dimension: int,
) -> np.ndarray:
    """Construct an array of generating points given values in one dimension.

    Parameters
    ----------
    spatial_dimension : int

    generating_values : :class:`numpy:numpy.ndarray`

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        A two-dimensional array of floats whose columns are the
        generating points per spatial dimension. The shape of the array
        is ``(n + 1, m)`` where ``n`` is the maximum polynomial degree
        in one dimension and ``m`` is the spatial dimension.
    """
    if generating_values.ndim == 1:
        generating_values = generating_values[:, np.newaxis]
    generating_points = np.tile(generating_values, (1, spatial_dimension))
    generating_points[:, ::2] *= -1
    return generating_points


def chebychev_2nd_order(n: int):  # 2nd order
    """Factory function of Chebychev points of the second kind.

    :param n: Degree of the point set, i.e. number of Chebychev points.
    :type n: int

    :return: Array of Chebychev points of the second kind.
    :rtype: np.ndarray

    .. todo::
        - rename this function
        - rename the parameter ``n``.
    """
    if n == 0:
        return np.zeros(1, dtype=FLOAT_DTYPE)
    if n == 1:
        return np.array([-1.0, 1.0], dtype=FLOAT_DTYPE)
    return np.cos(np.arange(n, dtype=FLOAT_DTYPE) * np.pi / (n - 1))


def gen_chebychev_2nd_order_leja_ordered(n: int):
    """Factory function of Leja ordered Chebychev points of the second kind.

    :param n: Degree of the point set, i.e. number of Chebychev points (plus one!).
    :type n: int

    :return: Array of Leja ordered Chebychev points of the second kind.
    :rtype: np.ndarray

    .. todo::
        - rename this function
        - rename the parameter ``n``.
        - refactor this function to remove all the loops.
        - make the arguments equivalent to ``chebychev_2nd_order``, i.e. if the degree ``n`` is passed, the number of points shall be ``n`` (not ``n+1``).
    """
    n = int(n)
    points1 = chebychev_2nd_order(n + 1)[::-1]
    points2 = points1  # TODO
    ord = np.arange(1, n + 1)

    lj = np.zeros([1, n + 1])
    lj[0] = 0
    m = 0

    for k in range(0, n):
        jj = 0
        for i in range(0, n - k):
            P = 1
            for j in range(k + 1):
                idx_pts = int(lj[0, j])
                P = P * (points1[idx_pts] - points1[ord[i]])
            P = np.abs(P)
            if P >= m:
                jj = i
                m = P
        m = 0
        lj[0, k + 1] = ord[jj]
        ord = np.delete(ord, jj)

    leja_points = np.zeros([n + 1, 1])
    for i in range(n + 1):
        leja_points[i, 0] = points2[int(lj[0, i])]
    return leja_points


GENERATING_FUNCTIONS = {
    "chebyshev": gen_points_chebyshev,
}
