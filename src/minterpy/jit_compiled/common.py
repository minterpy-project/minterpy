"""
A module with JIT-compiled code for common functionalities.

Most of the functions listed below are available in base Python or NumPy.
However, a common underlying reason to re-implement this in Numba is that
those functions cannot be called or called with the required arguments
within a JIT-compiled function.
"""
import numpy as np

from numba import njit

from minterpy.global_settings import (
    UINT32,
    UINT64,
    INT_DTYPE,
    I_1D,
    I_2D,
    F_1D,
    F_2D,
)


@njit(UINT64(UINT32, UINT32), cache=True)
def n_choose_r(n: int, r: int) -> int:
    """Compute the binominal coefficient (n choose r) iteratively.

    Parameters
    ----------
    n : int
        The total number of elements to choose from.
    r : int
        The number of elements to form the combinations.

    Returns
    -------
    int
        The number of ways to choose ``r`` elements from ``n`` elements
        (i.e., the binomial coefficient).

    Raises
    ------
    OverflowError
        If any of the arguments is signed (which does not make any sense
        anyway).

    Notes
    -----
    - This iterative implementation is an alternative to :py:func:`math.comb`
      (from the :py:mod:`math` module) such that it can be used inside
      a Numba JIT-ted function.
    """
    # Guardrails
    if r > n:
        return 0

    # Switch denominator
    if r > n - r:
        r = n - r

    # Divide at each iteration to avoid large number
    out = 1
    for i in range(r):
        out *= (n - i)
        out //= (i + 1)

    return out


@njit(I_2D(I_1D, UINT32), cache=True)
def combinations_iter(xx: np.ndarray, r: int) -> np.ndarray:
    """Return successive r-length combinations of elements as an array.

    Parameters
    ----------
    xx : :class:`numpy:numpy.ndarray`
        A one-dimensional array of integers to select from.
    r : int
        The number of elements to select from the array of integers.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        A two-dimensional array of integers; each rows is the selected
        combination. The total number of rows is :math:`n \choose r`
        where :math:`n` is the length of the input array.
        The r-length combinations are sorted lexicographically.

    Notes
    -----
    - This is iterative implementation is suitable for Numba as an alternative
      to :py:func:`itertools.combinations` (from the :py:mod:`itertools`
      module).
    - This is not a generator function; all possible combinations
      of length ``r`` will be returned at once. Unlike the function
      :py:func:`itertools.combinations`, the output of this function is
      a two-dimensional array instead of a *tuple* generator.
    - This function is created to avoid using a generator function inside
      a Numba JIT-ted function; generator functions causes memory leaks.
    - In the way this function is used by the relevant part in Minterpy, ``r``
      is always close to the total length of ``xx`` so there is no explosions
      in the number of elements.
    """
    # Safeguard
    if r == 0:
        return np.empty(shape=(0, r), dtype=INT_DTYPE)

    # Allocate output array
    n = len(xx)
    num_combs = n_choose_r(n, r)
    out = np.empty(shape=(num_combs, r), dtype=INT_DTYPE)

    # Initialization of the selection array
    idx = np.empty(r, dtype=INT_DTYPE)
    for j in range(r):
        idx[j] = j
    i = r - 1  # start with the last element to change
    cur = 0

    while idx[0] < n - r + 1:

        # Update pointing index of the selection array
        while i > 0 and idx[i] == n - r + i:
            i -= 1

        # Fill in a combination
        for j in range(r):
            out[cur, j] = xx[idx[j]]
        cur += 1

        # Update the selection array itself
        idx[i] += 1
        while i < r - 1:
            idx[i + 1] = idx[i] + 1
            i += 1

    return out


@njit(F_1D(F_1D, F_2D), cache=True)
def dot(a: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """Compute vector-matrix dot product with contiguous arrays.

    Parameters
    ----------
    a : :class:`numpy:numpy.ndarray`
        A one-dimensional array of length ``n``.
    bb : :class:`numpy:numpy.ndarray`
        A two-dimensional array of shape ``(n, m)``.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        A one-dimensional array of length ``m``.
    """
    a = np.ascontiguousarray(a)
    bb = np.ascontiguousarray(bb)

    return np.dot(a, bb)


@njit(I_1D(I_2D))
def get_max_columnwise(xx: np.ndarray) -> np.ndarray:
    """Get the maximum from each column of a two-dimensional integers array.

    Parameters
    ----------
    xx : :class:`numpy:numpy.ndarray`
        A two-dimensional integer array.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        A one-dimensional integer array, each element contains the maximum
        of each column of the input array.

    Notes
    -----
    - This is to imitate a particular way of calling :func:`numpy:numpy.max`,
      that is, ``np.max(arr, axis=0)``. Such a call cannot be used inside
      a Numba JIT-ted function as Numba does not support passing the second
      argument.
    - The function is NJIT-ted with integers input/output; it can't be used
      with other types.
    """
    out = np.empty(xx.shape[1], dtype=INT_DTYPE)
    for j in range(xx.shape[1]):
        out[j] = np.max(xx[:, j])

    return out
