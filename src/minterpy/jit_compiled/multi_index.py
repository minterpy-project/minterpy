"""
A module with JIT-compiled utility functions related to the multi-index set.
"""
import numpy as np

from numba import njit

from minterpy.global_settings import B_1D, B_DTYPE, I_2D, INT_DTYPE


@njit(I_2D(I_2D, I_2D), cache=True)
def cross_and_sum(
    indices_1: np.ndarray,
    indices_2: np.ndarray,
) -> np.ndarray:
    """Create a cross product of multi-indices and sum the pairs.

    Parameters
    ----------
    indices_1 : :class:`numpy:numpy.ndarray`
        First two-dimensional integers array of multi-indices.
    indices_2 : :class:`numpy:numpy.ndarray`
        Second two-dimensional integers array of multi-indices.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        An array that contains the pair-wise sums of cross-products between
        the two array.

    Notes
    -----
    - The two arrays must have the same number of columns.
    """
    # --- Create output array
    n_1 = len(indices_1)
    n_2 = len(indices_2)
    m = indices_1.shape[1]
    out = np.empty(shape=(n_1 * n_2, m), dtype=INT_DTYPE)

    # --- Loop over arrays, cross and sum them
    i = 0
    for index_1 in indices_1:
        for index_2 in indices_2:
            out[i] = index_1 + index_2
            i += 1

    return out


@njit(B_1D(I_2D), cache=True)
def unique_indices(indices: np.ndarray) -> np.ndarray:
    """Get the boolean mask of unique elements from an array of multi-indices.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Two-dimensional integer array of multi-indices to process.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        A boolean mask array that indicates unique elements from the input
        array.

    Notes
    -----
    - The input multi-indices must already be lexicographically sorted.
    - It turns out that getting the unique elements from a two-dimensional
      array is expensive, more than sorting lexicographically large array.
    """
    nr_indices = len(indices)

    # Output array must be initialized with False value
    out = np.zeros(nr_indices, dtype=B_DTYPE)

    # First index is always unique
    out[0] = True

    # Loop over indices
    for i in range(nr_indices - 1):
        if np.any(indices[i] != indices[i + 1]):
            out[i + 1] = True

    return out
