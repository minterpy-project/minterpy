"""
A module with JIT-compiled utility functions to process and manipulate
multi-index set of exponents.

Notes
-----
- All multi-index sets of exponents are always assumed to be two-dimensional
  non-negative integer arrays. There is no explicit check of this conditions
  in the following functions.
"""
import numpy as np

from numba import njit, void
from minterpy.global_settings import (
    B_1D,
    B_DTYPE,
    B_TYPE,
    I_1D,
    I_2D,
    INT,
    INT_DTYPE,
    NOT_FOUND,
)


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


@njit(B_TYPE(I_1D, I_1D), cache=True)
def is_lex_smaller_or_equal(index_1: np.ndarray, index_2: np.ndarray) -> bool:
    """Check if an index is lexicographically smaller than or equal to another.

    Parameters
    ----------
    index_1 : :class:`numpy:numpy.ndarray`
        A given multi-index, a one-dimensional array of length ``m``.
    index_2 : :class:`numpy:numpy.ndarray`
        Another multi-index, a one dimensional array of length ``m``.

    Returns
    -------
    bool
        Return `True` if ``index_1 <= index_2`` lexicographically,
        otherwise `False`.

    Notes
    -----
    - By default, Numba disables the bound-checking for performance reason.
      Therefore, if the two input arrays are of inconsistent shapes, no
      exception will be raised and the results cannot be trusted.

    Examples
    --------
    >>> my_index_1 = np.array([1, 2, 3])  # "Reference index"
    >>> my_index_2 = np.array([1, 2, 3])  # Equal
    >>> is_lex_smaller_or_equal(my_index_1, my_index_2)
    True
    >>> my_index_3 = np.array([2, 4, 5])  # Larger
    >>> is_lex_smaller_or_equal(my_index_1, my_index_3)
    True
    >>> my_index_4 = np.array([0, 3, 2])  # Smaller
    >>> is_lex_smaller_or_equal(my_index_1, my_index_4)
    False
    """
    spatial_dimension = len(index_1)
    # lexicographic: Iterate backward from the highest dimension
    for m in range(spatial_dimension - 1, -1, -1):
        if index_1[m] > index_2[m]:
            # index_1 is lexicographically larger
            return False

        if index_1[m] < index_2[m]:
            # index_1 is Lexicographically smaller
            return True

    # index_1 is lexicographically equal
    return True


@njit(B_TYPE(I_2D), cache=True)
def is_lex_sorted(indices: np.ndarray) -> bool:
    """Check if an array of multi-indices is lexicographically sorted.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of multi-indices, a two-dimensional non-negative integer array
        of shape ``(N, m)``, where ``N`` is the number of multi-indices
        and ``m`` is the number of spatial dimensions.

    Returns
    -------
    bool
        ``True`` if the multi-indices is lexicographically sorted, and
        ``False`` otherwise

    Notes
    -----
    - If there are any duplicate entries (between rows),
      an array of multi-indices does not have a lexicographical ordering.

    Examples
    --------
    >>> my_indices = np.array([[0, 2, 0]])  # single entry
    >>> is_lex_sorted(my_indices)
    True
    >>> my_indices = np.array([[0, 0], [1, 0], [0, 2]])  # already sorted
    >>> is_lex_sorted(my_indices)
    True
    >>> my_indices = np.array([[1, 0], [0, 0], [0, 2]])  # unsorted
    >>> is_lex_sorted(my_indices)
    False
    >>> my_indices = np.array([[0, 0], [2, 0], [2, 0]])  # duplicate entries
    >>> is_lex_sorted(my_indices)
    False
    """
    nr_indices = indices.shape[0]

    # --- Single entry is always lexicographically ordered
    if nr_indices <= 1:
        return True

    # --- Loop over the multi-indices and find any unsorted entry or duplicate
    index_1 = indices[0, :]
    for n in range(1, nr_indices):
        index_2 = indices[n, :]

        if is_lex_smaller_or_equal(index_2, index_1):
            # Unsorted entry or duplicates
            return False

        index_1 = index_2

    return True


@njit(INT(I_2D, I_1D), cache=True)
def search_lex_sorted(indices: np.ndarray, index: np.ndarray) -> int:
    """Find the position of a given entry within an array of multi-indices.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of lexicographically sorted multi-indices, a two-dimensional
        non-negative integer array of shape ``(N, m)``,
        where ``N`` is the number of multi-indices and ``m`` is the number
        of spatial dimensions.
    index : :class:`numpy:numpy.ndarray`
        Multi-index entry to check in ``indices``. The element is represented
        by a one-dimensional array of length ``m``, where ``m`` is the number
        of spatial dimensions.

    Returns
    -------
    int
        If ``index`` is present in ``indices``, its position in ``indices``
        is returned (the row number). Otherwise, a global constant
        ``NOT_FOUND`` is returned instead.

    Notes
    -----
    - ``indices`` must be lexicographically sorted.
    - This function is a binary search implementation that exploits
      a lexicographically sorted array of multi-indices.
      The time complexity of the implementation is :math:`O(m\log{N})`.
    - By Minterpy convention, duplicate entries are not allowed in
      a lexicographically sorted multi-indices. However, having duplicate
      entries won't stop the search. In that case, the search returns
      the position of the first match but cannot guarantee which one is that
      from the duplicates.

    Examples
    --------
    >>> my_indices = np.array([
    ... [0, 0, 0],  # 0
    ... [1, 0, 0],  # 1
    ... [2, 0, 0],  # 2
    ... [0, 0, 1],  # 3
    ... ])
    >>> my_index_1 = np.array([2, 0, 0])  # is present in my_indices
    >>> search_lex_sorted(my_indices, my_index_1)
    2
    >>> my_index_2 = np.array([0, 1, 0])  # is not present in my_indices
    >>> search_lex_sorted(my_indices, my_index_2)
    -1
    """
    nr_indices = indices.shape[0]
    if nr_indices == 0:
        # Zero-length multi-indices has no entry
        return NOT_FOUND

    # Initialize the search
    out = NOT_FOUND
    low = 0
    high = nr_indices - 1

    # Start the binary search
    while low <= high:

        mid = (high + low) // 2

        if is_lex_smaller_or_equal(indices[mid], index):
            # NOTE: Equality must be checked here because the function
            #       `is_lex_smaller_or_equal()` cannot check just for smaller.
            if is_lex_smaller_or_equal(index, indices[mid]):
                return mid

            low = mid + 1

        else:
            high = mid - 1

    return out


@njit(B_TYPE(I_2D, I_1D), cache=True)
def is_index_contained(indices: np.ndarray, index: np.ndarray) -> bool:
    """Check if a multi-index entry is present in a set of multi-indices.

    Parameters
    ----------
    indices : :class:`numpy:numpy.ndarray`
        Array of lexicographically sorted multi-indices, a two-dimensional
        non-negative integer array of shape ``(N, m)``,
        where ``N`` is the number of multi-indices and ``m`` is the number
        of spatial dimensions.
    index : :class:`numpy:numpy.ndarray`
        Multi-index entry to check in the set. The element is represented
        by a one-dimensional array of length ``m``,
        where ``m`` is the number of spatial dimensions.

    Returns
    -------
    bool
        ``True`` if the entry ``index`` is contained in the set ``indices``
        and ``False`` otherwise.

    Notes
    -----
    - The implementation is based on the binary search and therefore
      ``indices`` must be lexicographically sorted.

    Examples
    --------
    >>> my_indices = np.array([
    ... [0, 0, 0],  # 0
    ... [1, 0, 0],  # 1
    ... [2, 0, 0],  # 2
    ... [0, 0, 1],  # 3
    ... ])
    >>> is_index_contained(my_indices, np.array([1, 0, 0]))  # is present
    True
    >>> is_index_contained(my_indices, np.array([0, 1, 2]))  # is not present
    False
    """
    return search_lex_sorted(indices, index) != NOT_FOUND


@njit(B_TYPE(I_2D, I_2D), cache=True)
def all_indices_are_contained(subset_indices: np.ndarray, indices: np.ndarray) -> bool:
    """Checks if a set of indices is a subset of (or equal to) another set of indices.

    :param subset_indices: one set of multi indices.
    :param indices: another set of multi indices.
    :return: ``True`` if ``subset_indices`` is a subset or equal to ``indices``, ``False`` otherwise.

    Notes
    -----
    Exploits the lexicographical order of the indices to abort early -> not testing all indices.
    """
    nr_exp, dim = indices.shape
    nr_exp_subset, dim_subset = subset_indices.shape
    if nr_exp == 0 or nr_exp_subset == 0:
        raise ValueError("empty index set")
    if dim != dim_subset:
        raise ValueError("dimensions do not match.")
    if nr_exp < nr_exp_subset:
        return False

    # return True when all candidate indices are contained
    match_idx = -1
    for i in range(nr_exp_subset):
        candidate_index = subset_indices[i, :]
        indices2search = indices[match_idx + 1 :, :]  # start from the next one
        match_idx = search_lex_sorted(indices2search, candidate_index)
        if match_idx == NOT_FOUND:
            return False
    return True


@njit(void(I_2D, I_2D, I_1D), cache=True)
def fill_match_positions(larger_idx_set, smaller_idx_set, positions):
    """Finds matching positions (array indices) for multi index entries in two multi indices.

    :param larger_idx_set: the larger set of multi indices
    :param smaller_idx_set: the smaller set of multi indices
    :param positions: an array with positions (array indices) of multi index entries in `larger_idx_set`` that matches
                      each of the entry in ``smaller_idx_set``.

    """
    search_pos = -1
    nr_exp_smaller, spatial_dimension = smaller_idx_set.shape
    for i in range(nr_exp_smaller):
        idx1 = smaller_idx_set[i, :]
        while 1:
            search_pos += 1
            idx2 = larger_idx_set[search_pos, :]
            if is_lex_smaller_or_equal(idx1, idx2) and is_lex_smaller_or_equal(idx2, idx1):
                # NOTE: testing for equality directly is faster, but only in the case of equality (<- rare!)
                #   most of the times the index won't be smaller and the check can be performed with fewer comparisons
                positions[i] = search_pos
                break


if __name__ == "__main__":
    import doctest
    doctest.testmod()