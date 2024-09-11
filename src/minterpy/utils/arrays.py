"""
This module contains common numerical routines that operate on NumPy arrays.
"""

from __future__ import annotations

from typing import Union

import numpy as np


def lp_norm(arr, p, axis=None, keepdims: bool = False):
    """Robust lp-norm function.

    Works essentially like ``numpy.linalg.norm``, but is numerically stable for big arguments.

    :param arr: Input array.
    :type arr: np.ndarray

    :param axis: If axis is an integer, it specifies the axis of x along which to compute the vector norms. If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed. If axis is None then either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The default is :class:`None`.
    :type axis: {None, int, 2-tuple of int}, optional

    :param keepdims: If this is set to True, the axes which are normed over are left in the result as dimensions with size one. With this option the result will broadcast correctly against the original ``arr``.
    :type keepdims: bool, optional
    """

    a = np.abs(arr).max()
    if a == 0.0:  # NOTE: avoid division by 0
        return 0.0
    return a * np.linalg.norm(arr / a, p, axis, keepdims)


def cartesian_product(*arrays: np.ndarray) -> np.ndarray:
    """
    Build the cartesian product of any number of 1D arrays.

    :param arrays: List of 1D array_like.
    :type arrays: list

    :return: Array of all combinations of elements of the input arrays (a cartesian product).
    :rtype: np.ndarray

    Examples
    --------
    >>> x = np.array([1,2,3])
    >>> y = np.array([4,5])
    >>> cartesian_product(x,y)
    array([[1, 4],
           [1, 5],
           [2, 4],
           [2, 5],
           [3, 4],
           [3, 5]])

    >>> s= np.array([0,1])
    >>> cartesian_product(s,s,s,s)
    array([[0, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0],
           [0, 0, 1, 1],
           [0, 1, 0, 0],
           [0, 1, 0, 1],
           [0, 1, 1, 0],
           [0, 1, 1, 1],
           [1, 0, 0, 0],
           [1, 0, 0, 1],
           [1, 0, 1, 0],
           [1, 0, 1, 1],
           [1, 1, 0, 0],
           [1, 1, 0, 1],
           [1, 1, 1, 0],
           [1, 1, 1, 1]])

    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def lp_sum(arr: np.ndarray, p: Union[float, int]) -> Union[float, int]:
    """
    Sum of powers, i.e. lp-norm to the lp-degree.

    :param arr: 2D-array to be lp-summed
    :type arr: np.ndarray

    :param p: Power for each element in the lp-sum
    :type p: Real

    :return: lp-sum over the last axis of the input array powered by the given power
    :rtype: np.ndarray

    Notes
    -----
    - equivalent to ```lp_norm(arr,p,axis=-1)**p``` but more stable then the implementation using `np.linalg.norm` (see `numpy #5697`_  for more informations)

    .. _numpy #5697:
        https://github.com/numpy/numpy/issues/5697
    """
    return np.sum(np.power(arr, p), axis=-1)


def make_coeffs_2d(coefficients: np.ndarray) -> np.ndarray:
    """Make coefficients array 2d.

    Parameters
    ----------
    coefficients: np.ndarray with coefficients

    Returns
    -------
    Returns a 2d array in the case of both single and multiple polynomials

    Notes
    -----
    This function is similar to np.atleast_2d, but adds the extra dimension differently.
    """

    coeff_shape = coefficients.shape
    if len(coeff_shape) == 1:  # 1D: a single polynomial
        coefficients = np.expand_dims(coefficients,-1)  # reshape to 2D

    return coefficients


def expand_dim(
    xx: np.ndarray,
    target_dim: int,
    new_values: np.ndarray = None,
) -> np.ndarray:
    """Expand the dimension of a given 2D array filled with given values.

    Parameters
    ----------
    xx : :class:`numpy:numpy.ndarray`
        Input array (exponents array or interpolating grid array) which will
        be expanded; it must be a two-dimensional array.
    target_dim : int
        The target dimension up to which the array will be expanded.
        The value must be larger than or equal to the dimension of the current
        array.
    new_values : :class:`numpy:numpy.ndarray`, optional
       The new values for the expanded dimensions; the values will be tiled
       to fill in the expanded dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        Exponents or grid array with expanded dimension (i.e., additional
        columns).

    Raises
    ------
    ValueError
        If the number of dimension of the input array is not equal to 2;
        if the target number of columns is less than the number of columns
        of the input array;
        or if the number of values for the new columns is inconsistent
        (not equal the number of rows of the input array).

    Notes
    -----
    - The term `dimension` here refers to dimensionality of exponents or
      interpolating grid; in other words, it refers to the number of columns
      of such arrays.

    Examples
    --------
    >>> array = np.array([[0, 0], [1, 0], [0, 1]])  # 2 columns / "dimensions"
    >>> expand_dim(array, 4)  # expand to 4 columns / "dimensions"
    array([[0, 0, 0, 0],
           [1, 0, 0, 0],
           [0, 1, 0, 0]])
    >>> expand_dim(array, 4, np.array([3, 2]))  # expand with tiled values
    array([[0, 0, 3, 2],
           [1, 0, 3, 2],
           [0, 1, 3, 2]])
    """
    # Check the dimension of the input array
    if xx.ndim != 2:
        raise ValueError(
            f"The exponent or grid array must be of dimension 2! "
            f"Instead got {xx.ndim}."
        )

    # Get the shape of the input array
    num_rows, num_columns = xx.shape

    # --- Dimension contraction (smaller target), raises an exception
    if target_dim < num_columns:
        # TODO maybe build a reduce fun. which removes dims where all exps 0
        raise ValueError(
            f"Can't expand the exponent or grid array dimension "
            f"from {num_columns} to {target_dim}."
        )

    # --- No dimension expansion (same target)
    if target_dim == num_columns:
        # Return the input array (identical)
        return xx

    # --- Dimension expansion
    diff_dim = target_dim - num_columns
    if new_values is None:
        new_values = np.zeros(
            (num_rows, diff_dim),
            dtype=xx.dtype
        )
    else:
        new_values = np.atleast_1d(new_values)
        if len(new_values) != diff_dim:
            raise ValueError(
                f"The given set of new values {new_values} does not have "
                f"enough elements to fill the extra columns! "
                f"<{diff_dim}> required, got <{len(new_values)}> instead."
            )

        # Tile the new values according to the shape of the input array
        new_values = np.require(
            np.tile(new_values, (num_rows, 1)), dtype=xx.dtype
        )

    return np.append(xx, new_values, axis=1)


def is_unique(xx: np.ndarray) -> bool:
    """Return ``True`` if the input array has unique values.

    Parameters
    ----------
    xx : :class:`numpy:numpy.ndarray`
        The one-dimensional array to be checked.

    Returns
    -------
    bool
        ``True`` if the values in the array are unique.

    Examples
    --------
    >>> is_unique(np.array([0, 1, 2, 3, 4, 5]))
    True
    >>> is_unique(np.array([0, 1, 2, 3, 4, 5, 5, 6]))
    False
    >>> is_unique(np.array([[0, 1, 2, 3, 4]])) # two-dimensional, squeeze-able
    True
    >>> is_unique(np.array([0.0]))
    True
    >>> is_unique(np.array([[1, 0], [1, 0]]))
    Traceback (most recent call last):
    ...
    ValueError: The input array must be one-dimensional
    """
    xx = np.atleast_1d(np.squeeze(xx))
    if xx.ndim > 1:
        raise ValueError("The input array must be one-dimensional")

    return len(np.unique(xx)) == len(xx)


if __name__ == "__main__":
    import doctest
    doctest.testmod()