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
