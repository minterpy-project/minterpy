"""
This module contains functions to verify relevant Minterpy quantities.
"""

from typing import Any, List, Optional, Sized, Tuple, Type, TypeVar, Union

import numpy as np
from _warnings import warn

from minterpy.global_settings import DEBUG, DEFAULT_DOMAIN, FLOAT_DTYPE, INT_DTYPE


def verify_domain(domain, spatial_dimension):
    """Building and verification of domains.

    This function builds a suitable domain as the cartesian product of a one-
    dimensional domain, or verifies the domain shape, of a multivariate domain is
    passed. If None is passed, the default domain is build from [-1,1].

    :param domain: Either one-dimensional domain ``(min,max)``, or a stack of domains for each domain with shape ``(spatial_dimension,2)``. If :class:`None` is passed, the ``DEFAULT_DOMAIN`` is repeated for each spatial dimentsion.
    :type domain: array_like, None
    :param spatial_dimension: Dimentsion of the domain space.
    :type spatial_dimension: int

    :return verified_domain: Stack of domains for each dimension with shape ``(spatial_dimension,2)``.
    :rtype: np.ndarray
    :raise ValueError: If no domain with the expected shape can be constructed from the input.

    """
    if domain is None:
        domain = np.repeat(DEFAULT_DOMAIN[:, np.newaxis], spatial_dimension, axis=1)
    domain = np.require(domain, dtype=FLOAT_DTYPE)
    if domain.ndim == 1:
        domain = np.repeat(domain[:, np.newaxis], spatial_dimension, axis=1)
    check_shape(domain, shape=(2, spatial_dimension))
    return domain


def check_type(obj: Any, expected_type: Type[Any]):
    """Check if the given input is of expected type.
    
    Parameters
    ----------
    obj : Any
        An instance to be checked.
    expected_type : Type[Any]
        The expected type (i.e., class).

    Raises
    ------
    TypeError
        If the given instance is not of the expected type.

    Notes
    -----
    - The purpose of this function is to unify type-checking procedure that
      raises an exception with a common message to avoid repetition across
      the codebase.

    Examples
    --------
    >>> check_type(1, int)  # 1 is an int
    >>> check_type(np.array([1, 2]), np.ndarray)
    >>> check_type("1.0", float)  # a string is not a float
    Traceback (most recent call last):
    ...
    TypeError: Expected <class 'float'>, but got <class 'str'> instead.
    """
    if not isinstance(obj, expected_type):
        raise TypeError(
            f"Expected {expected_type}, but got {type(obj)} instead.",
        )


def check_dtype(a: np.ndarray, expected_dtype):
    """Verify if the input array has the expected dtype.

    :param a: Array to be checked.
    :type a: np.ndarray

    :param expected_dtype: The dtype which is check against.
    :type expected_dtype: type

    :raise TypeError: if input array hasn't the expected dtype.

    .. todo::
        - use ``is not`` instead of ``!=``.

    """
    if a.dtype != expected_dtype:
        raise TypeError(
            f"input must be given as {expected_dtype} (encountered {a.dtype})"
        )


def check_dimensionality(
    xx: np.ndarray,
    dimensionality: Union[List[int], int],
) -> None:
    """Verify the dimensionality of a given array.

    Use this verification function when its expected dimensionality is known.

    Parameters
    ----------
    xx : np.ndarray
        A given array to verify.
    dimensionality : Union[List[int], int]
        The expected dimensionality (i.e., the number of dimensions)
        of the array. If given in a list then multiple dimensionality is
        allowed.

    Raises
    ------
    ValueError
        If the input array is not of the expected dimension.

    Examples
    --------
    >>> check_dimensionality(np.array([1, 2, 3]), dimensionality=1)
    >>> yy = np.array([
    ...     [1, 2, 3, 4],
    ...     [5, 6, 7, 8],
    ... ])
    >>> check_dimensionality(yy, dimensionality=2)
    >>> check_dimensionality(yy, dimensionality=1)  # Wrong dimensionality
    Traceback (most recent call last):
    ...
    ValueError: 1-D array is expected; got instead 2-D.
    >>> check_dimensionality(yy, dimensionality=[1, 2])
    """
    if isinstance(dimensionality, list):
        invalid = not any([xx.ndim == dim for dim in dimensionality])
    else:
        invalid = xx.ndim != dimensionality
    if invalid:
        raise ValueError(
            f"{dimensionality}-D array is expected; got instead {xx.ndim}-D."
        )


def check_shape(xx: np.ndarray, shape: Tuple[int, ...]):
    """Verify the shape of a given array.

    Use this verification function when its expected shape (given as a tuple)
    is known.

    Parameters
    ----------
    xx : np.ndarray
        A given array to verify.
    shape : Tuple[int, ...]
        The expected shape of the array.

    Raises
    ------
    ValueError
        If the input array is not of the expected shape.

    Examples
    --------
    >>> check_shape(np.array([1, 2, 3]), shape=(3, ))
    >>> yy = np.array([
    ...     [1, 2, 3, 4],
    ...     [5, 6, 7, 8],
    ... ])
    >>> check_shape(yy, shape=(2, 4))
    >>> check_shape(yy, shape=(1, 5))  # Wrong shape
    Traceback (most recent call last):
    ...
    ValueError: Array of shape (1, 5) is expected; got instead (2, 4).
    >>> check_shape(yy, shape=(2, 4, 1))  # Wrong dimensionality
    Traceback (most recent call last):
    ...
    ValueError: 3-D array is expected; got instead 2-D.
    """
    # Check dimensionality
    check_dimensionality(xx, dimensionality=len(shape))

    # Check shape
    if xx.shape != shape:
        raise ValueError(
            f"Array of shape {shape} is expected; got instead {xx.shape}."
        )


def check_values(xx: Union[int, float, np.ndarray], **kwargs):
    """Verify that the input has neither ``NaN`` nor ``inf`` values.

    Parameters
    ----------
    xx : Union[int, float, :class:`numpy:numpy.ndarray`]
        The scalar or array to be checked.
    **kwargs
        Keyword arguments with Boolean as values, if ``True`` then the invalid
        value is allowed. The keys are ``nan`` (check for ``NaN`` values),
        ``inf`` (check for ``inf`` values), ``zero`` (check for 0 values),
        and ``negative`` (check for negative values).
        If any of those set to ``True``,
        the given value will raise an exception.
        The default is ``NaN`` and ``inf`` values are not allowed, while
        zero or negative values are allowed.

    Raises
    ------
    ValueError
        If the scalar value or the given array contains any of the specified
        invalid values(e.g., ``NaN``, ``inf``, zero, or negative).

    Examples
    --------
    >>> check_values(10)  # valid
    >>> check_values(np.nan)  # Default, no nan
    Traceback (most recent call last):
    ...
    ValueError: Invalid value(s) (NaN, inf, negative, zero).
    >>> check_values(np.inf)  # Default, no inf
    Traceback (most recent call last):
    ...
    ValueError: Invalid value(s) (NaN, inf, negative, zero).
    >>> check_values(np.zeros((3, 2)), zero=False)  # No zero value is allowed
    Traceback (most recent call last):
    ...
    ValueError: Invalid value(s) (NaN, inf, negative, zero).
    >>> check_values(-10, negative=False)  # No negative value is allowed
    Traceback (most recent call last):
    ...
    ValueError: Invalid value(s) (NaN, inf, negative, zero).
    """
    # Parse keyword arguments
    kwargs = dict((key.lower(), val) for key, val in kwargs.items())
    nan = kwargs.get("nan", False)
    inf = kwargs.get("inf", False)
    zero = kwargs.get("zero", True)
    negative = kwargs.get("negative", True)

    # Check values
    is_nan = False if nan else np.any(np.isnan(xx))
    is_inf = False if inf else np.any(np.isinf(xx))
    is_zero = False if zero else np.any(xx == 0)
    is_negative = False if negative else np.any(xx < 0)

    if is_nan or is_inf or is_zero or is_negative:
        raise ValueError(
            "Invalid value(s) (NaN, inf, negative, zero)."
        )


DOMAIN_WARN_MSG2 = "the grid points must fit the interpolation domain [-1;1]^m."
DOMAIN_WARN_MSG = (
    "this may lead to unexpected behaviour, "
    "e.g. rank deficiencies in the regression matrices, etc. ."
)


def check_domain_fit(points: np.ndarray):
    """Checks weather a given array of points is properly formatted and spans the standard domain :math:`[-1,1]^m`.

    .. todo::
        - maybe remove the warnings.
        - generalise to custom ``internal_domain``

    :param points: array to be checked. Here ``m`` is the dimenstion of the domain and ``k`` is the number of points.
    :type points: np.ndarray, shape = (m, k)
    :raises ValueError: if the grid points do not fit into the domain :math:`[-1;1]^m`.
    :raises ValueError: if less than one point is passed.

    """
    # check first if the sample points are valid
    check_type(points, np.ndarray)
    check_values(points)
    # check weather the points lie outside of the domain
    sample_max = np.max(points, axis=1)
    if not np.allclose(np.maximum(sample_max, 1.0), 1.0):
        raise ValueError(DOMAIN_WARN_MSG2 + f"violated max: {sample_max}")
    sample_min = np.min(points, axis=1)
    if not np.allclose(np.minimum(sample_min, -1.0), -1.0):
        raise ValueError(DOMAIN_WARN_MSG2 + f"violated min: {sample_min}")
    check_dimensionality(points, dimensionality=2)
    nr_of_points, m = points.shape
    if nr_of_points == 0:
        raise ValueError("at least one point must be given")
    if nr_of_points == 1:
        return  # one point cannot span the domain
    if DEBUG:
        # check weather the points span the hole domain
        max_grid_val = np.max(sample_max)
        if not np.isclose(max_grid_val, 1.0):
            warn(
                f"the highest encountered value in the given points is {max_grid_val}  (expected 1.0). "
                + DOMAIN_WARN_MSG
            )
        min_grid_val = np.min(sample_min)
        if not np.isclose(min_grid_val, -1.0):
            warn(
                f"the smallest encountered value in the given points is {min_grid_val} (expected -1.0). "
                + DOMAIN_WARN_MSG
            )


def is_real_scalar(x: Union[int, float, np.integer, np.floating]) -> bool:
    """Check if a given value is a real scalar number.

    Parameters
    ----------
    x : Union[int, float, numpy.integer, numpy.floating]
        The variable to be checked.

    Returns
    -------
    bool
        ``True`` if the variable is a scalar (an ``int``, ``float``,
        `numpy.integer`, or `numpy.floating`), ``False`` otherwise.

    Examples
    --------
    >>> is_real_scalar(1)  # int
    True
    >>> is_real_scalar(10.0)  # float
    True
    >>> is_real_scalar(np.array([1])[0])  # numpy.int64
    True
    >>> is_real_scalar(np.array([1]))  # numpy.ndarray
    False
    >>> is_real_scalar(np.array([123.0])[0])  # numpy.float64
    True
    >>> is_real_scalar(1+5j)  # complex
    False
    """
    return isinstance(x, (int, float, np.integer, np.floating))


def shape_eval_output(eval_output: np.ndarray) -> np.ndarray:
    """Shape the output of polynomial evaluation according to the convention.

    Parameters
    ----------
    eval_output : :class:`numpy:numpy.ndarray`
        The output of polynomial evaluation as a one- or two-dimensional array
        to be shaped. The length of the array is ``N``, i.e., the number of
        query points.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The shaped array, a one-dimensional array for a polynomial with
        a single set of coefficients or a two-dimensional array for
        a polynomial with a multiple sets. The number of columns in the latter
        is the number of coefficient sets. Both have length of the number
        of query points.

    Notes
    ------
    - The output array is at least one-dimensional.

    Examples
    --------
    >>> shape_eval_output(np.array([[1.0], [2.0], [3.0], [4.0]]))
    array([1., 2., 3., 4.])
    >>> shape_eval_output(np.array([[1.0, 2.0, 3.0, 4.0]]))
    array([[1., 2., 3., 4.]])
    >>> shape_eval_output(np.array([[1.0, 2.0], [3.0, 4.0]]))
    array([[1., 2.],
           [3., 4.]])
    """
    # Handle case of single point evaluation with multiple coefficient sets
    if eval_output.ndim == 2:
        if eval_output.shape[1] == 1:
            # Has a single coefficient set (as a column vector)
            return np.atleast_1d(eval_output.squeeze())
        else:
            # Multiple coefficient sets (as a matrix)
            return eval_output

    return np.atleast_1d(eval_output)


# --- Verifications of Minterpy Value Objects
def verify_spatial_dimension(spatial_dimension: int) -> int:
    """Verify if the value of a given spatial dimension is valid.

    Parameters
    ----------
    spatial_dimension : int
        Spatial dimension to verify; the value of a spatial dimension must be
        strictly positive (> 0). ``spatial_dimension`` may not necessarily be
        an `int` but it must be a single whole number.

    Returns
    -------
    int
        Verified spatial dimension. If the input is not an `int`,
        the function does a type conversion to an `int` if possible.

    Raises
    ------
    TypeError
        If ``spatial_dimension`` is not of a correct type, i.e., its
        strict-positiveness cannot be verified or the conversion to `int`
        cannot be carried out.
    ValueError
        If ``spatial_dimension`` is, for example, not a positive
        or a whole number.

    Examples
    --------
    >>> verify_spatial_dimension(2)  # int
    2
    >>> verify_spatial_dimension(3.0)  # float but whole
    3
    >>> verify_spatial_dimension(np.array([1])[0])  # numpy.int64
    1
    """
    try:
        # Must be strictly positive
        check_values(spatial_dimension, negative=False, zero=False)

        # Other type than int may be acceptable if it's a whole number
        if spatial_dimension % 1 != 0:
            raise ValueError("Spatial dimension must be a whole number.")

        # Make sure that it's an int (whole number checked must come first!)
        spatial_dimension = int(spatial_dimension)

    except TypeError as err:
        custom_message = "Invalid type for spatial dimension!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = (
            f"{spatial_dimension} is invalid for spatial dimension!"
        )
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return spatial_dimension


def verify_poly_degree(poly_degree: int) -> int:
    """Verify if the value of a given polynomial degree is valid.

    Parameters
    ----------
    poly_degree : int
        Polynomial degree to verify; the value of a polynomial degree must be
        non-negative (>= 0). ``poly_degree`` may not necessarily be
        an `int` but it must be a single whole number.

    Returns
    -------
    int
        Verified polynomial degree. If the input is not an `int`,
        the function does a type conversion to an `int` if possible.

    Raises
    ------
    TypeError
        If ``poly_degree`` is not of a correct type, i.e., its
        non-negativeness cannot be verified or the conversion to `int`
        cannot be carried out.
    ValueError
        If ``poly_degree`` is, for example, not a positive
        or a whole number.

    Examples
    --------
    >>> verify_poly_degree(0)  # int
    0
    >>> verify_poly_degree(1.0)  # float but whole
    1
    >>> verify_poly_degree(np.array([2])[0])  # numpy.int64
    2
    """
    try:
        # Must be non-negative
        check_values(poly_degree, negative=False)

        # Other type than int may be acceptable if it's a whole number
        if poly_degree % 1 != 0:
            raise ValueError("Poly. degree must be a whole number.")

        # Make sure that it's an int (whole number checked must come first!)
        poly_degree = int(poly_degree)

    except TypeError as err:
        custom_message = "Invalid type for poly. degree!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = f"{poly_degree} is invalid for poly. degree! "
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return poly_degree


def verify_lp_degree(lp_degree: float) -> float:
    """Verify that the value of a given lp-degree is valid.

    Parameters
    ----------
    lp_degree : float
        A given :math:`p` of the :math:`l_p`-norm (i.e., :math:`l_p`-degree)
        to verify. The value of an ``lp_degree`` must be strictly positive, but
        may not necessarily be a `float`.

    Returns
    -------
    float
        Verified lp-degree value. If the input is not a `float`, the function
        does a type conversion to a `float` if possible.

    Raises
    ------
    TypeError
        If ``lp_degree`` is not of correct type, i.e., its strict-positiveness
        cannot be verified or the conversion to `float` cannot be carried
        out.
    ValueError
        If ``lp_degree`` is, for example, a non strictly positive value.

    Examples
    --------
    >>> verify_lp_degree(2.5)  # float
    2.5
    >>> verify_lp_degree(3)  # int
    3.0
    >>> verify_lp_degree(np.array([1])[0])  # numpy.int64
    1.0
    """
    try:
        # Must be strictly positive, infinity is allowed
        check_values(lp_degree, inf=True, negative=False, zero=False)

        # Make sure that it's a float
        lp_degree = float(lp_degree)

    except TypeError as err:
        custom_message = "Invalid type for lp-degree!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = (
            f"{lp_degree} is invalid for lp-degree (must be > 0)!"
        )
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return lp_degree


def verify_poly_coeffs(coeffs: np.ndarray, num_monomials: int) -> np.ndarray:
    """Verify that the given polynomial(s) coefficients are valid.

    Parameters
    ----------
    coeffs : :class:`numpy:numpy.ndarray`
        The polynomial coefficients to verify. Other container sequences
        may be accepted as long as it can be converted to a
        :class:`numpy:numpy.ndarray` of `numpy.float64`.
    num_monomials : int
        The number of monomials in the polynomials, used as the basis
        to verify the length of the coefficients.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        Verified polynomial coefficients. If the input is not of
        `numpy.float64`, the function does a type conversion
        to the type if possible. This type is expected by Numba functions.

    Raises
    ------
    TypeError
        If ``coeffs`` is not of correct type or the conversion to
        :class:`numpy:numpy.ndarray` or `numpy.float64`
        cannot be carried out.
    ValueError
        If ``coeffs`` contains inf or nan's or if the dimensionality of
        the coefficients is incorrect (not 1 or 2).

    Examples
    --------
    >>> verify_poly_coeffs(np.array([1, 2, 3]), 3)  # numpy.int64
    array([1., 2., 3.])
    >>> verify_poly_coeffs(np.array([10., 20.]), 2)  # numpy.float64
    array([10., 20.])
    >>> verify_poly_coeffs(np.array([[1., 2.], [3., 4.]]), 2)  # multi-dim
    array([[1., 2.],
           [3., 4.]])
    >>> verify_poly_coeffs(1.0, 1)  # a scalar
    array([1.])
    """
    try:
        # Convert the coefficients as a numpy.float64 (expected by Numba)
        coeffs = np.atleast_1d(np.array(coeffs)).astype(np.float64)

        # The dimension of the array (it may be one- or two-dimensional)
        check_dimensionality(coeffs, dimensionality=[1, 2])

        # The values must not contain NaN or inf
        check_values(coeffs, nan=False, inf=False, zero=True, negative=True)

        # The length must be the same as the number of monomials
        if len(coeffs) != num_monomials:
            raise ValueError(
                f"The number of coefficients ({len(coeffs)}) does not match "
                f"the number of monomials ({num_monomials})."
            )

        # A single set is stored as one-dimensional array
        if coeffs.ndim > 1 and coeffs.shape[1] == 1:
            coeffs = coeffs.reshape(-1)

    except TypeError as err:
        custom_message = "Invalid type for polynomial coefficients!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = "Invalid values in the polynomial coefficients!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return coeffs


def verify_poly_domain(
    domain: np.ndarray,
    spatial_dimension: int,
) -> np.ndarray:
    r"""Verify that the given polynomial domain is valid.

    Examples
    --------
    >>> verify_poly_domain(np.array([[1], [2]]), 1)  # integer array
    array([[1.],
           [2.]])
    >>> verify_poly_domain(np.array([[1, 2], [2, 3]]), 2)
    array([[1., 2.],
           [2., 3.]])
    >>> verify_poly_domain([3, 2], 1) # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ValueError: The upper bounds must be strictly larger than the lower
    bounds. Invalid values in the polynomial domain!
    """
    try:
        # The domain must be a NumPy ndarray
        domain = np.atleast_2d(np.array(domain)).astype(np.float64)
        if domain.shape[0] == 1:
            # Column array
            domain = domain.T

        # The dimension of the array must be two-dimensional
        check_dimensionality(domain, dimensionality=2)

        # The values must not contain inf
        check_values(domain, nan=False, inf=True, zero=True, negative=True)

        # The length must be two (lower and upper bounds)
        if domain.shape[0] != 2:
            raise ValueError(
                f"The domain is defined by {domain.shape[0]} numbers "
                "instead of by 2 (lower and upper bounds)."
            )

        # The number of columns must be the same as the dimension
        if domain.shape[1] != spatial_dimension:
            raise ValueError(
                f"The dimension of the domain ({domain.shape[1]}) does not "
                f"match the required dimension ({spatial_dimension})."
            )

        # The lower bounds must be smaller than the upper bounds
        if np.any(domain[1, :] - domain[0, :] <= 0):
            raise ValueError(
                "The upper bounds must be strictly larger than "
                "the lower bounds."
            )

    except TypeError as err:
        custom_message = "Invalid type for polynomial domain!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = "Invalid values in the polynomial domain!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return domain


def verify_query_points(xx: np.ndarray, spatial_dimension: int) -> np.ndarray:
    r"""Verify if the values of the query points for evaluation are valid.

    Parameters
    ----------
    xx : :class:`numpy:numpy.ndarray`
        A one- or two-dimensional array of query points at which a polynomial
        is evaluated. The length of the array is ``N``, i.e., the number
        of query points.
    spatial_dimension : int
        The spatial dimension of the polynomial (``m``).
        The shape of the query points array must be consistent with this.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        A two-dimensional array of ``numpy.float64`` with a length of ``N``
        and a number of columns of ``m``. If the dtype of the array is not of
        `numpy.float64`, the function does a type conversion if possible.

    Raises
    ------
    TypeError
        If ``xx`` is not of a correct type.
    ValueError
        If ``xx`` is, for example, has an inconsistent number of columns or
        it contains nan's.

    Examples
    --------
    >>> verify_query_points(1, 1)  # a scalar integer
    array([[1.]])
    >>> verify_query_points([3., 4., 5.], 1)  # a list
    array([[3.],
           [4.],
           [5.]])
    >>> verify_query_points([[3, 4, 5]], 3)  # a list of lists of integers
    array([[3., 4., 5.]])
    >>> verify_query_points(np.array([1., 2., 3.]), 1)  # 1 dimension
    array([[1.],
           [2.],
           [3.]])
    >>> verify_query_points(np.array([[1., 2.], [3., 4.]]), 2)  # 2 dimensions
    array([[1., 2.],
           [3., 4.]])
    >>> verify_query_points(np.array([1, 2, 3]), 1)  # integer
    array([[1.],
           [2.],
           [3.]])
    >>> verify_query_points(np.array(["a", "b"]), 1)
    Traceback (most recent call last):
    ...
    ValueError: could not convert string to float: 'a' Invalid values in query points array!

    Notes
    -----
    - The conversion to ``numpy.float64`` is required because some evaluation
      routines rely on Numba whose input types are pre-determined.
    """
    try:
        # Attempt to convert to a NumPy array of the correct dtype
        # NOTE: Use np.asarray instead of np.array to avoid unnecessary copy
        xx = np.atleast_1d(np.asarray(xx, dtype=FLOAT_DTYPE))

        # Check dimensionality
        check_dimensionality(xx, dimensionality=[1, 2])

        # Check spatial dimension
        if xx.ndim == 1:
            dim = 1
        else:
            dim = xx.shape[1]
        dim_is_consistent = dim == spatial_dimension
        if not dim_is_consistent:
            raise ValueError(
                "Inconsistent dimension of query points "
                f"(got {dim}, expected {spatial_dimension})."
            )

        # Check the values inside (allow inf but not nan)
        check_values(xx, nan=False, inf=True, zero=True, negative=True)

        # Make sure the array has a correct shape
        num_points = len(xx)
        xx = np.reshape(xx, (num_points, spatial_dimension))

    except TypeError as err:
        custom_message = "Invalid type for query points array!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = f"Invalid values in query points array!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return xx


def verify_poly_power(power: int) -> int:
    """Verify if the value of a given polynomial power is valid.

    This function verify the value of ``power`` in the expression
    ``poly**power``.

    Parameters
    ----------
    power : int
        Polynomial power to verify; the value of a polynomial power must be
        a scalar non-negative (>= 0). ``poly_power`` may not necessarily be
        an `int` but it must be a single whole number.

    Returns
    -------
    int
        Verified polynomial power. If the input is not an `int`,
        the function does a type conversion to an `int` if possible.

    Raises
    ------
    TypeError
        If ``poly_power`` is not of a correct type, e.g., it's not a real
        scalar, or its non-negativeness cannot be verified,
        or the conversion to `int` cannot be carried out.
    ValueError
        If ``poly_degree`` is, for example, not a positive
        or a whole number.

    Examples
    --------
    >>> verify_poly_power(0)  # int
    0
    >>> verify_poly_power(1.0)  # float but whole
    1
    >>> verify_poly_power(np.array([2])[0])  # numpy.int64
    2
    """
    try:
        # Must be a real scalar
        if not is_real_scalar(power):
            raise TypeError("Polynomial power must be a scalar.")

        # Must be positive
        check_values(power, negative=False)

        # Other type than int may be acceptable if it's a whole number
        if power % 1 != 0:
            raise ValueError("Polynomial power must be a whole number.")

        # Make sure that it's an int (whole number checked must come first!)
        power = int(power)

    except TypeError as err:
        custom_message = f"Invalid type for polynomial power (got {power})!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    except ValueError as err:
        custom_message = f"{power} is invalid for polynomial power!"
        err.args = _add_custom_exception_message(err.args, custom_message)
        raise err

    return power


def _add_custom_exception_message(
    exception_args: Tuple[str, ...],
    custom_message: str
) -> Tuple[str, ...]:
    """Prepend a custom message to an exception message.

    Parameters
    ----------
    exception_args : Tuple[str, ...]
        The arguments of the raised exception.
    custom_message : str
        The custom message to be prepended.

    Returns
    -------
    Tuple[str, ...]
        Modified exception arguments.
    """
    if not exception_args:
        arg = custom_message
    else:
        arg = f"{exception_args[0]} {custom_message}"
    exception_args = (arg,) + exception_args[1:]

    return exception_args


def dummy(*args, **kwargs) -> None:
    """A placeholder function to indicate a feature that is not supported.

    .. warning::
      This feature is not implemented yet!

    Raises
    ------
    NotImplementedError
        Any time this function or method is called.
    """
    raise NotImplementedError("This feature is not yet implemented!")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
