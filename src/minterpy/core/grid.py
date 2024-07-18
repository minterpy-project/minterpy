"""
Module for the generating points (which provides the unisolvent nodes).

How-To Guides
-------------

The relevant section of the :doc:`docs </how-to/grid/index>`
contains several how-to guides related to instances of the `Grid` class
demonstrating their usages and features.

----

"""
from copy import copy, deepcopy
from typing import Callable, Optional, Union

import numpy as np

from minterpy.global_settings import ARRAY, INT_DTYPE
from minterpy.gen_points import GENERATING_FUNCTIONS, gen_points_from_values

from minterpy.core.multi_index import MultiIndexSet
from minterpy.core.tree import MultiIndexTree
from minterpy.utils.arrays import is_unique
from minterpy.utils.verification import (
    check_type,
    check_values,
    check_dimensionality,
    check_domain_fit,
)

__all__ = ["Grid"]

# Default generating function
DEFAULT_FUN = "chebyshev"

# Type alias
GEN_FUNCTION = Callable[[int, int], np.ndarray]


def _gen_unisolvent_nodes(multi_index, generating_points):
    """
    .. todo::
        - document this function but ship it to utils first.
    """
    return np.take_along_axis(generating_points, multi_index.exponents, axis=0)


# TODO implement comparison operations based on multi index comparison operations and the generating values used
class Grid:
    """A class representing the nodes on which interpolating polynomials live.

    Instances of this class provide the data structure for the unisolvent
    nodes, i.e., points in a hypercube that uniquely determine
    a multi-dimensional interpolating polynomial
    (of a specified multi-index set).

    Parameters
    ----------
    multi_index : MultiIndexSet
        The multi-index set of exponents of multi-dimensional polynomials
        that the Grid should support.
    generating_function : Union[GEN_FUNCTION, str], optional
        The generating function to construct an array of generating points.
        One of the built-in generating functions may be selected via
        a string as a key to a dictionary.
        This parameter is optional; if neither this parameter nor
        ``generating_points`` is specified, the default generating function
        based on the Leja-ordered Chebyshev-Lobatto nodes is selected.
    generating_points : :class:`numpy:numpy.ndarray`, optional
        The generating points of the interpolation grid, a two-dimensional
        array of floats whose columns are the generating points
        per spatial dimension. The shape of the array is ``(n + 1, m)``
        where ``n`` is the maximum degree of all one-dimensional polynomials
        (i.e., the maximum exponent) and ``m`` is the spatial dimension.
        This parameter is optional. If not specified, the generating points
        are created from the default generating function. If specified,
        then the points must be consistent with any non-``None`` generating
        function.

    Notes
    -----
    - The ``Callable`` as a ``generating_function`` must accept as its
      arguments two integers, namely, the maximum exponent (``n``) of all
      of the multi-index set of polynomial exponents and the spatial dimension
      (``m``). Furthermore, it must return an array of shape ``(n + 1, m)``
      whose values are unique per column.
    - The multi-index set to construct a :class:`Grid` instance may not be
      downward-closed. However, building a :class:`.MultiIndexTree` used
      in the transformation between polynomials in the Newton and Lagrange
      bases requires a downward-closed multi-index set.
    - The notion of unisolvent nodes, strictly speaking, relies on the
      downward-closedness of the multi-index set. If the set is not
      downward-closed then unisolvency cannot be guaranteed.
    """
    def __init__(
        self,
        multi_index: MultiIndexSet,
        generating_function: Optional[Union[GEN_FUNCTION, str]] = None,
        generating_points: Optional[np.ndarray] = None,
    ):

        # --- Arguments processing

        # Process and assign the multi-index set argument
        self._multi_index = _process_multi_index(multi_index)

        # If generating_function and points not specified,
        # use the default generating function
        no_gen_function = generating_function is None
        no_gen_points = generating_points is None
        if no_gen_function and no_gen_points:
            generating_function = DEFAULT_FUN

        # Process and assign the generating function
        self._generating_function = _process_generating_function(
            generating_function
        )

        # Assign and verify the generating points argument
        if no_gen_points:
            generating_points = self._create_generating_points()
        else:
            # Create a copy to avoid accidental changes from the outside
            generating_points = generating_points.copy()
        self._generating_points = generating_points
        self._verify_generating_points()

        # --- Post-assignment verifications

        # Verify the maximum exponent
        self._verify_grid_max_exponent()

        # Verify generating function and points again if both are specified
        if not no_gen_function and not no_gen_points:
            self._verify_matching_gen_function_and_points()

        # --- Lazily-evaluated properties
        self._unisolvent_nodes = None
        self._tree = None

    # --- Factory methods
    @classmethod
    def from_degree(
        cls,
        spatial_dimension: int,
        poly_degree: int,
        lp_degree: float,
        generating_function: Optional[Union[GEN_FUNCTION, str]] = None,
        generating_points: Optional[np.ndarray] = None,
    ):
        r"""Create an instance of Grid with a complete multi-index set.

        A complete multi-index set denoted by :math:`A_{m, n, p}` contains
        all the exponents
        :math:`\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_m) \in \mathbb{N}^m`
        such that the :math:`l_p`-norm
        :math:`|| \boldsymbol{\alpha} ||_p \leq n`, where:

        - :math:`m`: the spatial dimension
        - :math:`n`: the polynomial degree
        - :math:`p`: the `l_p`-degree

        Parameters
        ----------
        spatial_dimension : int
            Spatial dimension of the multi-index set (:math:`m`); the value of
            ``spatial_dimension`` must be a positive integer (:math:`m > 0`).
        poly_degree : int
            Polynomial degree of the multi-index set (:math:`n`); the value of
            ``poly_degree`` must be a non-negative integer (:math:`n \geq 0`).
        lp_degree : float
            :math:`p` of the :math:`l_p`-norm (i.e., the :math:`l_p`-degree)
            that is used to define the multi-index set. The value of
            ``lp_degree`` must be a positive float (:math:`p > 0`).
        generating_function : Union[GEN_FUNCTION, str], optional
            The generating function to construct an array of generating points.
            One of the built-in generating functions may be selected via
            a string key.
            This parameter is optional; if neither this parameter nor
            ``generating_points`` is specified, the default generating function
            based on the Leja-ordered Chebyshev-Lobatto nodes is selected.
        generating_points : :class:`numpy:numpy.ndarray`, optional
            The generating points of the interpolation grid, a two-dimensional
            array of floats whose columns are the generating points
            per spatial dimension. The shape of the array is ``(n + 1, m)``
            where ``n`` is the maximum degree of all one-dimensional
            polynomials (i.e., the maximum exponent) and ``m`` is the spatial
            dimension. This parameter is optional. If not specified,
            the generating points are created from the default generating
            function. If specified, then the points must be consistent
            with any non-``None`` generating function.

        Returns
        -------
        Grid
            A new instance of the `Grid` class initialized with a complete
            multi-index set (:math:`A_{m, n, p}`) and with the given
            generating function and generating points.
        """
        # Create a complete multi-index set
        mi = MultiIndexSet.from_degree(
            spatial_dimension,
            poly_degree,
            lp_degree,
        )

        # Create an instance of Grid
        return cls(mi, generating_function, generating_points)

    @classmethod
    def from_function(
        cls,
        multi_index: MultiIndexSet,
        generating_function: Union[GEN_FUNCTION, str],
    ) -> "Grid":
        """Create an instance of Grid with a given generating function.

        Parameters
        ----------
        multi_index : MultiIndexSet
            The multi-index set of exponents of multi-dimensional polynomials
            that the Grid should support.
        generating_function: Union[GEN_FUNCTION, str]
            The generating function to construct an array of generating points.
            The function should accept as its arguments two integers, namely,
            the maximum exponent of the multi-index set of exponents and
            the spatial dimension and returns an array of shape ``(n + 1, m)``
            where ``n`` is the one-dimensional polynomial degree
            and ``m`` is the spatial dimension.
            Alternatively, a string as a key to dictionary of built-in
            generating functions may be specified.

        Returns
        -------
        Grid
            A new instance of the `Grid` class initialized with the given
            generating function.
        """
        return cls(multi_index, generating_function=generating_function)

    @classmethod
    def from_points(
        cls,
        multi_index: MultiIndexSet,
        generating_points: np.ndarray,
    ) -> "Grid":
        """Create an instance of Grid from an array of generating points.

        Parameters
        ----------
        multi_index : MultiIndexSet
            The multi-index set of exponents of multi-dimensional polynomials
            that the Grid should support.
        generating_points : :class:`numpy:numpy.ndarray`
            The generating points of the interpolation grid, a two-dimensional
            array of floats whose columns are the generating points
            per spatial dimension. The shape of the array is ``(n + 1, m)``
            where ``n`` is the maximum polynomial degree in all dimensions
            (i.e., the maximum exponent) and ``m`` is the spatial dimension.
            The values in each column must be unique.

        Returns
        -------
        Grid
            A new instance of the `Grid` class initialized
            with the given multi-index set and generating points.
        """
        return cls(multi_index, generating_points=generating_points)

    @classmethod
    def from_value_set(
        cls,
        multi_index: MultiIndexSet,
        generating_values: np.ndarray,
    ):
        """Create an instance of Grid from an array of generating values.

        A set of generating values is one-dimensional interpolation points.

        Parameters
        ----------
        multi_index : MultiIndexSet
            The multi-index set of polynomial exponents that defines the Grid.
            The set, in turn, defines the polynomials the Grid can support.
        generating_values : :class:`numpy:numpy.ndarray`
            The one-dimensional generating points of the interpolation grid,
            a one-dimensional array of floats of length ``(n + 1, )``
            where ``n`` is the maximum exponent of the multi-index set.
            The values in the array must be unique.

        Returns
        -------
        Grid
            A new instance of the `Grid` class initialized
            with the given multi-index set and generating values.

        Notes
        -----
        - An array of generating points are created based on tiling
          the generating values to the required spatial dimension.
        """
        # Create the generating points from the generating values
        spatial_dimension = multi_index.spatial_dimension
        if generating_values.ndim == 2 and generating_values.shape[1] > 1:
            raise ValueError(
                "Only one set of generating values can be provided; "
                f"Got {generating_values.shape[1]} instead"
            )
        # Make sure it is one-dimensional array
        if generating_values.ndim >= 1:
            generating_values = generating_values.reshape(-1)
        generating_points = gen_points_from_values(
            generating_values,
            spatial_dimension,
        )

        return cls(multi_index, generating_points=generating_points)

    # --- Properties
    @property
    def multi_index(self) -> MultiIndexSet:
        """The multi-index set of exponents associated with the Grid.

        The multi-index set of a Grid indicates the largest interpolating
        polynomial the Grid can support.

        Returns
        -------
        MultiIndexSet
            A multi-index set of polynomial exponents associated with the Grid.
        """
        return self._multi_index

    @property
    def generating_function(self) -> Optional[GEN_FUNCTION]:
        """The generating function of the interpolation Grid.

        Returns
        -------
        Optional[GEN_FUNCTION]
            The generating function of the interpolation Grid which is used
            to construct the array of generating points.

        Notes
        -----
        - If the generating function is ``None`` then the Grid may not be
          manipulated that results in a grid of a higher degree or dimension.
        """
        return self._generating_function

    @property
    def generating_points(self) -> np.ndarray:
        """The generating points of the interpolation Grid.

        The generating points of the interpolation grid are one two main
        ingredients of constructing unisolvent nodes (the other being the
        multi-index set of exponents).

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            A two-dimensional array of floats whose columns are the
            generating points per spatial dimension. The shape of the array
            is ``(n + 1, m)`` where ``n`` is the maximum exponent of the
            multi-index set of exponents and ``m`` is the spatial dimension.
        """
        return self._generating_points

    @property
    def max_exponent(self) -> int:
        """The maximum exponent of the interpolation grid.

        Returns
        -------
        int
            The maximum exponent of the interpolation grid is the maximum
            degree of any one-dimensional polynomials the grid can support.
        """
        return len(self.generating_points) - 1

    @property
    def unisolvent_nodes(self) -> np.ndarray:
        """The array of unisolvent nodes.

        For a definition of unisolvent nodes, see
        :ref:`fundamentals/introduction:The notion of unisolvence` in the docs.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The unisolvent nodes as a two-dimensional array of floats.
            The shape of the array is ``(N, m)`` where ``N`` is the number of
            elements in the multi-index set and ``m`` is the spatial dimension.
        """
        if self._unisolvent_nodes is None:  # lazy evaluation
            self._unisolvent_nodes = _gen_unisolvent_nodes(
                self.multi_index, self.generating_points
            )
        return self._unisolvent_nodes

    @property
    def spatial_dimension(self):
        """Dimension of the domain space.

        This attribute is propagated from ``multi_index``.

        :return: The dimension of the domain space, where the polynomial will live on.
        :rtype: int

        """
        return self.multi_index.spatial_dimension

    @property
    def tree(self):
        """The used :class:`MultiIndexTree`.

        :return: The :class:`MultiIndexTree` which is connected to this :class:`Grid` instance.
        :rtype: MultiIndexTree

        .. todo::
            - is this really necessary?

        """
        if self._tree is None:  # lazy evaluation
            self._tree = MultiIndexTree(self)
        return self._tree

    @property
    def has_generating_function(self) -> bool:
        """Return ``True`` if the instance has a generating function.

        Returns
        -------
        bool
            ``True`` if the instance has a generating function assigned to it,
            and ``False`` otherwise.
        """
        return self.generating_function is not None

    # --- Instance methods
    def _new_instance_if_necessary(self, multi_indices_new: MultiIndexSet) -> "Grid":
        """Constructs new grid instance only if the multi indices have changed

        :param new_indices: :class:`MultiIndexSet` instance for the ``grid``, needs to be a subset of the current ``multi_index``.
        :type new_indices: MultiIndexSet

        :return: Same :class:`Grid` instance if ``multi_index`` stays the same, otherwise new polynomial instance with the new ``multi_index``.
        :rtype: Grid
        """
        multi_indices_old = self.multi_index
        # TODO: Following MR !69, the MultiIndexSet will always be a new
        # instance, revise this for consistency.
        if multi_indices_new is multi_indices_old:
            return self
        # construct new:
        return self.__class__(multi_indices_new, self.generating_points)

    def make_complete(self) -> "Grid":
        """completes the multi index within this :class:`Grid` instance.

        :return: completed :class:`Grid` instance
        :rtype: Grid

        Notes
        -----
        - This is required e.g. for building a multi index tree (DDS scheme)!


        """
        multi_indices_new = self.multi_index.make_complete(inplace=False)
        return self._new_instance_if_necessary(multi_indices_new)

    def add_points(self, exponents: ARRAY) -> "Grid":
        """Extend ``grid`` and ``multi_index``

        Adds points ``grid`` and exponents to ``multi_index`` related to a given set of additional exponents.

        :param exponents: Array of exponents added.
        :type exponents: np.ndarray

        :return: New ``grid`` with the added exponents.
        :rtype: Grid

        .. todo::
            - this is boilerplate, since similar code appears in :class:`MultivariatePolynomialSingleABC`.
        """
        exponents = np.require(exponents, dtype=INT_DTYPE)
        if np.max(exponents) > self.max_exponent:
            # TODO 'enlarge' the grid, increase the degree, ATTENTION:
            raise ValueError(
                f"trying to add point with exponent {np.max(exponents)} "
                f"but the grid is only of degree {self.max_exponent}"
            )

        multi_indices_new = self.multi_index.add_exponents(exponents)
        return self._new_instance_if_necessary(multi_indices_new)

    def expand_dim(self, target_dimension: Union[int, "Grid"]) -> "Grid":
        """Expand the dimension of the Grid.

        Parameters
        ----------
        target_dimension : Union[Grid, int]
            The new spatial dimension. It must be larger than or equal
            to the current dimension of the Grid. Alternatively,
            another instance of Grid whose dimension is higher can also
            be specified as a target dimension.


        Returns
        -------
        Grid
            The Grid with expanded dimension.

        Raises
        ------
        ValueError
            If an instance is expanded to a dimension that cannot be supported
            either by the available generating function or generating points.
            If the target dimension is a `Grid`, the exception is raised
            when there are inconsistencies in either generating function
            or points.
        """
        # Expand the dimension to the target Grid instance
        if isinstance(target_dimension, Grid):
            return _expand_dim_to_target_grid(self, target_dimension)

        # Expand to the target dimension
        return _expand_dim_to_target_dim(self, target_dimension)

    # --- Special methods: Copies
    # copying
    def __copy__(self):
        """Creates of a shallow copy.

        This function is called, if one uses the top-level function ``copy()`` on an instance of this class.

        :return: The copy of the current instance.
        :rtype: Grid

        See Also
        --------
        copy.copy
            copy operator form the python standard library.
        """
        return self.__class__(
            self._multi_index,
            generating_function=self._generating_function,
            generating_points=self._generating_points,
        )

    def __deepcopy__(self, mem):
        """Creates of a deepcopy.

        This function is called, if one uses the top-level function ``deepcopy()`` on an instance of this class.

        :return: The deepcopy of the current instance.
        :rtype: Grid

        See Also
        --------
        copy.deepcopy
            copy operator form the python standard library.

        """
        return self.__class__(
            deepcopy(self._multi_index),
            generating_function=deepcopy(self._generating_function),
            generating_points=deepcopy(self._generating_points),
        )

    # --- Dunder method: Callable instance
    def __call__(self, fun: Callable, *args, **kwargs) -> np.ndarray:
        """Evaluate the given function on the unisolvent nodes of the grid.

        Parameters
        ----------
        fun : Callable
            The given function to evaluate. The function must accept as its
            first argument a two-dimensional array and return as its output
            an array of the same length as the input array.
        *args
            Additional positional arguments passed to the given function.
        **kwargs
            Additional keyword arguments passed to the given function.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The values of the given function evaluated on the unisolvent nodes
            (i.e., the coefficients of the polynomial in the Lagrange basis).
        """
        # No need for type checking the argument; rely on Python to raise any
        # exceptions when a problematic 'fun' is called on the nodes.
        return fun(self.unisolvent_nodes, *args, **kwargs)

    # --- Dunder methods: Rich comparison
    def __eq__(self, other: "Grid") -> bool:
        """Compare two instances of Grid for exact equality in value.

        Two instances of :class:`Grid` class is equal in value if and only if:

        - both the underlying multi-index sets are equal, and
        - both the generating points are equal, and
        - both the generating functions are equal.

        Parameters
        ----------
        other : Grid
            An instance of :class:`Grid` that is to be compared with
            the current instance.

        Returns
        -------
        bool
            ``True`` if the two instances are equal in value,
            ``False`` otherwise.
        """
        # Checks are from the cheapest to the most expensive for early exit
        # (multi-index equality check is the most expensive one)

        # Check for consistent type
        if not isinstance(other, Grid):
            return False

        # Generating function equality
        if self.generating_function != other.generating_function:
            return False

        # Generating points equality
        if not np.array_equal(self.generating_points, other.generating_points):
            return False

        # Multi-index set equality
        if self.multi_index != other.multi_index:
            return False

        return True

    # --- Private internal methods: not to be called directly from outside
    def _create_generating_points(self) -> np.ndarray:
        """Construct generating points from the generating function.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The generating points of the interpolation grid, a two-dimensional
            array of floats whose columns are the generating points
            per spatial dimension. The shape of the array is ``(n + 1, m)``
            where ``n`` is the maximum exponent of the multi-index set of
            exponents and ``m`` is the spatial dimension.
        """
        multi_index = self._multi_index
        poly_degree = multi_index.max_exponent
        spatial_dimension = multi_index.spatial_dimension

        generating_function = self._generating_function

        return generating_function(poly_degree, spatial_dimension)

    def _verify_generating_points(self):
        """Verify if the generating points are valid.

        Raises
        ------
        ValueError
            If the points are not of the correct dimension, contains
            NaN's or inf's, do not fit the standard domain, or the values
            per column are not unique.
        TypeError
            If the points are not given in the correct type.
        """
        # Check array dimension
        check_dimensionality(self._generating_points, dimensionality=2)
        # No NaN's and inf's
        check_values(self._generating_points)
        # Check domain fit
        check_domain_fit(self._generating_points)
        # Check dimension against spatial dimension of the set
        gen_points_dim = self._generating_points.shape[1]
        if gen_points_dim < self.spatial_dimension:
            raise ValueError(
                "Dimension mismatch between the generating points "
                f"({gen_points_dim}) and the multi-index set "
                f"({self.spatial_dimension})"
            )
        # Check the uniqueness of values column-wise
        are_unique = all([is_unique(xx) for xx in self._generating_points.T])
        if not are_unique:
            raise ValueError(
                "One or more columns of the generating points are not unique"
            )

    def _verify_matching_gen_function_and_points(self):
        """Verify if the generation function and points match.

        Raises
        ------
        ValueError
            If the generating function generates different points than
            then ones that are provided.
        """
        gen_points = self.generating_function(
            self.max_exponent,
            self.spatial_dimension,
        )

        if not np.array_equal(gen_points, self.generating_points):
            raise ValueError(
                "The generating function generates points that are "
                "inconsistent with the generating points"
            )

    def _verify_grid_max_exponent(self):
        """Verify if the Grid max. exponent is consistent with the multi-index.

        Raises
        ------
        ValueError
            If the maximum exponent of the Grid is smaller than the maximum
            exponent of the multi-index set of polynomial exponents.

        Notes
        -----
        - While it perhaps makes sense to store the maximum exponent of each
          dimension as the instance property instead of the maximum over all
          dimensions, this has no use because the generating points have
          a uniform length in every dimension. For instance, if the maximum
          exponent per dimension of a two-dimensional polynomial are
          ``[5, 3]``, the stored generating points remain ``(5, 2)`` instead of
          two arrays having lengths of ``5`` and ``3``, respectively.
        """
        # The maximum exponent in any dimension of the multi-index set
        # indicates the largest degree of one-dimensional polynomial
        # the grid needs to support.
        max_exponent_multi_index = self.multi_index.max_exponent

        # Both must be consistent; "smaller" multi-index may be contained
        # in a larger grid, but not the other way around.
        if max_exponent_multi_index > self.max_exponent:
            raise ValueError(
                f"A grid of a maximum exponent {self.max_exponent} "
                "cannot consist of multi-indices with a maximum exponent "
                f"of {max_exponent_multi_index}"
            )


# --- Internal helper functions
def _process_multi_index(multi_index: MultiIndexSet) -> MultiIndexSet:
    """Process the MultiIndexSet given as an argument to Grid constructor.

    Parameters
    ----------
    multi_index : MultiIndexSet
        The multi-index set as input argument to the Grid constructor to be
        processed.

    Returns
    -------
    MultiIndexSet
        The same instance of :class:`MultiIndexSet` if processing does not
        raise any exceptions.

    Raises
    ------
    TypeError
        If the argument is not an instance of :class:`MultiIndexSet`.
    ValueError
        If the argument is an empty instance of :class:`MultiIndexSet`.
    """
    check_type(multi_index, MultiIndexSet, "The multi-index set")

    # MultiIndexSet for a Grid cannot be an empty set
    if len(multi_index) == 0:
        raise ValueError("MultiIndexSet must not be empty!")

    return multi_index


def _process_generating_function(
    generating_function: Optional[Union[GEN_FUNCTION, str]],
) -> Optional[GEN_FUNCTION]:
    """Process the generating function given as argument to Grid constructor.

    Parameters
    ----------
    generating_function : Union[GEN_FUNCTION, str], optional
        The generating function to be processed, either ``None``,
        a dictionary key as string for selecting from the built-in functions,
        or a callable.

    Returns
    -------
    GEN_FUNCTION, optional
        The generating function as a callable or ``None`` if not specified.
    """
    # None is specified
    if generating_function is None:
        return generating_function

    # Get the built-in generating function
    if isinstance(generating_function, str):
        return GENERATING_FUNCTIONS[generating_function]

    if callable(generating_function):
        return generating_function

    raise TypeError(
        f"The generating function {generating_function} is not callable"
    )


def _expand_dim_to_target_dim(
    origin_grid: "Grid",
    target_dimension: int,
) -> "Grid":
    """Expand the dimension of a given Grid to a target dimension.

    Parameters
    ----------
    origin_grid : Grid
        The `Grid` instance whose dimension is to be expanded.
    target_dimension : Grid
        The target dimension; must be equal to or larger than the current
        dimension of the `Grid` instance.

    Returns
    -------
    Grid
        The grid with an expanded dimension.

    Raises
    ------
    ValueError
        If the target dimension cannot be accommodated by the available
        generating points.
    """
    # Expand the dimension of the multi-index set
    mi_expanded = origin_grid.multi_index.expand_dim(target_dimension)

    # Check if a generating function is available
    if origin_grid.has_generating_function:
        return origin_grid.__class__.from_function(
            mi_expanded,
            origin_grid.generating_function,
        )

    # Check if the available generating points can accommodate higher dimension
    gen_points_dim = origin_grid.generating_points.shape[1]
    if gen_points_dim >= target_dimension:
        return origin_grid.__class__.from_points(
            mi_expanded,
            origin_grid.generating_points,
        )

    raise ValueError(
        f"The available dimension of the generating points ({gen_points_dim} "
        f"can't accommodate target dimension ({target_dimension})"
    )


def _expand_dim_to_target_grid(
    origin_grid: "Grid",
    target_grid: "Grid",
) -> "Grid":
    """Expand the dimension of a given Grid to the dimension of another.

    Parameters
    ----------
    origin_grid : Grid
        The grid whose dimension is to be expanded.
    target_grid : Grid
        The grid whose dimension is the base for expansion.

    Returns
    -------
    Grid
        The grid with an expanded dimension.

    Raises
    ------
    ValueError
        If the generating functions are not compatible (when available) or
        if the generating points are not compatible.
    """
    # Create expanded multi-index set
    target_dim = target_grid.spatial_dimension
    mi_expanded = origin_grid.multi_index.expand_dim(target_dim)

    if _have_gen_functions(origin_grid, target_grid):
        # Check if the Grid instances have compatible generating functions
        if _have_compatible_gen_functions(origin_grid, target_grid):
            # The functions are compatible
            return origin_grid.__class__.from_function(
                mi_expanded,
                target_grid.generating_function,
            )
        else:
            raise ValueError(
                "Grid has an inconsistent generating function with target Grid"
            )

    # Check if the Grid instances have compatible generating points
    if _have_compatible_gen_points(origin_grid, target_grid):
        # Get the largest generating points from the two
        gen_points = _get_larger_gen_points(origin_grid, target_grid)
        return origin_grid.__class__.from_points(
            mi_expanded,
            gen_points,
        )

    # Points are inconsistent
    raise ValueError(
        "Grid has inconsistent generating points with the target Grid"
    )


def _have_gen_functions(*grids) -> bool:
    """Check if a sequence of Grid instances all have generating function."""
    return all(grd.has_generating_function for grd in grids)


def _have_compatible_gen_functions(grid_1: "Grid", grid_2: "Grid") -> bool:
    """Check if two grids have compatible generating functions.

    Parameters
    ----------
    grid_1 : Grid
        First instance of `Grid` to compare.
    grid_2 : Grid
        Second instance of `Grid` to compare.
    """
    # There is no way in Python to check for equality to what functions do
    return (
        grid_1.has_generating_function
        and grid_2.has_generating_function
        and grid_1.generating_function == grid_2.generating_function
    )


def _have_compatible_gen_points(grid_1: "Grid", grid_2: "Grid") -> bool:
    """Check if two grids have compatible generating points.

    Parameters
    ----------
    grid_1 : Grid
        First `Grid` instance to compare.
    grid_2 : Grid
        Second `Grid` instance to compare.

    Returns
    -------
    bool
        ``True`` if all generating points in the common spatial dimension
        of the two `Grid` instances are equal; ``False`` otherwise.
    """
    dim_1 = grid_1.spatial_dimension
    dim_2 = grid_2.spatial_dimension
    dim = np.min([dim_1, dim_2])

    gen_points_1 = grid_1.generating_points[:, :dim]
    gen_points_2 = grid_2.generating_points[:, :dim]

    return np.array_equal(gen_points_1, gen_points_2)


def _get_larger_gen_points(grid_1: "Grid", grid_2: "Grid") -> np.ndarray:
    """Get the larger array of generating points from two Grid instances.

    Parameters
    ----------
    grid_1 : Grid
        First `Grid` instance to check.
    grid_2 : Grid
        Second `Grid` instance to check.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The generating points with the larger number of columns.

    Notes
    -----
    - It is assumed that the generating points are consistent (i.e.,
      equal up to the common dimension/columns).
    """
    dim_1 = grid_1.generating_points.shape[1]
    dim_2 = grid_2.generating_points.shape[1]
    grids = [grid_1, grid_2]
    idx = np.argmax([dim_1, dim_2])

    return grids[idx].generating_points
