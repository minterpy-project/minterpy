"""
This module contains the implementation of the `Grid` class.

The `Grid` class represents the interpolation grid on which interpolating
polynomials live.

Background information
======================

An interpolation grid is defined by its multi-index set, generating points,
and generating function. The generating function, when provided,
defines the generating points, which are the one-dimensional interpolation
nodes in each dimension. The multi-index set, together with
the generating points, specifies the unisolvent nodes---these are the points
where the function to be interpolated is evaluated.

More detailed background information can be found in
:doc:`/fundamentals/interpolation-at-unisolvent-nodes`.

Generating Points Convention
-----------------------------

The generating points are stored as a two-dimensional array of shape
``(n + 1, m)``:

- **Columns** represent spatial dimensions: each column contains the
  one-dimensional interpolation nodes for that specific dimension
- **Rows** represent polynomial degrees: row ``i`` contains nodes for
  polynomial degree ``i`` (from degree 0 to degree ``n``)

**Important**: Currently, generating points must be defined in the normalized
domain :math:`[-1, 1]^m`. When a custom user domain is specified,
the Grid handles transformation automatically during function evaluation.

Example structure for a 2D grid with maximum degree 3::

    generating_points = [
        [x0_dim1, x0_dim2],  # Degree 0 nodes
        [x1_dim1, x1_dim2],  # Degree 1 nodes
        [x2_dim1, x2_dim2],  # Degree 2 nodes
        [x3_dim1, x3_dim2],  # Degree 3 nodes
    ]
    # All values must be in [-1, 1]

The generating function, when provided, produces this array structure.
Different generating function (e.g., Chebyshev-Lobatto, equidistant, Leja)
creates different node distributions within :math:`[-1, 1]`, each with distinct
approximation properties.

How-To Guides
=============

The relevant section of the :doc:`docs </how-to/grid/index>`
contains several how-to guides related to instances of the `Grid` class
demonstrating their usages and features.

----

"""
from copy import copy, deepcopy
from typing import Callable, Optional, Union, Tuple

import numpy as np

from minterpy.global_settings import ARRAY, INT_DTYPE
from minterpy.gen_points import GENERATING_FUNCTIONS, gen_points_from_values

from minterpy.core.domain import Domain
from minterpy.core.multi_index import MultiIndexSet
from minterpy.core.tree import MultiIndexTree
from minterpy.utils.arrays import is_unique
from minterpy.utils.verification import (
    check_type,
    check_values,
    check_dimensionality,
)

__all__ = ["Grid"]

# Default generating function
DEFAULT_FUN = "chebyshev"

# Type alias
GEN_FUNCTION = Callable[[int, int], np.ndarray]


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
    domain : Domain, optional
        The domain of the interpolation grid. This parameter is optional.
        If not specified, a normalized domain in the hypercube of
        :math:`[-1, 1]^m` is created.

    Notes
    -----
    - The ``Callable`` as a ``generating_function`` must accept as its
      arguments two integers, namely, the maximum exponent (``n``) of all
      of the multi-index set of polynomial exponents and the spatial dimension
      (``m``). Furthermore, it must return an array of shape ``(n + 1, m)``
      whose values are unique per column.
    - Generating points array has shape ``(n + 1, m)`` where columns
      represent spatial dimensions and rows represent polynomial degrees.
      All values must be in the normalized domain :math:`[-1, 1]^m`.
      Different generating functions create different node distributions within
      this normalized space.
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
        domain: Optional[Domain] = None,
    ):

        # --- Arguments processing

        # Process and assign the multi-index set argument
        self._multi_index = _process_multi_index(multi_index)

        # Process and assign the domain
        if domain is None:
            domain = Domain.identity(self.multi_index.spatial_dimension)

        self._domain = _process_domain(domain, multi_index)

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
        # Note: Domain is already processed
        if no_gen_points:
            generating_points = self._create_generating_points()
        else:
            # Create a copy to avoid accidental changes from the outside
            generating_points = generating_points.copy()
        self._generating_points = self._verify_generating_points(
            generating_points
        )

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
        domain: Optional[Domain] = None,
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
        domain : Domain, optional
            The domain of the interpolation grid. This parameter is optional.
            If not specified, a normalized domain in the hypercube of
            :math:`[-1, 1]^m` is created.

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
        return cls(mi, generating_function, generating_points, domain=domain)

    @classmethod
    def from_function(
        cls,
        multi_index: MultiIndexSet,
        generating_function: Union[GEN_FUNCTION, str],
        domain: Optional[Domain] = None,
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
        domain : Domain, optional
            The domain of the interpolation grid. This parameter is optional.
            If not specified, a normalized domain in the hypercube of
            :math:`[-1, 1]^m` is created.

        Returns
        -------
        Grid
            A new instance of the `Grid` class initialized with the given
            generating function.
        """
        return cls(
            multi_index,
            generating_function=generating_function,
            domain=domain,
        )

    @classmethod
    def from_points(
        cls,
        multi_index: MultiIndexSet,
        generating_points: np.ndarray,
        domain: Optional[Domain] = None,
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
        domain : Domain, optional
            The domain of the interpolation grid. This parameter is optional.
            If not specified, a normalized domain in the hypercube of
            :math:`[-1, 1]^m` is created.

        Returns
        -------
        Grid
            A new instance of the `Grid` class initialized
            with the given multi-index set and generating points.
        """
        return cls(
            multi_index,
            generating_points=generating_points,
            domain=domain,
        )

    @classmethod
    def from_value_set(
        cls,
        multi_index: MultiIndexSet,
        generating_values: np.ndarray,
        domain: Optional[Domain] = None,
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
        domain : Domain, optional
            The domain of the interpolation grid. This parameter is optional.
            If not specified, a normalized domain in the hypercube of
            :math:`[-1, 1]^m` is created.

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

        return cls(
            multi_index,
            generating_points=generating_points,
            domain=domain,
        )

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
    def domain(self) -> Domain:
        """The domain associated with the Grid.

        The domain represents the rectangular bounds of the Grid.

        Returns
        -------
        Domain
            The domain of the interpolation grid.
        """
        return self._domain

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
        """The array of unisolvent nodes in the normalized domain.

        The unisolvent nodes are provided in the normalized domain
        :math:`[-1, 1]^m`.

        For a definition of unisolvent nodes,
        see :doc:`/fundamentals/unisolvence` in the docs.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The unisolvent nodes in [-1, 1]^m as a two-dimensional array
            of floats. The shape of the array is ``(N, m)``
            where ``N`` is the number of elements in the multi-index set
            and ``m`` is the spatial dimension.

        Notes
        -----
        - When evaluating functions via ``grid(func)``,
          the nodes are automatically transformed to the user-defined domain.
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
    def is_complete(self) -> bool:
        """Return ``True`` if the instance has a complete multi-index set.

        Returns
        -------
        bool
            ``True`` if the underlying multi-index set of the instance
            is a complete index set and ``False`` otherwise.
        """
        return self.multi_index.is_complete

    @property
    def is_downward_closed(self) -> bool:
        """Return ``True`` if the instance has a downward-closed multi-indices.

        Returns
        -------
        bool
            ``True`` if the underlying multi-index set of the instance
            is a downward-closed set and ``False`` otherwise.
        """
        return self.multi_index.is_downward_closed

    # --- Instance methods
    def add_exponents(self, exponents: np.ndarray) -> "Grid":
        """Add a set of exponents to the underlying multi-index set.

        Parameters
        ----------
        exponents : `numpy:numpy.ndarray`
            Array of integers to be added. A single element of multi-index set
            may be specified as a one-dimensional array with the length
            of the spatial dimension. Multiple elements must be specified
            as a two-dimensional array with the spatial dimension as
            the number of columns

        Returns
        -------
        Grid
            A new instance of `Grid` with the updated multi-index set.

        Notes
        -----
        - The set of exponents are added lexicographically.
        """
        # Add the set of exponents
        mi_added = self.multi_index.add_exponents(exponents)

        return self._new_instance(mi_added)

    def expand_dim(self, target: Union[int, "Grid"]) -> "Grid":
        """Expand the dimension of the Grid to a target dimension or Grid.

        This method creates a new Grid with a higher spatial dimension by
        expanding both the underlying multi-index set and the domain.
        The expansion can target either a specific integer dimension
        or match the dimension of another (compatible) Grid instance.

        Parameters
        ----------
        target : Union[Grid, int]
            The target for for dimension expansion:

            - If int: The new spatial dimension (must be >= current dimension).
              For non-normalized domains, expansion to an integer is not
              allowed.
            - If Grid: Another Grid instance whose dimension to match.
              The grids must have compatible generating functions or points.

        Returns
        -------
        Grid
            A new Grid instance with expanded dimension, containing:

            - Expanded multi-index set
            - Expanded domain
            - Compatible generating function or points

        Raises
        ------
        ValueError
            If the target dimension is smaller than the current dimension,
            if expanding a non-normalized domain to an integer dimension,
            if the generating points cannot accommodate the target dimension,
            or if the Grid is not compatible with the target Grid.
        """
        # Expand the dimension to the target Grid instance
        if isinstance(target, Grid):
           return self._expand_to_grid(target)

        return self._expand_to_dimension(target)

    def has_compatible_gen_function(self, other: "Grid") -> bool:
        """Check if two grids have a compatible generating function.

        Two grids have a compatible generating function if each has
        a generating function and they are an identical function.

        Parameters
        ----------
        other : Grid
            Another Grid instance to check compatibility with.

        Returns
        -------
        bool
            ``True`` if the generating functions are compatible,
            ``False`` otherwise.
        """
        if self._has_generating_function and other._has_generating_function:
            # Check if the grid instances have compatible generating functions
            if self.generating_function == other.generating_function:
                return True

        return False

    def has_compatible_gen_points(self, other: "Grid") -> bool:
        """Check if two grids have compatible generating points.

        Two grids have compatible generating points if their generating point
        arrays match in the common dimensions (columns) and common degrees
        (rows). Compatibility is checked by comparing the overlapping subarray.

        Parameters
        ----------
        other : Grid
            Another Grid instance to check compatibility with.

        Returns
        -------
        bool
            ``True`` if the generating points are compatible in their common
            dimensions and degrees, ``False`` otherwise.

        Notes
        -----
        - Compatibility is checked only for the intersection of dimensions
          and degrees
        - For example, a grid with 3D generating points can be compatible with
          a 2D grid if their first 2 dimensions match; or a grid with degree 5
          can be compatible with degree 3 if the first 4 rows (degrees 0-3)
          match
        """
        # Find common dimensions and degrees to compare
        common_dim = min(self.spatial_dimension, other.spatial_dimension)
        common_deg = min(
            self.generating_points.shape[0],
            other.generating_points.shape[0]
        )

        # Extract overlapping subarrays
        gen_points_self = self.generating_points[:common_deg, :common_dim]
        gen_points_other = other.generating_points[:common_deg, :common_dim]

        # Check if they match
        return np.array_equal(gen_points_self, gen_points_other)

    def is_compatible(self, other: "Grid") -> bool:
        """Check if two instances of Grid are compatible.

        Two instances of Grid are compatible if they have:

        - The same generating function (when both have one), OR
        - Compatible generating points (matching values in common dimensions,
          i.e., columns and common degrees, i.e., rows)


        Parameters
        ----------
        other : Grid
            Another Grid instance to check compatibility with.

        Returns
        -------
        bool
            ``True`` if the generating data is compatible, ``False`` otherwise.

        Notes
        -----
        - This method checks ONLY the compatibility of the underlying
          generating functions or points of the two instances of Grid.
        """
        return (
            self.has_compatible_gen_function(other) or
            self.has_compatible_gen_points(other)
        )

    def make_complete(self) -> "Grid":
        """Complete the underlying multi-index set of the `Grid` instance.

        Returns
        -------
        Grid
            A new instance of `Grid` whose underlying multi-index set is
            a complete set with respect to the spatial dimension, polynomial
            degree, and :math:`l_p`-degree.

        Notes
        -----
        - Calling the function always returns a new instance. If the index-set
          is already complete, a deep copy of the current instance
          is returned.
        """
        if self.is_complete:
            # This is a deep copy
            return deepcopy(self)

        # Complete the index set -> New instance
        mi_complete = self.multi_index.make_complete()

        return self._new_instance(mi_complete)

    def make_downward_closed(self) -> "Grid":
        """Make the underlying multi-index set downward-closed.

        Returns
        -------
        Grid
            A new instance of `Grid` whose underlying multi-index set is
            a downward-closed set.
        Notes
        -----
        - Calling the function always returns a new instance. If the index-set
          is already downward-closed, a deep copy of the current instance
          is returned.
        """
        if self.is_downward_closed:
            # This is a deep copy
            return deepcopy(self)

        # Make the index set downward-closed -> New instance
        mi_downward_closed = self.multi_index.make_downward_closed()

        return self._new_instance(mi_downward_closed)

    # --- Special methods: Copies
    # copying
    def __copy__(self):
        """Creates a shallow copy of the instance.

        This function is called when using the top-level function ``copy()``
        on an instance of this class.

        Returns
        -------
        Grid
            A shallow copy of the current instance.

        See Also
        --------
        copy.copy
            Copy operator from the Python standard library.
        """
        return self.__class__(
            self._multi_index,
            generating_function=self._generating_function,
            generating_points=self._generating_points,
            domain=self._domain,
        )

    def __deepcopy__(self, mem):
        """Create of a deepcopy.

        This function is called if one uses the top-level function
        ``deepcopy()`` on an instance of this class.

        Returns
        -------
        Grid
            A deepcopy of the current instance where the underlying
            multi-index set, the domain, and the generating points
            are deepcopied.

        See Also
        --------
        copy.deepcopy
            copy function from the Python standard library.
        """
        # Create a new instance with a deep-copied multi-index set and domain
        multi_index = deepcopy(self._multi_index, mem)
        domain = deepcopy(self._domain, mem)

        return self._new_instance(multi_index, domain)

    # --- Dunder method: Callable instance
    def __call__(self, fun: Callable, *args, **kwargs) -> np.ndarray:
        """Evaluate a function on the unisolvent nodes in the given domain.

        The function is evaluated at the unisolvent nodes transformed from
        the canonical domain :math:`[-1, 1]^m` to the user-defined domain.
        This allows users to define functions in their natural coordinate
        system without manually handling domain transformations.

        Parameters
        ----------
        fun : Callable
            The function to evaluate. Must accept as its first argument a
            two-dimensional array of shape ``(N, m)`` and return an array of
            length ``N``, where ``N`` is the number of unisolvent nodes
            and ``m`` is the spatial dimension.
        *args
            Additional positional arguments passed to the function.
        **kwargs
            Additional keyword arguments passed to the function.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The function values evaluated at the unisolvent nodes in the
            user domain. These values correspond to the coefficients of
            the polynomial in the Lagrange basis.

        Notes
        -----
        - The unisolvent nodes are automatically transformed from
          :math:`[-1, 1]^m` to the given domain before evaluation.
        - If the domain is normalized (:math:`[-1, 1]^m`),
          there no transformation takes place.
        """
        # No need for type checking the argument; rely on Python to raise any
        # exceptions when problematic 'fun' is called on the nodes.
        if self.domain.is_identity:
            xx = self.unisolvent_nodes
        else:
            xx = self.domain.map_from_internal(self.unisolvent_nodes)

        return fun(xx, *args, **kwargs)

    # --- Dunder methods: Rich comparison
    def __eq__(self, other: "Grid") -> bool:
        """Compare two instances of Grid for exact equality in value.

        Two instances of :class:`Grid` class is equal in value if and only if:

        - both the underlying multi-index sets are equal, and
        - both the generating points are equal, and
        - both the generating functions are equal, and
        - both the underlying domains are equal.

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

        # Domain equality
        if self.domain != other.domain:
            return False

        # Generating points equality
        if not np.array_equal(self.generating_points, other.generating_points):
            return False

        # Multi-index set equality
        if self.multi_index != other.multi_index:
            return False

        return True

    # --- Dunder methods: Arithmetics
    def __mul__(self, other: "Grid") -> "Grid":
        """Multiply two instances of Grid via the ``*`` operator.

        Multiplying two instances of Grid creates a product grid with
        a multi-index set that is the product of the underlying sets
        of the two operands, and a domain that is the union of the two
        underlying domains.

        Parameters
        ----------
        other : Grid
            The second Grid operand for multiplication.

        Returns
        -------
        Grid
            The product grid with:

            - Product of the two multi-index sets
            - Union of the two domains
            - Compatible generating function or points

        Notes
        -----
        - This operation provides the infrastructure for polynomial
          multiplication: if polynomial ``p1`` is defined on ``grid1``
          and ``p2`` on ``grid2``, then their product ``(p1 * p2)``
          is naturally defined on ``(grid1 * grid2)``.
        """
        # Multiply the underlying multi-index sets
        mi_product = self.multi_index * other.multi_index
        domain_union = self.domain | other.domain

        #return self.merge(other, mi_product)
        return self._combine_with(other, mi_product, domain_union)

    def __or__(self, other: "Grid") -> "Grid":
        """Combine two instances of Grid via the ``|`` operator.

        Combining two instances of Grid creates a union grid with a multi-index
        set that is the union of the underlying sets of the two operands,
        and a domain that is the union of the two underlying domains.

        Parameters
        ----------
        other : Grid
            The second Grid operand for union.

        Returns
        -------
        Grid
            The union grid with:

            - Union of the two multi-index sets
            - Union of the two domains
            - Compatible generating function or points

        Notes
        -----
        - This operation provides the infrastructure for polynomial addition:
          if polynomial ``p1`` is defined on ``grid1`` and ``p2`` on ``grid2``,
          then their sum ``(p1 + p2)`` is naturally defined
          on ``(grid1 | grid2)``.
        """
        # Add (union) the underlying multi-index sets
        mi_union = self.multi_index | other.multi_index
        domain_union = self.domain | other.domain

        return self._combine_with(other, mi_union, domain_union)

    # --- Private internal methods: not to be called directly from outside
    def _combine_with(
        self,
        other: "Grid",
        multi_index: MultiIndexSet,
        domain: Domain,
    ) -> "Grid":
        """Combine two compatible grids with specified multi-index and domain.

        This is a low-level helper used by grid operations, e.g., ``__mul__``,
        ``__or__``. The multi-index set and domain are already computed by the
        caller; this method picks the compatible generating data and constructs
        a new instance of Grid.

        Parameters
        ----------
        other : Grid
            The other grid to combine with
        multi_index : MultiIndexSet
            The multi-index for the result
        domain : Domain
            The domain for the result

        Returns
        -------
        Grid
            New grid constructed with the given multi-index and domain
        """
        # Get the generating data
        gen_fun, gen_points  = self._pick_generating_data(other)

        if gen_fun is None:
            return Grid.from_points(multi_index, gen_points, domain)

        return Grid.from_function(multi_index, gen_fun, domain)

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

    def _expand_to_grid(self, target: "Grid") -> "Grid":
        """Expand the dimension of the Grid to match that of a target Grid.

        This method expands the dimension of the current Grid instance to
        match the spatial dimension of the target Grid. Both the underlying
        multi-index set and domain are expanded, and the grids must have
        compatible generating functions or points.

        Parameters
        ----------
        target : Grid
            The target Grid whose dimension to match. Must have compatible
            generating functions or points with the current Grid instance.

        Returns
        -------
        Grid
            A new instance of Grid with:

            - Dimension matching the target Grid
            - Expanded multi-index set
            - Expanded domain
            - Compatible generating function or points for both grids
        """
        # Expand the domain
        domain_expanded = self.domain.expand_dim(target.domain)

        # Expand the multi-index set
        target_dimension = target.spatial_dimension
        mi_expanded = self.multi_index.expand_dim(target_dimension)

        # Combine with the target Grid and validates compatibility
        return self._combine_with(target, mi_expanded, domain_expanded)

    def _expand_to_dimension(self, target: int) -> "Grid":
        """Expand the dimension of the Grid to a target dimension.

        This method expands the dimension of the current Grid instance to
        the specified dimension (given as integer).
        The domain must be normalized to allow expansion, as the new dimension
        is inferred to have the same normalized bounds.

        Parameters
        ----------
        target : int
            The target dimension to expand the Grid to. Must be greater than
            or equal to the current dimension.

        Returns
        -------
        Grid
            A new instance of Grid with:

            - Dimension equal to the target dimension
            - Expanded multi-index set
            - Expanded domain
            - Same generating function or points as the current instance
        """
        # Expand the domain
        domain_expanded = self.domain.expand_dim(target)

        # Expand the multi-index set
        mi_expanded = self.multi_index.expand_dim(target)

        # Construct a new instance with the expanded components
        if self._has_generating_function:
            return self.__class__.from_function(
                mi_expanded, self.generating_function, domain_expanded,
            )

        return self.__class__.from_points(
            mi_expanded,
            self.generating_points,
            domain_expanded,
        )

    @property
    def _has_generating_function(self) -> bool:
        """Return ``True`` if the instance has a generating function.

        Returns
        -------
        bool
            ``True`` if the instance has a generating function assigned to it,
            and ``False`` otherwise.
        """
        return self.generating_function is not None

    def _verify_generating_points(
        self,
        generating_points: np.ndarray,
    ) -> np.ndarray:
        """Validate generating points.

        Parameters
        ----------
        generating_points : np.ndarray
            The given generating points to validate, a 2D array of
            shape ``(n + 1, m)`` where ``n`` is the maximum degree of the
            the grid in any dimension and ``m`` is the maximum spatial
            dimension of the grid.

        Returns
        -------
        np.ndarray
            Validated generating points.

        Raises
        ------
        ValueError
            If the points are not of the correct dimension, contain
            NaN's or inf's, do not match with the dimension of the grid,
            the values per column are not unique, or the points are not
            within the internal domain.
        TypeError
            If the points are not given in the correct type.

        Notes
        -----
        - Generating points may contain more columns (i.e., spatial dimension)
          than the grid itself (as defined by the dimension of the multi-index
          set). If there's no generating function, this indicates the maximum
          dimension the current grid can be expanded to.
        """
        # --- Type and structure checks
        check_type(generating_points, np.ndarray)
        check_dimensionality(generating_points, dimensionality=2)
        check_values(generating_points)  # No NaN's and inf's

        # --- Dimension check
        gen_points_dim = generating_points.shape[1]
        if gen_points_dim < self.spatial_dimension:
            raise ValueError(
                "Dimension mismatch between generating points "
                f"({gen_points_dim}) and the grid ({self.spatial_dimension})"
            )

        # --- Uniqueness check (column-wise)
        if not all(is_unique(col) for col in generating_points.T):
            raise ValueError(
                "One or more columns of the generating points are not unique"
            )

        # --- Internal domain containment check
        gen_points_ = generating_points[:, :self.spatial_dimension]
        if not np.all(self.domain.contains(gen_points_, internal=True)):
            raise ValueError(
                "Generating points are not contained in the internal domain"
            )

        return generating_points

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

    def _new_instance(
        self,
        multi_index: MultiIndexSet,
        domain: Optional[Domain] = None,
    ) -> "Grid":
        """Construct a new grid instance with a new multi-index set and domain.

        Parameters
        ----------
        multi_index : MultiIndexSet
            The multi-index set of the new instance.
        domain : Domain, optional
            The domain of the new instance. If not specified, the domain of
            the current grid will be used.

        Returns
        -------
        Grid
            A new instance of `Grid` with the given multi-index set.

        Notes
        -----
        - The new instance will have the same underlying generating function
          and generating points.
        - If the new multi-index set cannot be accommodated either by
          the generating function or the generating points, an exception
          will be raised.
        """
        if domain is None:
            domain = self.domain

        if self._has_generating_function:
            return self.__class__.from_function(
                multi_index,
                self._generating_function,
                domain=domain,
            )

        return self.__class__.from_points(
            multi_index,
            self._generating_points,
            domain=domain,
        )

    def _pick_generating_data(
        self,
        other: "Grid",
    ) -> Tuple[Optional[GEN_FUNCTION], Optional[np.ndarray]]:
        """Pick generating data from two compatible grids.

        This private method validates that the two grids have compatible
        generating functions or generating points, then selects the appropriate
        generating data to use for constructing a combined grid.
        If both grids have compatible generating functions,
        the generating function; otherwise, the larger set of generating points
        is used.

        Parameters
        ----------
        other : Grid
            Another instance of Grid to pick compatible generating data from.

        Returns
        -------
        Tuple[Optional[GEN_FUNCTION], Optional[np.ndarray]]
            A tuple containing either (Only one element of the tuple will be
            non-None):

            - (generating_function, None) if both grids have compatible
              functions
            - (None, generating_points) if grids have compatible points.

        Raises
        ------
        ValueError
            If the grids are not compatible.

        Notes
        -----
        - This is a low-level helper method used internally by grid combination
          operations (__mul__, __or__, expand_dim). Compatibility is determined
          by the is_compatible() method.
        """
        if self.has_compatible_gen_function(other):
            return self.generating_function, None

        if self.has_compatible_gen_points(other):
            gen_points = _get_larger_gen_points(self, other)
            return None, gen_points

        raise ValueError(
                "Cannot pick generating data from incompatible grids. "
                "Grids must have matching generating functions or compatible "
                "generating points."
            )


# --- Internal helper functions
def _gen_unisolvent_nodes(multi_index, generating_points):
    """
    .. todo::
        - document this function but ship it to utils first.
    """
    return np.take_along_axis(generating_points, multi_index.exponents, axis=0)

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
    check_type(multi_index, MultiIndexSet)

    # MultiIndexSet for a Grid cannot be an empty set
    if len(multi_index) == 0:
        raise ValueError("MultiIndexSet must not be empty!")

    return multi_index


def _process_domain(domain: Domain, multi_index: MultiIndexSet) -> Domain:
    """Process the Domain given as an argument to Grid constructor.

    Parameters
    ----------
    domain : Domain
        The domain as input argument to the Grid constructor to be
        processed.
    multi_index : MultiIndexSet
        The multi-index set as input argument to the Grid constructor to be
        processed.

    Returns
    -------
    Domain
        The same instance of :class:`Domain` if processing does not
        raise any exceptions.

    Raises
    ------
    TypeError
        If the domain argument is not an instance of :class:`Domain`.
    ValueError
        If the domain argument does not have the same spatial dimension as
        the multi-index set.
    """
    check_type(domain, Domain)

    # Spatial dimensions must be consistent
    if domain.spatial_dimension != multi_index.spatial_dimension:
        raise ValueError(
            f"Spatial dimension of the domain ({domain.spatial_dimension}) is "
            "inconsistent with that of the multi-index set "
            f"({multi_index.spatial_dimension})"
        )

    return domain


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