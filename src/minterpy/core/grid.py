"""
Module for the generating points (which provides the unisolvent nodes)
"""
from copy import copy, deepcopy
from typing import Callable, Optional, Union

import numpy as np

from minterpy.global_settings import ARRAY, INT_DTYPE
from minterpy.gen_points import GENERATING_FUNCTIONS, gen_points_from_values

from minterpy.core.multi_index import MultiIndexSet
from minterpy.core.tree import MultiIndexTree
from minterpy.utils.arrays import expand_dim
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
        The multi-index set of polynomial exponents that the Grid should
        support.
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
        where ``n`` is the maximum polynomial degree in all dimensions
        and ``m`` is the spatial dimension.
        This parameter is optional. If not specified, the generating points
        are created from the default generating function. If specified,
        then the points must be consistent with any non-``None`` generating
        function.

    Notes
    -----
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

        # Process and assign the generating points argument
        if no_gen_points:
            generating_points = self._create_generating_points()
        self._generating_points = _process_generating_points(generating_points)

        # --- Post-assignment verifications

        # Verify the polynomial degree of the Grid
        self._verify_grid_poly_degree()

        # Verify generating function and points again if both are specified
        if not no_gen_function and not no_gen_points:
            self._verify_matching_gen_function_and_points()

        # --- Lazily-evaluated properties
        self._unisolvent_nodes = None
        self._tree = None

    # --- Factory methods
    # TODO rename: name is misleading. a generator is something different in python:
    #   cf. https://wiki.python.org/moin/Generators
    @classmethod
    def from_generator(
        cls,
        multi_index: MultiIndexSet,
        generating_function: Callable,
    ) -> "Grid":
        """
        Constructor from a factory method for the ``generating_values``.

        :param multi_index: The :class:`MultiIndexSet` this ``grid`` is based on.
        :type multi_index: MultiIndexSet

        :param generating_function: Factory method for the ``generating_values``. This functions gets a polynomial degree and returns a set of generating values of this degree.
        :type generating_function: callable

        :return: Instance of :class:`Grid` for the given input.
        :rtype: Grid


        """
        generating_points = generating_function(
            multi_index.poly_degree,
            multi_index.spatial_dimension,
        )

        # TODO: Revise this
        return cls.from_value_set(multi_index, generating_points)

    @classmethod
    def from_value_set(cls, multi_index: MultiIndexSet, generating_values: ARRAY):
        """
        Constructor from given ``generating_values``.

        :param multi_index: The :class:`MultiIndexSet` this ``grid`` is based on.
        :type multi_index: MultiIndexSet

        :param generating_values: Generating values the :class:`Grid` instance shall be based on. The input shape needs to be one-dimensional.
        :type generating_function: np.ndarray

        :return: Instance of :class:`Grid` for the given input.
        :rtype: Grid
        """
        spatial_dimension = multi_index.spatial_dimension
        generating_points = gen_points_from_values(
            generating_values,
            spatial_dimension,
        )
        return cls(
            multi_index,
            generating_points=generating_points,
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
    def generating_function(self) -> Optional[Callable[[int, int], np.ndarray]]:
        """The generating function of the interpolation Grid.

        Returns
        -------
        Optional[Callable]
            The generating function of the interpolation Grid which is used
            to generate the generating points.

        Notes
        -----
        - If the generating function is ``None`` then the Grid may not be
          manipulated that results in a grid of a higher degree.
        """
        return self._generating_function

    @property
    def generating_points(self) -> np.ndarray:
        """The generating points of the interpolation Grid.

        The generating points of the interpolation grid are the ingredients
        of constructing unisolvent nodes.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            A two-dimensional array of floats whose columns are the
            generating points per spatial dimension. The shape of the array
            is ``(n + 1, m)`` where ``n`` is the maximum polynomial degree
            in any dimension and ``m`` is the spatial dimension.
        """
        return self._generating_points

    @property
    def poly_degree(self) -> int:
        """The polynomial degree of the interpolation Grid.

        Returns
        -------
        int
            The polynomial degree of the interpolation Grid is the maximum
            polynomial degree of one-dimensional polynomials in any dimension
            that the Grid can support.

        Notes
        -----
        - Unlike the polynomial degree associated with a multi-index set,
          the polynomial degree of the Grid does not depend on the notion
          of ``lp_degree`` as it is based only on one dimension.
        """
        return len(self.generating_points) - 1

    @property
    def unisolvent_nodes(self):
        """Array of unidolvent nodes.

        For a definition of unisolvent nodes, see the mathematical introduction.

        :return: Array of the unisolvent nodes. If None were given, the output is lazily build from ``multi_index`` and ``generation_points``.
        :rtype: np.ndarray


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

    # --- Instance methods
    def apply_func(self, func, out=None):
        """This function is not implemented yet and will raise a :class:`NotImplementedError` if called.

        Apply a given (universal) function on this :class:`Grid` instance.

        :param func: The function, which will be evaluated on the grid points.
        :type func: callable
        :raise NotImplementedError: if called, since it is not implemented yet.

        :param out: The array, where the result of the function evaluation will be stored. If given, the ``out`` array will be changed inplace, otherwise the a new one will be initialised.
        :type out: np.ndarray, optional


        .. todo::
            - implement an evaluation function for :class:`Grid` instances.
            - think about using the numpy interface for universal funcions.

        """
        # apply func to unisolvent nodes and return the func values, or store them alternatively in out
        raise NotImplementedError

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
        if np.max(exponents) > self.poly_degree:
            # TODO 'enlarge' the grid, increase the degree, ATTENTION:
            raise ValueError(
                f"trying to add point with exponent {np.max(exponents)} "
                f"but the grid is only of degree {self.poly_degree}"
            )

        multi_indices_new = self.multi_index.add_exponents(exponents)
        return self._new_instance_if_necessary(multi_indices_new)

    def expand_dim(self, new_dimension: int) -> "Grid":
        """Expand the dimension of the Grid.

        Parameters
        ----------
        new_dimension : int
            The new spatial dimension. It must be larger than or equal
            to the current dimension of the Grid.

        Returns
        -------
        Grid
            The Grid with expanded dimension.

        Notes
        -----
        - If no generating function is available, then the generating points
          are directly expanded by appending zeros in the last column.
        """
        # Expand the dimension of the multi-index set
        multi_index = self.multi_index.expand_dim(new_dimension)

        if self.generating_function is None:
            # If generating function is not available,
            # directly expand the generating points
            gen_points = expand_dim(self.generating_points, new_dimension)

            # Return a new instance
            return self.__class__(
                multi_index,
                generating_points=gen_points,
                generating_function=self._generating_function,
            )

        # Return a new instance
        return self.__class__(
            multi_index,
            generating_function=copy(self._generating_function),
        )

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
        # Check for consistent type
        if not isinstance(other, Grid):
            return False

        # Multi-index set equality
        if self.multi_index != other.multi_index:
            return False

        # Generating points equality
        if not np.array_equal(self.generating_points, other.generating_points):
            return False

        if self.generating_function != other.generating_function:
            return False

        return True

    # --- Private internal methods: not to be called directly from outside
    def _verify_matching_gen_function_and_points(self):
        """Verify if the generation function and points match.

        Raises
        ------
        ValueError
            If the generating function generates different points than
            then ones that are provided.
        """
        gen_points = self.generating_function(
            self.poly_degree,
            self.spatial_dimension,
        )

        if not np.array_equal(gen_points, self.generating_points):
            raise ValueError(
                "The generating function generates points that are "
                "inconsistent with the generating points"
            )

    def _create_generating_points(self) -> np.ndarray:
        """Construct generating points from the generating function.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The generating points of the interpolation grid, a two-dimensional
            array of floats whose columns are the generating points
            per spatial dimension. The shape of the array is ``(n + 1, m)``
            where ``n`` is the maximum polynomial degree in all dimensions
            and ``m`` is the spatial dimension.
        """
        multi_index = self._multi_index
        poly_degree = np.max(multi_index.exponents)
        spatial_dimension = multi_index.spatial_dimension

        generating_function = self._generating_function

        return generating_function(poly_degree, spatial_dimension)

    def _verify_grid_poly_degree(self):
        """Verify if the degree of the Grid is consistent with the multi-index.

        Raises
        ------
        ValueError
            If the degree of the Grid is smaller than the maximum
            one-dimensional polynomial degree in all dimensions specified
            in the corresponding multi-index set of polynomial exponents.

        Notes
        -----
        - While it perhaps makes sense to store the polynomial degrees of each
          dimension as the instance property instead of the maximum over all
          dimensions, this has no use because the generating points have
          a uniform length in every dimension. For instance, if the maximum
          polynomial degrees per dimension of a two-dimensional polynomial are
          ``[5, 3]``, the stored generating points remain ``(5, 2)`` instead of
          two arrays having lengths of ``5`` and ``3``, respectively.
        """
        # The maximum polynomial degree in any dimension of the multi-index
        # set indicates the largest degree of one-dimensional polynomial
        # the grid needs to support.
        poly_degree_multi_index = np.max(self.multi_index.exponents)

        # Both must be consistent; "smaller" multi-index may be contained
        # in a larger grid, but not the other way around.
        if poly_degree_multi_index > self.poly_degree:
            raise ValueError(
                f"A grid of a polynomial degree {self.poly_degree} "
                "cannot consist of multi-indices with a maximum polynomial "
                "degree {poly_degree_multi_index}"
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


def _process_generating_points(
    generating_points: Optional[np.ndarray],
) -> np.ndarray:
    """Process the generating points given as an argument to Grid constructor.

    Parameters
    ----------
    generating_points : :class:`numpy:numpy.ndarray`, optional
        The generating points of the grid, a two-dimensional array of floats
        of shape ``(n, m)`` where ``n`` is the number of points and ``m``
        is the number of spatial dimension.
    """
    # Check type
    check_type(generating_points, np.ndarray, "The generating points")
    # Check array dimension
    check_dimensionality(generating_points, dimensionality=2)
    # No NaN's and inf's
    check_values(generating_points)
    # Check domain fit
    check_domain_fit(generating_points)

    # Return processed generating points
    return generating_points
