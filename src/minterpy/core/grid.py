"""
Module for the generating points (which provides the unisolvent nodes)
"""
from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from minterpy.global_settings import ARRAY, INT_DTYPE
from minterpy.gen_points import gen_chebychev_2nd_order_leja_ordered, gen_points_chebyshev, gen_points_from_values

from minterpy.core.multi_index import MultiIndexSet
from minterpy.core.tree import MultiIndexTree
from minterpy.utils.verification import (
    check_type,
    check_values,
    check_dimensionality,
    check_domain_fit,
)

__all__ = ["Grid"]


def _gen_unisolvent_nodes(multi_index, generating_points):
    """
    .. todo::
        - document this function but ship it to utils first.
    """
    return np.take_along_axis(generating_points, multi_index.exponents, axis=0)


DEFAULT_GRID_VAL_GEN_FCT = gen_points_chebyshev


# TODO implement comparison operations based on multi index comparison operations and the generating values used
class Grid:
    """Datatype for the nodes some polynomial bases are defined on.

    For a definition of these nodes (refered to as unisolvent nodes), see the mathematical introduction.

    Notes
    -----
    - The multi-index set to construct a :class:`Grid` may not be
      downward-closed. However, building a :class:`.MultiIndexTree` used
      in the transformation between polynomials in the Newton and Lagrange
      bases requires a downward-closed multi-index set.

    .. todo::
        - insert a small introduction to the purpose of :class:`Grid` here.
        - refactor the exposed attributes (each needs at least a getter)
        - naming issues for ``generating_points`` and ``generating_values``
    """

    # TODO make all attributes read only!

    _unisolvent_nodes: Optional[ARRAY] = None

    def __init__(
        self,
        multi_index: MultiIndexSet,
        generating_points: Optional[ARRAY] = None,
    ):
        # Process and assign the multi-index set argument
        self._multi_index = _process_multi_index(multi_index)

        self._generating_function = DEFAULT_GRID_VAL_GEN_FCT

        # Process and assign the generating points argument
        self._generating_points = _process_generating_points(
            generating_points,
            self._multi_index,
            self._generating_function,
        )

        self.poly_degree = len(self._generating_points)
        # check if multi index and generating values fit together
        if self.multi_index.poly_degree > self.poly_degree:
            raise ValueError(
                f"a grid of degree {self.poly_degree} "
                f"cannot consist of indices with degree {self.multi_index.poly_degree}"
            )
        # TODO check if values and points fit together
        # TODO redundant information.

        self._tree: Optional[MultiIndexTree] = None

    # --- Factory methods
    # TODO rename: name is misleading. a generator is something different in python:
    #   cf. https://wiki.python.org/moin/Generators
    @classmethod
    def from_generator(
        cls,
        multi_index: MultiIndexSet,
        generating_function: Callable = DEFAULT_GRID_VAL_GEN_FCT,
    ):
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
        return self.__class__(self.multi_index, self.generating_points)

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
            deepcopy(self.multi_index),
            generating_points=deepcopy(self.generating_points),
        )

    # --- Dunder methods: Rich comparison
    def __eq__(self, other: "Grid") -> bool:
        """Compare two instances of Grid for exact equality in value.

        Two instances of :class:`Grid` class is equal in value if and only if
        both the underlying multi-index sets are equal and
        the generating points are equal.

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

        return True


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


def _process_generating_points(
    generating_points: Optional[np.ndarray],
    multi_index: MultiIndexSet,
    generating_function: Optional[Callable],
) -> np.ndarray:
    """Process the generating points given as an argument to Grid constructor.
    """

    if generating_points is None:
        poly_degree = multi_index.poly_degree
        spatial_dimension = multi_index.spatial_dimension
        generating_points = generating_function(poly_degree, spatial_dimension)

    # Check type
    check_type(generating_points, np.ndarray)
    # Check array dimension
    check_dimensionality(generating_points, dimensionality=2)
    # No NaN's and inf's
    check_values(generating_points)
    # Check domain fit
    check_domain_fit(generating_points)

    # Return processed generating points
    return generating_points
