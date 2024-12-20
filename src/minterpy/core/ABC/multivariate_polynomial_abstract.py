"""
This module contains the abstract base classes for all polynomial base classes.

All concrete implementations of polynomial bases must inherit from
the abstract base class. This ensures a consistent interface across
all polynomials. As a result, additional features can be developed without
needing to reference specific polynomial classes,
while allowing each concrete class to manage its own implementation details.

See e.g. :PEP:`3119` for further explanations on the topic.

----

"""
import abc
import numpy as np

from copy import copy, deepcopy
from typing import List, Optional, Tuple, Union

from minterpy.global_settings import ARRAY, SCALAR
from minterpy.core.grid import Grid
from minterpy.core.multi_index import MultiIndexSet
from minterpy.utils.verification import (
    check_type,
    check_values,
    is_real_scalar,
    check_shape,
    shape_eval_output,
    verify_domain,
    verify_poly_coeffs,
    verify_poly_domain,
    verify_poly_power,
    verify_query_points,
)
from minterpy.utils.multi_index import find_match_between

__all__ = ["MultivariatePolynomialABC", "MultivariatePolynomialSingleABC"]


class MultivariatePolynomialABC(abc.ABC):
    """the most general abstract base class for multivariate polynomials.

    Every data type which needs to behave like abstract polynomial(s) should subclass this class and implement all the abstract methods.
    """

    @property
    @abc.abstractmethod
    def coeffs(self) -> ARRAY:  # pragma: no cover
        """Abstract container which stores the coefficients of the polynomial.

        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @coeffs.setter
    def coeffs(self, value):
        pass

    @property
    @abc.abstractmethod
    def num_active_monomials(self):  # pragma: no cover
        """Abstract container for the number of monomials of the polynomial(s).

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten
        by the concrete implementation.
        """
        pass

    @property
    @abc.abstractmethod
    def spatial_dimension(self):  # pragma: no cover
        """Abstract container for the dimension of space where the polynomial(s) live on.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @property
    @abc.abstractmethod
    def unisolvent_nodes(self):  # pragma: no cover
        """Abstract container for unisolvent nodes the polynomial(s) is(are) defined on.

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _eval(
        poly: "MultivariatePolynomialABC",
        xx: np.ndarray,
        **kwargs,
    ) -> np.ndarray:  # pragma: no cover
        """Abstract method to the polynomial evaluation function.

        Parameters
        ----------
        poly : MultivariatePolynomialABC
            A concrete instance of a polynomial class that can be evaluated
            on a set of query points.
        xx : :class:`numpy:numpy.ndarray`
            The set of query points to evaluate as a two-dimensional array
            of shape ``(k, m)`` where ``k`` is the number of query points and
            ``m`` is the spatial dimension of the polynomial.
        **kwargs
            Additional keyword-only arguments that change the behavior of
            the underlying evaluation (see the concrete implementation).

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The values of the polynomial evaluated at query points.

            - If there is only a single polynomial (i.e., a single set of
              coefficients), then a one-dimensional array of length ``k``
              is returned.
            - If there are multiple polynomials (i.e., multiple sets
              of coefficients), then a two-dimensional array of shape
              ``(k, np)`` is returned where ``np`` is the number of
              coefficient sets.

        Notes
        -----
        - This is a placeholder of the ABC, which is overwritten
          by the concrete implementation.

        See Also
        --------
        __call__
            The dunder method as a syntactic sugar to evaluate
            the polynomial(s) instance on a set of query points.
        """
        pass

    def __call__(self, xx: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the polynomial on a set of query points.

        The function is called when an instance of a polynomial is called with
        a set of query points, i.e., :math:`p(\mathbf{X})` where
        :math:`\mathbf{X}` is a matrix of values with :math:`k` rows
        and each row is of length :math:`m` (i.e., a point in
        :math:`m`-dimensional space).

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            The set of query points to evaluate as a two-dimensional array
            of shape ``(k, m)`` where ``k`` is the number of query points and
            ``m`` is the spatial dimension of the polynomial.
        **kwargs
            Additional keyword-only arguments that change the behavior of
            the underlying evaluation (see the concrete implementation).

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The values of the polynomial evaluated at query points.

            - If there is only a single polynomial (i.e., a single set of
              coefficients), then a one-dimensional array of length ``k``
              is returned.
            - If there are multiple polynomials (i.e., multiple sets
              of coefficients), then a two-dimensional array of shape
              ``(k, np)`` is returned where ``np`` is the number of
              coefficient sets.

        Notes
        -----
        - The function calls the concrete implementation of the static method
          ``_eval()``.

        See Also
        --------
        _eval
            The underlying static method to evaluate the polynomial(s) instance
            on a set of query points.

        TODO
        ----
        - Possibly built-in rescaling between ``user_domain`` and
          ``internal_domain``. An idea: use sklearn min max scaler
          (``transform()`` and ``inverse_transform()``)
        """
        # Verify query points
        xx = verify_query_points(xx, self.spatial_dimension)

        # Evaluate using concrete static method
        yy = self._eval(self, xx, **kwargs)

        # Follow the convention of output shape from an evaluation
        return shape_eval_output(yy)

    # anything else any polynomial must support
    # TODO mathematical operations? abstract
    # TODO copy operations. abstract


class MultivariatePolynomialSingleABC(MultivariatePolynomialABC):
    """abstract base class for "single instance" multivariate polynomials

    Attributes
    ----------
    multi_index : MultiIndexSet
        The multi-indices of the multivariate polynomial.
    internal_domain : array_like
        The domain the polynomial is defined on (basically the domain of the unisolvent nodes).
        Either one-dimensional domain (min,max), a stack of domains for each
        domain with shape (spatial_dimension,2).
    user_domain : array_like
        The domain where the polynomial can be evaluated. This will be mapped onto the ``internal_domain``.
        Either one-dimensional domain ``min,max)`` a stack of domains for each
        domain with shape ``(spatial_dimension,2)``.

    Notes
    -----
    the grid with the corresponding indices defines the "basis" or polynomial space a polynomial is part of.
    e.g. also the constraints for a Lagrange polynomial, i.e. on which points they must vanish.
    ATTENTION: the grid might be defined on other indices than multi_index! e.g. useful for defining Lagrange coefficients with "extra constraints"
    but all indices from multi_index must be contained in the grid!
    this corresponds to polynomials with just some of the Lagrange polynomials of the basis being "active"
    """

    # __doc__ += __doc_attrs__

    _coeffs: Optional[ARRAY] = None

    @staticmethod
    @abc.abstractmethod
    def generate_internal_domain(
        internal_domain, spatial_dimension
    ):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def generate_user_domain(user_domain, spatial_dimension):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    # TODO static methods should not have a parameter "self"
    @staticmethod
    @abc.abstractmethod
    def _add(poly_1, poly_2):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _sub(self, other):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _mul(poly_1, poly_2, **kwargs):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _div(self, other):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _pow(self, pow):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _scalar_add(poly, scalar):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    def _gen_grid_default(multi_index):
        """Return the default :class:`Grid` for a given :class:`MultiIndexSet` instance.

        For the default values of the Grid class, see :class:`minterpy.Grid`.


        :param multi_index: An instance of :class:`MultiIndexSet` for which the default :class:`Grid` shall be build
        :type multi_index: MultiIndexSet
        :return: An instance of :class:`Grid` with the default optional parameters.
        :rtype: Grid
        """
        return Grid(multi_index)

    @staticmethod
    @abc.abstractmethod
    def _partial_diff(
        poly: MultivariatePolynomialABC,
        dim: int,
        order: int,
        **kwargs,
    ) -> "MultivariatePolynomialSingleABC":  # pragma: no cover
        """Abstract method for differentiating poly. on a given dim. and order.

        Parameters
        ----------
        poly : MultivariatePolynomialABC
            The instance of polynomial to differentiate.
        dim : int
            Spatial dimension with respect to which the differentiation
            is taken. The dimension starts at 0 (i.e., the first dimension).
        order : int
            Order of partial derivative.
        **kwargs
            Additional keyword-only arguments that change the behavior of
            the underlying differentiation (see the concrete implementation).

        Returns
        -------
        MultivariatePolynomialSingleABC
            A new polynomial instance that represents the partial derivative
            of the original polynomial of the given order of derivative with
            respect to the specified dimension.

        Notes
        -----
        - The concrete implementation of this static method is called when
          the public method ``partial_diff()`` is called on an instance.

        See also
        --------
        partial_diff
            The public method to differentiate the polynomial of a specified
            order of derivative with respect to a given dimension.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _diff(
        poly: MultivariatePolynomialABC,
        order: np.ndarray,
        **kwargs,
    ) -> "MultivariatePolynomialSingleABC":  # pragma: no cover
        """Abstract method for diff. poly. on given orders w.r.t each dim.

        Parameters
        ----------
        poly : MultivariatePolynomialABC
            The instance of polynomial to differentiate.
        order : :class:`numpy:numpy.ndarray`
            A one-dimensional integer array specifying the orders of derivative
            along each dimension. The length of the array must be ``m`` where
            ``m`` is the spatial dimension of the polynomial.
        **kwargs
            Additional keyword-only arguments that change the behavior of
            the underlying differentiation (see the concrete implementation).

        Returns
        -------
        MultivariatePolynomialSingleABC
            A new polynomial instance that represents the partial derivative
            of the original polynomial of the specified orders of derivative
            along each dimension.

        Notes
        -----
        - The concrete implementation of this static method is called when
          the public method ``diff()`` is called on an instance.

        See also
        --------
        diff
            The public method to differentiate the polynomial instance on
            the given orders of derivative along each dimension.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _integrate_over(
        poly: "MultivariatePolynomialABC",
        bounds: Optional[np.ndarray],
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """Abstract method for definite integration.

        Parameters
        ----------
        poly : MultivariatePolynomialABC
            The instance of polynomial to integrate.
        bounds : Union[List[List[float]], np.ndarray], optional
            The bounds of the integral, an ``(m, 2)`` array where ``m``
            is the number of spatial dimensions. Each row corresponds to
            the bounds in a given dimension.
            If not given, then the canonical bounds :math:`[-1, 1]^m` will
            be used instead.
        **kwargs
            Additional keyword-only arguments that change the behavior of
            the underlying integration (see the respective concrete
            implementations).

        Returns
        -------
        Union[:py:class:`float`, :class:`numpy:numpy.ndarray`]
            The integral value of the polynomial over the given bounds.
            If only one polynomial is available, the return value is of
            a :py:class:`float` type.

        Notes
        -----
        - The concrete implementation of this static method is called when
          the public method ``integrate_over()`` is called on an instance.

        See Also
        --------
        integrate_over
            The public method to integrate the polynomial instance over
            the given bounds.
        """
        pass

    # --- Constructors
    def __init__(
        self,
        multi_index: Union[MultiIndexSet, ARRAY],
        coeffs: Optional[ARRAY] = None,
        internal_domain: Optional[ARRAY] = None,
        user_domain: Optional[ARRAY] = None,
        grid: Optional[Grid] = None,
    ):

        if multi_index.__class__ is MultiIndexSet:
            if len(multi_index) == 0:
                raise ValueError("MultiIndexSet must not be empty!")
            self.multi_index = multi_index
        else:
            # TODO should passing multi indices as ndarray be supported?
            self.multi_index = MultiIndexSet(multi_index)

        nr_monomials, spatial_dimension = self.multi_index.exponents.shape
        self.coeffs = coeffs  # calls the setter method and checks the input shape

        if internal_domain is not None:
            check_type(internal_domain, np.ndarray)
            check_values(internal_domain)
            check_shape(internal_domain, shape=(2, spatial_dimension))
        self.internal_domain = self.generate_internal_domain(
            internal_domain, self.multi_index.spatial_dimension
        )

        if user_domain is not None:  # TODO not better "external domain"?!
            check_type(user_domain, np.ndarray)
            check_values(user_domain)
            check_shape(user_domain, shape=(2, spatial_dimension))
        self.user_domain = self.generate_user_domain(
            user_domain, self.multi_index.spatial_dimension
        )

        # TODO make multi_index input optional? otherwise use the indices from grid
        # TODO class method from_grid
        if grid is None:
            grid = self._gen_grid_default(self.multi_index)
        if type(grid) is not Grid:
            raise ValueError(f"unexpected type {type(grid)} of the input grid")

        if not grid.multi_index.is_superset(self.multi_index):
            raise ValueError(
                "the multi indices of a polynomial must be a subset of the indices of the grid in use"
            )
        self.grid: Grid = grid
        # weather or not the indices are independent from the grid ("basis")
        # TODO this could be enconded by .active_monomials being None
        self.indices_are_separate: bool = self.grid.multi_index != self.multi_index
        self.active_monomials: Optional[ARRAY] = None  # 1:1 correspondence
        if self.indices_are_separate:
            # store the position of the active Lagrange polynomials with respect to the basis indices:
            self.active_monomials = find_match_between(
                self.multi_index.exponents, self.grid.multi_index.exponents
            )

    # --- Factory methods
    @classmethod
    def from_degree(
        cls,
        spatial_dimension: int,
        poly_degree: int,
        lp_degree: int,
        coeffs: Optional[ARRAY] = None,
        internal_domain: ARRAY = None,
        user_domain: ARRAY = None,
    ):
        """Initialise Polynomial from given coefficients and the default construction for given polynomial degree, spatial dimension and :math:`l_p` degree.

        :param spatial_dimension: Dimension of the domain space of the polynomial.
        :type spatial_dimension: int

        :param poly_degree: The degree of the polynomial, i.e. the (integer) supremum of the :math:`l_p` norms of the monomials.
        :type poly_degree: int

        :param lp_degree: The :math:`l_p` degree used to determine the polynomial degree.
        :type lp_degree: int

        :param coeffs: coefficients of the polynomial. These shall be 1D for a single polynomial, where the length of the array is the number of monomials given by the ``multi_index``. For a set of similar polynomials (with the same number of monomials) the array can also be 2D, where the first axis refers to the monomials and the second axis refers to the polynomials.
        :type coeffs: np.ndarray

        :param internal_domain: the internal domain (factory) where the polynomials are defined on, e.g. :math:`[-1,1]^d` where :math:`d` is the dimension of the domain space. If a ``callable`` is passed, it shall get the dimension of the domain space and returns the ``internal_domain`` as an :class:`np.ndarray`.
        :type internal_domain: np.ndarray or callable
        :param user_domain: the domain window (factory), from which the arguments of a polynomial are transformed to the internal domain. If a ``callable`` is passed, it shall get the dimension of the domain space and returns the ``user_domain`` as an :class:`np.ndarray`.
        :type user_domain: np.ndarray or callable

        """
        return cls(
            MultiIndexSet.from_degree(spatial_dimension, poly_degree, lp_degree),
            coeffs,
            internal_domain,
            user_domain,
        )

    @classmethod
    def from_poly(
        cls,
        polynomial: "MultivariatePolynomialSingleABC",
        new_coeffs: Optional[ARRAY] = None,
    ) -> "MultivariatePolynomialSingleABC":
        """constructs a new polynomial instance based on the properties of an input polynomial

        useful for copying polynomials of other types


        :param polynomial: input polynomial instance defining the properties to be reused
        :param new_coeffs: the coefficients the new polynomials should have. using `polynomial.coeffs` if `None`
        :return: new polynomial instance with equal properties

        Notes
        -----
        The coefficients can also be assigned later.
        """
        p = polynomial
        if new_coeffs is None:  # use the same coefficients
            new_coeffs = p.coeffs

        return cls(p.multi_index, new_coeffs, p.internal_domain, p.user_domain, p.grid)

    @classmethod
    def from_grid(
        cls,
        grid: Grid,
        coeffs: Optional[np.ndarray] = None,
        internal_domain: Optional[np.ndarray] = None,
        user_domain: Optional[np.ndarray] = None,
    ):
        """Create an instance of polynomial with a `Grid` instance.

        Parameters
        ----------
        grid : Grid
            The grid on which the polynomial is defined.
        coeffs : :class:`numpy:numpy.ndarray`, optional
            The coefficients of the polynomial(s); a one-dimensional array
            with the same length as the length of the multi-index set or
            a two-dimensional array with each column corresponds to the
            coefficients of a single polynomial on the same grid.
            This parameter is optional, if not specified the polynomial
            is considered "uninitialized".
        internal_domain  : :class:`numpy:numpy.ndarray`, optional
            The internal domain of the polynomial(s).
        user_domain : :class:`numpy:numpy.ndarray`, optional
            The user domain of the polynomial(s).

        Returns
        -------
        MultivariatePolynomialSingleABC
            An instance of polynomial defined on the given grid.
        """
        return cls(
            multi_index=grid.multi_index,
            coeffs=coeffs,
            internal_domain=internal_domain,
            user_domain=user_domain,
            grid=grid,
        )

    # --- Properties
    @property
    def coeffs(self) -> np.ndarray:
        """The coefficients of the polynomial(s).

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            One- or two-dimensional array that contains the polynomial
            coefficients. Coefficients of multiple polynomials having common
            structure are stored in a two-dimensional array of shape ``(N, P)``
            where ``N`` is the number of monomials and ``P`` is the number
            of polynomials.

        Raises
        ------
        ValueError
            If the coefficients of an uninitialized polynomial are accessed.

        Notes
        -----
        - ``coeffs`` may be assigned with `None` to indicate an uninitialized
           polynomial. Accessing such coefficients, however,
           raises an exception. Many operations involving polynomial instances,
           require the instance to be initialized and raising the exception
           here provides a common single point of failure.
        """
        if self._coeffs is None:
            raise ValueError(
                "Coefficients of an uninitialized polynomial "
                "cannot be accessed."
            )

        return self._coeffs

    @coeffs.setter
    def coeffs(self, value: Optional[np.ndarray]) -> None:
        # setters shall not have docstrings. See numpydoc class example.
        if value is None:
            # `None` indicates an uninitialized polynomial
            self._coeffs = None
            return

        # Verify and assign the coefficient values
        expected_num_monomials = self.num_active_monomials
        self._coeffs = verify_poly_coeffs(value, expected_num_monomials)

    @property
    def num_active_monomials(self) -> int:
        """The number of active monomials of the polynomial(s).

        The multi-index set that directly defines a polynomial and the grid
        (where the polynomial lives) may differ. Active monomials are
        the monomials that are defined by the multi-index set not by the one
        in the grid.

        Returns
        -------
        int
            The number of active monomials.
        """
        return len(self.multi_index)

    # --- Special methods: Rich comparison
    def __eq__(self, other: "MultivariatePolynomialSingleABC") -> bool:
        """Compare two concrete polynomial instances for exact equality.

        Two polynomial instances are equal if and only if:

        - both are of the same concrete class, *and*
        - the underlying multi-index sets are equal, *and*
        - the underlying grid instances are equal, *and*
        - the coefficients of the polynomials are equal.

        Parameters
        ----------
        other : MultivariatePolynomialSingleABC
            Another instance of concrete implementation of
            `MultivariatePolynomialSingleABC` to compare with

        Returns
        -------
        bool
            ``True`` if the current instance is equal to the other instance,
            ``False`` otherwise.
        """
        # The instances are of different concrete classes
        if not isinstance(self, type(other)):
            return False

        # The underlying multi-index sets are equal
        if self.multi_index != other.multi_index:
            return False

        # The underlying grid instances are equal
        if self.grid != other.grid:
            return False

        # The coefficients are both None
        if self._coeffs is None and other._coeffs is None:
            return True

        # The coefficients of the polynomials are equal
        if not np.array_equal(self._coeffs, other._coeffs):
            return False

        return True

    # --- Special methods: Unary numeric
    def __neg__(self) -> "MultivariatePolynomialSingleABC":
        """Negate the polynomial(s) instance.

        This function is called when a polynomial is negated via
        the ``-`` operator, e.g., ``-P``.

        Returns
        -------
        MultivariatePolynomialSingleABC
            New polynomial(s) instance with negated coefficients.

        Notes
        -----
        - The resulting polynomial is a deep copy of the original polynomial.
        - ``-P`` is not the same as ``-1 * P``, the latter of which is a scalar
          multiplication. In this case, however, the result is the same;
          it returns a new instance with negated coefficients.
        """
        self_copy = deepcopy(self)
        self_copy._coeffs = -1 * self_copy._coeffs

        return self_copy

    def __pos__(self) -> "MultivariatePolynomialSingleABC":
        """Plus sign the polynomial(s) instance.

        This function is called when a polynomial is plus signed via
        the ``+`` operator, e.g., ``+P``.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The same polynomial

        Notes
        -----
        - ``+P`` is not the same as ``1 * P``, the latter of which is a scalar
          multiplication. In this case, the result actually differs because
          the scalar multiplication ``1 * P`` returns a new instance of
          polynomial even though the coefficients are not altered.
        """
        return self

    # --- Special methods: Arithmetic operators
    def __add__(self, other: Union["MultivariatePolynomialSingleABC", SCALAR]):
        """Add the polynomial(s) with another polynomial(s) or a real scalar.

        This function is called when:

        - two polynomials are added: ``P1 + P2``, where ``P1`` (i.e., ``self``)
          and ``P2`` (``other``) are both instances of a concrete polynomial
          class.
        - a polynomial is added with a real scalar number: ``P1 + a``,
          where ``a`` (``other``) is a real scalar number.

        Polynomials are closed under scalar addition, meaning that
        the result of the addition is also a polynomial with the same
        underlying multi-index set; only the coefficients are altered.

        Parameters
        ----------
        other : Union[MultivariatePolynomialSingleABC, SCALAR]
            The right operand, either an instance of polynomial (of the same
            concrete class as the right operand) or a real scalar number.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of the addition, an instance of summed polynomial.

        Notes
        -----
        - The concrete implementation of polynomial-polynomial and polynomial-
          scalar addition is delegated to the respective polynomial concrete
          class.
        """
        # Handle scalar addition
        if is_real_scalar(other):
            return self._scalar_add(self, other)

        # Verify the operands before conducting addition
        poly_1, poly_2 = self._verify_operands(other, operation="+ or -")

        return self._add(poly_1, poly_2)

    def __sub__(self, other: Union["MultivariatePolynomialSingleABC", SCALAR]):
        """Subtract the polynomial(s) with another poly. or a real scalar.

        This function is called when:

        - two polynomials are subtracted: ``P1 - P2``, where ``P1`` and ``P2``
          are both instances of a concrete polynomial class.
        - a polynomial is added with a real scalar number: ``P1 - a``,
          where ``a`` is a real scalar number.

        Polynomials are closed under scalar subtraction, meaning that
        the result of the subtraction is also a polynomial with the same
        underlying multi-index set; only the coefficients are altered.

        Parameters
        ----------
        other : Union[MultivariatePolynomialSingleABC, SCALAR]
            The right operand, either an instance of polynomial (of the same
            concrete class as the right operand) or a real scalar number.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of the subtraction, an instance of subtracted
            polynomial.

        Notes
        -----
        - Under the hood subtraction is an addition operation with a negated
          operand on the right; no separate concrete implementation is used.
        """
        # Handle scalar addition
        if is_real_scalar(other):
            return self._scalar_add(self, -other)

        return self.__add__(-other)

    def __mul__(self, other: Union["MultivariatePolynomialSingleABC", SCALAR]):
        """Multiply the polynomial(s) with another polynomial or a real scalar.

        This function is called when:

        - two polynomials are multiplied: ``P1 * P2``, where ``P1`` and ``P2``
          are both instances of a concrete polynomial class.
        - a polynomial is multiplied with a real scalar number: ``P1 * a``,
          where ``a`` is a real scalar number.

        Polynomials are closed under scalar multiplication, meaning that
        the result of the multiplication is also a polynomial with the same
        underlying multi-index set; only the coefficients are altered.

        Parameters
        ----------
        other : Union[MultivariatePolynomialSingleABC, SCALAR]
            The right operand, either an instance of polynomial (of the same
            concrete class as the right operand) or a real scalar number.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of the multiplication, an instance of multiplied
            polynomial.

        Notes
        -----
        - The concrete implementation of polynomial-polynomial multiplication
          is delegated to the respective polynomial concrete class.
        """
        # Multiplication by a real scalar number
        if is_real_scalar(other):
            return _scalar_mul(self, other)

        # Verify the operands before conducting multiplication
        poly_1, poly_2 = self._verify_operands(other, operation="*")

        return self._mul(poly_1, poly_2)

    def __truediv__(self, other: SCALAR) -> "MultivariatePolynomialSingleABC":
        """Divide an instance of polynomial with a real scalar number (``/``).

        Parameters
        ----------
        other : Union[MultivariatePolynomialSingleABC, SCALAR]
            The right operand of the (true) division expression,
            a real scalar number.

        Returns
        -------
        MultivariatePolynomialSingleABC
            An instance of polynomial, the result of (true) scalar division
            of a polynomial.
        """
        if is_real_scalar(other):
            return _scalar_truediv(self, other)

        return self._div(self, other)

    def __floordiv__(self, other: SCALAR) -> "MultivariatePolynomialSingleABC":
        """Divide an instance of polynomial with a real scalar number (``//``).

        Parameters
        ----------
        other : Union[MultivariatePolynomialSingleABC, SCALAR]
            The right operand of the (floor) division expression,
            a real scalar number.

        Returns
        -------
        MultivariatePolynomialSingleABC
            An instance of polynomial, the result of (floor) scalar division
            of a polynomial.
        """
        if is_real_scalar(other):
            return _scalar_floordiv(self, other)

        return self._div(self, other)

    def __pow__(self, power: int):
        """Take the polynomial instance to the given power.

        Parameters
        ----------
        power : int
            The power in the exponentiation expression; the value must
            be a non-negative real scalar whole number. The value may not
            strictly be an integer as long as it is a whole number
            (e.g., :math:`2.0` is acceptable).

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of exponentiation, an instance of a concrete polynomial
            class.

        Notes
        -----
        - Exponentiation by zero returns a constant polynomial whose
          coefficients are zero except for the constant term with respect to
          the multi-index set which is given a value of :math:`1.0`.
          In the case of polynomials in the Lagrange basis whose no constant
          term with respect to the multi-index set, all coefficients are set to
          :math:`1.0`.
        """
        # Check if power is valid
        power = verify_poly_power(power)

        # Iterative exponentiation
        if power == 0:
            return self * 0 + 1

        result = copy(self)
        for _ in range(power - 1):
            result = result * self

        return result

    # --- Special methods: Reversed arithmetic operation
    def __radd__(self, other: SCALAR):
        """Right-sided addition of the polynomial(s) with a real scalar number.

        This function is called for the expression ``a + P`` where ``a``
        and ``P`` is a real scalar number and an instance of polynomial,
        respectively.

        Parameters
        ----------
        other : SCALAR
            A real scalar number (the left operand) to be added to
            the polynomial.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of adding the scalar value to the polynomial.

        Notes
        -----
        - If the left operand is not a real scalar number, the right-sided
          addition is not explicitly supported, and it will rely on
          the `__add__()` method of the left operand.
        """
        # Addition of a real scalar number by a polynomial
        if is_real_scalar(other):
            return self._scalar_add(self, other)

        # Right-sided addition with other types is not explicitly supported;
        # it will rely on the left operand '__add__()' method
        return NotImplemented

    def __rsub__(self, other: SCALAR):
        """Right-sided subtraction of the polynomial(s) with a real scalar.

        This function is called for the expression ``a - P`` where ``a``
        and ``P`` is a real scalar number and an instance of polynomial,
        respectively.

        Parameters
        ----------
        other : SCALAR
            A real scalar number (the left operand) to be substracted by
            the polynomial.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of subtracting a scalar value by the polynomial.

        Notes
        -----
        - If the left operand is not a real scalar number, the right-sided
          subtraction is not explicitly supported, and it will rely on
          the `__add__()` method of the left operand.
        - This operation relies on the negation of a polynomial and scalar
          addition
        """
        # Subtraction of a real scalar number by a polynomial
        if is_real_scalar(other):
            return self._scalar_add(-self, other)

        # Right-sided subtraction with other types is not explicitly supported;
        # it will rely on the left operand '__sub__()' method
        return NotImplemented

    def __rmul__(self, other: SCALAR):
        """Right sided multiplication of the polynomial(s) with a real scalar.

        This function is called if a real scalar number is multiplied
        with a polynomial like ``a * P`` where ``a`` and ``P`` are a scalar
        and a polynomial instance, respectively.

        Parameters
        ----------
        other : SCALAR
            The left operand, a real scalar number.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of the multiplication, an instance of multiplied
            polynomial.
        """
        # Multiplication by a real scalar number
        if is_real_scalar(other):
            return _scalar_mul(self, other)

        # Right-sided multiplication with other types is not explicitly
        # supported; it will rely on the left operand '__mul__()' method
        return NotImplemented

    # --- Special methods: copies
    def __copy__(self):
        """Creates of a shallow copy.

        This function is called, if one uses the top-level function ``copy()`` on an instance of this class.

        :return: The copy of the current instance.
        :rtype: MultivariatePolynomialSingleABC

        See Also
        --------
        copy.copy
            copy operator form the python standard library.
        """
        return self.__class__(
            self.multi_index,
            self._coeffs,
            self.internal_domain,
            self.user_domain,
            self.grid,
        )

    def __deepcopy__(self, mem):
        """Creates of a deepcopy.

        This function is called, if one uses the top-level function ``deepcopy()`` on an instance of this class.

        :return: The deepcopy of the current instance.
        :rtype: MultivariatePolynomialSingleABC

        See Also
        --------
        copy.deepcopy
            copy operator form the python standard library.

        """
        return self.__class__(
            deepcopy(self.multi_index),
            deepcopy(self._coeffs),
            deepcopy(self.internal_domain),
            deepcopy(self.user_domain),
            deepcopy(self.grid),
        )

    # Special methods: Collection emulation
    def __len__(self) -> int:
        """Return the number of polynomials in the instance.

        Returns
        -------
        int
            The number of polynomial in the instance. A single instance of
            polynomial may contain multiple polynomials with different
            coefficient values but sharing the same underlying multi-index set
            and grid.
        """
        if self.coeffs.ndim == 1:
            return 1

        return self.coeffs.shape[1]

    @property
    def spatial_dimension(self):
        """Spatial dimension.

        The dimension of space where the polynomial(s) live on.

        :return: Dimension of domain space.
        :rtype: int

        Notes
        -----
        This is propagated from the ``multi_index.spatial_dimension``.
        """
        return self.multi_index.spatial_dimension

    @property
    def unisolvent_nodes(self):
        """Unisolvent nodes the polynomial(s) is(are) defined on.

        For definitions of unisolvent nodes see the mathematical introduction.

        :return: Array of unisolvent nodes.
        :rtype: np.ndarray

        Notes
        -----
        This is propagated from from ``self.grid.unisolvent_nodes``.
        """
        return self.grid.unisolvent_nodes

    # --- Instance methods
    def _new_instance_if_necessary(
        self, new_grid, new_indices: Optional[MultiIndexSet] = None
    ) -> "MultivariatePolynomialSingleABC":
        """Constructs a new instance only if the multi indices have changed.

        :param new_grid: Grid instance the polynomial is defined on.
        :type new_grid: Grid

        :param new_indices: :class:`MultiIndexSet` instance for the polynomial(s), needs to be a subset of the current ``multi_index``. Default is :class:`None`.
        :type new_indices: MultiIndexSet, optional

        :return: Same polynomial instance if ``grid`` and ``multi_index`` stay the same, otherwise new polynomial instance with the new ``grid`` and ``multi_index``.
        :rtype: MultivariatePolynomialSingleABC
        """
        prev_grid = self.grid
        if new_grid is prev_grid:
            return self
        # grid has changed
        if new_indices is None:
            # the active monomials (and coefficients) stay equal
            new_indices = self.multi_index
            new_coeffs = self._coeffs
        else:
            # also the active monomials change
            prev_indices = self.multi_index
            if not prev_indices.is_subset(new_indices):
                raise ValueError(
                    "an index set of a polynomial can only be expanded, "
                    "but the old indices contain multi indices not present in the new indices."
                )

            # convert the coefficients correctly:
            if self._coeffs is None:
                new_coeffs = None
            else:
                new_coeffs = np.zeros(len(new_indices))
                idxs_of_old = find_match_between(
                    prev_indices.exponents, new_indices.exponents
                )
                new_coeffs[idxs_of_old] = self._coeffs

        new_poly_instance = self.__class__(new_indices, new_coeffs, grid=new_grid)
        return new_poly_instance

    def make_complete(self) -> "MultivariatePolynomialSingleABC":
        """returns a possibly new polynomial instance with a complete multi index set.

        :return: completed polynomial, where additional coefficients setted to zero.
        :rtype: MultivariatePolynomialSingleABC

        Notes
        -----
        - the active monomials stay equal. only the grid ("basis") changes
        - in the case of a Lagrange polynomial this could be done by evaluating the polynomial on the complete grid
        """
        grid_completed = self.grid.make_complete()
        return self._new_instance_if_necessary(grid_completed)

    def add_points(self, exponents: ARRAY) -> "MultivariatePolynomialSingleABC":
        """Extend ``grid`` and ``multi_index``

        Adds points ``grid`` and exponents to ``multi_index`` related to a given set of additional exponents.

        :param exponents: Array of exponents added.
        :type exponents: np.ndarray

        :return: New polynomial with the added exponents.
        :rtype: MultivariatePolynomialSingleABC

        """
        # replace the grid with an independent copy with the new multi indices
        # ATTENTION: the grid might be defined on other indices than multi_index!
        #   but all indices from multi_index must be contained in the grid!
        # -> make sure to add all new additional indices also to the grid!
        grid_new = self.grid.add_exponents(exponents)
        multi_indices_new = None
        if self.indices_are_separate:
            multi_indices_new = self.multi_index.add_exponents(exponents)
        return self._new_instance_if_necessary(grid_new, multi_indices_new)

    # def make_derivable(self) -> "MultivariatePolynomialSingleABC":
    #     """ convert the polynomial into a new polynomial instance with a "derivable" multi index set
    #  NOTE: not meaningful since derivation requires complete index sets anyway?
    #     """
    #     new_indices = self.multi_index.make_derivable()
    #     return self._new_instance_if_necessary(new_indices)

    def expand_dim(
        self,
        target_dimension: Union["MultivariatePolynomialSingleABC", int],
        extra_internal_domain: Optional[np.ndarray] = None,
        extra_user_domain: Optional[np.ndarray] = None,
    ):
        """Expand the spatial dimension of the polynomial instance.

        Parameters
        ----------
        target_dimension : Union[int, MultivariatePolynomialSingleABC]
            The new spatial dimension. It must be larger than or equal to
            the current dimension of the polynomial. Alternatively, another
            instance of polynomial that has a higher dimension, a consistent
            underlying `Grid` instance is consistent, and a matching domain
            can also be specified as a target dimension.
        extra_internal_domain : :class:`numpy:numpy.ndarray`, optional
            The additional internal domains for the expanded polynomial.
            This parameter is optional; if not specified, the values are either
            taken from the domain of the higher-dimensional polynomial or
            from the domain of the other dimensions.
        extra_user_domain : :class:`numpy:numpy.ndarray`, optional
            The additional user domains for the expanded polynomial.
            This parameter is optional; if not specified, the values are either
            taken from the domain of the higher-dimensional polynomial or
            from the domain of the other dimensions.

        Returns
        -------
        MultivariatePolynomialSingleABC
            A new instance of polynomial whose spatial dimension has been
            expanded to the target.

        Raises
        ------
        ValueError
            If the target dimension is an `int`, the exception is raised
            when the user or internal domains cannot be extrapolated to
            a higher dimension. If the target dimension is an instance of
            `MultivariatePolynomialSingleABC`, the exception is raised when
            the user or internal domains do no match.
            In both cases, an exception may also be raised by attempting
            to expand the dimension of the underlying `Grid` or `MultiIndexSet`
            instances.
        """
        if isinstance(target_dimension, MultivariatePolynomialSingleABC):
            return _expand_dim_to_target_poly(self, target_dimension)

        return _expand_dim_to_target_dim(
            self,
            target_dimension,
            extra_internal_domain,
            extra_user_domain,
        )

    def partial_diff(
        self,
        dim: int,
        order: int = 1,
        **kwargs,
    ) -> "MultivariatePolynomialSingleABC":
        """Return the partial derivative poly. at the given dim. and order.

        Parameters
        ----------
        dim : int
            Spatial dimension with respect to which the differentiation
            is taken. The dimension starts at 0 (i.e., the first dimension).
        order : int
            Order of partial derivative.
        **kwargs
            Additional keyword-only arguments that change the behavior of
            the underlying differentiation (see the respective concrete
            implementations).

        Returns
        -------
        MultivariatePolynomialSingleABC
            A new polynomial instance that represents the partial derivative
            of the original polynomial of the specified order of derivative
            and with respect to the specified dimension.

        Notes
        -----
        - This method calls the concrete implementation of the abstract
          method ``_partial_diff()`` after input validation.

        See Also
        --------
        _partial_diff
            The underlying static method to differentiate the polynomial
            instance of a specified order of derivative and with respect to
            a specified dimension.
        """

        # Guard rails for dim
        if not np.issubdtype(type(dim), np.integer):
            raise TypeError(f"dim <{dim}> must be an integer")

        if dim < 0 or dim >= self.spatial_dimension:
            raise ValueError(
                f"dim <{dim}> for spatial dimension <{self.spatial_dimension}>"
                f" should be between 0 and {self.spatial_dimension-1}"
            )

        # Guard rails for order
        if not np.issubdtype(type(dim), np.integer):
            raise TypeError(f"order <{order}> must be a non-negative integer")

        if order < 0:
            raise ValueError(f"order <{order}> must be a non-negative integer")

        return self._partial_diff(self, dim, order, **kwargs)

    def diff(
        self,
        order: np.ndarray,
        **kwargs,
    ) -> "MultivariatePolynomialSingleABC":
        """Return the partial derivative poly. of given orders along each dim.

        Parameters
        ----------
        order : :class:`numpy:numpy.ndarray`
            A one-dimensional integer array specifying the orders of derivative
            along each dimension. The length of the array must be ``m`` where
            ``m`` is the spatial dimension of the polynomial.
        **kwargs
            Additional keyword-only arguments that change the behavior of
            the underlying differentiation (see the respective concrete
            implementations).

        Returns
        -------
        MultivariatePolynomialSingleABC
            A new polynomial instance that represents the partial derivative
            of the original polynomial of the specified orders of derivative
            along each dimension.

        Notes
        -----
        - This method calls the concrete implementation of the abstract
          method ``_diff()`` after input validation.

        See Also
        --------
        _diff
            The underlying static method to differentiate the polynomial
            of specified orders of derivative along each dimension.
        """

        # convert 'order' to numpy 1d array if it isn't already. This allows type checking below.
        order = np.ravel(order)

        # Guard rails for order
        if not np.issubdtype(order.dtype.type, np.integer):
            raise TypeError(f"order of derivative <{order}> can only be non-negative integers")

        if np.any(order < 0):
            raise ValueError(f"order of derivative <{order}> cannot have negative values")

        if len(order) != self.spatial_dimension:
            raise ValueError(f"inconsistent number of elements in 'order' <{len(order)}>,"
                             f"expected <{self.spatial_dimension}> corresponding to each spatial dimension")

        return self._diff(self, order, **kwargs)

    def integrate_over(
        self,
        bounds: Optional[Union[List[List[float]], np.ndarray]] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """Compute the definite integral of the polynomial over the bounds.

        Parameters
        ----------
        bounds : Union[List[List[float]], np.ndarray], optional
            The bounds of the integral, an ``(m, 2)`` array where ``m``
            is the number of spatial dimensions. Each row corresponds to
            the bounds in a given dimension.
            If not given, then the canonical bounds :math:`[-1, 1]^m` will
            be used instead.
        **kwargs
            Additional keyword-only arguments that change the behavior of
            the underlying integration (see the respective concrete
            implementations).

        Returns
        -------
        Union[:py:class:`float`, :class:`numpy:numpy.ndarray`]
            The integral value of the polynomial over the given bounds.
            If only one polynomial is available, the return value is of
            a :py:class:`float` type.

        Raises
        ------
        ValueError
            If the bounds either of inconsistent shape or not in
            the :math:`[-1, 1]^m` domain.

        Notes
        -----
        - This method calls the concrete implementation of the abstract
          method ``_integrate_over()`` after input validation.

        See Also
        --------
        _integrate_over
            The underlying static method to integrate the polynomial instance
            over the given bounds.

        TODO
        ----
        - The default fixed domain [-1, 1]^M may in the future be relaxed.
          In that case, the domain check below along with the concrete
          implementations for the poly. classes must be updated.
        """
        num_dim = self.spatial_dimension
        if bounds is None:
            # The canonical bounds are [-1, 1]^M
            bounds = np.ones((num_dim, 2))
            bounds[:, 0] *= -1

        if isinstance(bounds, list):
            bounds = np.atleast_2d(bounds)

        # --- Bounds verification
        # Shape
        if bounds.shape != (num_dim, 2):
            raise ValueError(
                "The bounds shape is inconsistent! "
                f"Given {bounds.shape}, expected {(num_dim, 2)}."
            )
        # Domain fit, i.e., in [-1, 1]^M
        if np.any(bounds < -1) or np.any(bounds > 1):
            raise ValueError("Bounds are outside [-1, 1]^M domain!")

        # --- Compute the integrals
        # If the lower and upper bounds are equal, immediately return 0
        if np.any(np.isclose(bounds[:, 0], bounds[:, 1])):
            return 0.0

        value = self._integrate_over(self, bounds, **kwargs)

        try:
            # One-element array (one set of coefficients), just return the item
            return value.item()
        except ValueError:
            return value

    # --- Public utility methods
    def has_matching_dimension(
        self,
        other: "MultivariatePolynomialSingleABC",
    ) -> bool:
        """Return ``True`` if the polynomials have matching dimensions.

        Parameters
        ----------
        other : MultivariatePolynomialSingleABC
            The second instance of polynomial to compare.

        Returns
        -------
        bool
            ``True`` if the two spatial dimensions match, ``False`` otherwise.
        """
        return self.spatial_dimension == other.spatial_dimension

    def has_matching_domain(
        self,
        other: "MultivariatePolynomialSingleABC",
        tol: float = 1e-16,
    ) -> bool:
        """Return ``True`` if the polynomials have matching domains.

        Parameters
        ----------
        other : MultivariatePolynomialSingleABC
            The second instance of polynomial to compare.
        tol : float, optional
            The tolerance used to check for matching domains.
            Default is 1e-16.

        Returns
        -------
        bool
            ``True`` if the two domains match, ``False`` otherwise.

        Notes
        -----
        - The method checks both the internal and user domains.
        - If the dimensions of the polynomials do not match, the comparison
          is carried out up to the smallest matching dimension.
        """
        # Get the dimension to deal with unmatching dimension
        dim_1 = self.spatial_dimension
        dim_2 = other.spatial_dimension
        dim = np.min([dim_1, dim_2])  # Check up to the smallest matching dim.

        # Check matching internal domain
        internal_domain_1 = self.internal_domain[:, :dim]
        internal_domain_2 = other.internal_domain[:, :dim]
        has_matching_internal_domain = np.less_equal(
            np.abs(internal_domain_1 - internal_domain_2),
            tol,
        )

        # Check matching user domain
        user_domain_1 = self.user_domain[:, :dim]
        user_domain_2 = other.user_domain[:, :dim]
        has_matching_user_domain = np.less_equal(
            np.abs(user_domain_1 - user_domain_2),
            tol,
        )

        # Checking both domains
        has_matching_domain = np.logical_and(
            has_matching_internal_domain,
            has_matching_user_domain,
        )

        return np.all(has_matching_domain)

    # --- Private utility methods: Not supposed to be called from the outside
    def _match_dims(
        self,
        other: "MultivariatePolynomialSingleABC",
    ) -> Tuple[
        "MultivariatePolynomialSingleABC",
        "MultivariatePolynomialSingleABC",
    ]:
        """Match the dimension of two polynomials.

        Parameters
        ----------
        other : MultivariatePolynomialSingleABC
            An instance polynomial whose dimension is to match with the current
            polynomial instance.

        Returns
        -------
        Tuple[MultivariatePolynomialSingleABC, MultivariatePolynomialSingleABC]
            The two instances of polynomials whose dimensions have been
            matched.

        Raises
        ------
        ValueError
            If the dimension of one of the polynomial instance can't be
            matched due to, for instance, incompatible domain.

        Notes
        -----
        - If both polynomials have matching dimension and domains, then
          the function return the two polynomials as they are.
        """
        if self.has_matching_dimension(other) and \
                self.has_matching_domain(other):
            # Dimension and domain match, no need to do anything
            return self, other

        # Otherwise expand the lower dimension polynomial to a higher dimension
        if self.spatial_dimension > other.spatial_dimension:
            other_expanded = other.expand_dim(self)
            return self, other_expanded
        else:
            self_expanded = self.expand_dim(other)
            return self_expanded, other

    def _verify_operands(
        self,
        other: "MultivariatePolynomialSingleABC",
        operation: str,
    ) -> Tuple[
         "MultivariatePolynomialSingleABC",
         "MultivariatePolynomialSingleABC",
         ]:
        """Verify the operands are valid before moving on."""
        # Only supported for polynomials of the same concrete class
        if self.__class__ != other.__class__:
            raise TypeError(
                f"Unsupported operand type(s) for {operation}: "
                f"'{self.__class__}' and '{other.__class__}'"
            )

        # Check if the number of coefficients is consistent
        if len(self) != len(other):
            raise ValueError(
                "Cannot add polynomials with inconsistent "
                "number of coefficient sets"
            )

        poly_1, poly_2 = self._match_dims(other)

        return poly_1, poly_2


def _scalar_mul(
    poly: MultivariatePolynomialSingleABC,
    scalar: Union[SCALAR, np.ndarray],
) -> MultivariatePolynomialSingleABC:
    """Multiply the polynomial by a (real) scalar value.

    Parameters
    ----------
    poly : MultivariatePolynomialSingleABC
        The polynomial instance to be multiplied.
    scalar : Union[SCALAR, np.ndarray]
        The real scalar value to multiply the polynomial by.
        Multiple scalars may be specified as an array as long as the length
        is consistent with the length of the polynomial instance.

    Returns
    -------
    MultivariatePolynomialSingleABC
        The multiplied polynomial.

    Notes
    -----
    - This is a concrete implementation applicable to all concrete
      implementations of polynomial due to the universal rule of
      scalar-polynomial multiplication.
    """
    poly_copy = deepcopy(poly)
    poly_copy.coeffs *= scalar

    return poly_copy


def _scalar_truediv(
    poly: MultivariatePolynomialSingleABC,
    other: Union[SCALAR, np.ndarray],
) -> MultivariatePolynomialSingleABC:
    """True divide the polynomial by a real scalar value.

    Parameters
    ----------
    poly : MultivariatePolynomialSingleABC
        The polynomial instance to be divided.
    scalar : Union[SCALAR, np.ndarray]
        The real scalar value to divide the polynomial by.
        Multiple scalars may be specified as an array as long as the length
        is consistent with the length of the polynomial instance.

    Returns
    -------
    MultivariatePolynomialSingleABC
        The divided polynomial.

    Notes
    -----
    - This is a concrete implementation applicable to all concrete
      implementations of polynomial due to the universal rule of
      polynomial-scalar division.
    """
    poly_copy = deepcopy(poly)
    poly_copy.coeffs /= other

    return poly_copy


def _scalar_floordiv(
    poly: MultivariatePolynomialSingleABC,
    other: Union[SCALAR, np.ndarray],
) -> MultivariatePolynomialSingleABC:
    """Floor divide the polynomial by a real scalar value.

    Parameters
    ----------
    poly : MultivariatePolynomialSingleABC
        The polynomial instance to be divided.
    scalar : Union[SCALAR, np.ndarray]
        The real scalar value to divide the polynomial by.
        Multiple scalars may be specified as an array as long as the length
        is consistent with the length of the polynomial instance.

    Returns
    -------
    MultivariatePolynomialSingleABC
        The divided polynomial.

    Notes
    -----
    - This is a concrete implementation applicable to all concrete
      implementations of polynomial due to the universal rule of
      polynomial-scalar division.
    """
    poly_copy = deepcopy(poly)
    poly_copy.coeffs //= other

    return poly_copy


def _has_consistent_number_of_polys(
    poly_1: "MultivariatePolynomialSingleABC",
    poly_2: "MultivariatePolynomialSingleABC",
) -> bool:
    """Check if two polynomials have a consistent number of coefficient sets.
    """
    coeffs_1 = poly_1.coeffs
    coeffs_2 = poly_2.coeffs

    ndim_1 = coeffs_1.ndim
    ndim_2 = coeffs_2.ndim

    if (ndim_1 == 1) and (ndim_2 == 1):
        return True

    has_same_dims = coeffs_1.ndim == coeffs_2.ndim

    try:
        has_same_cols = coeffs_1.shape[1] == coeffs_2.shape[1]
    except IndexError:
        return False

    return has_same_dims and has_same_cols


def _expand_dim_to_target_poly(
    origin_poly: "MultivariatePolynomialSingleABC",
    target_poly: "MultivariatePolynomialSingleABC",
) -> "MultivariatePolynomialSingleABC":
    """Expand the dimension of the polynomial to the dimension of another.

    Parameters
    ----------
    origin_poly : MultivariatePolynomialSingleABC
        The polynomial whose spatial dimension is to be expanded.
    target_poly : MultivariatePolynomialSingleABC
        The polynomial whose spatial dimension is the target dimension.

    Returns
    -------
    MultivariatePolynomialSingleABC
        A new instance of polynomial with an expanded dimension.

    Notes
    -----
    - The extra internal and user domains of the resulting instance takes
      the values from the target polynomial.
    """
    if not origin_poly.has_matching_domain(target_poly):
        raise ValueError(
            "Polynomial cannot be expanded to the dimension of the target "
            "due to non-matching domain."
        )

    # Domains and dimensions match: return a copy
    if origin_poly.has_matching_dimension(target_poly):
        return copy(origin_poly)

    # Otherwise: expand the dimension

    # Get the dimensions
    origin_dimension = origin_poly.spatial_dimension
    target_dimension = target_poly.spatial_dimension

    # Expand the dimension underlying multi-index set to the target dimension
    mi = origin_poly.multi_index.expand_dim(target_dimension)

    # Expand the dimension of the underlying grid to the target grid
    grd = origin_poly.grid.expand_dim(target_poly.grid)

    # Expand the dimension of the internal domain (use values from the larger)
    origin_internal_domain = origin_poly.internal_domain
    target_internal_domain = target_poly.internal_domain
    internal_domain = np.c_[
        origin_internal_domain,
        target_internal_domain[:, origin_dimension:],
    ]

    # Expand the dimension of the user domain (use values from the larger)
    origin_user_domain = origin_poly.user_domain
    target_user_domain = target_poly.user_domain
    user_domain = np.c_[
        origin_user_domain,
        target_user_domain[:, origin_dimension:],
    ]

    # NOTE: There is no need to verify the domains again because they are
    # taken from the properties of polynomial instances (already verified)

    # Return a new instance
    try:
        # The instance is initialized with coefficients
        return origin_poly.__class__(
            multi_index=mi,
            coeffs=origin_poly.coeffs,
            internal_domain=internal_domain,
            user_domain=user_domain,
            grid=grd,
        )
    except ValueError:
        # The instance has no coefficients
        return origin_poly.__class__(
            multi_index=mi,
            coeffs=None,
            internal_domain=internal_domain,
            user_domain=user_domain,
            grid=grd,
        )


def _expand_dim_to_target_dim(
    origin_poly: "MultivariatePolynomialSingleABC",
    target_dimension: int,
    extra_internal_domain: Optional[np.ndarray] = None,
    extra_user_domain: Optional[np.ndarray] = None,
) -> "MultivariatePolynomialSingleABC":
    """Expand the dimension of the polynomial to the target dimension.

    Parameters
    ----------
    origin_poly : MultivariatePolynomialSingleABC
        The polynomial whose spatial dimension is to be expanded.
    target_dimension : int
        The target dimension to which the given polynomial will be expanded.
    extra_internal_domain : :class:`numpy:numpy.ndarray`, optional
        The additional internal domains for the expanded dimensions.
    extra_user_domain : :class:`numpy:numpy.ndarray`, optional
        The additional user domains for the expanded dimensions.

    Returns
    -------
    MultivariatePolynomialSingleABC
        A new instance of polynomial with an expanded dimension.

    Raises
    ------
    ValueError
        If ``extra_internal_domain`` and ``extra_user_domain`` are both
        ``None`` and the domains of the origin polynomial are not uniform
        such that the domains cannot be extrapolated to higher dimension.
    """
    if origin_poly.spatial_dimension == target_dimension:
        return copy(origin_poly)

    # Expand the underlying multi-index set
    mi = origin_poly.multi_index.expand_dim(target_dimension)

    # Expand the dimension of the underlying grid
    grd = origin_poly.grid.expand_dim(target_dimension)

    # Expand the dimension of the internal domain
    origin_internal_domain = origin_poly.internal_domain
    target_internal_domain = _expand_domain(
        origin_internal_domain,
        target_dimension,
        extra_internal_domain,
    )

    # Expand the dimension of the user domain
    origin_user_domain = origin_poly.user_domain
    target_user_domain = _expand_domain(
        origin_user_domain,
        target_dimension,
        extra_user_domain,
    )

    # Return a new instance
    try:
        # The instance is initialized with coefficients
        return origin_poly.__class__(
            multi_index=mi,
            coeffs=origin_poly.coeffs,
            internal_domain=target_internal_domain,
            user_domain=target_user_domain,
            grid=grd,
        )
    except ValueError:
        # The instance has no coefficients
        return origin_poly.__class__(
            multi_index=mi,
            coeffs=None,
            internal_domain=target_internal_domain,
            user_domain=target_user_domain,
            grid=grd,
        )


def _is_domain_uniform(domain: np.ndarray):
    """Check if a given domain is non-uniform"""
    lb_uniform = np.unique(domain[0, :]).size == 1
    ub_uniform = np.unique(domain[1, :]).size == 1

    return lb_uniform and ub_uniform


def _expand_domain(
    origin_domain: np.ndarray,
    target_dimension: int,
    extra_domain: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Append additional polynomial domains to the origin domain column-wise.

    Parameters
    ----------
    origin_domain : :class:`numpy:numpy.ndarray`
        The origin polynomial domain to be expanded.
    target_dimension : int
        The target spatial dimension to which the given polynomial
        will be expanded.
    extra_domain : :class:`numpy:numpy.ndarray`, optional
        The additional domain to be added to form the target domain.
        This parameter is optional, if `None` is provided, the domain can only
        be expanded if ``origin_domain`` is uniform.
    """
    # Get the spatial dimension difference
    origin_dimension = origin_domain.shape[1]
    diff_dimension = target_dimension - origin_dimension

    # If no extra domain is provided
    if extra_domain is None:
        if _is_domain_uniform(origin_domain):
            extra_domain = np.repeat(
                origin_domain[:, 0][:, np.newaxis],
                repeats=diff_dimension,
                axis=1,
            )
        else:
            raise ValueError(
                "Non-uniform domain cannot be extrapolated "
                "for dimension expansion"
            )
    # Combine the extra domain
    target_domain = np.c_[origin_domain, extra_domain]
    # Verify the resulting domain
    target_domain = verify_poly_domain(target_domain, target_dimension)

    return target_domain
