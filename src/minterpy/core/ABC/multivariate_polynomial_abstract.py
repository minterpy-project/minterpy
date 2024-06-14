"""
Abstract base class for the various polynomial base classes.

This module contains the abstract base classes, from which all concrete implementations of polynomial classes shall subclass.
This ensures that all polynomials work with the same interface, so futher features can be formulated without referencing the concrete polynomial implementation. See e.g. :PEP:`3119` for further explanations on that topic.
"""
import abc
from copy import deepcopy
from typing import Any, List, Optional, Union

import numpy as np

from minterpy.global_settings import ARRAY, SCALAR

from ..grid import Grid
from ..multi_index import MultiIndexSet
from ..utils import expand_dim, find_match_between
from ..verification import (
    check_dimensionality,
    is_scalar,
    check_shape,
    check_type_n_values,
    verify_domain,
)

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
    def nr_active_monomials(self):  # pragma: no cover
        """Abstract container for the number of monomials of the polynomial(s).

        Notes
        -----
        This is a placeholder of the ABC, which is overwritten by the concrete implementation.
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
    def _eval(self, xx: np.ndarray, **kwargs) -> np.ndarray:  # pragma: no cover
        """Abstract method to the polynomial evaluation function.

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
        - Introduce input validation for xx as it is common across concrete
          implementations.
        - Possibly built-in rescaling between ``user_domain`` and
          ``internal_domain``. An idea: use sklearn min max scaler
          (``transform()`` and ``inverse_transform()``)
        """
        return self._eval(self, xx, **kwargs)

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
    def _add(self, other):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _sub(self, other):  # pragma: no cover
        # no docstring here, since it is given in the concrete implementation
        pass

    @staticmethod
    @abc.abstractmethod
    def _mul(self, other, **kwargs):  # pragma: no cover
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
    def _iadd(self, other):  # pragma: no cover
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

    # --- Concrete implementations of operations

    def _scalar_mul(
        self,
        other: SCALAR,
        inplace=False,
    ) -> Optional["MultivariatePolynomialSingleABC"]:
        """Multiply the polynomial by a (real) scalar value.

        Parameters
        ----------
        other : SCALAR
            The real scalar value to multiply the polynomial by.
        inplace : bool, optional
            ``True`` if the multiplication should be done in-place,
            ``False`` otherwise. The default is ``False``.

        Returns
        -------
        Optional[MultivariatePolynomialSingleABC]
            The multiplied polynomial if ``inplace`` is ``False`` (the
            default), otherwise ``None`` (the instance is modified in-place).

        Notes
        -----
        - If inplace is ``False``, a deep copy of the polynomial will be
          created whose coefficients are multiplied by the scalar,
          and then returned.
        - This is a concrete implementation applicable to all concrete
          implementations of polynomial due to the universal rule of
          scalar-polynomial multiplication.
        """
        if inplace:
            # Don't do 'self._coeffs *= other' because external coeffs
            # will change and cause a side effect.
            self._coeffs = self._coeffs * other
        else:

            self_copy = deepcopy(self)
            self_copy._coeffs *= other  # inplace is safe due to deepcopy above

            return self_copy

    def _scalar_add(
        self,
        other: SCALAR,
        inplace=False,
    ) -> Optional["MultivariatePolynomialSingleABC"]:
        """Add the polynomial with a real scalar value.

        Parameters
        ----------
        other : SCALAR
            The real scalar value to add the polynomial with.
        inplace : bool, optional
            ``True`` if the addition should be done in-place,
            ``False`` otherwise. The default is ``False``.

        Returns
        -------
        Optional[MultivariatePolynomialSingleABC]
            The summed polynomial if ``inplace`` is ``False`` (the
            default), otherwise ``None`` (the instance is modified in-place).

        Notes
        -----
        - Adding a real scalar value to a polynomial is equivalent to
          adding a constant polynomial whose coefficient value is the scalar
          value to the polynomial. The concrete implementation called by
          ``__add__()`` or ``__iadd__()`` is responsible for handling
          the polynomial-polynomial addition.
        """
        # Create a constant polynomial
        dim = self.spatial_dimension
        lp_degree = self.multi_index.lp_degree
        mi_constant = MultiIndexSet.from_degree(
            spatial_dimension=dim,
            poly_degree=0,
            lp_degree=lp_degree,
        )
        if self.coeffs.ndim == 1:
            coeffs_shape = (1,)
        else:
            coeffs_shape = (1, self.coeffs.shape[1])
        coeffs_constant = other * np.ones(
            coeffs_shape,
            dtype=self.coeffs.dtype,
        )
        poly_constant = type(self)(
            multi_index=mi_constant,
            coeffs=coeffs_constant,
            internal_domain=self.internal_domain,
            user_domain=self.user_domain,
            grid=self.grid,
        )

        # Call the relevant method
        if inplace:
            # Call back `__iadd__()` because it contains verification routines
            return self.__iadd__(poly_constant)
        else:
            # Call back `__add__()` because it contains verification routines
            return self.__add__(poly_constant)

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
            check_type_n_values(multi_index)  # expected ARRAY
            check_dimensionality(multi_index, dimensionality=2)
            self.multi_index = MultiIndexSet(multi_index)

        nr_monomials, spatial_dimension = self.multi_index.exponents.shape
        self.coeffs = coeffs  # calls the setter method and checks the input shape

        if internal_domain is not None:
            check_type_n_values(internal_domain)
            check_shape(internal_domain, shape=(spatial_dimension, 2))
        self.internal_domain = self.generate_internal_domain(
            internal_domain, self.multi_index.spatial_dimension
        )

        if user_domain is not None:  # TODO not better "external domain"?!
            check_type_n_values(user_domain)
            check_shape(user_domain, shape=(spatial_dimension, 2))
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
        self.indices_are_separate: bool = self.grid.multi_index is not self.multi_index
        self.active_monomials: Optional[ARRAY] = None  # 1:1 correspondence
        if self.indices_are_separate:
            # store the position of the active Lagrange polynomials with respect to the basis indices:
            self.active_monomials = find_match_between(
                self.multi_index.exponents, self.grid.multi_index.exponents
            )

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

        # The coefficients of the polynomials are equal
        if not np.array_equal(self.coeffs, other.coeffs):
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

    def __add__(
        self,
        other: Union["MultivariatePolynomialSingleABC", SCALAR],
    ) -> "MultivariatePolynomialSingleABC":
        """Add the polynomial(s) with another polynomial(s) or a real scalar.

        This function is called when:

        - two polynomials are added: ``P1 + P2``, where ``P1`` and ``P2``
          are both instances of a concrete polynomial class.
        - a polynomial is added with a real scalar number: ``P1 + a``,
          where ``a`` is a real scalar number.

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
        - The concrete implementation of polynomial-polynomial addition
          is delegated to the respective polynomial concrete class.

        See Also
        --------
        _add
            Concrete implementation of ``__add__``
        _scalar_add
            Concrete implementation of ``__add__`` when the right operand
            is a real scalar number.
        """
        # Handle scalar addition
        if is_scalar(other):
            return self._scalar_add(other, inplace=False)

        # Only supported for polynomials of the same concrete class
        if self.__class__ != other.__class__:
            raise TypeError(
                f"Addition operation is not implemented for "
                f"'{self.__class__}', '{other.__class__}'"
            )

        # Check if the number of coefficients remain consistent.
        if not _has_consistent_number_of_polys(self, other):
            raise ValueError(
                "Cannot add polynomials with inconsistent "
                "number of coefficient sets"
            )

        # Only do it if the dimension is matching and inplace
        if not self.has_matching_domain(other):
            raise ValueError(
                "Cannot add polynomials of different domains"
            )

        # Handle equal value
        if self == other:
            return self._scalar_mul(2.0)

        # Handle equal but negated value
        if self == -other:
            return self._scalar_mul(0.0)

        result = self._add(self, other)

        return result

    def __sub__(
        self,
        other: Union["MultivariatePolynomialSingleABC", SCALAR],
    ) -> "MultivariatePolynomialSingleABC":
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
          operand on the right; no separate concrete implementation is
          used.

        See Also
        --------
        _add
            Concrete implementation of ``__add__``
        _scalar_add
            Concrete implementation of ``__add__`` when the right operand
            is a real scalar number.
        """
        # Handle scalar addition
        if is_scalar(other):
            return self._scalar_add(-other, inplace=False)

        return self.__add__(-other)

    def __mul__(
        self,
        other: Union["MultivariatePolynomialSingleABC", SCALAR]
    ) -> "MultivariatePolynomialSingleABC":
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

        See Also
        --------
        _mul
            Concrete implementation of ``__mil__``
        _scalar_mul
            Concrete implementation of ``__mul__`` when the right operand
            is a real scalar number.
        """
        # Multiplication by a real scalar number
        if is_scalar(other):
            return self._scalar_mul(other, inplace=False)

        if isinstance(self, type(other)):
            # Check if the number of coefficients remain consistent.
            if not _has_consistent_number_of_polys(self, other):
                raise ValueError(
                    "Cannot multiply polynomials with inconsistent "
                    "number of coefficient sets"
                )
            # Only do it if the dimension is matching and inplace
            if not self.has_matching_domain(other):
                raise ValueError(
                    "Cannot multiply polynomials of different domains"
                )

            # Rely on the subclass concrete implementation (static method)
            return self._mul(self, other)

        return NotImplemented

    # --- Special methods: Reversed arithmetic operation

    def __radd__(
        self,
        other: SCALAR,
    ) -> "MultivariatePolynomialSingleABC":
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

        See Also
        --------
        _scalar_add
            Concrete implementation of ``__add__`` when the right operand
            is a real scalar number.
        """
        # Addition of a real scalar number by a polynomial
        if is_scalar(other):
            return self._scalar_add(other, inplace=False)

        # Right-sided addition with other types is not explicitly supported;
        # it will rely on the left operand '__add__()' method
        return NotImplemented

    def __rsub__(
        self,
        other: SCALAR,
    ) -> "MultivariatePolynomialSingleABC":
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

        See Also
        --------
        _scalar_add
            Concrete implementation of ``__add__`` when the right operand
            is a real scalar number.
        __neg__
            Negating a polynomial.
        """
        # Subtraction of a real scalar number by a polynomial
        if is_scalar(other):
            return (-self)._scalar_add(other, inplace=False)

        # Right-sided subtraction with other types is not explicitly supported;
        # it will rely on the left operand '__sub__()' method
        return NotImplemented

    def __rmul__(self, other: SCALAR) -> "MultivariatePolynomialSingleABC":
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

        See Also
        --------
        _scalar_mul
            Concrete implementation of ``__mul__`` when the right operand
            is a real scalar number.
        """
        # Multiplication by a real scalar number
        if is_scalar(other):
            return self._scalar_mul(other, inplace=False)

        # Right-sided multiplication with other types is not explicitly
        # supported; it will rely on the left operand '__mul__()' method
        return NotImplemented

    # --- Special methods: Augmented assignment arithmetic operators

    def __imul__(
        self,
        other: SCALAR,
    ) -> "MultivariatePolynomialSingleABC":
        """Multiply a polynomial with a real scalar in-place.

        This function is called when a polynomial is multiplied with a real
        scalar number in-place like ``P *= a`` where ``P`` ``a`` are
        a polynomial instance and a scalar, respectively.

        Parameters
        ----------
        other : SCALAR
            The right operand, a real scalar number.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of the multiplication, an instance of multiplied
            polynomial.

        See Also
        --------
        _scalar_mul
            Concrete implementation of ``__mul__`` when the right operand
            is a real scalar number.

        TODO
        ----
        - Add support for polynomial-polynomial multiplication in-place.
        """
        if is_scalar(other):
            self._scalar_mul(other, inplace=True)
            return self

        # TODO: Currently only multiplication with scalar is supported inplace
        if isinstance(self, type(other)):
            raise NotImplementedError

        return NotImplemented

    def __iadd__(
        self,
        other: Union["MultivariatePolynomialSingleABC", SCALAR],
    ) -> "MultivariatePolynomialSingleABC":
        """Add a polynomial with a poly. or a real scalar in-place.

        This function is called when a polynomial is added in-place,
        specifically when:

        - a polynomial is added with another polynomial like ``P1 += P2`` where
          ``P1`` and ``P2`` are both polynomial instances of the same concrete
          class.
        - a polynomial is added with a real scalar number like ``P1 += a``
          where ``a`` is a real scalar number.

        Parameters
        ----------
        other : Union[MultivariatePolynomialSingleABC, SCALAR]
            The right operand, a polynomial instance of the same concrete class
            or a real scalar number.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of the addition, an updated instance of summed
            polynomial.

        See Also
        --------
        _iadd
            Concrete implementation of ``__iadd__``.
        _scalar_add
            Concrete implementation of ``__mul__`` when the right operand
            is a real scalar number.
        """
        # Handle in-place addition by a scalar
        if is_scalar(other):
            return self._scalar_add(other, inplace=True)

        #  Only supported for polynomials of the same concrete class
        if self.__class__ != other.__class__:
            raise TypeError(
                f"Subtraction operation not implemented for "
                f"'{self.__class__}', '{other.__class__}'"
            )

        # Check if the number of coefficients remain consistent.
        if not _has_consistent_number_of_polys(self, other):
            raise ValueError(
                "Cannot subtract polynomials with inconsistent "
                "number of coefficient sets"
            )

        # Only do it if the dimension is matching and inplace
        if not self.has_matching_domain(other):
            raise ValueError(
                "Cannot subtract polynomials of different domains"
            )

        return self._iadd(self, other)

    def __isub__(
        self,
        other: Union["MultivariatePolynomialSingleABC", SCALAR],
    ) -> "MultivariatePolynomialSingleABC":
        """Subtract a polynomial with a poly. or a real scalar in-place.

        This function is called when a polynomial is subtracted in-place,
        specifically when:

        - a polynomial is subtracted with another polynomial like ``P1 += P2``
          where ``P1`` and ``P2`` are both polynomial instances of the same
          concrete class.
        - a polynomial is subtracted with a real scalar number like ``P1 += a``
          where ``a`` is a real scalar number.

        Parameters
        ----------
        other : Union[MultivariatePolynomialSingleABC, SCALAR]
            The right operand, a polynomial instance of the same concrete class
            or a real scalar number.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The result of the addition, an instance of subtracted polynomial.

        Notes
        -----
        - Under the hood subtraction is an addition operation with a negated
          operand on the right; no separate concrete implementation is
          used.

        See Also
        --------
        _iadd
            Concrete implementation of ``__iadd__``.
        _scalar_add
            Concrete implementation of ``__mul__`` when the right operand
            is a real scalar number.
        """
        # Handle in-place subtraction by a scalar
        if is_scalar(other):
            return self._scalar_add(-other, inplace=True)

        return self.__iadd__(-other)

    # copying
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

    @property
    def nr_active_monomials(self):
        """Number of active monomials of the polynomial(s).

        For caching and methods based on switching single monomials on and off, it is distigushed between active and passive monomials, where only the active monomials particitpate on exposed functions.

        :return: Number of active monomials.
        :rtype: int

        Notes
        -----
        This is usually equal to the "amount of coefficients". However the coefficients can also be a 2D array (representing a multitude of polynomials with the same base grid).
        """

        return len(self.multi_index)

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
    def coeffs(self) -> Optional[ARRAY]:
        """Array which stores the coefficients of the polynomial.

        With shape (N,) or (N, p) the coefficients of the multivariate polynomial(s), where N is the amount of monomials and p is the amount of polynomials.

        :return: Array of coefficients.
        :rtype: np.ndarray
        :raise ValueError: Raised if the coeffs are not initialised.

        Notes
        -----
        It is allowed to set the coefficients to `None` to represent a not yet initialised polynomial
        """
        if self._coeffs is None:
            raise ValueError(
                "trying to access an uninitialized polynomial (coefficients are `None`)"
            )
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value: Optional[ARRAY]):
        # setters shall not have docstrings. See numpydoc class example.
        if value is None:
            self._coeffs = None
            return
        check_type_n_values(value)
        if value.shape[0] != self.nr_active_monomials:
            raise ValueError(
                f"the amount of given coefficients <{value.shape[0]}> does not match "
                f"with the amount of monomials in the polynomial <{self.nr_active_monomials}>."
            )
        self._coeffs = value

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
        grid_new = self.grid.add_points(exponents)
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
        dim: int,
        extra_internal_domain: ARRAY = None,
        extra_user_domain: ARRAY = None,
    ):
        """Expand the spatial dimention.

        Expands the dimension of the domain space of the polynomial by adding zeros to the multi_indices
        (which is equivalent to the multiplication of ones to each monomial).
        Furthermore, the grid is now embedded in the higher dimensional space by pinning the grid arrays to the origin in the additional spatial dimension.

        :param dim: Number of additional dimensions.
        :type dim: int
        """
        diff_dim = dim - self.multi_index.spatial_dimension

        # If dim<spatial_dimension, i.e. expand_dim<0, exception is raised
        self.multi_index = self.multi_index.expand_dim(dim)

        grid = self.grid
        new_gen_pts = expand_dim(grid.generating_points, dim)
        new_gen_vals = expand_dim(grid.generating_values.reshape(-1, 1), dim)

        self.grid = Grid(self.multi_index, new_gen_pts, new_gen_vals)

        extra_internal_domain = verify_domain(extra_internal_domain, diff_dim)
        self.internal_domain = np.concatenate(
            (self.internal_domain, extra_internal_domain)
        )
        extra_user_domain = verify_domain(extra_user_domain, diff_dim)
        self.user_domain = np.concatenate((self.user_domain, extra_user_domain))

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
            The underluing static method to integrate the polynomial instance
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

    # Utility public methods

    def has_matching_domain(
        self,
        other: "MultivariatePolynomialSingleABC",
        tol: float = 1e-16,
    ) -> bool:
        """
        Check if two MultivariatePolynomialSingleABC objects have matching domains.

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
        internal_domain_1 = self.internal_domain[:dim, :]
        internal_domain_2 = other.internal_domain[:dim, :]
        has_matching_internal_domain = np.less_equal(
            np.abs(internal_domain_1 - internal_domain_2),
            tol,
        )

        # Check matching user domain
        user_domain_1 = self.user_domain[:dim, :]
        user_domain_2 = other.user_domain[:dim, :]
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
