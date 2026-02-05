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
from minterpy.core.domain import Domain
from minterpy.core.grid import Grid
from minterpy.core.multi_index import MultiIndexSet
from minterpy.utils.verification import (
    is_real_scalar,
    shape_eval_output,
    verify_poly_coeffs,
    verify_poly_power,
    verify_query_points,
)
from minterpy.utils.multi_index import find_match_between
from minterpy.utils.exceptions import DomainMismatchError

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

    def __call__(self, xx: np.ndarray, *, truncate_cols: bool = False, **kwargs) -> np.ndarray:
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
        truncate_cols : bool, optional
            If ``True``, then the last columns of ``xx`` are truncated to match
            the spatial dimension of the polynomial; otherwise all provided
            columns are attempted to be evaluated.
            The default is ``False``.
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
        if truncate_cols:
            xx = xx[:, :self.spatial_dimension]

        xx = verify_query_points(xx, self.spatial_dimension)

        # Evaluate using concrete static method
        yy = self._eval(self, xx, **kwargs)

        # Follow the convention of output shape from an evaluation
        return shape_eval_output(yy)

    # anything else any polynomial must support
    # TODO mathematical operations? abstract
    # TODO copy operations. abstract


class MultivariatePolynomialSingleABC(MultivariatePolynomialABC):
    """Abstract base class for "single instance" multivariate polynomials

    Attributes
    ----------
    multi_index : MultiIndexSet
        The multi-index set of the multivariate polynomial.
    coeffs : np.ndarray, optional
        Coefficients of the polynomial. If ``None``, the polynomial is
        uninitialized. Either a one-dimensional array of length equal to
        the number of monomials or a two-dimensional array of shape
        ``(num_monomials, num_polynomials)`` for multiple polynomials
        sharing the same multi-index set.
    grid : Grid, optional
        The underlying grid on which the (interpolating) polynomial is defined.
        If not given, the default Grid instance will be constructed.
    domain : Domain, optional
        The domain of the underlying grid. If not given, the domain will be
        derived from the default Grid instance.

    Notes
    -----
    - The multi-index set associated with ``grid`` may be different from
      ``multi_index``. All indices from ``multi_index`` must be contained
      in ``grid.multi_index``.
    """
    # __doc__ += __doc_attrs__

    _coeffs: Optional[ARRAY] = None

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
        grid: Optional[Grid] = None,
        domain: Optional[Domain] = None,
    ):
        # Verify and assign multi_index
        if multi_index.__class__ is MultiIndexSet:
            if len(multi_index) == 0:
                raise ValueError("MultiIndexSet must not be empty!")
            self.multi_index = multi_index
        else:
            # TODO should passing multi indices as ndarray be supported?
            self.multi_index = MultiIndexSet(multi_index)

        # Verify and assign grid (delegate to setter)
        self.coeffs = coeffs  # calls the setter method and checks the input shape

        # Verify and assign grid
        self._grid: Grid = _verify_grid(self.multi_index, grid, domain)

        # # TODO make multi_index input optional? otherwise use the indices from grid
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
        lp_degree: float,
        coeffs: Optional[ARRAY] = None,
        grid: Optional[Grid] = None,
        domain: Optional[Domain] = None,
    ):
        r"""Create a polynomial from polynomial degree specifications.

        This factory method constructs a polynomial with a complete multi-index
        set :math:`\mathcal{M}_{m, n, p}` defined by the spatial dimension
        :math:`m`, polynomial degree :math:`n`,
        and :math:`l_p` degree :math:`p`.

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
        coeffs : np.ndarray, optional
            Coefficients of the polynomial. If ``None``, the polynomial is
            uninitialized. Either one-dimensional array of length equal to
            the number of monomials, or two-dimensional array of shape
            ``(num_monomials, num_polynomials)`` for multiple polynomials
            sharing the same multi-index set.
        grid : Grid, optional
            The underlying grid on which the polynomial is defined.
            If not given, a default Grid instance will be constructed.
        domain : Domain, optional
            The domain of the underlying grid. If not given, the domain will be
            derived from the Grid instance (either provided or constructed).

        Returns
        -------
        MultivariatePolynomialSingleABC
            A new polynomial instance with the specified parameters of the
            complete multi-index set.
        """
        mi = MultiIndexSet.from_degree(
            spatial_dimension,
            poly_degree,
            lp_degree,
        )

        return cls(mi, coeffs, grid, domain)

    @classmethod
    def from_poly(
        cls,
        polynomial: "MultivariatePolynomialSingleABC",
        new_coeffs: Optional[ARRAY] = None,
    ) -> "MultivariatePolynomialSingleABC":
        """Create a new polynomial instance based on an existing polynomial.

        This factory method constructs a new polynomial by copying
        the structure (multi-index set, grid) from an existing polynomial,
        optionally with new coefficients.
        Useful for converting between polynomial types or creating polynomials
        with the same structure but different coefficients.

        Parameters
        ----------
        polynomial : MultivariatePolynomialSingleABC
            Input polynomial instance defining the properties to be reused
            (multi-index set and grid).
        new_coeffs : np.ndarray, optional
            The coefficients the new polynomial should have. If ``None``,
            uses a copy of ``polynomial.coeffs``.

        Returns
        -------
        MultivariatePolynomialSingleABC
            A new polynomial instance with the same structure as the input
            polynomial.

        Notes
        -----
        - The coefficients can also be assigned later if the polynomial is
          created uninitialized.

        TODO
        ----
        - Copying the coefficients of the given polynomial if ``new_coeffs``
          is ``None`` is not safe because the concrete class that calls this
          method may be different from the class of the given polynomial.
        """
        p = polynomial
        if new_coeffs is None:  # use the same coefficients
            new_coeffs = p.coeffs.copy()

        return cls(
            copy(p.multi_index),
            new_coeffs,
            copy(p.grid),
        )

    @classmethod
    def from_grid(
        cls,
        grid: Grid,
        coeffs: Optional[np.ndarray] = None,
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

        Returns
        -------
        MultivariatePolynomialSingleABC
            An instance of polynomial defined on the given grid.
        """
        return cls(
            multi_index=grid.multi_index,
            coeffs=coeffs,
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
    def grid(self) -> Grid:
        """The underlying grid of the polynomial(s).

        Returns
        -------
        Grid
            The underlying grid of the polynomial(s). Interpolating polynomials
            live on a Grid.
        """
        return self._grid

    @property
    def domain(self) -> Domain:
        """The domain associated with the Grid of the polynomial(s).

        The domain represents the rectangular bounds of the Grid.

        Returns
        -------
        Domain
            The domain of the interpolation grid.
        """
        return self._grid.domain

    @property
    def spatial_dimension(self) -> int:
        """The spatial dimension of the polynomial(s).

        Returns
        -------
        int
            The spatial dimension of the polynomial(s). This dimension may
            be less than the spatial dimension of the underlying grid.

        Notes
        -----
        - The value is propagated from the underlying ``multi_index``.
        """
        return self.multi_index.spatial_dimension

    @property
    def unisolvent_nodes(self) -> np.ndarray:
        """Unisolvent nodes on which the interpolating polynomial lives.

        Returns
        -------
        np.ndarray
            A two-dimensional array of shape ``(N, m)`` where
            ``N`` is the number of the multi-index set exponents and
            ``m`` is the spatial dimension.

        Notes
        -----
        - The value is propagated from the underlying ``grid``.
        """
        return self._grid.unisolvent_nodes

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
        - the underlying grid instances are equal (including the underlying
          domains and multi-index sets), *and*
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
        if self.__class__ != other.__class__:
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
            concrete class as the left operand) or a real scalar number.

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
        poly_1, poly_2 = self._verify_operands(other)

        # Compute the grid for the polynomial sum
        grd_sum = poly_1.grid | poly_2.grid

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
        poly_1, poly_2 = self._verify_operands(other)

        # Compute the grid for the polynomial product
        grd_prod = poly_1.grid * poly_2.grid

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
        # TODO: Use copy for each instance to propagate consistent behavior
        return self.__class__(
            self.multi_index,
            self._coeffs,
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
            deepcopy(self.grid),
        )

    # Special methods: Collection emulation
    def __len__(self) -> int:
        """Return the number of polynomials in the instance.

        Returns
        -------
        int
            The number of polynomials in the instance. A single instance of
            polynomial may contain multiple polynomials with different
            coefficient values but sharing the same underlying multi-index set
            and grid.
        """
        if self.coeffs.ndim == 1:
            return 1

        return self.coeffs.shape[1]

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
        target_dimension: int,
    ) -> "MultivariatePolynomialSingleABC":
        """Expand the spatial dimension of the polynomial instance.

        Parameters
        ----------
        target_dimension : int
            The new spatial dimension. It must be larger than or equal to the
            current spatial dimension of the polynomial.

        Returns
        -------
        MultivariatePolynomialSingleABC
            The new instance of polynomial with expanded dimension.
        """
        # Expand the dimension of the multi-index set
        mi = self.multi_index.expand_dim(target_dimension)

        # Expand the underlying grid if necessary
        if self.grid.spatial_dimension < target_dimension:
            grd = self.grid.expand_dim(target_dimension)
        else:
            grd = self.grid

        return self.__class__(
            mi,
            self._coeffs,
            grid=grd,
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

    # --- Private utility methods: Not supposed to be called from the outside
    def _verify_operands(
        self,
        other: "MultivariatePolynomialSingleABC",
    ) -> Tuple[
         "MultivariatePolynomialSingleABC",
         "MultivariatePolynomialSingleABC",
         ]:
        """Verify the operands are valid before moving on."""
        # Only supported for polynomials of the same concrete class
        if self.__class__ != other.__class__:
            raise TypeError(
                f"Unsupported operation for "
                f"'{self.__class__}' and '{other.__class__}'"
            )

        # Check if the number of coefficients is consistent
        if len(self) != len(other):
            raise ValueError(
                "Cannot add polynomials with inconsistent "
                "number of coefficient sets"
            )
        return self, other


def _verify_grid(
    multi_index: MultiIndexSet,
    grid: Optional[Grid],
    domain: Optional[Domain]
) -> Grid:
    """Verify the given Grid instance.

    Parameters
    ----------
    multi_index : MultiIndexSet
        The multi-index set of the polynomial.
    grid : Grid, optional
        Grid to be validated; If ``None``, the default Grid instance will be
        constructed.
    domain : Domain, optional
        Domain to create the grid or validation. If ``grid`` is ``None``,
        this domain is used to create the grid. If both ``grid``
        and ``domain`` are provided, they must match.

    Returns
    -------
    Grid
        Validated grid instance.

    Raises
    ------
    TypeError
        If ``grid`` is not a Grid instance.
    DomainMismatchError
        If both ``grid`` and ``domain`` are provided, but they don't match.
    ValueError
        If ``multi_index`` is not a subset of ``grid.multi_index``.
    """
    if grid is not None:

        if not isinstance(grid, Grid):
            raise TypeError(f"grid must be a Grid instance, got {type(grid)}")

        if domain is not None and domain != grid.domain:
            raise DomainMismatchError(
                f"grid domain {grid.domain} does not match "
                f"the given domain {domain}"
            )

        # A grid multi-index must be able to support the given polynomial
        if not grid.multi_index.is_superset(multi_index):
            raise ValueError(
                "The given multi-index set must be a subset of "
                "the indices of the given grid"
            )
    else:

        if domain is None:
            # Create a default grid instance
            grid = Grid(multi_index)
        else:
            grid = Grid(multi_index, domain=domain)

    return grid


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
