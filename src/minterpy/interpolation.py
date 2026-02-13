"""A top-level module with interfaces to conveniently create interpolants.

The main purpose of Minterpy is to interpolate given functions provided as
Python Callables.
This top-level module provides a convenient function named `interpolate`,
which outputs a Callable of the type `Interpolant`.
This represents the given function as a multidimensional (Newton) polynomial.

Moreover, it is also possible to construct an `Interpolator` instance.
This object precomputes and caches all the necessary ingredients
for the interpolation of any functions.

+-------------------+---------------------------------------------------------+
| Function / Class  | Description                                             |
+===================+=========================================================+
| `interpolate`     | Interpolate a given function                            |
+-------------------+---------------------------------------------------------+
| `Interpolant`     | Class that represents an interpolated function          |
+-------------------+---------------------------------------------------------+
| `Interpolator`    | Class that represents interpolators for given functions |
+-------------------+---------------------------------------------------------+

"""
import attrs
import numpy as np

from numpy.typing import ArrayLike
from typing import Callable, Optional

from minterpy.core import Grid, MultiIndexSet, Domain
from minterpy.dds import dds
from minterpy.polynomials import (
    NewtonPolynomial,
    LagrangePolynomial,
    CanonicalPolynomial,
    ChebyshevPolynomial,
)
from minterpy.transformations import NewtonToCanonical, NewtonToChebyshev
from minterpy.global_settings import DEFAULT_LP_DEG

__all__ = ["Interpolator", "Interpolant", "interpolate"]


class InterpolationError(Exception):
    """Exception raised if the interpolation went wrong."""
    pass


@attrs.define(frozen=True, order=False, eq=False)
class Interpolator:
    r"""The class for constructing polynomial interpolant.

    The class contains all the relevant parts for constructing a polynomial
    interpolant in the Newton basis of a given callable.
    The instance of this class is a callable; passing a function to it will
    return an interpolating polynomial.

    With an instance of this class, one can construct interpolants of the
    same underlying polynomial for different functions.

    Parameters
    ----------
    spatial_dimension : int
        The dimension of the interpolator.
    poly_degree : int
        The degree of the interpolating polynomial.
    lp_degree : float
        The degree :math:`p` of the :math:`l_p`-norm used to define
        the (multivariate) polynomial degree.
    bounds : array_like, optional
        The bounds of the domain space, an array of shape ``(m, 2)``,
        where ``m`` is the spatial dimension. Each row corresponds to the
        lower and upper bounds of the domain in the corresponding dimension.
        If not provided, the bounds are assumed to be :math:`[-1, 1]^m`.

    Attributes
    ----------
    multi_index : MultiIndexSet
        The multi-index set used to define the interpolating polynomial,
        a lexicographically complete multi-index set, i.e.,
        :math:`\mathcal{A}_{m, n, p}`, where :math:`m` is
        ``spatial_dimension``, :math:`n` is ``poly_degree``,
        and :math:`p` is the ``lp_degree``.
    grid : Grid
        The underlying interpolation grid with the domain defined
        by ``bounds``.
    """
    # Constructor parameters
    _spatial_dimension: int = attrs.field(repr=False)
    _poly_degree: int = attrs.field(repr=False)
    _lp_degree: float = attrs.field(repr=False)
    _bounds: Optional[ArrayLike] = attrs.field(default=None, repr=False)

    # --- Core properties (computed based on constructor parameters)
    multi_index: MultiIndexSet = attrs.field(init=False, repr=False)
    grid: Grid = attrs.field(init=False, repr=False)

    @multi_index.default
    def _multi_index_default(self) -> MultiIndexSet:
        return MultiIndexSet.from_degree(
            self._spatial_dimension,
            self._poly_degree,
            self._lp_degree
        )

    @grid.default
    def _grid_default(self) -> Grid:
        if self._bounds is not None:
            domain = Domain(self._bounds)
        else:
            domain = None
        return Grid(self.multi_index, domain=domain)

    # --- Derived properties
    @property
    def spatial_dimension(self) -> int:
        """The dimension of the interpolator."""
        return self.multi_index.spatial_dimension

    @property
    def poly_degree(self) -> int:
        """The degree of the interpolating polynomial (and multi-index set)."""
        return self.multi_index.poly_degree

    @property
    def lp_degree(self) -> float:
        """:math:`p` of :math:`l_p`-norm used to define the multi-index set."""
        return self.multi_index.lp_degree

    @property
    def domain(self) -> Domain:
        """The domain of the interpolating polynomial."""
        return self.grid.domain

    # --- Dunder methods
    def __repr__(self) -> str:
        return (
            f"Interpolator(spatial_dimension={self.spatial_dimension}, "
            f"poly_degree={self.poly_degree}, "
            f"lp_degree={self.lp_degree})"
        )

    def __call__(self, func: Callable) -> NewtonPolynomial:
        """Interpolate a given function and return an interpolating polynomial.

        Parameters
        ----------
        func : Callable
            The function to interpolate. It must accept as the first argument
            a :class:`numpy:numpy.ndarray` with shape ``(k, m)``, where ``k``
            is the number of evaluation points and ``m`` is the spatial
            dimension. The function returns a :class:`numpy:numpy.ndarray`
            with shape ``(k, )`` for scalar outputs and ``(k, d)``
            for vector outputs, where ``d``  is the output dimension.

        Returns
        -------
        NewtonPolynomial
            An interpolating polynomial of the function ``func`` in the Newton
            basis. The interpolating polynomial is constructed according
            to the underlying multi-index set and interpolation grid.

        Raises
        ------
        InterpolationError
            If anything goes wrong with the interpolation.
        """
        try:
            func_values = self.grid(func)
        except Exception as e:
            raise InterpolationError(e) from e

        return self.interpolate_values(func_values)

    def interpolate_values(self, func_values: ArrayLike) -> NewtonPolynomial:
        """Interpolate a given array of values at the unisolvent nodes.

        Parameters
        ----------
        func_values : array_like
            The function values at the unisolvent nodes (i.e., the Lagrange
            coefficients of the interpolating polynomial).

        Returns
        -------
        NewtonPolynomial
            An interpolating polynomial of the function ``func`` in the Newton
            basis. The interpolating polynomial is constructed according
            to the underlying multi-index set and interpolation grid.

        Raises
        ------
        InterpolationError
            If anything goes wrong with the interpolation.
        """
        try:
            coeffs = dds(func_values, self.grid.tree)
            # DDS returns shape (N, d) where N=#points, d=output_dim
            # For scalar functions (d=1), convert to 1D array of shape (N,)
            if coeffs.shape[1] == 1:
                coeffs = coeffs[:, 0]  # Most explicit and safe
        except Exception as e:
            raise InterpolationError(e) from e

        return NewtonPolynomial(self.multi_index, coeffs, grid=self.grid)


@attrs.define(frozen=True, order=False, eq=False)
class Interpolant:
    """A class representing the result of an interpolation of a given function.

    An instance of this class is a callable that interpolates a given function.
    It serves as an intermediate layer between function approximation
    and the corresponding interpolating polynomial representation; it can be
    used without any in-depth knowledge of the underlying polynomial
    representation.

    Parameters
    ----------
    func : Callable
        The function to interpolate. It must accept as the first argument
        a :class:`numpy:numpy.ndarray` with shape ``(k, m)``, where ``k``
        is the number of evaluation points and ``m`` is the spatial
        dimension. The function returns a :class:`numpy:numpy.ndarray`
        with shape ``(k, )`` for scalar outputs and ``(k, d)``
        for vector outputs, where ``d``  is the output dimension.
    interpolator : Interpolator
        The underlying setting for the interpolation.
    """
    # --- Constructor parameters
    func: Callable = attrs.field(repr=False)
    interpolator: Interpolator = attrs.field(repr=False)

    # --- Private attribute
    _func_values: np.ndarray = attrs.field(init=False, repr=False)
    _interpolation_poly: NewtonPolynomial = attrs.field(init=False, repr=False)

    @_func_values.default
    def _func_values_default(self):
        return self.interpolator.grid(self.func)

    @_interpolation_poly.default
    def _interpolation_poly_default(self):
        return self.interpolator.interpolate_values(self._func_values)

    # --- Factory method
    @classmethod
    def from_degree(
        cls,
        func: Callable,
        spatial_dimension: int,
        poly_degree: int,
        lp_degree: float,
        bounds: Optional[ArrayLike] = None,
    ) -> "Interpolant":
        r"""Create an interpolant with respect to a complete multi-index set.

        Parameters
        ----------
        func : Callable
            The function to interpolate. It must accept as the first argument
            a :class:`numpy:numpy.ndarray` with shape ``(k, m)``, where ``k``
            is the number of evaluation points and ``m`` is the spatial
            dimension. The function returns a :class:`numpy:numpy.ndarray`
            with shape ``(k, )`` for scalar outputs and ``(k, d)``
            for vector outputs, where ``d``  is the output dimension.
        spatial_dimension : int
            The dimension of the interpolator.
        poly_degree : int
            The degree of the interpolating polynomial.
        lp_degree : float
            The degree :math:`p` of the :math:`l_p`-norm used to define
            the (multivariate) polynomial degree.
        bounds : array_like, optional
            The bounds of the domain space, an array of shape ``(m, 2)``,
            where ``m`` is the spatial dimension. Each row corresponds to the
            lower and upper bounds of the domain in the corresponding dimension.
            If not provided, the bounds are assumed to be :math:`[-1, 1]^m`.

        Returns
        --------
        Interpolant
            An instance of interpolant of ``func`` using an interpolating
            polynomial with respect to complete multi-index set.

        Notes
        -----
        - ``spatial_dimension`` (:math:`m`), ``poly_degree`` (:math:`n`),
          ``lp_degree`` (:math:`p`) are used to construct the lexicographically
          complete multi-index set :math:`\mathcal{A}_{m, n, p}`.
        """
        return cls(
            func,
            Interpolator(spatial_dimension, poly_degree, lp_degree, bounds),
        )

    # --- Properties
    @property
    def spatial_dimension(self) -> int:
        """The dimension of the interpolator."""
        return self.interpolator.spatial_dimension

    @property
    def poly_degree(self) -> int:
        """The degree of the interpolating polynomial."""
        return self.interpolator.poly_degree

    @property
    def lp_degree(self) -> float:
        """:math:`p` of :math:`l_p`-norm used to define the multi-index set."""
        return self.interpolator.lp_degree

    @property
    def multi_index(self) -> MultiIndexSet:
        """The multi-index set defining the interpolating polynomial."""
        return self.interpolator.multi_index

    # --- Public methods
    def to_newton(self) -> NewtonPolynomial:
        """Return the interpolant as a polynomial in the Newton basis.

        Returns
        -------
        NewtonPolynomial
            The interpolating polynomial represented in the Newton basis.
        """
        return self._interpolation_poly

    def to_lagrange(self) -> LagrangePolynomial:
        """Return the interpolant as a polynomial in the Lagrange basis.

        Returns
        -------
        LagrangePolynomial
            The interpolating polynomial represented in the Lagrange basis.
        """
        return LagrangePolynomial.from_grid(
            self.interpolator.grid,
            self._func_values,
        )

    def to_canonical(self) -> CanonicalPolynomial:
        """Return the interpolant as a polynomial in the canonical basis.

        Returns
        -------
        CanonicalPolynomial
            The interpolating polynomial represented in the canonical
            (monomial) basis.
        """
        nwt_poly = self._interpolation_poly

        return NewtonToCanonical(nwt_poly)()

    def to_chebyshev(self) -> ChebyshevPolynomial:
        """Return the interpolant as a polynomial in the Chebyshev basis.

        Returns
        -------
        ChebyshevPolynomial
            The interpolating polynomial represented in the Chebyshev basis
            (of the first kind).
        """
        nwt_poly = self._interpolation_poly

        return NewtonToChebyshev(nwt_poly)()

    # --- Dunder methods
    def __call__(self, xx: ArrayLike, **kwargs) -> np.ndarray:
        """Evaluate the interpolant on a given array of points.

        Parameters
        ----------
        xx : array_like
            Query points to evaluate. Can be a scalar, list, or numpy array.
            The points will be standardized to a two-dimensional array of
            shape ``(k, m)`` where ``k`` is the number of query points and
            ``m`` is the spatial dimension of the polynomial.
        **kwargs
            Additional keyword-only arguments that change the behavior of
            the underlying evaluation (see the concrete implementation of the
            interpolating polynomial).

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The values of the polynomial evaluated at query points.
        """
        return self._interpolation_poly(xx)


def interpolate(
    func: Callable,
    spatial_dimension: int,
    poly_degree: int,
    lp_degree: float = DEFAULT_LP_DEG,
    bounds: Optional[ArrayLike] = None,
) -> Interpolant:
    r"""Interpolate a function using a complete multi-index set polynomial.

    Parameters
    ----------
    func : Callable
        The function to interpolate. It must accept as the first argument
        a :class:`numpy:numpy.ndarray` with shape ``(k, m)``, where ``k``
        is the number of evaluation points and ``m`` is the spatial
        dimension. The function returns a :class:`numpy:numpy.ndarray`
        with shape ``(k, )`` for scalar outputs and ``(k, d)``
        for vector outputs, where ``d``  is the output dimension.
    spatial_dimension : int
        The dimension of the interpolator.
    poly_degree : int
        The degree of the interpolating polynomial.
    lp_degree : float, optional
        The degree :math:`p` of the :math:`l_p`-norm used to define
        the (multivariate) polynomial degree.
    bounds : array_like, optional
        The bounds of the domain space, an array of shape ``(m, 2)``,
        where ``m`` is the spatial dimension. Each row corresponds to the
        lower and upper bounds of the domain in the corresponding dimension.
        If not provided, the bounds are assumed to be :math:`[-1, 1]^m`.

    Returns
    -------
    Interpolant
        The interpolant of ``func`` with respect to the lexicographically
        complete multi-index set :math:`\mathcal{A}_{m, n, p}` where :math:`m`
        is ``spatial_dimension``, :math:`n` is ``poly_degree``, and :math:`p`
        is ``lp_degree``.
    """
    m, n, p = spatial_dimension, poly_degree, lp_degree

    return Interpolant.from_degree(func, m, n, p, bounds)
