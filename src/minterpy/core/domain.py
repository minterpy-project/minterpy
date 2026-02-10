"""
This module contains the implementation of the `Domain` class.

The `Domain` class represents the domain of the polynomials, i.e., the set of
all input points for which the polynomials are defined. As the internals of
Minterpy work for interpolating polynomial in the so-called "internal"
domain (i.e., :math:`[-1, 1]^m`), the class provides convenient methods for:

- transformation of values to/from normalized domain in :math:`[-1, 1]^m`
- computing the scaling factor in polynomial differentiation
- computing the scaling factor in polynomial integration

The main assumption of the transformation is that the transformation is
affine applied to each dimension separately.
"""
import numpy as np

from typing import Optional, Union

from minterpy.global_settings import DEFAULT_DOMAIN, FLOAT_DTYPE
from minterpy.utils.verification import verify_domain_bounds
from minterpy.utils.exceptions import DomainMismatchError

__all__ = ["Domain"]


DEFAULT_ATOL = 1e-12
DEFAULT_RTOL = 1e-9


class Domain:
    """A class representing the domain of the polynomials.

    A domain of a polynomial is the set of all input points
    for which the polynomials are defined.
    A domain is defined by the lower and upper bounds per dimension.
    Currently, only domains with finite bounds are supported.

    Parameters
    ----------
    bounds : np.ndarray
        The domain bounds as a 2D array with shape ``(m, 2)``,
        where ``m`` is the spatial dimension.
        The first column contains the lower bounds and the second column
        the upper bounds across dimensions.
    """
    _INTERNAL_BOUNDS: np.ndarray = DEFAULT_DOMAIN

    def __init__(self, bounds: np.ndarray):

        # Verify and assign the bounds
        self._bounds = verify_domain_bounds(bounds)

        # Assign lazily-evaluated properties
        self._internal_bounds = None
        self._is_identity = None

    # --- Factory methods
    @classmethod
    def uniform(
        cls,
        spatial_dimension: int,
        lower: float,
        upper: float,
    ):
        r"""Create an instance with uniform bounds.

        Creates a hyper-rectangular domain :math:`[a, b]^m` where :math:`a`
        and :math:`b` are the lower and upper bounds for each dimension,
        respectively.

        Parameters
        ----------
        spatial_dimension : int
            The number of dimensions for the space.
        lower : float
            The lower bound of each dimension.
        upper : float
            The upper bound of each dimension.

        Returns
        -------
        Domain
            A new instance of the `Domain` class initialized with uniform
            bounds, i.e., the same lower and upper bound for all dimensions.
        """
        bounds = np.empty((spatial_dimension, 2), dtype=FLOAT_DTYPE)
        bounds[:, 0] = lower
        bounds[:, 1] = upper

        return cls(bounds)

    @classmethod
    def normalized(cls, spatial_dimension: int):
        r"""Create an instance with the default internal bounds.

        The default internal bounds are :math:`[-1, 1]^m`.

        Parameters
        ----------
        spatial_dimension : int
            The number of dimensions for the space.

        Returns
        -------
        Domain
            A new instance of the `Domain` class initialized with the default
            internal bounds, i.e., :math:`[-1, 1]^m` where :math:`m`
            is the spatial dimension.
        """
        internal_bounds = cls._INTERNAL_BOUNDS[np.newaxis, :]
        bounds = np.repeat(internal_bounds, spatial_dimension, axis=0)

        return cls(bounds)

    # --- Properties (public)
    @property
    def bounds(self) -> np.ndarray:
        """The domain bounds.

        Return
        ------
        np.ndarray
            The domain bounds as a 2D array with shape ``(m, 2)``,
            where ``m`` is the spatial dimension.
            The first column contains the lower bounds and the second column
            the upper bounds across dimensions.
        """
        return self._bounds

    @property
    def spatial_dimension(self):
        """Dimension of the domain space.

        Return
        ------
        int
            The dimension of the domain space.
        """
        return len(self.bounds)

    @property
    def lowers(self) -> np.ndarray:
        """The lower bounds of the domain.

        Returns
        -------
        np.ndarray
            The lower bounds of the domain as a one-dimensional array having
            shape ``(m, )``.
        """
        return self.bounds[:, 0]

    @property
    def uppers(self) -> np.ndarray:
        """The upper bounds of the domain.

        Returns
        -------
        np.ndarray
            The upper bounds of the domain as a one-dimensional array having
            shape ``(m, )``.
        """
        return self.bounds[:, 1]

    @property
    def widths(self) -> np.ndarray:
        """The widths of the domain.

        Returns
        -------
        np.ndarray
            The difference between the upper and lower bounds of the domain
            as a one-dimensional array having shape ``(m, )``.
        """
        return self.uppers - self.lowers

    @property
    def is_identity(self) -> bool:
        """Check whether the domain is (approx) equal to the internal domain.

        Returns
        -------
        bool
            ``True`` if the domain is an identity, ``False`` otherwise.

        Notes
        -----
        - This check uses numerical tolerances (``DEFAULT_RTOL`` and
          ``DEFAULT_ATOL``) for robustness against floating-point errors.
        - When the domain is an identity, transformations and scaling factors
          computation can usually be skipped.
        """
        if self._is_identity is None:
            # Get the default tolerances
            rtol = DEFAULT_RTOL
            atol = DEFAULT_ATOL

            lb = np.allclose(self.lowers, self._internal_lowers, rtol, atol)
            ub = np.allclose(self.uppers, self._internal_uppers, rtol, atol)

            self._is_identity = bool(lb and ub)

        return self._is_identity

    @property
    def is_uniform(self) -> bool:
        r"""Check whether the domain is uniform.

        A uniform domain has the same lower and upper bounds in all dimensions,
        i.e., the domain has the form :math:`[a, b]^m` for some :math:`a`
        and :math:`b`.

        Returns
        -------
        bool
            ``True`` if the domain is uniform, ``False`` otherwise.

        Notes
        -----
        - The dimension of a uniform domain can be expanded by extrapolating
          the bounds of the extra dimension from the bounds of the other
          dimensions.
        - A domain of dimension 1 is always uniform by definition.
        - A normalized domain is always uniform, but not vice versa.
        - This check uses numerical tolerances (``DEFAULT_RTOL`` and
          ``DEFAULT_ATOL``) for robustness against floating-point errors.
        """
        # Get the default tolerances
        rtol = DEFAULT_RTOL
        atol = DEFAULT_ATOL

        # Lower bound condition
        lb_c = np.allclose(self.lowers, self.lowers[0], rtol=rtol, atol=atol)

        # Upper bound condition
        ub_c = np.allclose(self.uppers, self.uppers[0], rtol=rtol, atol=atol)

        return bool(lb_c and ub_c)

    @property
    def internal_bounds(self) -> np.ndarray:
        """The bounds of the internal domain.

        Returns
        -------
        np.ndarray
            The bounds of the internal domain as a 2D array with shape
            ``(m, 2)``, where ``m`` is the spatial dimension.
            The first column contains the lower bounds and the second column
            the upper bounds across dimensions.
        """
        if self._internal_bounds is None:
            # Note: Use the default hard-coded internal bounds
            self._internal_bounds = np.repeat(
                self._INTERNAL_BOUNDS[np.newaxis, :],
                self.spatial_dimension,
                axis=0,
            )

        return self._internal_bounds

    # --- Properties (private)
    @property
    def _internal_lowers(self) -> np.ndarray:
        """The lower bounds of the internal domain.

       Returns
        -------
        np.ndarray
            The lower bounds of the internal domain as a one-dimensional array
            having shape ``(m, )``.
        """
        return self.internal_bounds[:, 0]

    @property
    def _internal_uppers(self) -> np.ndarray:
        """"The upper bounds of the internal domain.

        Returns
        -------
        np.ndarray
            The upper bounds of the internal domain as a one-dimensional array
            having shape ``(m, )``.
        """
        return self.internal_bounds[:, 1]

    @property
    def _internal_widths(self) -> np.ndarray:
        """"The widths of the internal domain.

        Returns
        -------
        np.ndarray
            The difference between the upper and lower bounds of the internal
            domain as a one-dimensional array having shape ``(m, )``.
        """
        return self._internal_uppers - self._internal_lowers

    # --- Instance methods
    def map_to_internal(
        self,
        xx: np.ndarray,
        validate: bool = False,
    ) -> np.ndarray:
        r"""Map input points to the internal domain.

        The default internal domain is :math:`[-1, 1]^m`.

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            The input points to be mapped to the internal domain.
        validate : bool, optional
            If True, validate that input points are within domain bounds, i.e.,
            no extrapolation is allowed.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The points in the internal domain, except when extrapolated
            (with ``validate=False``).
        """
        if validate and not np.all(self.contains(xx)):
            raise ValueError(
                "Input points are outside of domain bounds. "
                "Set validate=False to allow extrapolation."
            )

        ilb = self._internal_lowers
        iwidths = self._internal_widths

        return ilb + iwidths * (xx - self.lowers) / self.widths

    def map_from_internal(
        self,
        xx: np.ndarray,
        validate: bool = False,
    ) -> np.ndarray:
        r"""Map points in the internal domain to the original domain.

        The default internal domain is :math:`[-1, 1]^m`.

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            The input points in the internal domain to be mapped
            to the original domain.
        validate : bool, optional
            If True, validate that input points are within domain bounds, i.e.,
            no extrapolation is allowed.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The points in the original domain.
        """
        if validate and not np.all(self.contains(xx, internal=True)):
            raise ValueError(
                "Input points are outside of the internal domain. "
                "Set validate=False to allow extrapolation."
            )

        ilb = self._internal_lowers
        iwidths = self._internal_widths

        return self.lowers + (xx - ilb) / iwidths * self.widths

    def int_factor(self) -> float:
        """Compute the scaling factor for polynomial integration.

        Returns
        -------
        float
            The scaling factor for polynomial integration.

        Notes
        -----
        - Here, we assume that the integration is carried out over all
          dimensions.
        """
        if self.is_identity:
            return 1.0

        return float(np.prod(self.widths / self._internal_widths))

    def get_diff_factor(self, order: np.ndarray) -> float:
        """Compute the scaling factor for polynomial differentiation.

        Parameters
        ----------
        order : :class:`numpy:numpy.ndarray`
            A one-dimensional integer array specifying the order of derivatives
            along each dimension. The length of the array must be ``m`` where
            ``m`` is the spatial dimension of the polynomial.

        Returns
        -------
        float
            The scaling factor for polynomial differentiation.
        """
        if self.is_identity:
            return 1.0

        idx = order > 0
        iwidths = self._internal_widths

        return float(np.prod((iwidths[idx] / self.widths[idx])**(order[idx])))

    def partial_matching(
        self,
        other: "Domain",
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> bool:
        """Compare two instances of domain up to the common dimension.

        This method performs approximate equality checking between two domains
        using numerical tolerances, comparing only up to the minimum spatial
        dimension of the two domains.
        This is useful for checking compatibility of domains of different
        dimensions before, e.g., merging.

        Parameters
        ----------
        other : Domain
            An instance of :class:`Domain` that is to be compared with
            the current instance.
        rtol : float, optional
            The relative tolerance parameter. If not specified,
            the module-level default ``DEFAULT_RTOL`` is used.
        atol : float, optional
            The absolute tolerance parameter. If not specified,
            the module-level default ``DEFAULT_ATOL`` is used.

        Returns
        -------
        bool
            ``True`` if the two instances matches up to the common dimension,
            ``False`` otherwise.
        """
        # Get the default tolerances
        rtol = DEFAULT_RTOL if rtol is None else rtol
        atol = DEFAULT_ATOL if atol is None else atol

        dim = min(self.spatial_dimension, other.spatial_dimension)

        # "User" bounds
        bounds_match = np.allclose(
            self.bounds[:dim, :],
            other.bounds[:dim, :],
            rtol=rtol,
            atol=atol,
        )

        # Check internal bounds (currently identical for all instances,
        # but enables future generalization of internal coordinate system)
        internal_bounds_match = np.allclose(
            self.internal_bounds[:dim, :],
            other.internal_bounds[:dim, :],
            rtol=rtol,
            atol=atol,
        )

        return bool(bounds_match and internal_bounds_match)

    def contains(self, xx: np.ndarray, internal: bool = False) -> np.ndarray:
        """Check whether the input points are contained in the domain.

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            A set of values to be checked.
        internal : bool, optional
            Check against internal bounds instead of the domain bounds.
            The default is ``False``.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            A boolean array indicating whether each point is contained
            in the domain.
        """
        if internal:
            lb = self._internal_lowers
            ub = self._internal_uppers
        else:
            lb = self.lowers
            ub = self.uppers

        return np.all(lb <= xx, axis=1) & np.all(xx <= ub, axis=1)

    def expand_dim(self, target: Union[int, "Domain"]) -> "Domain":
        """Expand the dimension of the domain.

        Parameters
        ----------
        target : Union[int, Domain]
            The target dimension to expand to. If an integer, it specifies
            the new dimension. If a Domain instance, it specifies the domain
            whose dimension to expand to.

        Returns
        -------
        Domain
            A new instance of the :class:`Domain` class with expanded
            dimension.

        Raises
        ------
        ValueError
            If the target dimension is smaller than the current dimension or
            an expansion to a target integer is attempted for an unnormalized
            domain.
        DomainMismatchError
            If the target domain does not match the current domain.
        TypeError
            If the target is not an instance of Domain or int.

        Notes
        -----
        - If the target domain is the current instance, the current instance
          is returned.
        """
        if isinstance(target, int):

            if target < self.spatial_dimension:
                raise ValueError(
                    f"Target dimension {target} cannot be smaller than "
                    f"the current dimension {self.spatial_dimension}"
                )

            if not self.is_uniform:
                raise ValueError(
                    "Non-uniform domain cannot be expanded due to ambiguous "
                    "bounds for the extra dimension."
                )

            lb = self.lowers[0]
            ub = self.uppers[0]

            return self.__class__.uniform(target, lb, ub)

        if isinstance(target, Domain):

            if not self.partial_matching(target):
                raise DomainMismatchError(
                    "Target domain does not match the current domain"
                )

            # If there's no need to expand, return the current instance
            if self is target:
                return self

            if target.spatial_dimension < self.spatial_dimension:
                raise ValueError(
                    f"Target dimension {target.spatial_dimension} cannot be "
                    "smaller than the current dimension "
                    f"{self.spatial_dimension}"
                )

            return self.__class__(target.bounds.copy())

        raise TypeError(
            "Target domain must be an instance of Domain or int, "
            f"got {type(target)} instead"
        )


    # --- Dunder methods
    def __eq__(self, other: "Domain") -> bool:
        """Compare two instances of Domain for exact equality in value.

        Two instances of :class:`Domain` class are considered equal in value if
        and only if their underlying bounds arrays are exactly equal.
        In other words, this is a strict comparison without any numerical
        tolerance.

        Parameters
        ----------
        other : Domain
            An instance of :class:`Domain` that is to be compared with
            the current instance.

        Returns
        -------
        bool
            ``True`` if the two instances are equal in value,
            ``False`` otherwise.
        """
        if not isinstance(other, Domain):
            return False

        bounds_eq = np.array_equal(self.bounds, other.bounds)
        internal_bounds_eq = np.array_equal(
            self.internal_bounds,
            other.internal_bounds,
        )

        return bool(bounds_eq and internal_bounds_eq)

    def __or__(self, other: "Domain") -> "Domain":
        """Combine two instances of Domain via the ``|`` operator.

        Two instances of Domain may be combined via the ``|`` operator if they
        are partially matched.

        Parameters
        ----------
        other : Domain
            An instance of :class:`Domain` that is to be combined with
            the current instance.

        Returns
        -------
        Domain
            A new instance of :class:`Domain` that is the result of the
            combination of the two instances.

        Raises
        ------
        DomainMismatchError
            If the two domains are not partially matched.
        """
        if self.spatial_dimension > other.spatial_dimension:
            return other.expand_dim(self)

        return self.expand_dim(other)