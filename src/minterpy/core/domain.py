"""
This module contains the implementation of the `Domain` class.

The `Domain` class represents the domain of the polynomials, i.e., the set of
all input points for which the polynomials are defined. As the internals of
Minterpy work for interpolating polynomial in the so-called "normalized"
domain (i.e., :math:`[-1, 1]^m`), the class provides convenient methods for:

- transformation of values to/from normalized domain in :math:`[-1, 1]^m`
- computing the scaling factor in polynomial differentiation
- computing the scaling factor in polynomial integration

The main assumption of the transformation is that the transformation is
affine applied to each dimension separately.
"""
import numpy as np

from typing import Optional, Union

from minterpy.global_settings import FLOAT_DTYPE
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
        The domain bounds as a 2D array with shape (m, 2),
        where m is the spatial dimension.
        The first column contains the lower bounds and the second column
        the upper bounds across dimensions.
    """

    def __init__(self, bounds: np.ndarray):

        # Verify and assign the bounds
        self._bounds = verify_domain_bounds(bounds)

    # --- Factory methods
    @classmethod
    def uniform(
        cls,
        spatial_dimension: int,
        lower_bound: float,
        upper_bound: float,
    ):
        """Create an instance of Domain with uniform bounds across dimensions.

        Creates a hyper-rectangular domain :math:`[a, b]^m` where :math:`a`
        and :math:`b` are the lower and upper bounds for each dimension,
        respectively.

        Parameters
        ----------
        spatial_dimension : int
            The number of dimensions for the space.
        lower_bound : float
            The lower bound of each dimension.
        upper_bound : float
            The upper bound of each dimension.

        Returns
        -------
        Domain
            A new instance of the `Domain` class initialized with uniform
            bounds, i.e., the same lower and upper bound for all dimensions.
        """
        bounds = np.empty((spatial_dimension, 2), dtype=FLOAT_DTYPE)
        bounds[:, 0] = lower_bound
        bounds[:, 1] = upper_bound

        return cls(bounds)

    @classmethod
    def normalized(cls, spatial_dimension: int):
        """Create an instance of Domain with [-1, 1] bound across dimensions.

        Parameters
        ----------
        spatial_dimension : int
            The number of dimensions for the space.

        Returns
        -------
        Domain
            A new instance of the `Domain` class initialized with normalized
            bounds, i.e., :math:`[-1, 1]^m` where :math:`m`
            is the spatial dimension.
        """
        bounds = np.empty((spatial_dimension, 2), dtype=FLOAT_DTYPE)
        bounds[:, 0] = -1.0
        bounds[:, 1] = 1.0

        return cls(bounds)

    # --- Properties
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
    def lower_bounds(self) -> np.ndarray:
        """The lower bounds of the domain.

        Returns
        -------
        np.ndarray
            The lower bounds of the domain as a one-dimensional array having
            shape ``(m, )``.
        """
        return self.bounds[:, 0]

    @property
    def upper_bounds(self) -> np.ndarray:
        """The upper bounds of the domain.

        Returns
        -------
        np.ndarray
            The upper bounds of the domain as a one-dimensional array having
            shape ``(m, )``.
        """
        return self.bounds[:, 1]

    @property
    def domain_widths(self) -> np.ndarray:
        """The widths of the domain.

        Returns
        -------
        np.ndarray
            The difference between the upper and lower bounds of the domain
            as a one-dimensional array having shape ``(m, )``.
        """
        return self.upper_bounds - self.lower_bounds

    @property
    def is_normalized(self) -> bool:
        """Check whether the domain is normalized in :math:`[-1, 1]^m`.

        Returns
        -------
        bool
            ``True`` if the domain is normalized, ``False`` otherwise.

        Notes
        -----
        - This check uses numerical tolerances (``DEFAULT_RTOL`` and
          ``DEFAULT_ATOL``) for robustness against floating-point errors.
        """
        # Get the default tolerances
        rtol = DEFAULT_RTOL
        atol = DEFAULT_ATOL

        # Compute peak-to-peak width of the domain
        lb = bool(np.isclose(self.lower_bounds, -1.0, rtol, atol).all())
        ub = bool(np.isclose(self.upper_bounds, 1.0, rtol, atol).all())

        return lb and ub

    @property
    def is_uniform(self) -> bool:
        """Check whether the domain is uniform.

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
        - This check uses numerical tolerances (``DEFAULT_RTOL`` and
          ``DEFAULT_ATOL``) for robustness against floating-point errors.
        """
        # Get the default tolerances
        rtol = DEFAULT_RTOL
        atol = DEFAULT_ATOL

        # Lower bound condition
        lb = self.lower_bounds[0]
        lb_c = np.allclose(self.lower_bounds, lb, rtol=rtol, atol=atol)

        # Upper bound condition
        ub = self.upper_bounds[0]
        ub_c = np.allclose(self.upper_bounds, ub, rtol=rtol, atol=atol)

        # Compute peak-to-peak width of the domain
        ptp = np.ptp(self.domain_widths)
        ptp_c = np.allclose(ptp, 0.0, rtol=rtol, atol=atol)

        return bool(lb_c and ub_c and ptp_c)

    # --- Instance methods
    def map_to_normalized(
        self,
        xx: np.ndarray,
        validate: bool = False,
    ) -> np.ndarray:
        """Map input points to normalized :math:`[-1, 1]^m` domain.

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            The input points to be mapped to the normalized domain.
        validate : bool, optional
            If True, validate that input points are within domain bounds, i.e.,
            no extrapolation is allowed.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The normalized points in :math:`[-1, 1]^m` domain, except when
            extrapolated (with ``validate=False``).
        """
        if validate and not np.all(self.contains(xx)):
            raise ValueError(
                "Input points are outside of domain bounds. "
                "Set validate=False to allow extrapolation."
            )

        return -1 + 2 * (xx - self.lower_bounds) / self.domain_widths

    def map_from_normalized(
        self,
        xx: np.ndarray,
        validate: bool = False,
    ) -> np.ndarray:
        """Map normalized points in :math:`[-1, 1]^m` to the original domain.

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            The input points in the normalized domain to be mapped
            to the original domain.
        validate : bool, optional
            If True, validate that input points are within domain bounds, i.e.,
            no extrapolation is allowed.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            The points in the original domain.
        """
        if validate and not np.all((xx <= 1.0) & (xx >= -1.0)):
            raise ValueError(
                "Input points are outside of [-1, 1]^m domain. "
                "Set validate=False to allow extrapolation."
            )

        return self.lower_bounds + (xx + 1) / 2 * self.domain_widths

    def get_int_factor(self) -> float:
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
        return float(np.prod(self.domain_widths / 2.0))

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
        idx = order > 0

        return float(np.prod((2.0 / self.domain_widths[idx])**(order[idx])))

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

        return np.allclose(
            self.bounds[:dim, :],
            other.bounds[:dim, :],
            rtol=rtol,
            atol=atol,
        )

    def contains(self, xx: np.ndarray) -> np.ndarray:
        """Check whether the input points are contained in the domain.

        Parameters
        ----------
        xx : :class:`numpy:numpy.ndarray`
            A set of values to be checked.

        Returns
        -------
        :class:`numpy:numpy.ndarray`
            A boolean array indicating whether each point is contained
            in the domain.
        """
        return (
                np.all(self.lower_bounds <= xx, axis=1)
                & np.all(xx <= self.upper_bounds, axis=1)
        )

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

            lb = self.lower_bounds[0]
            ub = self.upper_bounds[0]

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

        return np.array_equal(self.bounds, other.bounds)

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
