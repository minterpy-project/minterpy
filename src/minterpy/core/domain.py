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

from minterpy.global_settings import FLOAT_DTYPE
from minterpy.utils.verification import verify_domain_bounds

__all__ = ["Domain"]


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
        self.bounds = verify_domain_bounds(bounds)

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
        """Check whether the domain is normalized.

        Returns
        -------
        bool
            ``True`` if the domain is normalized, ``False`` otherwise.
        """
        return (
            np.all(self.lower_bounds == -1.0) and
            np.all(self.upper_bounds == 1.0)
        )

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

    def partial_matching(self, other: "Domain") -> bool:
        """Compare two instances of domain up to the common dimension.

        Parameters
        ----------
        other : Domain
            An instance of :class:`Domain` that is to be compared with
            the current instance.

        Returns
        -------
        bool
            ``True`` if the two instances matches up to the common dimension,
            ``False`` otherwise.
        """
        dim = min(self.spatial_dimension, other.spatial_dimension)

        return np.all(self.bounds[:dim, :] == other.bounds[:dim, :])

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


    # --- Dunder methods
    def __eq__(self, other: "Domain") -> bool:
        """Compare two instances of Domain for exact equality in value.

        Two instances of :class:`Domain` class is equal in value if
        and only if both the underlying bounds are equal.

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
        return (
            self.spatial_dimension == other.spatial_dimension and
            np.all(self.bounds == other.bounds)
        )
