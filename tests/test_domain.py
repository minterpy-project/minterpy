import numpy as np
import pytest

from minterpy import Domain
from minterpy.utils.exceptions import InvalidDomainBoundsError

INVALID_BOUNDS = [
    {"type": "empty", "bounds": np.array([])},
    {"type": "nan", "bounds": np.array([[1, np.nan], [2, 3]])},
    {"type": "inf", "bounds": np.array([[1, 3], [2, np.inf]])},
    {"type": "too_few", "bounds": np.array([1])},
    {"type": "too_many", "bounds": np.array([1, 2, 3])},
    {"type": "wrong_shape", "bounds": np.array([[1], [2]])},
    {"type": "lower>upper", "bounds": np.array([3, 1])},
    {"type": "lower==upper", "bounds": np.array([[1, 1], [2, 2], [3, 3]])},
]


# Invalid values (`val`)
def _id_invalid_bounds(invalid_bounds):
    return f"val={invalid_bounds['type']:>12}"


@pytest.fixture(params=INVALID_BOUNDS, ids=_id_invalid_bounds)
def invalid_bounds(request):
    """Return invalid bounds fixture."""
    return request.param["bounds"]


@pytest.fixture
def random_bounds_valid(SpatialDimension):
    """Generate random valid bounds for testing."""
    bounds = np.random.uniform(low=0, high=5, size=(SpatialDimension, 2))
    bounds[:, 0] = -1 * bounds[:, 0]

    return bounds


@pytest.fixture
def random_bounds_pair(SpatialDimension):
    """Generate a pair of random valid bounds for testing."""
    bounds_1 = np.random.uniform(low=0, high=5, size=(SpatialDimension, 2))
    bounds_1[:, 0] = -1 * bounds_1[:, 0]

    bounds_2 = np.random.uniform(low=0, high=5, size=(SpatialDimension, 2))
    bounds_2[:, 0] = -1 * bounds_2[:, 0]

    return bounds_1, bounds_2


def generate_values_inbounds(random_bounds):
    """Generate random valid values for testing."""
    dim = len(random_bounds)
    lb, ub = random_bounds[:, 0], random_bounds[:, 1]
    bound_widths = ub - lb

    return lb + np.random.rand(10, dim) * bound_widths


def generate_values_outbounds(random_bounds):
    """Generate random out of bounds values for testing."""
    dim = len(random_bounds)
    xx = np.empty((10, dim))
    lb, ub = random_bounds[:, 0], random_bounds[:, 1]

    idx = np.random.randint(0, 10)
    for j in range(dim):
        for i in range(idx):
            xx[i, j] = lb[j] - np.random.rand()
        for i in range(10 - idx):
            xx[idx+i, j] = ub[j] + np.random.rand()

    return xx

class TestInit:
    """All tests related to the default construction of Domain."""

    def test_one_dimension(self):
        """Test that a 1D domain is correctly initialized."""
        bounds = np.array([-5, 5])
        domain = Domain(bounds)

        # Assertions
        assert domain.spatial_dimension == 1
        assert np.all(bounds == domain.bounds)

    def test_invalid_bounds_smaller_upper(self, invalid_bounds):
        """Test raising an exception for any invalid bounds."""

        with pytest.raises(InvalidDomainBoundsError):
            _ = Domain(invalid_bounds)


class TestFactoryMethods:
    """All tests related to the construction via factory methods."""

    def test_uniform_domain(self, SpatialDimension):
        """Test that a uniform domain is correctly initialized."""
        upper_bound = np.random.uniform(0, 5)
        lower_bound = -1 * upper_bound

        domain = Domain.uniform(SpatialDimension, lower_bound, upper_bound)

        # Assertions
        assert domain.spatial_dimension == SpatialDimension
        assert np.all(domain.bounds[:, 0] == lower_bound)
        assert np.all(domain.bounds[:, 1] == upper_bound)

    def test_normalized_domain(self, SpatialDimension):
        """Test that a normalized domain is correctly initialized."""
        domain = Domain.normalized(SpatialDimension)

        # Assertions
        assert domain.spatial_dimension == SpatialDimension
        assert np.all(domain.bounds[:, 0] == -1.)
        assert np.all(domain.bounds[:, 1] == 1.)


class TestProperties:
    """All tests related to the properties of Domain."""

    def test_spatial_dimension(self, random_bounds_valid):
        """Test the spatial dimension property."""
        my_dom = Domain(random_bounds_valid)

        # Assertion
        assert my_dom.spatial_dimension == random_bounds_valid.shape[0]

    def test_bounds(self, random_bounds_valid):
        """Test the bounds property."""
        my_dom = Domain(random_bounds_valid)

        # Assertions
        assert np.all(my_dom.bounds == random_bounds_valid)

    def test_lower_bounds(self, random_bounds_valid):
        """Test the lower bounds property."""
        my_dom = Domain(random_bounds_valid)

        # Assertion
        assert np.all(my_dom.lower_bounds == random_bounds_valid[:, 0])

    def test_upper_bounds(self, random_bounds_valid):
        """Test the upper bounds property."""
        my_dom = Domain(random_bounds_valid)

        # Assertion
        assert np.all(my_dom.upper_bounds == random_bounds_valid[:, 1])

    def test_domain_widths(self, random_bounds_valid):
        """Test the domain widths property."""
        my_dom = Domain(random_bounds_valid)

        # Assertion
        diff_bounds = random_bounds_valid[:, 1] - random_bounds_valid[:, 0]
        assert np.all(my_dom.domain_widths == diff_bounds)

    def test_normalized(self, SpatialDimension):
        """Test the normalized property for normalized domain."""
        my_dom_1 = Domain.normalized(SpatialDimension)
        my_dom_2 = Domain.uniform(SpatialDimension, -1, 1)
        bounds = np.ones((SpatialDimension, 2))
        bounds[:, 0] = -1
        my_dom_3 = Domain(bounds)

        # Assertions
        assert my_dom_1.is_normalized
        assert my_dom_2.is_normalized
        assert my_dom_3.is_normalized

    def test_not_normalized(self, random_bounds_valid):
        """"Test the normalized property for non-normalized domain."""
        my_dom = Domain(random_bounds_valid)

        # Assertion
        assert not my_dom.is_normalized


class TestMapValues:
    """All tests related to the mapping of values."""

    def test_to_normalized_edges(self, random_bounds_valid):
        """"Test the mapping of the edges to the normalized domain."""
        my_dom = Domain(random_bounds_valid)

        # Generate values at the edge of the domain
        xx = np.c_[my_dom.lower_bounds, my_dom.upper_bounds].T
        yy = my_dom.map_to_normalized(xx)

        # Assertions
        assert np.allclose(yy[0], -1.0)
        assert np.allclose(yy[1], 1.0)

    def test_from_normalized_edges(self, random_bounds_valid):
        """"Test the mapping of the edges from the normalized domain."""
        my_dom = Domain(random_bounds_valid)

        # Generate values at the edge of the normalized domain
        dim = my_dom.spatial_dimension
        xx = np.c_[-1 * np.ones(dim), np.ones(dim)].T
        yy = my_dom.map_from_normalized(xx)

        # Assertions
        assert np.allclose(yy[0], my_dom.lower_bounds)
        assert np.allclose(yy[1], my_dom.upper_bounds)

    def test_to_normalized_extrapolation(self, random_bounds_valid):
        """Test that extrapolation to normalized domain is correctly handled."""
        my_dom = Domain(random_bounds_valid)

        # Generate out of bounds values in the original domain
        xx = generate_values_outbounds(random_bounds_valid)
        yy = my_dom.map_to_normalized(xx)

        # Assertion
        assert np.any(yy <= -1.) or np.any(yy >= 1.)

    def test_from_normalized_extrapolation(self, random_bounds_valid):
        """Test that extrapolation from normalized domain is correctly handled.
        """
        my_dom = Domain(random_bounds_valid)

        # Generate out of bounds values in the normalized domain
        bounds = np.ones((my_dom.spatial_dimension, 2))
        bounds[:, 0] = -1
        xx = generate_values_outbounds(bounds)
        yy = my_dom.map_from_normalized(xx)

        # Assertion
        lb, ub = my_dom.lower_bounds, my_dom.upper_bounds
        assert np.any(yy <= lb) or np.any(yy >= ub)

    def test_to_normalized_valid(self, random_bounds_valid):
        """Test transformation to the normalized domain with valid values."""
        my_dom = Domain(random_bounds_valid)

        # Generate random valid input values
        xx = generate_values_inbounds(random_bounds_valid)
        # All the same for valid input values
        yy_1 = my_dom.map_to_normalized(xx)
        yy_2 = my_dom.map_to_normalized(xx, validate=True)
        yy_3 = my_dom.map_to_normalized(xx, validate=False)

        # Assertion
        assert np.all(yy_1 >= -1.) and np.all(yy_1 <= 1.)
        assert np.all(yy_2 >= -1.) and np.all(yy_2 <= 1.)
        assert np.all(yy_3 >= -1.) and np.all(yy_3 <= 1.)

    def test_to_normalized_invalid(self, random_bounds_valid):
        """Test transformation to the normalized domain with invalid values."""
        my_dom = Domain(random_bounds_valid)

        # Generate random invalid input values
        xx = generate_values_outbounds(random_bounds_valid)

        with pytest.raises(ValueError):
            _ = my_dom.map_to_normalized(xx, validate=True)

    def test_from_normalized_valid(self, random_bounds_valid):
        """Test transformation from the normalized domain with valid values."""
        my_dom = Domain(random_bounds_valid)

        # Generate random valid input values in the normalized domain
        xx = -1 + 2 * np.random.rand(10, my_dom.spatial_dimension)
        # All the same for valid input values
        yy_1 = my_dom.map_from_normalized(xx)
        yy_2 = my_dom.map_from_normalized(xx, validate=True)
        yy_3 = my_dom.map_from_normalized(xx, validate=False)

        # Assertion
        lb, ub = my_dom.lower_bounds, my_dom.upper_bounds
        assert np.all(yy_1 >= lb) and np.all(yy_1 <= ub)
        assert np.all(yy_2 >= lb) and np.all(yy_2 <= ub)
        assert np.all(yy_3 >= lb) and np.all(yy_3 <= ub)

    def test_from_normalized_validation_invalid(self, random_bounds_valid):
        """Test transformation from the normalized domain with invalid values.
        """
        my_dom = Domain(random_bounds_valid)

        # Generate random invalid input values in the normalized domain
        xx = -10 + 3 * np.random.rand(10, my_dom.spatial_dimension)

        # Assertion
        with pytest.raises(ValueError):
            _ = my_dom.map_from_normalized(xx, validate=True)

    def test_from_and_to(self, random_bounds_valid):
        """Test that values are correctly mapped from and to normalized."""
        my_dom = Domain(random_bounds_valid)

        # Generate random valid input values in the normalized domain
        xx_1 = -1 + 2 * np.random.rand(10, my_dom.spatial_dimension)
        yy = my_dom.map_from_normalized(xx_1)
        xx_2 = my_dom.map_to_normalized(yy)

        # Assertions
        assert np.allclose(xx_1, xx_2)

    def test_to_and_from(self, random_bounds_valid):
        """Test that values are correctly mapped to and from normalized."""
        my_dom = Domain(random_bounds_valid)

        # Generate random valid input values in the original domain
        xx_1 = generate_values_inbounds(random_bounds_valid)
        yy = my_dom.map_to_normalized(xx_1)
        xx_2 = my_dom.map_from_normalized(yy)

        # Assertions
        assert np.allclose(xx_1, xx_2)


class TestScalingFactor:
    """All tests related to the scaling factor."""

    def test_integration(self, random_bounds_valid):
        """Test the integration scaling factor."""
        my_dom = Domain(random_bounds_valid)
        diff_bounds = random_bounds_valid[:, 1] - random_bounds_valid[:, 0]

        assert np.isclose(
            my_dom.get_int_factor(),
            np.prod(diff_bounds) / 2**my_dom.spatial_dimension,
        )

    def test_integration_normalized(self, SpatialDimension):
        """Test the integration scaling factor for normalized domain."""
        dom = Domain.normalized(SpatialDimension)

        assert np.isclose(dom.get_int_factor(), 1.)

    def test_diff_zero_order(self, random_bounds_valid):
        """Test the differentiation scaling factor for 0th-order derivative."""
        my_dom = Domain(random_bounds_valid)

        # Assertions
        expected = 1.0  # Always 1.0
        dim = my_dom.spatial_dimension
        assert np.isclose(
            my_dom.get_diff_factor(np.zeros(dim, dtype=int)),
            expected,
        )

    def test_diff_normalized(self, SpatialDimension):
        """Test the differentiation scaling factor in the normalized domain."""
        my_dom = Domain.normalized(SpatialDimension)

        # Assertion
        expected = 1.0  # Always 1.0
        order = np.random.randint(0, 5, size=(SpatialDimension,))
        assert np.isclose(my_dom.get_diff_factor(order), expected)


class TestEquality:
    """All tests related to equality check."""

    def test_equal(self, random_bounds_valid):
        """Test strict equality of values between two instances."""
        my_dom_1 = Domain(random_bounds_valid)
        my_dom_2 = Domain(random_bounds_valid)

        # Assertions
        assert my_dom_1 is not my_dom_2
        assert my_dom_2 is not my_dom_1
        assert my_dom_1 == my_dom_2
        assert my_dom_2 == my_dom_1

    def test_not_equal_bounds(self, random_bounds_pair):
        """Test strict inequality of bounds between two instances."""
        bounds_1, bounds_2 = random_bounds_pair
        my_dom_1 = Domain(bounds_1)
        my_dom_2 = Domain(bounds_2)

        # Assertions
        assert my_dom_1 is not my_dom_2
        assert my_dom_2 is not my_dom_1
        assert my_dom_1 != my_dom_2
        assert my_dom_2 != my_dom_1

    def test_not_equal_dimensions(self, random_bounds_valid):
        """Test strict inequality of dimensions between two instances."""
        bounds_1 = random_bounds_valid
        bounds_2 = np.vstack([bounds_1, bounds_1[-1]])

        my_dom_1 = Domain(bounds_1)
        my_dom_2 = Domain(bounds_2)

        # Assertions
        assert my_dom_1 is not my_dom_2
        assert my_dom_2 is not my_dom_1
        assert my_dom_1 != my_dom_2
        assert my_dom_2 != my_dom_1


class TestPartialMatching:
    """All tests related to partial matching."""

    def test_partially_matched(self, random_bounds_valid):
        """Test partially matched domain."""
        bounds_1 = random_bounds_valid
        bounds_2 = np.vstack([bounds_1, bounds_1[-1]])

        my_dom_1 = Domain(bounds_1)
        my_dom_2 = Domain(bounds_2)

        # Assertions (symmetric)
        assert my_dom_1.partial_matching(my_dom_2)
        assert my_dom_2.partial_matching(my_dom_1)

    def test_no_partially_matched(self, random_bounds_pair):
        """Test no partially matched domain."""
        bounds_1, bounds_2 = random_bounds_pair

        my_dom_1 = Domain(bounds_1)
        if len(bounds_2) > 1:
            my_dom_2 = Domain(bounds_2[:-1])
        else:
            my_dom_2 = Domain(bounds_2)

        # Assertions (symmetric)
        assert not my_dom_1.partial_matching(my_dom_2)
        assert not my_dom_2.partial_matching(my_dom_1)


class TestContains:
    """All tests related to the contains check."""

    def test_contained(self, random_bounds_valid):
        """Test containment of sample inside the bounds."""
        my_dom = Domain(random_bounds_valid)

        # Generate random valid input values in the original domain
        xx = generate_values_inbounds(random_bounds_valid)

        # Assertion
        assert np.all(my_dom.contains(xx))

    def test_not_contained(self, random_bounds_valid):
        """Test containment of sample outside the bounds."""
        my_dom = Domain(random_bounds_valid)

        # Generate input values outside the original bound
        xx = generate_values_outbounds(random_bounds_valid)

        # Assertion
        assert not np.any(my_dom.contains(xx))
