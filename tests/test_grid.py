import numpy as np
import pytest

from minterpy import Grid, MultiIndexSet
from minterpy.gen_points import GENERATING_FUNCTIONS
from minterpy.core.grid import DEFAULT_FUN

from conftest import create_mi_pair_distinct


class TestInit:
    """All tests related to the default constructor of Grid."""

    @pytest.mark.parametrize("invalid_type", ["123", 1, np.array([1, 2, 3])])
    def test_with_invalid_multi_index_set(self, invalid_type):
        """Passing an invalid type of multi-index set raises an exception."""
        with pytest.raises(TypeError):
            Grid(invalid_type)

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_with_empty_multi_index_set(self, spatial_dimension, LpDegree):
        """Passing an empty multi-index set raises an exception."""
        # Create an empty set
        mi = MultiIndexSet(np.empty((0, spatial_dimension)), LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            Grid(mi)

    def test_with_invalid_gen_function(self, multi_index_mnp):
        """Invalid generating function raises an exception."""
        # Get the multi-index set
        mi = multi_index_mnp

        # Invalid generating function
        with pytest.raises(KeyError):
            Grid(mi, generating_function="1234")

        with pytest.raises(TypeError):
            Grid(mi, generating_function=[1, 2, 3])

    def test_with_valid_gen_function_and_points(self, multi_index_mnp):
        """Valid generating function and points are passed as arguments."""
        # Get the multi-index set
        mi = multi_index_mnp

        # Set up the generating function and points (the default)
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.poly_degree, mi.spatial_dimension)

        # Create Grids
        grd_1 = Grid(
            mi,
            generating_function=gen_function,
            generating_points=gen_points,
        )
        grd_2 = Grid(mi)

        # Assertion
        assert grd_1 == grd_2
        assert grd_1.generating_function == grd_2.generating_function

    def test_with_invalid_gen_function_and_points(self, multi_index_mnp):
        """Invalid generating function and points are given.

        They are invalid because the generating function does not reproduce
        the given generating points.
        """
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Set up the generating function and points (the default)
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.poly_degree, mi.spatial_dimension)

        # A generating function that is inconsistent with the gen. points above
        def _gen_fun(poly_degree, spatial_dimension):
            xx = np.linspace(-0.99, 0.99, poly_degree + 1)
            return np.tile(xx, spatial_dimension)

        # Assertion
        with pytest.raises(ValueError):
            Grid(
                mi,
                generating_function=_gen_fun,
                generating_points=gen_points,
            )


class TestInitGenPoints:
    """Tests construction with generating points."""
    def test_with_gen_points(self, multi_index_mnp):
        """Create a Grid with a specified generating points."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an array of generating points (from the default)
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.poly_degree, mi.spatial_dimension)

        # Create a Grid
        grd_1 = Grid(mi)  # Use the same default for the generating points
        grd_2 = Grid(mi, generating_points=gen_points)

        # Assertions
        assert grd_2 != grd_1  # Not equal because generating functions differ
        assert np.all(grd_1.generating_points == grd_2._generating_points)
        assert grd_2.multi_index == grd_1.multi_index
        assert grd_2.generating_function is None

    def test_larger_mi_invalid(self, SpatialDimension, PolyDegree, LpDegree):
        """Larger complete multi-index set than the grid raises an exception.

        Notes
        -----
        - The construction is expected to fail because a complete set of
          a given degree will have that degree as the maximum degree in any
          given dimension regardless of the lp-degree. If a grid is constructed
          with a degree less than the given degree of multi-index set, the
          grid cannot support the polynomials specified by the multi-index.
        """
        # create a multi-index set of a larger degree
        mi = MultiIndexSet.from_degree(
            SpatialDimension,
            PolyDegree + 1,
            LpDegree,
            )

        # Create an array of generating points with a lesser degree
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(PolyDegree, SpatialDimension)

        # Creating a Grid raises an exception
        with pytest.raises(ValueError):
            Grid(mi, generating_points=gen_points)

    def test_larger_mi_valid(self, SpatialDimension, PolyDegree, LpDegree):
        """Multi-index set with larger poly. degree doesn't raise an exception.

        Notes
        -----
        - Creating a multi-index set with the same exponents but with lower
          lp-degree tends to increase the polynomial degree of set.
          However, the polynomial degree of the grid is about the maximum
          degree of one-dimensional polynomials any dimension, so it should not
          matter if the polynomial degree of the multi-index set is larger than
          the degree of the grid as long as the grid has a degree larger than
          or equal to maximum degree of the multi-index set in any dimension.
        """
        # create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, np.inf)
        mi = MultiIndexSet(mi.exponents, LpDegree)

        # Create an array of generating points with lesser degree
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(PolyDegree, SpatialDimension)

        # Creating a Grid raises an exception
        grd = Grid(mi, generating_points=gen_points)

        # Assertions
        assert grd.poly_degree == PolyDegree
        assert grd.poly_degree <= mi.poly_degree

    def test_smaller_mi(self, multi_index_mnp):
        """Smaller complete multi-index set than the grid degree is okay."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an array of generating points with a higher degree
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.poly_degree + 1, mi.spatial_dimension)

        # Create a Grid instance
        grd = Grid(mi, generating_points=gen_points)

        # Assertions
        assert grd.poly_degree == mi.poly_degree + 1
        assert grd.poly_degree > mi.poly_degree  # only for a complete set


class TestExpandDim:
    """All tests related to the dimension expansion of a Grid instance."""
    def test_default_same_dim(self, multi_index_mnp):
        """Test the default behavior of expanding to the same dimension."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        grd = Grid(mi)

        # Expand the dimension: Same dimension
        grd_expanded = grd.expand_dim(grd.spatial_dimension)

        # Assertion
        assert grd == grd_expanded

    def test_default_diff_dim(self, multi_index_mnp):
        """Test the default behavior of expanding to a higher dimension."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        grd = Grid(mi)

        # Expand the dimension: Higher dimension
        new_dim = grd.spatial_dimension + 1
        grd_expanded = grd.expand_dim(new_dim)

        # Assertions
        assert grd_expanded != grd
        assert grd_expanded.spatial_dimension == new_dim
        assert grd_expanded.multi_index == mi.expand_dim(new_dim)

    def test_no_gen_fun_same_dim(self, multi_index_mnp):
        """Test expanding to the same dimension w/o a generating function."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.poly_degree, mi.spatial_dimension)
        grd = Grid(mi, generating_points=gen_points, generating_function=None)

        # Expand the dimension: Same dimension
        grd_expanded = grd.expand_dim(grd.spatial_dimension)

        # Assertion
        assert grd == grd_expanded

    def test_no_gen_fun_diff_dim(self, multi_index_mnp):
        """Test expanding to a higher dimension w/o a generating function."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.poly_degree, mi.spatial_dimension)
        grd = Grid(mi, generating_points=gen_points, generating_function=None)

        # Expand the dimension: Higher dimension
        new_dim = grd.spatial_dimension + 1
        grd_expanded = grd.expand_dim(new_dim)

        # Assertions
        assert np.array_equal(
            grd_expanded.generating_points[:, :new_dim - 1],
            grd.generating_points[:, :new_dim],
        )
        # New dimension is zero
        assert np.all(grd_expanded.generating_points[:, new_dim - 1] == 0.0)


class TestEquality:
    """All tests related to equality check of Grid instances."""

    def test_equal(self, SpatialDimension, PolyDegree, LpDegree):
        """Test equality of two Grid instances."""
        # Create a common multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create two Grid instances equal in value
        grd_1 = Grid(mi)
        grd_2 = Grid(mi)

        # Assertions
        assert grd_1 is not grd_2  # Not identical instances
        assert grd_1 == grd_2  # but equal in value
        assert grd_2 == grd_1  # symmetric property

    def test_inequal_multi_index(self):
        """Test inequality of two Grid instances due to different multi-index.
        """
        # Create two different multi-index set
        mi_1, mi_2 = create_mi_pair_distinct()

        # Create two Grid instances
        grd_1 = Grid(mi_1)
        grd_2 = Grid(mi_2)

        # Assertions
        assert grd_1 is not grd_2  # Not identical instances
        assert grd_1 != grd_2  # Not equal in values
        assert grd_2 != grd_1  # symmetric property

    def test_inequal_gen_points(self, SpatialDimension, PolyDegree, LpDegree):
        """Test inequality of two Grid instances due to diff. gen. points."""
        # Create a common multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create two Grid instances with different generating points
        # Chebyshev points
        grd_1 = Grid(mi)
        # Equidistant points
        grd_2 = Grid.from_value_set(
            mi,
            np.linspace(-0.99, 0.99, PolyDegree+1)[:, np.newaxis],
        )

        # Assertions
        assert grd_1 is not grd_2  # Not identical instances
        assert grd_1 != grd_2  # Not equal in values
        assert grd_2 != grd_1  # symmetric property

    def test_inequality_inconsistent_type(self):
        """Test inequality check with inconsistent types."""
        # Create a multi-index set
        mi = MultiIndexSet.from_degree(3, 2, 2.0)

        # Create a Grid instance
        grd = Grid(mi)

        # Assertions
        assert grd != mi
        assert grd != "123"
        assert grd != 1
        assert grd != 10.0
        assert grd != np.random.rand(len(mi), 3)
