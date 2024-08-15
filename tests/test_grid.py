import numpy as np
import pytest

from minterpy import Grid, MultiIndexSet
from minterpy.gen_points import (
    GENERATING_FUNCTIONS,
    gen_chebychev_2nd_order_leja_ordered,
    gen_points_from_values,
)
from minterpy.core.grid import DEFAULT_FUN
from minterpy.utils.multi_index import get_exponent_matrix

from conftest import create_mi_pair_distinct


def _fun_one_out(xx: np.ndarray, sum: bool = False):
    """Test function for calling an instance of grid."""
    if sum:
        return np.sum(xx, axis=1)

    return np.prod(xx, axis=1)


def _fun_multi_out(xx: np.ndarray):
    """Return the same output as input."""
    return xx  # xx is assumed to be multi-dimensional


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
        gen_points = gen_function(mi.max_exponent, mi.spatial_dimension)

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
        gen_points = gen_function(mi.max_exponent, mi.spatial_dimension)

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
        gen_points = gen_function(mi.max_exponent, mi.spatial_dimension)

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
          lp-degree tends to increase the polynomial degree of the set.
          However, the maximum exponent of the grid is about the maximum
          degree of one-dimensional polynomials any dimension, so it should not
          matter if the polynomial degree of the multi-index set is larger than
          the degree of the grid as long as the grid has a degree larger than
          or equal to maximum degree of the multi-index set in any dimension.
        """
        # create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, np.inf)
        # recreate with smaller lp-degree
        mi = MultiIndexSet(mi.exponents, LpDegree)

        # Create an array of generating points with lesser degree
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(PolyDegree, SpatialDimension)

        # Create an instance of Grid
        grd = Grid(mi, generating_points=gen_points)

        # Assertions
        assert grd.max_exponent == PolyDegree
        assert grd.max_exponent <= mi.poly_degree

    def test_smaller_mi(self, multi_index_mnp):
        """Smaller complete multi-index set than the grid degree is okay."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an array of generating points with a higher degree
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.max_exponent + 1, mi.spatial_dimension)

        # Create a Grid instance
        grd = Grid(mi, generating_points=gen_points)

        # Assertions
        assert grd.max_exponent > np.max(mi.exponents)


class TestInitFrom:
    """All tests related to the factory methods."""
    def test_from_degree(self, SpatialDimension, PolyDegree, LpDegree):
        """Tests the `from_degree()` method with default values."""
        # Create complete multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create instances of Grid
        grd_1 = Grid(mi)
        grd_2 = Grid.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Assertions
        assert grd_1 == grd_2
        assert grd_2 == grd_1

    def test_from_degree_with_gen_function(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Tests the `from_degree()` method with generating function."""
        # Create complete multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create instances of Grid
        grd_1 = Grid(mi, generating_function=DEFAULT_FUN)
        grd_2 = Grid.from_degree(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            generating_function=DEFAULT_FUN,
        )

        # Assertions
        assert grd_1 == grd_2
        assert grd_2 == grd_1

    def test_from_degree_with_gen_points(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Tests the `from_degree()` method with generating points."""
        # Create complete multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an array of generating points
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.poly_degree, mi.spatial_dimension)

        # Create instances of Grid
        grd_1 = Grid(mi, generating_points=gen_points)
        grd_2 = Grid.from_degree(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            generating_points=gen_points,
        )

        # Assertions
        assert grd_1 == grd_2
        assert grd_2 == grd_1

    def test_from_degree_with_gen_function_and_points(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Tests the `from_degree()` method with generating function & points.
        """
        # Create complete multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an array of generating points
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.poly_degree, mi.spatial_dimension)

        # Create instances of Grid
        grd_1 = Grid(mi)
        grd_2 = Grid.from_degree(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            generating_function=gen_function,
            generating_points=gen_points,
        )

        # Assertions
        assert grd_1 == grd_2
        assert grd_2 == grd_1

    def test_from_gen_function(self, multi_index_mnp):
        """Test the `from_function()` method."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Get the default generating function
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]

        # Create instances of Grid
        grd_1 = Grid(mi, generating_function=gen_function)
        grd_2 = Grid.from_function(mi, gen_function)

        # Assertion
        assert grd_1 == grd_2
        assert grd_2 == grd_1

    def test_from_gen_function_with_str(self, multi_index_mnp):
        """Test the `from_function()` method with string selection."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create instances of Grid
        grd_1 = Grid(mi, generating_function=DEFAULT_FUN)
        grd_2 = Grid.from_function(mi, DEFAULT_FUN)

        # Assertion
        assert grd_1 == grd_2
        assert grd_2 == grd_1

    def test_from_gen_function_invalid_non_unique(self, multi_index_mnp):
        """Test the `from_function()` method."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        if mi.poly_degree == 0:
            pytest.skip("Generating points of length 1 is always unique")

        # Get the default generating function
        gen_function = lambda x, y: np.ones((x + 1, y))

        # Create instances of Grid
        with pytest.raises(ValueError):
            Grid.from_function(mi, gen_function)

    def test_from_gen_points(self, multi_index_mnp):
        """Test the `from_points()` method."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an array of generating points
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.max_exponent, mi.spatial_dimension)

        # Create instances of Grid
        grd_1 = Grid(mi, generating_points=gen_points)
        grd_2 = Grid.from_points(mi, gen_points)

        # Assertions
        assert grd_1 == grd_2
        assert grd_2 == grd_1

    def test_from_gen_points_invalid_wrong_shape(self, multi_index_mnp):
        """Test invalid call to the `from_points()` due to dimension mismatch.
        """
        # Get the complete multi-index set
        mi = multi_index_mnp

        if mi.spatial_dimension == 1:
            pytest.skip("Dimension 1 has no relevant lesser dimension.")

        # Create an array of generating points
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.max_exponent, mi.spatial_dimension - 1)

        # Create an instance of Grid
        with pytest.raises(ValueError):
            Grid.from_points(mi, gen_points)

    def test_from_gen_points_invalid_non_unique(self, multi_index_mnp):
        """Test invalid call to the `from_points()` due to non-unique values.
        """
        # Get the complete multi-index set
        mi = multi_index_mnp

        if mi.max_exponent == 0:
            pytest.skip("Generating points of length 1 is always unique")

        # Create an array of generating points
        gen_points = np.ones((mi.max_exponent + 1, mi.spatial_dimension))

        # Create an instance of Grid
        with pytest.raises(ValueError):
            Grid.from_points(mi, gen_points)

    def test_from_value_set(self, multi_index_mnp):
        """Test the `from_value_set()` method."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an array of generating values (the default 1d generating
        # function) and the corresponding generating points
        gen_values = gen_chebychev_2nd_order_leja_ordered(mi.max_exponent)
        gen_points = gen_points_from_values(gen_values, mi.spatial_dimension)

        # Create instances of Grid
        grd_1 = Grid(mi, generating_points=gen_points)
        grd_2 = Grid.from_value_set(mi, gen_values)

        # Assertions
        assert grd_1 == grd_2
        assert grd_2 == grd_1

    def test_from_value_set_invalid_wrong_shape(self, multi_index_mnp):
        """Test invalid call to `from_value_set()` due to wrong shape."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an array of generating values (the default 1d generating
        # function) and the corresponding generating points (with higher dim.)
        gen_values = gen_chebychev_2nd_order_leja_ordered(mi.max_exponent)
        gen_points = gen_points_from_values(gen_values, mi.spatial_dimension+1)

        # Create an instance of Grid
        with pytest.raises(ValueError):
            Grid.from_value_set(mi, gen_points)

    def test_from_value_set_invalid_non_unique(self, multi_index_mnp):
        """Test invalid call to `from_value_set()` due to non-unique values."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        if mi.max_exponent == 0:
            pytest.skip("Generating values of length 1 is always unique")

        # Create an array of generating values
        gen_values = np.ones(mi.max_exponent + 1)

        # Create an instance of Grid
        with pytest.raises(ValueError):
            Grid.from_value_set(mi, gen_values)


class TestUnisolventNodes:
    """All tests related to the unisolvent nodes property."""
    def test_unisolvent_nodes(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the property of unisolvent nodes of complete set"""
        # Create a Grid instance
        grd = Grid.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Get the relevant properties
        mi = grd.multi_index
        unisolvent_nodes = grd.unisolvent_nodes
        gen_points = grd.generating_points

        # Assertions
        assert unisolvent_nodes.shape == (len(mi), mi.spatial_dimension)
        # The condition below only applies for a complete multi-index set
        # i.e., all generating points appear in the unisolvent nodes
        # Transpose to iterate dimension (column)-wise
        assert all(
            [
                np.all(np.unique(xx) == np.sort(yy))
                for xx, yy in zip(unisolvent_nodes.T, gen_points.T)
            ]
        )


class TestCall:
    """All tests related to calling an instance with a callable."""
    def test_call_multi_dim_output(self, multi_index_mnp):
        """Test calling on a valid callable that returns a multi-dim arrray."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        grd = Grid(mi)

        # Call the Grid instance
        lag_coeffs = grd(_fun_multi_out)

        # Assertion
        assert lag_coeffs.shape == (len(mi), mi.spatial_dimension)

    def test_call_one_dim_output(self, multi_index_mnp):
        """Test calling on a valid callable that returns a one-dim array."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        grd = Grid(mi)

        # Call the Grid instance
        lag_coeffs_1 = grd(_fun_one_out, True)  # pass a pos. argument
        lag_coeffs_2 = grd(_fun_one_out, sum=True)  # pass a keyword argument

        # Assertions
        assert len(lag_coeffs_1) == len(mi)
        assert len(lag_coeffs_2) == len(mi)
        assert np.array_equal(lag_coeffs_1, lag_coeffs_2)

    @pytest.mark.parametrize("invalid_function", [1, 2.0, "3.5"])
    def test_call_invalid_function(self, multi_index_mnp, invalid_function):
        """Test calling on an invalid function."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        grd = Grid(mi)

        # Assertion
        with pytest.raises(TypeError):
            grd(invalid_function)


class TestExpandDim:
    """All tests related to the dimension expansion of a Grid instance."""
    def test_target_dim_same_dim(self, multi_index_mnp):
        """Test the default behavior of expanding to the same dimension."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        grd = Grid(mi)

        # Expand the dimension: Same dimension (always a valid operation)
        grd_expanded = grd.expand_dim(grd.spatial_dimension)

        # Assertions
        assert grd == grd_expanded  # Equal in value
        assert grd is not grd_expanded  # Not an identical instance

    def test_target_dim_with_gen_fun(self, multi_index_mnp):
        """Test the default behavior of expanding to a higher dimension."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        grd = Grid(mi)

        # Expand the dimension: Higher dimension (always valid w/ gen. func)
        new_dim = grd.spatial_dimension + 1
        grd_expanded = grd.expand_dim(new_dim)

        # Assertions
        assert grd_expanded != grd
        assert grd_expanded.spatial_dimension == new_dim
        assert grd_expanded.multi_index == mi.expand_dim(new_dim)

    def test_target_dim_same_dim_with_gen_points(self, multi_index_mnp):
        """Test expanding to the same dimension with generating points."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.max_exponent, mi.spatial_dimension)
        grd = Grid.from_points(mi, gen_points)

        # Expand the dimension: Same dimension (always a valid operation)
        grd_expanded = grd.expand_dim(grd.spatial_dimension)

        # Assertions
        assert grd == grd_expanded  # Equal in value
        assert grd is not grd_expanded  # Not an identical instances

    def test_target_dim_with_gen_points_valid(self, multi_index_mnp):
        """Test expanding the dimension with valid generating points."""
        # Get the complete multi-index set
        mi = multi_index_mnp
        target_dim = mi.spatial_dimension + 1

        # Create generating points with a higher dimension
        gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_fun(mi.max_exponent, target_dim)

        # Create a Grid instance
        grd = Grid.from_points(mi, gen_points)

        # Expand the dimension of the Grid
        grd_expanded = grd.expand_dim(target_dim)

        # Assertion
        assert grd_expanded.spatial_dimension == target_dim

    def test_target_dim_with_gen_points_invalid(self, multi_index_mnp):
        """Test expanding the dimension with invalid generating points."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create generating points with the same dimension
        gen_function = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_function(mi.max_exponent, mi.spatial_dimension)

        # Create a Grid instance
        grd = Grid.from_points(mi, gen_points)

        # Expand the dimension: Higher dimension
        with pytest.raises(ValueError):
            grd.expand_dim(grd.spatial_dimension + 1)

    def test_target_grid_same_dim(self, multi_index_mnp):
        """Test expanding the dimension to the dimension of a target grid
        whose dimension is the same.
        """
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create instances of Grid
        origin_grid = Grid(mi)
        target_grid = Grid(mi)

        # Expand the grid
        expanded_grid = origin_grid.expand_dim(target_grid)

        # Assertion
        assert expanded_grid == target_grid

    def test_target_grid_with_gen_points_valid(self, multi_index_mnp):
        """Test expanding the dimension to the dimension of a target grid
        with a missing generating function but valid generating points.
        """
        # Create multi-indices
        origin_mi = multi_index_mnp
        target_mi = origin_mi.expand_dim(origin_mi.spatial_dimension + 1)

        # Get the ingredients for a Grid
        gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_fun(
            target_mi.max_exponent,
            target_mi.spatial_dimension,
        )

        # Create instances of Grid
        origin_grid = Grid.from_function(origin_mi, gen_fun)
        target_grid = Grid.from_points(target_mi, gen_points)

        # Expand the dimension
        expanded_grid = origin_grid.expand_dim(target_grid)

        # Assertion
        expanded_dim = expanded_grid.spatial_dimension
        target_dim = target_grid.spatial_dimension
        assert expanded_dim == target_dim

    def test_target_grid_with_larger_origin_gen_points(self, multi_index_mnp):
        """Test expanding the dimension to the dimension of a target grid
        but the origin generating points already have a larger dimension.
        """
        # Create multi-indices
        origin_mi = multi_index_mnp
        target_mi = origin_mi.expand_dim(origin_mi.spatial_dimension + 1)

        # Get the ingredients for a Grid
        gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]
        origin_gen_points = gen_fun(
            origin_mi.max_exponent,
            origin_mi.spatial_dimension + 1,
        )
        target_gen_points = gen_fun(
            target_mi.max_exponent,
            target_mi.spatial_dimension,
        )

        # Create instances of Grid
        origin_grid = Grid.from_points(origin_mi, origin_gen_points)
        target_grid = Grid.from_points(target_mi, target_gen_points)

        # Expand the dimension
        expanded_grid = origin_grid.expand_dim(target_grid)

        # Assertions
        expanded_dim = expanded_grid.spatial_dimension
        target_dim = target_grid.spatial_dimension
        assert expanded_dim == target_dim
        assert np.all(
            expanded_grid.generating_points == origin_grid.generating_points
        )

    def test_target_grid_with_gen_points_invalid(self, multi_index_mnp):
        """Test expanding the dimension to the dimension of a target grid
        whose generating points are inconsistent; it should raise an exception.
        """
        # Create multi-indices
        origin_mi = multi_index_mnp
        target_mi = origin_mi.expand_dim(origin_mi.spatial_dimension + 1)

        # Get the ingredients for a Grid
        gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_values = np.linspace(-0.99, 0.99, target_mi.max_exponent + 1)
        gen_points = np.tile(
            gen_values[:, np.newaxis],
            target_mi.spatial_dimension,
        )

        # Create instances of Grid
        origin_grid = Grid.from_function(origin_mi, gen_fun)
        target_grid = Grid.from_points(target_mi, gen_points)

        # Assertion
        with pytest.raises(ValueError):
            origin_grid.expand_dim(target_grid)

    def test_target_grid_with_gen_fun_valid(self, multi_index_mnp):
        """Test expanding the dimension to the dimension of a target grid
        whose generating functions are consistent.
        """
        # Create multi-indices
        origin_mi = multi_index_mnp
        target_mi = origin_mi.expand_dim(origin_mi.spatial_dimension + 1)

        # Get the ingredients for a Grid
        origin_gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]
        target_gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]

        # Create instances of Grid
        origin_grid = Grid.from_function(origin_mi, origin_gen_fun)
        target_grid = Grid.from_function(target_mi, target_gen_fun)

        # Expand the grid
        expanded_grid = origin_grid.expand_dim(target_grid)

        # Assertions
        assert expanded_grid.spatial_dimension == target_grid.spatial_dimension
        assert np.all(
            expanded_grid.generating_points == target_grid.generating_points
        )

    def test_target_grid_with_gen_fun_invalid(self, multi_index_mnp):
        """Test expanding the dimension to the dimension of a target grid
        whose generating functions are inconsistent; this should raise
        an exception.
        """
        # Create multi-indices
        origin_mi = multi_index_mnp
        target_mi = origin_mi.expand_dim(origin_mi.spatial_dimension + 1)

        # Get the ingredients for a Grid
        origin_gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]
        target_gen_fun = lambda x, y: origin_gen_fun(x, y)

        # Create instances of Grid
        origin_grid = Grid.from_function(origin_mi, origin_gen_fun)
        target_grid = Grid.from_function(target_mi, target_gen_fun)

        # Assertion
        with pytest.raises(ValueError):
            origin_grid.expand_dim(target_grid)


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

    def test_unequal_multi_index(self, SpatialDimension, PolyDegree, LpDegree):
        """Test inequality of two Grid instances due to different multi-index.
        """
        # Create two different multi-index set
        mi_1 = MultiIndexSet.from_degree(3, 2, 1.0)
        mi_2 = MultiIndexSet.from_degree(3, 2, 2.0)

        # Create two Grid instances
        grd_1 = Grid(mi_1)
        grd_2 = Grid(mi_2)

        # Assertions
        assert grd_1 is not grd_2  # Not identical instances
        assert grd_1 != grd_2  # Not equal in values
        assert grd_2 != grd_1  # symmetric property

    def test_unequal_gen_points(self, SpatialDimension, PolyDegree, LpDegree):
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

        # Assertions: Return False if one of the operands is inconsistent
        assert grd != mi
        assert grd != "123"
        assert grd != 1
        assert grd != 10.0
        assert grd != np.random.rand(len(mi), 3)


class TestMultiplication:
    """All tests related to the multiplication of two Grid instances."""
    def test_square(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the multiplication with the itself."""
        # Create a complete multi_index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an instance of Grid
        grd = Grid(mi)

        # Create a product Grid
        grd_prod = grd * grd

        # Assertions
        assert grd_prod.multi_index == mi * mi
        assert len(grd_prod.unisolvent_nodes) == len(mi * mi)

    def test_with_gen_points(self):
        """Test the multiplication of instances having only gen. points."""
        # Create a pair of distinct complete multi-index sets
        mi_1, mi_2 = create_mi_pair_distinct()  # the second has higher dim.

        # Get the ingredients for a Grid
        gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]
        # The product of two maximum exponents
        max_exponent = mi_1.max_exponent + mi_2.max_exponent
        gen_points_1 = gen_fun(max_exponent, mi_1.spatial_dimension)
        gen_points_2 = gen_fun(max_exponent, mi_2.spatial_dimension)

        # Create instances of Grid
        grd_1 = Grid.from_points(mi_1, gen_points_1)
        grd_2 = Grid.from_points(mi_2, gen_points_2)

        # Multiply the Grid
        grd_prod = grd_1 * grd_2

        # Assertions
        assert grd_prod.multi_index == mi_1 * mi_2
        assert len(grd_prod.unisolvent_nodes) == len(mi_1 * mi_2)
        assert np.all(grd_prod.generating_points == gen_points_2)

    @pytest.mark.parametrize("invalid_value", [1.0, 2, "123", np.array([1])])
    def test_invalid(self, multi_index_mnp, invalid_value):
        """Test the multiplication with an invalid value"""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid instance
        grd = Grid(mi)

        # Assertion
        with pytest.raises(AttributeError):
            grd * invalid_value


class TestUnion:
    """All tests related to taking the union of `Grid` instances."""
    def test_self(self, SpatialDimension, PolyDegree, LpDegree):
        """Test taking the union of an instance with the itself."""
        # Create a complete multi_index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an instance of Grid
        grd = Grid(mi)

        # Create a union Grid
        grd_union = grd | grd

        # Assertions
        assert grd_union == grd
        assert grd == grd_union

    def test_with_gen_points(self):
        """Test taking the union of instances having only generating points."""
        # Create a pair of distinct complete multi-index sets
        mi_1, mi_2 = create_mi_pair_distinct()  # the second has higher dim.

        # Get the ingredients for a Grid
        gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]
        # The max of two maximum exponents
        max_exponent = np.max([mi_1.max_exponent, mi_2.max_exponent])
        gen_points_1 = gen_fun(max_exponent, mi_1.spatial_dimension)
        gen_points_2 = gen_fun(max_exponent, mi_2.spatial_dimension)

        # Create instances of Grid
        grd_1 = Grid.from_points(mi_1, gen_points_1)
        grd_2 = Grid.from_points(mi_2, gen_points_2)

        # Multiply the Grid
        grd_prod = grd_1 | grd_2

        # Assertions
        assert grd_prod.multi_index == mi_1 | mi_2
        assert len(grd_prod.unisolvent_nodes) == len(mi_1 | mi_2)
        assert np.all(grd_prod.generating_points == gen_points_2)

    @pytest.mark.parametrize("invalid_value", [1.0, 2, "123", np.array([1])])
    def test_invalid(self, multi_index_mnp, invalid_value):
        """Test the multiplication with an invalid value"""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a Grid instance
        grd = Grid(mi)

        # Assertion
        with pytest.raises(AttributeError):
            grd | invalid_value


class TestAddExponents:
    """All tests related to the method to add a set of exponents."""
    def test_identical(self, multi_index_mnp):
        """Test adding identical set of exponents to a Grid."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an instance of Grid
        grd = Grid(mi)

        # Add the same exponents
        grd_added = grd.add_exponents(mi.exponents)

        # Assertion
        assert grd == grd_added
        assert grd_added == grd

    def test_too_large_exponent(self, multi_index_mnp):
        """Test adding an exponent that cannot be supported by the grid."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an array of generating points
        gen_fun = GENERATING_FUNCTIONS[DEFAULT_FUN]
        gen_points = gen_fun(mi.max_exponent, mi.spatial_dimension)

        # Create an instance of Grid
        grd = Grid.from_points(mi, gen_points)

        # Add a new exponent
        exponent = np.zeros(mi.spatial_dimension)
        exponent[0] = mi.max_exponent + 1

        with pytest.raises(ValueError):
            grd.add_exponents(exponent)

    def test_set_diff(self, multi_index_mnp):
        """Test adding the set difference to a Grid."""
        # Get the complete multi-index set
        mi_1 = multi_index_mnp
        exponents_1 = mi_1.exponents

        # Create a larger set of exponents
        exponents_2 = get_exponent_matrix(
            mi_1.spatial_dimension,
            mi_1.poly_degree * 2,
            mi_1.lp_degree
        )
        mi_2 = MultiIndexSet(exponents_2, mi_1.lp_degree)

        # Compute the set difference between the larger set and the smaller set
        exponents_diff = np.array(
            list(set(map(tuple, exponents_2)) - set(map(tuple, exponents_1)))
        )

        # Create Grid instances
        grd_1 = Grid(mi_1)
        grd_2 = Grid(mi_2)

        # Add the set difference
        grd_1_added = grd_1.add_exponents(exponents_diff)

        # Assertion
        assert grd_1_added == grd_2
        assert grd_2 == grd_1_added


class TestMakeComplete:
    """All tests related to make the Grid complete."""
    def test_already_complete(self, multi_index_mnp):
        """Test making an already complete grid complete."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an instance of Grid
        grd = Grid(mi)

        # Make the Grid complete
        grd_complete = grd.make_complete()

        # Assertions
        assert grd.is_complete  # Already complete
        assert grd_complete == grd
        assert grd == grd_complete
        assert grd is not grd_complete

    def test_incomplete(self, multi_index_incomplete):
        """Test making an incomplete grid complete."""
        # Get the incomplete multi-index set
        mi = multi_index_incomplete

        # Create an instance of Grid
        grd = Grid(mi)

        # Make the Grid complete
        grd_complete = grd.make_complete()

        # Assertions
        assert not grd.is_complete
        assert grd_complete.is_complete


class TestMakeDownwardClosed:
    """All tests related to make the Grid downward-closed."""
    def test_already_downward_closed(self, multi_index_mnp):
        """Test making an already downward-closed grid downward-closed."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create an instance of Grid
        grd = Grid(mi)

        # Make the Grid downward_closed
        grd_downward_closed = grd.make_downward_closed()

        # Assertions
        assert grd.is_downward_closed  # Already downward_closed
        assert grd_downward_closed == grd
        assert grd == grd_downward_closed
        assert grd is not grd_downward_closed

    def test_non_downward_closed(self, multi_index_non_downward_closed):
        """Test making a non-downward-closed grid downward-closed."""
        # Get the non-downward multi-index set
        mi = multi_index_non_downward_closed

        # Create an instance of Grid
        grd = Grid(mi)

        # Make the Grid downward-closed
        grd_downward_closed = grd.make_downward_closed()

        # Assertions
        assert not grd.is_downward_closed
        assert grd_downward_closed.is_downward_closed


class TestIsCompatible:
    """All tests related to compatibility check of Grid instances."""
    def test_compatible(self, multi_index_mnp):
        """Test compatible instances; equal instances are compatible."""
        # Get a common multi-index set
        mi = multi_index_mnp

        # Create two Grid instances equal in value
        grd_1 = Grid(mi)
        grd_2 = Grid(mi)

        # Assertions
        assert grd_1 == grd_2
        assert grd_1.is_compatible(grd_2)
        assert grd_2.is_compatible(grd_1)  # Commutativity must hold

    def test_unequal_multi_index(self, multi_index_mnp_pair):
        """Test that different multi-index with the same generating points
        are compatible.
        """
        # Get a pair of multi-index sets
        mi_1, mi_2 = multi_index_mnp_pair

        # Create a common generating values
        poly_degree = np.max([mi_1.poly_degree, mi_2.poly_degree]) + 10
        gen_points = np.linspace(-0.99, 0.99, poly_degree)[:, np.newaxis]

        # Create two instances of Grid
        grd_1 = Grid.from_value_set(mi_1, gen_points)
        grd_2 = Grid.from_value_set(mi_2, gen_points)

        # Assertions
        assert grd_1.is_compatible(grd_2)
        assert grd_2.is_compatible(grd_1)

    def test_unequal_gen_points(self, multi_index_mnp):
        """Test that different generating points cause incompatible grid."""
        # Get a common multi-index set
        mi = multi_index_mnp

        # Create two Grid instances with different generating points
        # Chebyshev points
        grd_1 = Grid(mi)
        # Equidistant points
        grd_2 = Grid.from_value_set(
            mi,
            np.linspace(-0.99, 0.99, mi.poly_degree + 1)[:, np.newaxis],
        )

        # Assertions
        assert grd_1 != grd_2  # Not equal in values
        assert not grd_1.is_compatible(grd_2)
        assert not grd_2.is_compatible(grd_1)  # Commutativity must hold

    def test_unequal_gen_function(self, multi_index_mnp):
        """Test that having different generating function is incompatible."""
        # Get a common multi-index set
        mi = multi_index_mnp

        # Create a generating function
        def _custom_gen_function(poly_degree, spatial_dimension):
            xx = np.linspace(-1.0, 1.0, poly_degree + 1)[:, np.newaxis]
            if xx.ndim == 1:
                xx = xx[:, np.newaxis]
            generating_points = np.tile(xx, (1, spatial_dimension))
            generating_points[:, ::2] *= -1

            return generating_points

        # Create two instances of Grid
        grd_1 = Grid(mi)
        grd_2 = Grid.from_function(mi, _custom_gen_function)

        # Assertion
        assert not grd_1.is_compatible(grd_2)
        assert not grd_2.is_compatible(grd_1)  # Commutativity must hold
