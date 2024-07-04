import numpy as np
import pytest

from minterpy import Grid, MultiIndexSet
from minterpy.core.grid import DEFAULT_GRID_VAL_GEN_FCT

from conftest import create_mi_pair_distinct


class TestInit:
    """All tests related to the default constructor of Grid."""

    @pytest.mark.parametrize("invalid_type", ["123", 1, np.array([1, 2, 3])])
    def test_multi_index_type_error(self, invalid_type):
        """Passing an invalid type of multi-index set raises an exception."""
        with pytest.raises(TypeError):
            Grid(invalid_type)

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set(self, spatial_dimension, LpDegree):
        """Passing an empty multi-index set raises an exception."""
        # Create an empty set
        mi = MultiIndexSet(np.empty((0, spatial_dimension)), LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            Grid(mi)

    def test_from_gen_points(self, SpatialDimension, PolyDegree, LpDegree):
        """Create a Grid with a specified generating points."""
        # Create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an array of generating points (from the default)
        gen_points = DEFAULT_GRID_VAL_GEN_FCT(PolyDegree, SpatialDimension)

        # Create a Grid
        grd_1 = Grid(mi)  # Use the same default for the generating points
        grd_2 = Grid(mi, generating_points=gen_points)

        # Assertions
        assert grd_1 == grd_2
        assert grd_2 == grd_1

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
        gen_points = DEFAULT_GRID_VAL_GEN_FCT(PolyDegree, SpatialDimension)

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
        gen_points = DEFAULT_GRID_VAL_GEN_FCT(PolyDegree, SpatialDimension)

        # Creating a Grid raises an exception
        grd = Grid(mi, generating_points=gen_points)

        # Assertions
        assert grd.poly_degree == PolyDegree
        assert grd.poly_degree <= mi.poly_degree

    def test_smaller_mi(self, SpatialDimension, PolyDegree, LpDegree):
        """Smaller complete multi-index set than the grid degree is okay."""
        # create a multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create an array of generating points with lesser degree
        gen_points = DEFAULT_GRID_VAL_GEN_FCT(PolyDegree + 1, SpatialDimension)

        # Creating a Grid raises an exception
        grd = Grid(mi, generating_points=gen_points)

        # Assertions
        assert grd.poly_degree == PolyDegree + 1
        assert grd.poly_degree > mi.poly_degree  # only for mnp set


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
