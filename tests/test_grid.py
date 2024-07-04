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
