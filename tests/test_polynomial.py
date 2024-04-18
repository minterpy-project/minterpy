"""
Testing module for the common functionalities of the Minterpy polynomials.

Notes
-----
- Specific behaviors from different concrete polynomial implementations are
  tested in separate testing modules.
- Lagrange polynomials cannot be evaluated directly on a query point.
"""
import numpy as np
import pytest
from conftest import assert_call, num_polynomials, polynomial_class

from minterpy import (
    LagrangePolynomial,
    MultiIndexSet,
    Grid,
)


class TestInitialization:
    """All tests related to the initialization of a polynomial instance."""

    def test_without_coeffs(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization without coefficients."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        assert_call(polynomial_class, mi)

    def test_with_coeffs(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization with coefficients."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)

        assert_call(polynomial_class, mi, coeffs)

    def test_with_grid(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization with a valid grid."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        grd = Grid(mi)
        coeffs = np.arange(len(mi), dtype=float)

        assert_call(polynomial_class, mi, grid=grd)
        assert_call(polynomial_class, mi, coeffs, grid=grd)

    def test_with_invalid_grid(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization with an invalid grid."""
        mi = MultiIndexSet.from_degree(
            SpatialDimension,
            PolyDegree + 1,
            LpDegree
        )
        # Create a "smaller" grid the the polynomial (invalid)
        mi_grid = MultiIndexSet.from_degree(
            SpatialDimension,
            PolyDegree,
            LpDegree
        )
        grd = Grid(mi_grid)
        coeffs = np.arange(len(mi), dtype=float)

        with pytest.raises(ValueError):
            polynomial_class(mi, coeffs, grid=grd)

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set(self, spatial_dimension, polynomial_class, LpDegree):
        """Test initialization with an empty multi-index set."""
        # Create an empty set
        mi = MultiIndexSet(np.empty((0, spatial_dimension)), LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            polynomial_class(mi)


class TestEvaluation:
    """All tests related to the evaluation of a polynomial instance."""

    def test_with_coeffs(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test the evaluation of an 'initialized' polynomial."""
        # Create a polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)
        poly = polynomial_class(mi, coeffs)

        xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
        if isinstance(poly, LagrangePolynomial):
            with pytest.raises(NotImplementedError):
                poly(xx_test)
        else:
            assert_call(poly, xx_test)

    def test_without_coeffs(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test the evaluation of an 'uninitialized' polynomial."""
        # Create a polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        poly = polynomial_class(mi)

        xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
        if isinstance(poly, LagrangePolynomial):
            with pytest.raises(NotImplementedError):
                poly(xx_test)
        else:
            with pytest.raises(ValueError):
                poly(xx_test)

    def test_multiple_coeffs(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test the evaluation of multiple polys (multiple set of coeffs.)"""
        # Create polynomials (with multiple sets of coefficients)
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)
        coeffs = np.repeat(coeffs[:, None], num_polynomials, axis=1)
        poly = polynomial_class(mi, coeffs)

        xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
        if isinstance(poly, LagrangePolynomial):
            with pytest.raises(NotImplementedError):
                poly(xx_test)
        else:
            yy_test = poly(xx_test)
            np.all(yy_test == yy_test[:, 0])  # due to identical coefficients
