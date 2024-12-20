"""
Testing module for interpolation.py

Here the functionality of the respective attribute is not tested.
"""

import numpy as np
import pytest
from conftest import (
    LpDegree,
    NrPoints,
    PolyDegree,
    SpatialDimension,
    assert_call,
    assert_grid_equal,
    assert_interpolant_almost_equal,
    assert_multi_index_equal,
    assert_polynomial_almost_equal,
    build_random_newton_polynom,
    build_rnd_points,
)
from numpy.testing import assert_, assert_almost_equal

import minterpy as mp
from minterpy import Interpolant, Interpolator, interpolate

# test construction


def test_init_interpolator(SpatialDimension, PolyDegree, LpDegree):
    assert_call(Interpolator, SpatialDimension, PolyDegree, LpDegree)
    interpolator = Interpolator(SpatialDimension, PolyDegree, LpDegree)
    groundtruth_multi_index = mp.MultiIndexSet.from_degree(
        SpatialDimension, PolyDegree, LpDegree
    )
    groundtruth_grid = mp.Grid(groundtruth_multi_index)
    assert_multi_index_equal(interpolator.multi_index, groundtruth_multi_index)
    assert_grid_equal(interpolator.grid, groundtruth_grid)


def test_init_interpolant(SpatialDimension, PolyDegree, LpDegree):
    assert_call(
        Interpolant,
        lambda x: x[:, 0],
        Interpolator(SpatialDimension, PolyDegree, LpDegree),
    )
    assert_call(
        Interpolant.from_degree,
        lambda x: x[:, 0],
        SpatialDimension,
        PolyDegree,
        LpDegree,
    )
    interpolant_default = Interpolant(
        lambda x: x[:, 0], Interpolator(SpatialDimension, PolyDegree, LpDegree)
    )
    interpolant_from_degree = Interpolant.from_degree(
        lambda x: x[:, 0], SpatialDimension, PolyDegree, LpDegree
    )
    assert_interpolant_almost_equal(interpolant_default, interpolant_from_degree)


def test_call_interpolate(SpatialDimension, PolyDegree, LpDegree):
    assert_call(interpolate, lambda x: x[:, 0], SpatialDimension, PolyDegree, LpDegree)
    interpolant = interpolate(lambda x: x[:, 0], SpatialDimension, PolyDegree, LpDegree)
    assert_(isinstance(interpolant, Interpolant))


# test if the interpolator can interpolate
def test_interpolator(SpatialDimension, PolyDegree, LpDegree):
    groundtruth_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )
    interpolator = Interpolator(SpatialDimension, PolyDegree, LpDegree)
    res_from_newton_poly = interpolator(groundtruth_poly)
    res_from_canonical_poly = interpolator(mp.NewtonToCanonical(groundtruth_poly)())
    assert_polynomial_almost_equal(res_from_newton_poly, groundtruth_poly)
    assert_polynomial_almost_equal(res_from_canonical_poly, groundtruth_poly)


# test if the interpolant interpolates
def test_interpolant(NrPoints, SpatialDimension, PolyDegree, LpDegree):
    rnd_points = build_rnd_points(NrPoints, SpatialDimension)
    groundtruth_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )
    groundtruth = groundtruth_poly(rnd_points)
    interpolant = Interpolant.from_degree(
        groundtruth_poly, SpatialDimension, PolyDegree, LpDegree
    )
    res = interpolant(rnd_points)
    assert_almost_equal(res, groundtruth)


# test if the interpolate does what it promisses
def test_interpolate(NrPoints, SpatialDimension, PolyDegree, LpDegree):
    rnd_points = build_rnd_points(NrPoints, SpatialDimension)
    groundtruth_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )
    groundtruth = groundtruth_poly(rnd_points)
    interpolant = interpolate(groundtruth_poly, SpatialDimension, PolyDegree, LpDegree)
    res = interpolant(rnd_points)
    assert_almost_equal(res, groundtruth)


def _fun(xx: np.ndarray) -> np.ndarray:
    """Dummy function for testing interpolant."""
    return np.sum(xx, axis=1)


class TestPoly:
    """All tests related to the accessing the polynomial of an interpolant."""

    def test_to_newton(self, SpatialDimension, PolyDegree, LpDegree):
        """Test obtaining the interpolating polynomial in the Newton basis."""
        # Interpolate a function
        interpol = interpolate(_fun, SpatialDimension, PolyDegree, LpDegree)
        poly_1 = interpol.to_newton()

        # Create a reference
        grd = mp.Grid.from_degree(SpatialDimension, PolyDegree, LpDegree)
        lag_coeffs = grd(_fun)
        # 'interpolate()' use DDS (don't use LagrangeToNewton in the test
        # as the results won't be identical, very close to, but not identical)
        nwt_coeffs = mp.dds.dds(lag_coeffs, grd.tree)
        poly_2 = mp.NewtonPolynomial.from_grid(grd, nwt_coeffs)

        assert poly_1 == poly_2

    def test_to_lagrange(self, SpatialDimension, PolyDegree, LpDegree):
        """Test obtaining the interpolating polynomial in the Newton basis."""
        # Interpolate a function
        interpol = interpolate(_fun, SpatialDimension, PolyDegree, LpDegree)
        poly_1 = interpol.to_lagrange()

        # Create a reference
        grd = mp.Grid.from_degree(SpatialDimension, PolyDegree, LpDegree)
        lag_coeffs = grd(_fun)
        poly_2 = mp.LagrangePolynomial.from_grid(grd, lag_coeffs)

        assert poly_1 == poly_2

    def test_to_canonical(self, SpatialDimension, PolyDegree, LpDegree):
        """Test obtaining the interpolating polynomial in the Newton basis."""
        # Interpolate a function
        interpol = interpolate(_fun, SpatialDimension, PolyDegree, LpDegree)
        poly_1 = interpol.to_canonical()

        # Create a reference
        grd = mp.Grid.from_degree(SpatialDimension, PolyDegree, LpDegree)
        lag_coeffs = grd(_fun)
        lag_coeffs = grd(_fun)
        # 'interpolate()' use DDS (don't use LagrangeToNewton in the test
        # as the results won't be identical, very close to, but not identical)
        nwt_coeffs = mp.dds.dds(lag_coeffs, grd.tree)
        nwt_poly = mp.NewtonPolynomial.from_grid(grd, nwt_coeffs)
        poly_2 = mp.NewtonToCanonical(nwt_poly)()

        assert poly_1 == poly_2

    def test_to_chebyshev(self, SpatialDimension, PolyDegree, LpDegree):
        """Test obtaining the interpolating polynomial in the Newton basis."""
        # Interpolate a function
        interpol = interpolate(_fun, SpatialDimension, PolyDegree, LpDegree)
        poly_1 = interpol.to_chebyshev()

        # Create a reference
        grd = mp.Grid.from_degree(SpatialDimension, PolyDegree, LpDegree)
        lag_coeffs = grd(_fun)
        lag_coeffs = grd(_fun)
        # 'interpolate()' use DDS (don't use LagrangeToNewton in the test
        # as the results won't be identical, very close to, but not identical)
        nwt_coeffs = mp.dds.dds(lag_coeffs, grd.tree)
        nwt_poly = mp.NewtonPolynomial.from_grid(grd, nwt_coeffs)
        poly_2 = mp.NewtonToChebyshev(nwt_poly)()

        assert poly_1 == poly_2
