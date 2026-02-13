"""
Testing module for interpolation.py

Here the functionality of the respective attribute is not tested.
"""

import numpy as np
import pytest

from conftest import build_random_newton_polynom

import minterpy as mp
from minterpy import (
    Interpolant,
    Interpolator,
    interpolate,
    Domain,
    Grid,
    NewtonPolynomial,
    LagrangePolynomial,
    ChebyshevPolynomial,
    CanonicalPolynomial,
)

#######################
# Internal functions  #
#######################

def func(xx):
    """Dummy function for testing interpolant."""
    return np.repeat(np.sum(xx, axis=1)[:, np.newaxis], repeats=5, axis=1)

##########
# Tests  #
##########

class TestInterpolator:
    """All tests related to the Interpolator class."""

    def test_init(self, multi_index_mnp, domain):
        """Test the initialization of an Interpolator instance."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds
        grd = mp.Grid(mi, domain=domain)

        # Create an instance of Interpolator
        interpolator = Interpolator(m, n, p, bounds)

        # Assertions
        assert interpolator.multi_index == mi
        assert interpolator.grid == grd
        assert interpolator.domain == domain
        assert interpolator.spatial_dimension == m
        assert interpolator.poly_degree == n
        assert interpolator.lp_degree == p

    def test_init_default_bound(self, multi_index_mnp):
        """Test the initialization of an instance with default bounds."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree

        # Create an instance of Interpolator
        interpolator = Interpolator(m, n, p)

        # Create a default Domain
        domain = Domain.normalized(m)

        # Assertions
        assert interpolator.domain == domain

    def test_call(self, multi_index_mnp, domain):
        """Test the evaluation of an interpolator instance."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds
        grd = mp.Grid(mi, domain=domain)

        # Create an instance of Interpolator
        interpolator = Interpolator(m, n, p, bounds)

        # Interpolate the function
        interpol_1 = interpolator(func)

        # Create a reference interpolant
        interpol_2 = NewtonPolynomial(mi, interpol_1.coeffs, grid=grd)

        # Assertion
        assert interpol_1 == interpol_2

    def test_identity(self, multi_index_mnp, domain):
        """Test interpolating a Newton polynomial in the Newton basis."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds

        # Create a groundtruth polynomial
        groundtruth_poly = build_random_newton_polynom(m, n, p, domain)

        # Create an interpolator and interpolate the groundtruth polynomial
        interpolator = Interpolator(m, n, p, bounds)
        interpolant_poly = interpolator(groundtruth_poly)

        # Assertions
        assert isinstance(interpolant_poly, type(groundtruth_poly))
        assert interpolant_poly.multi_index == groundtruth_poly.multi_index
        assert interpolant_poly.grid == groundtruth_poly.grid
        assert np.allclose(interpolant_poly.coeffs, groundtruth_poly.coeffs)


class TestInterpolant:
    """All tests related to the Interpolant class."""

    def test_init(self, multi_index_mnp, domain):
        """Test default construction of an Interpolant instance."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds

        # Create interpolant instances
        interpolant = Interpolant.from_degree(func, m, n, p, bounds)

        # Assertions
        assert interpolant.spatial_dimension == m
        assert interpolant.poly_degree == n
        assert interpolant.lp_degree == p
        assert interpolant.multi_index == mi

    def test_poly(self, multi_index_mnp, domain):
        """Test construction of the underlying interpolating polynomial."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds

        # Create an interpolator
        interpolator = Interpolator(m, n, p, bounds)

        # Create interpolant instances
        interpolant_1 = Interpolant(func, interpolator)
        interpolant_2 = Interpolant.from_degree(func, m, n, p, bounds)

        # Assertions
        assert interpolant_1.to_newton() == interpolator(func)
        assert interpolant_2.to_newton() == interpolator(func)

    def test_call(self, multi_index_mnp, domain):
        """Test calling an interpolant instance."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds

        # Create a reference polynomial
        reference_poly = build_random_newton_polynom(m, n, p, domain)

        # Interpolate the groundtruth polynomial
        interpolant = Interpolant.from_degree(reference_poly, m, n, p, bounds)

        # Create a set of random test points
        lb, ub = domain.lowers, domain.uppers
        xx_test = lb + (ub - lb) * np.random.rand(100, m)

        # Assertion
        assert np.allclose(interpolant(xx_test), reference_poly(xx_test))

    def test_to_newton(self, multi_index_mnp, domain):
        """Test getting the interpolating polynomial in the Newton basis."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds

        # Interpolate a function
        interpolant = interpolate(func, m, n, p, bounds)
        poly_1 = interpolant.to_newton()

        # Create a reference polynomial
        poly_2 = NewtonPolynomial(mi, poly_1.coeffs, domain=domain)

        # Assertion
        assert poly_1 == poly_2

    def test_to_lagrange(self, multi_index_mnp, domain):
        """Test getting the interpolating polynomial in the Lagrange basis."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds

        # Interpolate a function
        interpolant = interpolate(func, m, n, p, bounds)
        poly_1 = interpolant.to_lagrange()

        # Create a reference polynomial
        grd = Grid(mi, domain=domain)
        coeffs = grd(func)
        poly_2 = LagrangePolynomial(mi, coeffs, grid=grd)

        # Assertion
        assert poly_1 == poly_2

    def test_to_canonical(self, multi_index_mnp, domain):
        """Test getting the interpolating polynomial in the canonical basis."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds

        # Interpolate a function
        interpolant = interpolate(func, m, n, p, bounds)
        poly_1 = interpolant.to_canonical()

        # Create a reference polynomial
        coeffs = poly_1.coeffs
        poly_2 = CanonicalPolynomial(mi, coeffs, domain=domain)

        # Assertion
        assert poly_1 == poly_2

    def test_to_chebyshev(self, multi_index_mnp, domain):
        """Test getting the interpolating polynomial in the Chebyshev basis."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds

        # Interpolate a function
        interpolant = interpolate(func, m, n, p, bounds)
        poly_1 = interpolant.to_chebyshev()

        # Create a reference polynomial
        coeffs = poly_1.coeffs
        poly_2 = ChebyshevPolynomial(mi, coeffs, domain=domain)

        # Assertion
        assert poly_1 == poly_2

class TestInterpolate:
    """All tests related to the interpolate function."""

    def test_call(self, multi_index_mnp, domain):
        """Test calling the function."""
        # Fetch the relevant parameters for construction
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree
        bounds = domain.bounds

        # Create an interpolant instance
        interpolant_1 = Interpolant.from_degree(func, m, n, p, bounds)
        interpolant_2 = interpolate(func, m, n, p, bounds)

        # Create a set of random test points
        lb, ub = domain.lowers, domain.uppers
        xx_test = lb + (ub - lb) * np.random.rand(100, m)

        # Assertion (must be identical)
        assert isinstance(interpolant_2, Interpolant)
        assert np.allclose(interpolant_1(xx_test), interpolant_2(xx_test))
