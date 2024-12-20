"""
Testing module for lagrange_polynomial.py

"""
import numpy as np
import pytest

from minterpy import (
    CanonicalPolynomial,
    LagrangePolynomial,
    MultiIndexSet,
    Grid,
)
from minterpy.transformations import LagrangeToCanonical, LagrangeToNewton
from minterpy.utils.multi_index import make_complete


def test_evaluation(rand_poly_mnp_lag):
    """Test the evaluation of an instance of Lagrange polynomial."""
    # Generate random query points
    xx = -1 + 2 * np.random.rand(100, rand_poly_mnp_lag.spatial_dimension)

    with pytest.raises(NotImplementedError):
        _ = rand_poly_mnp_lag(xx)


class TestAddition:
    """All tests related to scalar addition for Lagrange polynomial instances.
    """
    def test_poly_add(self, rand_poly_mnp_lag):
        """Test adding two Lagrange polynomial."""
        # Get a random Lagrange polynomial instance
        poly = rand_poly_mnp_lag

        # Self addition
        with pytest.raises(NotImplementedError):
            _ = poly + poly

    def test_scalar_add(self, rand_poly_mnp_lag):
        """Test adding a Lagrange polynomial w/ an arbitrary real scalar.

        Notes
        -----
        - The test verifies the expression: ``poly + scalar``.
        """
        # Get a random Lagrange polynomial instance
        poly = rand_poly_mnp_lag

        # Generate a random scalar
        scalar = np.random.rand(1)[0]

        # Subtract with the scalar
        poly_sum_1 = poly + scalar
        poly_sum_2 = scalar + poly  # Commutativity must hold

        # Compute the reference
        coeffs_ref = poly.coeffs.copy()
        coeffs_ref += scalar  # apply to all the coefficients

        # Assertion
        assert np.all(coeffs_ref == poly_sum_1.coeffs)
        assert np.all(coeffs_ref == poly_sum_2.coeffs)


class TestSubtraction:
    """All tests related to scalar subtraction for Lagrange polynomials."""
    def test_poly_sub(self, rand_poly_mnp_lag):
        """Test subtracting a Lagrange polynomial with itself."""
        # Get a random Lagrange polynomial instance
        poly = rand_poly_mnp_lag

        # Self subtraction
        with pytest.raises(NotImplementedError):
            _ = poly - poly

    def test_scalar_sub(self, rand_poly_mnp_lag):
        """Test subtracting a Lagrange polynomial w/ an arbitrary real scalar.

        Notes
        -----
        - The test verifies the expression: ``poly - scalar``.
        """
        # Get a random Lagrange polynomial instance
        poly = rand_poly_mnp_lag

        # Generate a random scalar
        scalar = np.random.rand(1)[0]

        # Subtract with the scalar
        poly_sum_1 = poly - scalar
        poly_sum_2 = -scalar + poly  # Commutativity must hold

        # Compute the reference
        coeffs_ref = poly.coeffs.copy()
        coeffs_ref -= scalar  # only apply to all the coefficients

        # Assertion
        assert np.all(coeffs_ref == poly_sum_1.coeffs)
        assert np.all(coeffs_ref == poly_sum_2.coeffs)

    def test_scalar_rsub(self, rand_poly_mnp_lag):
        """Test right-sided subtraction of a scalar with a Lagrange polynomial.

        Notes
        -----
        - The test verifies the expression: ``scalar - poly``.
        """
        # Get a random Lagrange polynomial instance
        poly = rand_poly_mnp_lag

        # Generate a random scalar
        scalar = np.random.rand(1)[0]

        # Subtract with the scalar
        poly_sum_1 = scalar - poly
        poly_sum_2 = -poly + scalar  # Commutativity must hold

        # Compute the reference
        coeffs_ref = -1 * poly.coeffs.copy()
        coeffs_ref += scalar  # only apply to all the coefficients

        # Assertion
        assert np.all(coeffs_ref == poly_sum_1.coeffs)
        assert np.all(coeffs_ref == poly_sum_2.coeffs)


def test_mul_poly(rand_poly_mnp_lag):
    """Test general polynomial multiplication; not implemented."""
    # Get a random Lagrange polynomial instance
    poly = rand_poly_mnp_lag

    with pytest.raises(NotImplementedError):
        _ = poly * poly


def test_exponentiation(rand_poly_mnp_lag):
    """Test general exponentiation."""
    # Get a random Lagrange polynomial instance
    poly = rand_poly_mnp_lag

    with pytest.raises(NotImplementedError):
        _ = poly**3


def test_integrate_over_bounds_invalid_shape(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid shape."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = CanonicalPolynomial(mi, lag_coeffs)

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = np.random.rand(SpatialDimension + 3, 2)
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        lag_poly.integrate_over(bounds)


def test_integrate_over_bounds_invalid_domain(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid domain."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = CanonicalPolynomial(mi, lag_coeffs)

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = 2 * np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        lag_poly.integrate_over(bounds)


def test_integrate_over_bounds_equal(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with equal bounds (should be zero)."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # Create bounds (one of them has lb == ub)
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1
    idx = np.random.choice(SpatialDimension)
    bounds[idx, 0] = bounds[idx, 1]

    # Compute the integral
    ref = 0.0
    value = lag_poly.integrate_over(bounds)

    # Assertion
    assert isinstance(value, float)
    assert np.isclose(ref, value)


def test_integrate_over_bounds_flipped(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with flipped bounds."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = CanonicalPolynomial(mi, lag_coeffs)

    # Compute the integral
    value_1 = lag_poly.integrate_over()

    # Flip bounds
    bounds = np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1
    bounds[:, [0, 1]] = bounds[:, [1, 0]]

    # Compute the integral with flipped bounds
    value_2 = lag_poly.integrate_over(bounds)

    if np.mod(SpatialDimension, 2) == 1:
        # Odd spatial dimension flips the sign
        assert np.isclose(value_1, -1 * value_2)
    else:
        assert np.isclose(value_1, value_2)


def test_integrate_over_list_as_bounds(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test integrate over with bounds specified with lists."""
    # Create a Lagrange polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = CanonicalPolynomial(mi, lag_coeffs)

    # Compute the integral
    value_1 = lag_poly.integrate_over()

    # Flip bounds
    bounds = [[-1, 1] for _ in range(SpatialDimension)]

    # Compute the integral with flipped bounds
    value_2 = lag_poly.integrate_over(bounds)

    # Assertion
    assert np.isclose(value_1, value_2)


def test_integrate_over(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration in different basis (sanity check)."""
    # Create a Canonical polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    lag_coeffs = np.random.rand(len(mi))
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # Transform to other polynomial bases
    nwt_poly = LagrangeToNewton(lag_poly)()
    can_poly = LagrangeToCanonical(lag_poly)()

    # Compute the integral
    value_lag = lag_poly.integrate_over()
    value_nwt = nwt_poly.integrate_over()
    # NOTE: Canonical integration won't work in high degree
    value_can = can_poly.integrate_over()

    # Assertions
    assert np.isclose(value_lag, value_nwt)
    assert np.isclose(value_lag, value_can)

    # Create bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    value_lag = lag_poly.integrate_over(bounds)
    value_nwt = nwt_poly.integrate_over(bounds)
    value_can = can_poly.integrate_over(bounds)

    # Assertions
    assert np.isclose(value_lag, value_nwt)
    assert np.isclose(value_lag, value_can)


def test_integrate_over_sum_function(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration for a simple sum function."""
    # Create a Lagrange interpolating polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    grd = Grid(mi)
    lag_coeffs = np.sum(grd.unisolvent_nodes, axis=1)
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # With the default bounds
    if PolyDegree > 0:
        ref = 0.0
        value = lag_poly.integrate_over()

        # Assertion
        assert np.isclose(ref, value)

    # With non-symmetric bounds (non-cancelling)
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    if PolyDegree > 0:
        # Rhe reference from analytical results for non-cancelling bounds
        ref = 0.0
        for i in range(SpatialDimension):
            ref += (
                np.diff(bounds[i] ** 2)
                * np.prod(np.diff(np.delete(bounds, i, axis=0)))
            )
        ref *= 0.5
        value = lag_poly.integrate_over(bounds)

        # Assertion
        assert np.isclose(ref, value)


def test_integrate_over_product_function(
    SpatialDimension, LpDegree,
):
    """Test polynomial integration for a simple product function."""
    fun = lambda xx: np.prod(xx, axis=1)

    # Create a Lagrange interpolating polynomial
    exp = np.ones((1, SpatialDimension), dtype=int)
    exp_completed = make_complete(exp, LpDegree)
    mi = MultiIndexSet(exp_completed, LpDegree)
    grd = Grid(mi)
    lag_coeffs = fun(grd.unisolvent_nodes)
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # --- With the default bounds

    # Compute the integral without bounds
    ref = 0.0
    value = lag_poly.integrate_over()

    # Assertion
    assert np.isclose(ref, value)

    # --- With non-symmetric bounds (non-cancelling)

    # Set up bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    ref = 0.5**SpatialDimension * np.prod(np.diff(bounds**2))
    value = lag_poly.integrate_over(bounds)

    # Assertion
    assert np.isclose(ref, value)


def test_integrate_over_multiple_polynomials(
    SpatialDimension, LpDegree,
):
    """Test integration with multiple polynomials."""
    num_polys = 6
    factors = np.arange(1, num_polys + 1)[np.newaxis, :]
    fun = lambda xx: np.prod(xx, axis=1)[:, np.newaxis] * factors

    # Create a Lagrange interpolating polynomial
    exp = np.ones((1, SpatialDimension), dtype=int)
    exp_completed = make_complete(exp, LpDegree)
    mi = MultiIndexSet(exp_completed, LpDegree)
    grd = Grid(mi)
    lag_coeffs = fun(grd.unisolvent_nodes)
    lag_poly = LagrangePolynomial(mi, lag_coeffs)

    # --- With the default bounds

    # Compute the integral without bounds
    ref = 0.0
    value = lag_poly.integrate_over()

    # Assertion
    assert len(value) == num_polys
    assert np.allclose(ref, value)

    # --- With non-symmetric bounds (non-cancelling)

    # Set up bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    ref = factors * 0.5 ** SpatialDimension * np.prod(np.diff(bounds ** 2))
    value = lag_poly.integrate_over(bounds)

    # Assertion
    assert len(value) == num_polys
    assert np.allclose(ref, value)
