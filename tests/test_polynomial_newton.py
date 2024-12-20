"""
testing module for canonical_polynomial.py

The subclassing is not tested here, see tesing module `test_polynomial.py`
"""
import numpy as np
import pytest
from conftest import (
    assert_polynomial_almost_equal,
    build_rnd_coeffs,
    build_rnd_points,
    build_random_newton_polynom,
)
from numpy.testing import assert_almost_equal

from minterpy.global_settings import INT_DTYPE
from minterpy.utils.polynomials.newton import eval_newton_polynomials
from minterpy import Grid, MultiIndexSet

from minterpy import (
    NewtonPolynomial,
    NewtonToCanonical,
    CanonicalToNewton,
    NewtonToLagrange,
)


@pytest.fixture(params=["numpy", "numba", "numba-par"])
def diff_backend(request):
    return request.param


def test_eval(multi_index_mnp, NrPoints, num_polynomials):
    """Test the evaluation of Newton polynomials."""

    coeffs = build_rnd_coeffs(multi_index_mnp, num_polynomials)
    poly = NewtonPolynomial(multi_index_mnp, coeffs)
    pts = build_rnd_points(NrPoints, multi_index_mnp.spatial_dimension)

    # Evaluate
    res = poly(pts)

    trafo_n2c = NewtonToCanonical(poly)
    canon_poly = trafo_n2c()
    groundtruth = canon_poly(pts)
    assert_almost_equal(res, groundtruth)


def test_eval_batch(multi_index_mnp, num_polynomials, BatchSizes):
    """Test the evaluation on Newton polynomials in batches of query points."""

    #TODO: This is a temporary test as the 'batch_size' parameter is not
    #      opened in the higher-level interface, i.e., 'newton_poly(xx)'

    # Create a random coefficient values
    newton_coeffs = build_rnd_coeffs(multi_index_mnp, num_polynomials)
    grid = Grid(multi_index_mnp)
    generating_points = grid.generating_points
    exponents = multi_index_mnp.exponents

    # Create test query points
    xx = build_rnd_points(421, multi_index_mnp.spatial_dimension)

    # Evaluate the polynomial in batches
    yy_newton = eval_newton_polynomials(
        xx, newton_coeffs, exponents, generating_points, batch_size=BatchSizes
    )
    if num_polynomials == 1:
        yy_newton = yy_newton.reshape(-1)

    # Create a reference results from canonical polynomial evaluation
    newton_poly = NewtonPolynomial(multi_index_mnp, newton_coeffs)
    canonical_poly = NewtonToCanonical(newton_poly)()
    yy_canonical = canonical_poly(xx)

    # Assert
    assert_almost_equal(yy_newton, yy_canonical)


class TestDiff:
    """All tests related to the differentiation of polys. in the Newton basis.
    """

    def test_zero_derivative(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
        diff_backend,
    ):
        """Test taking the 0th-order derivative of polynomials."""
        # Create a random Newton polynomial
        newton_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # A derivative of order zero along all dimensions should be equivalent
        # to the same polynomial
        orders = np.zeros(SpatialDimension, dtype=INT_DTYPE)
        zero_order_diff_newt = newton_poly.diff(orders)

        # Assertion
        assert_polynomial_almost_equal(zero_order_diff_newt, newton_poly)

    def test_vs_canonical(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
        diff_backend,
    ):
        """Test comparing the gradient with that computed in canonical basis.
        """
        # Create a random Newton polynomial
        newton_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # Transform to the canonical basis
        trafo_n2c = NewtonToCanonical(newton_poly)
        canon_poly = trafo_n2c()

        # Differentiate in the canonical basis and transform back
        diff_order = np.ones(SpatialDimension, dtype=INT_DTYPE)
        can_diff_poly = canon_poly.diff(diff_order)
        trafo_c2n = CanonicalToNewton(can_diff_poly)
        newt_can_diff_poly = trafo_c2n()

        # Differentiate the original polynomial
        newt_diff_poly = newton_poly.diff(diff_order, backend=diff_backend)

        # Assertion
        assert_polynomial_almost_equal(newt_can_diff_poly, newt_diff_poly)

    def test_partial_diff(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        diff_backend,
    ):
        """Test taking the partial derivative of polynomials."""
        # Create a random Newton polynomial
        newton_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # Check partial derivative on each dimension by comparing it
        # with the partial derivative in the canonical basis
        for dim in range(SpatialDimension):
            # Transform to the canonical basis
            trafo_n2c = NewtonToCanonical(newton_poly)
            canon_poly = trafo_n2c()
            # ...differentiate
            can_diff_poly = canon_poly.partial_diff(dim)
            # ...and transform back
            trafo_c2n = CanonicalToNewton(can_diff_poly)
            newt_can_diff_poly = trafo_c2n()

            # Differentiate the original polynomial
            newt_diff_poly = newton_poly.partial_diff(
                dim,
                backend=diff_backend,
            )

            # Assertion
            assert_polynomial_almost_equal(newt_can_diff_poly, newt_diff_poly)

    def test_unsupported_backend(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
        diff_backend,
    ):
        """Test unsupported backend to differentiate Newton polynomials."""
        # Create a Newton polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        nwt_coeffs = np.random.rand(len(mi))
        nwt_poly = NewtonPolynomial(mi, nwt_coeffs)

        # Attempt to differentiate with a non-supported back-end
        unsupported_backend = "numdumb"
        with pytest.raises(NotImplementedError):
            nwt_poly.partial_diff(0, backend=unsupported_backend)

        with pytest.raises(NotImplementedError):
            nwt_poly.diff(
                order=np.ones(SpatialDimension, dtype=np.int_),
                backend=unsupported_backend,
            )


def test_integrate_over_bounds_invalid_shape(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid shape."""
    # Create a Newton polynomial
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = np.random.rand(SpatialDimension + 3, 2)
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        nwt_poly.integrate_over(bounds)


def test_integrate_over_bounds_invalid_domain(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with bounds of invalid domain."""
    # Create a Newton polynomial
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create bounds (outside the canonical domain of [-1, 1]^M)
    bounds = 2 * np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1

    with pytest.raises(ValueError):
        nwt_poly.integrate_over(bounds)


def test_integrate_over_bounds_equal(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with equal bounds (should be zero)."""
    # Create a Newton polynomial
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Create bounds (one of them has lb == ub)
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1
    idx = np.random.choice(SpatialDimension)
    bounds[idx, 0] = bounds[idx, 1]

    # Compute the integral
    ref = 0.0
    value = nwt_poly.integrate_over(bounds)

    # Assertion
    assert np.isclose(ref, value)


def test_integrate_over_bounds_flipped(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration with specified and valid bounds."""
    # Create a Newton polynomial
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Compute the integral
    value_1 = nwt_poly.integrate_over()

    # Flip bounds
    bounds = np.ones((SpatialDimension, 2))
    bounds[:, 0] *= -1
    bounds[:, [0, 1]] = bounds[:, [1, 0]]

    # Compute the integral with flipped bounds
    value_2 = nwt_poly.integrate_over(bounds)

    if np.mod(SpatialDimension, 2) == 1:
        # Odd spatial dimension flips the sign
        assert np.isclose(value_1, -1 * value_2)
    else:
        assert np.isclose(value_1, value_2)


def test_integrate_over(
    SpatialDimension, PolyDegree, LpDegree
):
    """Test polynomial integration in different basis (sanity check)."""
    # Create a Canonical polynomial
    nwt_poly = build_random_newton_polynom(
        SpatialDimension, PolyDegree, LpDegree
    )

    # Transform to other polynomial bases
    lag_poly = NewtonToLagrange(nwt_poly)()
    can_poly = NewtonToCanonical(nwt_poly)()

    # Compute the integral
    value_nwt = nwt_poly.integrate_over()
    value_lag = lag_poly.integrate_over()
    # NOTE: Canonical integration won't work in high degree
    value_can = can_poly.integrate_over()

    # Assertions
    assert np.isclose(value_nwt, value_lag)
    assert np.isclose(value_nwt, value_can)

    # Create bounds
    bounds = np.random.rand(SpatialDimension, 2)
    bounds[:, 0] *= -1

    # Compute the integral with bounds
    value_lag = lag_poly.integrate_over(bounds)
    value_nwt = nwt_poly.integrate_over(bounds)
    value_can = can_poly.integrate_over(bounds)

    # Assertions
    assert np.isclose(value_nwt, value_lag)
    assert np.isclose(value_nwt, value_can)
