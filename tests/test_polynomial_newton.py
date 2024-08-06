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


def test_eval(MultiIndices, NrPoints, num_polynomials):
    """Test the evaluation of Newton polynomials."""

    coeffs = build_rnd_coeffs(MultiIndices, num_polynomials)
    poly = NewtonPolynomial(MultiIndices, coeffs)
    pts = build_rnd_points(NrPoints, MultiIndices.spatial_dimension)

    # Evaluate
    res = poly(pts)

    trafo_n2c = NewtonToCanonical(poly)
    canon_poly = trafo_n2c()
    groundtruth = canon_poly(pts)
    assert_almost_equal(res, groundtruth)


def test_eval_batch(MultiIndices, num_polynomials, BatchSizes):
    """Test the evaluation on Newton polynomials in batches of query points."""

    #TODO: This is a temporary test as the 'batch_size' parameter is not
    #      opened in the higher-level interface, i.e., 'newton_poly(xx)'

    # Create a random coefficient values
    newton_coeffs = build_rnd_coeffs(MultiIndices, num_polynomials)
    grid = Grid(MultiIndices)
    generating_points = grid.generating_points
    exponents = MultiIndices.exponents

    # Create test query points
    xx = build_rnd_points(421, MultiIndices.spatial_dimension)

    # Evaluate the polynomial in batches
    yy_newton = eval_newton_polynomials(
        xx, newton_coeffs, exponents, generating_points, batch_size=BatchSizes
    )
    if num_polynomials == 1:
        yy_newton = yy_newton.reshape(-1)

    # Create a reference results from canonical polynomial evaluation
    newton_poly = NewtonPolynomial(MultiIndices, newton_coeffs)
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


class TestPolyMultiplication:
    """All tests related to the poly.-poly. multiplication in the Newton basis.
    """
    def test_identity(self, SpatialDimension, PolyDegree, LpDegree):
        """Test multiplicative identity of polynomial multiplication.

        A polynomial multiplied with a constant polynomial with coefficient
        value of 1.0 should result in the same polynomial in value.
        """
        # Create a Newton polynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension, PolyDegree, LpDegree
        )

        # Create a constant identity Newton polynomial
        exponents = np.zeros((1, SpatialDimension), dtype=np.int_)
        mi = MultiIndexSet(exponents, LpDegree)
        coeffs = np.array([1.0])
        nwt_poly_constant = NewtonPolynomial(mi, coeffs)

        # Multiplication
        nwt_poly_prod_1 = nwt_poly * nwt_poly_constant
        nwt_poly_prod_2 = nwt_poly_constant * nwt_poly

        # Assertions
        assert nwt_poly == nwt_poly_prod_1
        assert nwt_poly == nwt_poly_prod_2

    def test_identity_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test multiplicative identity with multiple sets of coefficients.

        A polynomial with multiple sets of coefficients multiplied with
        a constant polynomial with coefficient values of 1.0 should return
        the same polynomial in value.
        """
        # Create a Newton polynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # Create a constant identity Newton polynomial
        exponents = np.zeros((1, SpatialDimension), dtype=np.int_)
        mi = MultiIndexSet(exponents, LpDegree)
        coeffs = np.ones((1, num_polynomials))
        nwt_poly_constant = NewtonPolynomial(mi, coeffs)

        # Multiplication
        nwt_poly_prod_1 = nwt_poly * nwt_poly_constant
        nwt_poly_prod_2 = nwt_poly_constant * nwt_poly

        # Assertions
        assert nwt_poly == nwt_poly_prod_1
        assert nwt_poly == nwt_poly_prod_2

    def test_constant_sanity(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the multiplication with an arbitrary constant polynomial.

        A polynomial multiplied with a constant polynomial should return
        a polynomial multiplied with a scalar.
        """
        # Create a Newton polynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # Create a constant identity Newton polynomial
        exponents = np.zeros((1, SpatialDimension), dtype=np.int_)
        mi = MultiIndexSet(exponents, LpDegree)
        coeffs = np.random.rand(1)
        nwt_poly_constant = NewtonPolynomial(mi, coeffs)

        # Multiplication
        nwt_poly_prod_1 = nwt_poly * nwt_poly_constant
        nwt_poly_prod_2 = nwt_poly_constant * nwt_poly
        nwt_poly_prod_3 = nwt_poly * coeffs[0]

        # Assertions
        assert nwt_poly_prod_3 == nwt_poly_prod_1
        assert nwt_poly_prod_3 == nwt_poly_prod_2

    def test_eval(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the evaluation results of polynomial multiplication."""
        # Create two Newton polynomials
        nwt_poly_1 = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )
        # Increase both the spatial dimension and poly. degree so
        # the polynomial differ
        nwt_poly_2 = build_random_newton_polynom(
            SpatialDimension+1,
            PolyDegree+1,
            LpDegree,
        )

        # Generate random test points
        xx_test = -1 + 2 * np.random.rand(10000, SpatialDimension+1)

        # Compute reference results
        yy_1 = nwt_poly_1(xx_test[:, :SpatialDimension])
        yy_2 = nwt_poly_2(xx_test[:, :SpatialDimension+1])
        yy_ref = yy_1 * yy_2

        # Multiply polynomial
        nwt_poly_prod = nwt_poly_1 * nwt_poly_2
        yy_prod = nwt_poly_prod(xx_test)

        # Assertion
        assert np.allclose(yy_ref, yy_prod)

    def test_eval_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test the evaluation results of poly. multiplication with multiple
        sets of coefficients.
        """
        # Create two Newton polynomials
        nwt_poly_1 = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )
        # Increase both the spatial dimension and poly. degree so
        # the polynomial differ
        nwt_poly_2 = build_random_newton_polynom(
            SpatialDimension + 1,
            PolyDegree + 1,
            LpDegree,
            num_polynomials,
        )

        # Generate random test points
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension+1)

        # Compute reference results
        yy_1 = nwt_poly_1(xx_test[:, :SpatialDimension])
        yy_2 = nwt_poly_2(xx_test[:, :SpatialDimension+1])
        yy_ref = yy_1 * yy_2

        # Multiply polynomial
        nwt_poly_prod = nwt_poly_1 * nwt_poly_2
        yy_prod = nwt_poly_prod(xx_test)

        # Assertion
        assert np.allclose(yy_ref, yy_prod)


class TestScalarAddition:
    """All tests related to polynomial-scalar addition."""
    def test_sanity(self, SpatialDimension, PolyDegree, LpDegree):
        """Test adding additive identity."""
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # Addition with additive identity
        nwt_poly_sum_1 = nwt_poly + 0
        nwt_poly_sum_2 = 0 + nwt_poly  # Commutativity holds

        # Assertions
        assert nwt_poly == nwt_poly_sum_1
        assert nwt_poly == nwt_poly_sum_2

    def test_sanity_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test adding additive identity on polynomial with multiple sets of
        coefficients.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # Addition with additive identity
        nwt_poly_sum_1 = nwt_poly + 0
        nwt_poly_sum_2 = 0 + nwt_poly  # Commutativity holds

        # Assertions
        assert nwt_poly == nwt_poly_sum_1
        assert nwt_poly == nwt_poly_sum_2

    def test_inplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test inplace scalar addition (not implemented)."""
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # In-place addition
        with pytest.raises(NotImplementedError):
            nwt_poly += 5

    def test_eval(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the evaluation of a polynomial summed with a scalar."""
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension)

        # Compute reference results
        yy_ref = nwt_poly(xx_test) + 5

        # Summed a polynomial
        yy_1 = (nwt_poly + 5)(xx_test)
        yy_2 = (5 + nwt_poly)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)

    def test_eval_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test the evaluation of a polynomial with multiple sets of
        coefficients summed with a scalar.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension)

        # Compute reference results
        yy_ref = nwt_poly(xx_test) + 5

        # Summed a polynomial
        yy_1 = (nwt_poly + 5)(xx_test)
        yy_2 = (5 + nwt_poly)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)


class TestScalarSubtraction:
    """All tests related to polynomial-scalar subtraction."""
    def test_sanity(self, SpatialDimension, PolyDegree, LpDegree):
        """Test adding additive identity."""
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # Addition with additive identity
        nwt_poly_sum_1 = nwt_poly - 0
        nwt_poly_sum_2 = -0 + nwt_poly  # Commutativity holds

        # Assertions
        assert nwt_poly == nwt_poly_sum_1
        assert nwt_poly == nwt_poly_sum_2

    def test_sanity_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test adding additive identity on polynomial with multiple sets of
        coefficients.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # Addition with additive identity
        nwt_poly_sum_1 = nwt_poly - 0
        nwt_poly_sum_2 = -0 + nwt_poly  # Commutativity holds

        # Assertions
        assert nwt_poly == nwt_poly_sum_1
        assert nwt_poly == nwt_poly_sum_2

    def test_rsub_single_poly(self, SpatialDimension, PolyDegree, LpDegree):
        """Test right-sided subtraction of a polynomial with a single set of
        coefficients by a scalar.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # Right-sided subtraction
        nwt_poly_sub = 5 - nwt_poly

        # Reference
        yy_ref = -1 * nwt_poly.coeffs.copy()
        yy_ref[0] += 5
        nwt_poly_ref = NewtonPolynomial(
            nwt_poly.multi_index,
            yy_ref,
            user_domain=nwt_poly.user_domain,
            internal_domain=nwt_poly.internal_domain,
            grid=nwt_poly.grid,
        )

        # Assertion
        assert nwt_poly_ref == nwt_poly_sub

    def test_rsub_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test right-sided subtraction of a polynomial with multiple sets
        of coefficients by a scalar.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # Right-sided subtraction
        nwt_poly_sub = 5 - nwt_poly

        # Reference
        yy_ref = -1 * nwt_poly.coeffs.copy()
        yy_ref[0] += 5
        nwt_poly_ref = NewtonPolynomial(
            nwt_poly.multi_index,
            yy_ref,
            user_domain=nwt_poly.user_domain,
            internal_domain=nwt_poly.internal_domain,
            grid=nwt_poly.grid,
        )

        # Assertion
        assert nwt_poly_ref == nwt_poly_sub

    def test_inplace(self, SpatialDimension, PolyDegree, LpDegree):
        """Test inplace scalar addition (not implemented)."""
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # In-place addition
        with pytest.raises(NotImplementedError):
            nwt_poly -= 5

    def test_eval(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the evaluation of a polynomial subtracted by a scalar."""
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension)

        # Compute reference results
        yy_ref = nwt_poly(xx_test) - 5

        # Summed a polynomial
        yy_1 = (nwt_poly - 5)(xx_test)
        yy_2 = (-5 + nwt_poly)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)

    def test_eval_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test the evaluation of a polynomial with multiple sets of
        coefficients subtracted by a scalar.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension)

        # Compute reference results
        yy_ref = nwt_poly(xx_test) - 5

        # Summed a polynomial
        yy_1 = (nwt_poly - 5)(xx_test)
        yy_2 = (-5 + nwt_poly)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)


class TestPolyAddition:
    """All tests related to polynomial-polynomial addition."""
    def test_self(self, SpatialDimension, PolyDegree, LpDegree):
        """Test adding the same polynomial instance."""
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # Self addition
        nwt_poly_sum = nwt_poly + nwt_poly

        # Assertion
        assert nwt_poly_sum == 2 * nwt_poly

    def test_self_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test adding the same polynomial instance with multiple sets of
        coefficients.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # Self addition
        nwt_poly_sum = nwt_poly + nwt_poly

        # Assertion
        assert nwt_poly_sum == 2 * nwt_poly

    def test_eval(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the evaluation of a polynomial summed with another polynomial.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly_1 = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )
        # Increase both the spatial dimension and poly. degree so
        # the polynomial differ
        nwt_poly_2 = build_random_newton_polynom(
            SpatialDimension + 1,
            PolyDegree + 1,
            LpDegree,
        )

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension+1)

        # Compute reference results
        yy_r1 = nwt_poly_1(xx_test[:, :SpatialDimension])
        yy_r2 = nwt_poly_2(xx_test[:, :SpatialDimension+1])
        yy_ref = yy_r1 + yy_r2

        # Summed a polynomial
        yy_1 = (nwt_poly_1 + nwt_poly_2)(xx_test)
        yy_2 = (nwt_poly_2 + nwt_poly_1)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)

    def test_eval_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test the evaluation of a polynomial with multiple sets of
        coefficients summed with another polynomial.
        """
        # Create instances of NewtonPolynomial
        nwt_poly_1 = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )
        # Increase both the spatial dimension and poly. degree so
        # the polynomial differ
        nwt_poly_2 = build_random_newton_polynom(
            SpatialDimension+1,
            PolyDegree+1,
            LpDegree,
            num_polynomials,
        )

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension + 1)

        # Compute reference results
        yy_r1 = nwt_poly_1(xx_test[:, :SpatialDimension])
        yy_r2 = nwt_poly_2(xx_test[:, :SpatialDimension + 1])
        yy_ref = yy_r1 + yy_r2

        # Summed a polynomial
        yy_1 = (nwt_poly_1 + nwt_poly_2)(xx_test)
        yy_2 = (nwt_poly_2 + nwt_poly_1)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)


class TestPolySubtraction:
    """All tests related to polynomial-polynomial subtraction."""
    def test_self(self, SpatialDimension, PolyDegree, LpDegree):
        """Test adding the same polynomial instance."""
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )

        # Self addition
        nwt_poly_sum = nwt_poly - nwt_poly

        # Assertion
        assert nwt_poly_sum == 0 * nwt_poly

    def test_self_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test subtracting the same polynomial instance with multiple sets of
        coefficients.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )

        # Self addition
        nwt_poly_sum = nwt_poly - nwt_poly

        # Assertion
        assert nwt_poly_sum == 0 * nwt_poly

    def test_eval(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the evaluation of a polynomial subtracted by another
        polynomial.
        """
        # Create an instance of NewtonPolynomial
        nwt_poly_1 = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
        )
        # Increase both the spatial dimension and poly. degree so
        # the polynomial differ
        nwt_poly_2 = build_random_newton_polynom(
            SpatialDimension + 1,
            PolyDegree + 1,
            LpDegree,
        )

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension+1)

        # Compute reference results
        yy_r1 = nwt_poly_1(xx_test[:, :SpatialDimension])
        yy_r2 = nwt_poly_2(xx_test[:, :SpatialDimension+1])
        yy_ref = yy_r1 - yy_r2

        # Summed a polynomial
        yy_1 = (nwt_poly_1 - nwt_poly_2)(xx_test)
        yy_2 = (-nwt_poly_2 + nwt_poly_1)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)

    def test_eval_multiple_polys(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test the evaluation of a polynomial with multiple sets of
        coefficients subtracted by another polynomial.
        """
        # Create instances of NewtonPolynomial
        nwt_poly_1 = build_random_newton_polynom(
            SpatialDimension,
            PolyDegree,
            LpDegree,
            num_polynomials,
        )
        # Increase both the spatial dimension and poly. degree so
        # the polynomial differ
        nwt_poly_2 = build_random_newton_polynom(
            SpatialDimension+1,
            PolyDegree+1,
            LpDegree,
            num_polynomials,
        )

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension + 1)

        # Compute reference results
        yy_r1 = nwt_poly_1(xx_test[:, :SpatialDimension])
        yy_r2 = nwt_poly_2(xx_test[:, :SpatialDimension + 1])
        yy_ref = yy_r1 - yy_r2

        # Summed a polynomial
        yy_1 = (nwt_poly_1 - nwt_poly_2)(xx_test)
        yy_2 = (-nwt_poly_2 + nwt_poly_1)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)
