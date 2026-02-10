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
