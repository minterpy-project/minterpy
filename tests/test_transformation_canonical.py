"""
Testing module for the transformation of canonical polynomials.

This module contains tests that are specific to the canonical polynomials;
common functionalities and behaviors of transformation instances
and the corresponding operators are tested in a separate module.

Notes
-----
- Canonical basis is known to be unstable at high polynomial degree.
  If the tested polynomial degrees defined in the conftest.py increase,
  the tests may fail.
"""
import numpy as np
import pytest

from conftest import build_rnd_coeffs

from minterpy import (
    MultiIndexSet,
    CanonicalPolynomial,
    CanonicalToLagrange,
    CanonicalToNewton,
    CanonicalToChebyshev,
)


class TestDownwardClosed:
    """All transformation tests involving polys. with downward-closed sets."""
    def test_to_lagrange(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the Lagrange basis."""
        # Create a canonical polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        can_poly = CanonicalPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = can_poly.unisolvent_nodes
        lag_coeffs_ref = can_poly(unisolvent_nodes)

        # Transform to the Lagrange basis
        lag_poly = CanonicalToLagrange(can_poly)()
        lag_coeffs = lag_poly.coeffs

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_newton(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the Newton basis."""
        # Create a canonical polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        can_poly = CanonicalPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = can_poly.unisolvent_nodes
        lag_coeffs_ref = can_poly(unisolvent_nodes)

        # Transform to the Newton basis
        nwt_poly = CanonicalToNewton(can_poly)()
        lag_coeffs = nwt_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_chebyshev(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the Chebyshev basis."""
        # Create a canonical polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        can_poly = CanonicalPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = can_poly.unisolvent_nodes
        lag_coeffs_ref = can_poly(unisolvent_nodes)

        # Transform to the Chebyshev basis
        cheb_poly = CanonicalToChebyshev(can_poly)()
        lag_coeffs =  cheb_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)


class TestNonDownwardClosed:
    """All transformation tests involving polys. with non-downward-closed sets.

    Notes
    -----
    - Any transformation from canonical polynomial strictly requires
      a downward-closed multi-index set.
    """
    def test_to_lagrange(self, multi_index_non_downward_closed):
        """Test the transformation to the Lagrange basis."""
        # Get the non-downward-closed multi-index set
        mi = multi_index_non_downward_closed
        assert not mi.is_downward_closed

        # Create a canonical polynomial
        coeffs = build_rnd_coeffs(mi)
        can_poly = CanonicalPolynomial(mi, coeffs)

        # Transform to the Lagrange basis
        with pytest.raises(ValueError):
            CanonicalToLagrange(can_poly)()

    def test_to_newton(self, multi_index_non_downward_closed):
        """Test the transformation to the Newton basis."""
        # Get the non-downward-closed multi-index set
        mi = multi_index_non_downward_closed
        assert not mi.is_downward_closed

        # Create a canonical polynomial
        coeffs = build_rnd_coeffs(mi)
        can_poly = CanonicalPolynomial(mi, coeffs)

        # Transform to the Newton basis
        with pytest.raises(ValueError):
            CanonicalToNewton(can_poly)()

    def test_to_chebyshev(self, multi_index_non_downward_closed):
        """Test the transformation to the Chebyshev basis."""
        # Get the non-downward-closed multi-index set
        mi = multi_index_non_downward_closed
        assert not mi.is_downward_closed

        # Create a canonical polynomial
        coeffs = build_rnd_coeffs(mi)
        can_poly = CanonicalPolynomial(mi, coeffs)

        # Transform to the Newton basis
        with pytest.raises(ValueError):
            CanonicalToChebyshev(can_poly)()
