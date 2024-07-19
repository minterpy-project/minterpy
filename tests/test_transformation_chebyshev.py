"""
Testing module for the transformation of Chebyshev polys. of the first kind.

This module contains tests that are specific to the Chebyshev polynomials;
common functionalities and behavior of transformation instances
and the corresponding operators are tested in a separate module.
"""
import numpy as np
import pytest

from conftest import build_rnd_coeffs

from minterpy import (
    MultiIndexSet,
    ChebyshevPolynomial,
    ChebyshevToLagrange,
    ChebyshevToNewton,
    ChebyshevToCanonical,
)


class TestDownwardClosed:
    """All transformation tests involving polys. with downward-closed sets."""
    def test_to_lagrange(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the Lagrange basis."""
        # Create a Chebyshev polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        cheb_poly = ChebyshevPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = cheb_poly.unisolvent_nodes
        lag_coeffs_ref = cheb_poly(unisolvent_nodes)

        # Transform to the Lagrange basis
        lag_poly = ChebyshevToLagrange(cheb_poly)()
        lag_coeffs = lag_poly.coeffs

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_newton(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the Newton basis."""
        # Create a Chebyshev polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        cheb_poly = ChebyshevPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = cheb_poly.unisolvent_nodes
        lag_coeffs_ref = cheb_poly(unisolvent_nodes)

        # Transform to the Newton basis
        nwt_poly = ChebyshevToNewton(cheb_poly)()
        lag_coeffs = nwt_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_canonical(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the canonical basis."""
        # Create a Chebyshev polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        cheb_poly = ChebyshevPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = cheb_poly.unisolvent_nodes
        lag_coeffs_ref = cheb_poly(unisolvent_nodes)

        # Transform to the canonical basis
        can_poly = ChebyshevToCanonical(cheb_poly)()
        lag_coeffs = can_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)


class TestNonDownwardClosed:
    """All transformation tests involving polys. with non-downward-closed sets.
    """
    def test_to_lagrange(self, multi_index_non_downward_closed):
        """Test the transformation to the Lagrange basis."""
        # Get the non-downward-closed multi-index set
        mi = multi_index_non_downward_closed
        assert not mi.is_downward_closed

        # Create a Chebyshev polynomial
        coeffs = build_rnd_coeffs(mi)
        cheb_poly = ChebyshevPolynomial(mi, coeffs)

        # Transform to the Lagrange basis
        lag_poly = ChebyshevToLagrange(cheb_poly)()

        # Evaluate the polynomials at the unisolvent nodes
        unisolvent_nodes = cheb_poly.unisolvent_nodes
        lag_coeffs_ref = cheb_poly(unisolvent_nodes)
        lag_coeffs = lag_poly.coeffs

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_newton(self, multi_index_non_downward_closed):
        """Test the transformation to the Newton basis."""
        # Get the non-downward-closed multi-index set
        mi = multi_index_non_downward_closed
        assert not mi.is_downward_closed

        # Create a Chebyshev polynomial
        coeffs = build_rnd_coeffs(mi)
        cheb_poly = ChebyshevPolynomial(mi, coeffs)

        # Transform to the Newton basis
        nwt_poly = ChebyshevToNewton(cheb_poly)()

        # Evaluate the polynomials at the unisolvent nodes
        unisolvent_nodes = cheb_poly.unisolvent_nodes
        lag_coeffs_ref = cheb_poly(unisolvent_nodes)
        lag_coeffs = nwt_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_canonical(self, multi_index_non_downward_closed):
        """Test the transformation to the canonical basis.

        Notes
        -----
        - Transformation to the canonical basis strictly requires
          a downward-closed multi-index set.
        """
        # Get the non-downward-closed multi-index set
        mi = multi_index_non_downward_closed
        assert not mi.is_downward_closed

        # Create a Chebyshev polynomial
        coeffs = build_rnd_coeffs(mi)
        cheb_poly = ChebyshevPolynomial(mi, coeffs)

        # Transform to the canonical basis
        with pytest.raises(ValueError):
            ChebyshevToCanonical(cheb_poly)()
