"""
Testing module for the transformation of Newton polynomials.

This module contains tests that are specific to the Newton polynomials;
common functionalities and behavior of transformation instances
and the corresponding operators are tested in a separate module.

The transformation of polynomials in the Newton bases to the Lagrange
bases are available in two flavors: barycentric and naive versions.
These two should be tested.
"""
import numpy as np
import pytest

from conftest import build_rnd_coeffs, create_non_downward_closed_multi_index

from minterpy import (
    MultiIndexSet,
    NewtonPolynomial,
    NewtonToLagrange,
    NewtonToCanonical,
    NewtonToChebyshev,
)
from minterpy.transformations.utils import (
    _build_newton_to_lagrange_bary,
    _build_newton_to_lagrange_naive,
)


class TestDownwardClosed:
    """All transformation tests involving polys. with downward-closed sets."""
    def test_to_lagrange(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the default transformation to the Lagrange basis."""
        # Create a Newton polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        nwt_poly = NewtonPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = nwt_poly.grid.unisolvent_nodes
        lag_coeffs_ref = nwt_poly(unisolvent_nodes)

        # Transform
        lag_coeffs = NewtonToLagrange(nwt_poly)().coeffs

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_canonical(self,  SpatialDimension, PolyDegree, LpDegree):
        """Test the default transformation to the Canonical basis.

        Notes
        -----
        - If the tested polynomial degrees defined in the conftest.py increase,
          the canonical polynomial may become unstable and the test fails.
        """
        # Create a Newton polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        nwt_poly = NewtonPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = nwt_poly.grid.unisolvent_nodes
        lag_coeffs_ref = nwt_poly(unisolvent_nodes)

        # Transform to the canonical basis
        can_poly = NewtonToCanonical(nwt_poly)()
        lag_coeffs = can_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_chebyshev(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the default transformation to the Chebyshev basis."""
        # Create a Newton polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        nwt_poly = NewtonPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = nwt_poly.grid.unisolvent_nodes
        lag_coeffs_ref = nwt_poly(unisolvent_nodes)

        # Transform to the Chebyshev basis
        cheb_poly = NewtonToChebyshev(nwt_poly)()
        lag_coeffs = cheb_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)


class TestNonDownwardClosed:
    """All transformation tests involving polys. with non-downward-closed sets.

    Notes
    -----
    - Transformation of polynomials having a non-downward-close multi-index set
      is separately tested because Minterpy may use different methods
      for the transformation. Furthermore, some transformation strictly
      requires that the polynomials having a downward-close multi-index set.
    """
    def test_to_lagrange(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the Lagrange basis.

        Notes
        -----
        - If the multi-index set is not downward-closed,
          the naive transformation operator is automatically selected.
        """
        mi = create_non_downward_closed_multi_index(
            SpatialDimension,
            PolyDegree,
            LpDegree
        )
        assert not mi.is_downward_closed

        # Create a Newton polynomial
        coeffs = build_rnd_coeffs(mi)
        nwt_poly = NewtonPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = nwt_poly.grid.unisolvent_nodes
        lag_coeffs_ref = nwt_poly(unisolvent_nodes)

        # Transform
        lag_coeffs = NewtonToLagrange(nwt_poly)().coeffs

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_canonical(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the canonical basis.

        Notes
        -----
        - Newton to Canonical transformation strictly requires the multi-index
          set to be downward-closed.
        """
        mi = create_non_downward_closed_multi_index(
            SpatialDimension,
            PolyDegree,
            LpDegree
        )
        assert not mi.is_downward_closed

        # Create a Newton polynomial
        coeffs = build_rnd_coeffs(mi)
        nwt_poly = NewtonPolynomial(mi, coeffs)

        # Newton to Canonical requires the multi-indices to be downward-closed
        with pytest.raises(ValueError):
            NewtonToCanonical(nwt_poly)()

    def test_to_chebyshev(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the Chebyshev basis."""
        mi = create_non_downward_closed_multi_index(
            SpatialDimension,
            PolyDegree,
            LpDegree
        )
        assert not mi.is_downward_closed

        # Create a Newton polynomial
        coeffs = build_rnd_coeffs(mi)
        nwt_poly = NewtonPolynomial(mi, coeffs)

        # Evaluate the polynomial at the unisolvent nodes
        unisolvent_nodes = nwt_poly.grid.unisolvent_nodes
        lag_coeffs_ref = nwt_poly(unisolvent_nodes)

        # Transform to the Chebyshev basis
        cheb_poly = NewtonToChebyshev(nwt_poly)()
        lag_coeffs = cheb_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)


def test_newton2lagrange_naive_vs_bary(SpatialDimension, PolyDegree, LpDegree):
    """Test the naive and barycentric Newton to Lagrange transformations."""
    # Create a Newton polynomial
    mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
    coeffs = build_rnd_coeffs(mi)
    nwt_poly = NewtonPolynomial(mi, coeffs)

    # Compute the polynomial values at the unisolvent nodes
    unisolvent_nodes = nwt_poly.grid.unisolvent_nodes
    lag_coeffs_ref = nwt_poly(unisolvent_nodes)

    # Construct a transformation object
    transformer_n2l = NewtonToLagrange(nwt_poly)

    # Naive transformation
    operator_n2l_naive = _build_newton_to_lagrange_naive(transformer_n2l)
    lag_coeffs_naive = operator_n2l_naive @ nwt_poly.coeffs

    # Barycentric transformation
    operator_n2l_baryc = _build_newton_to_lagrange_bary(transformer_n2l)
    lag_coeffs_baryc = operator_n2l_baryc @ nwt_poly.coeffs

    # Assertions
    assert np.allclose(lag_coeffs_ref, lag_coeffs_naive)
    assert np.allclose(lag_coeffs_ref, lag_coeffs_baryc)
    assert np.allclose(lag_coeffs_naive, lag_coeffs_baryc)
