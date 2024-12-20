"""
Testing module for the transformation of Lagrange polynomials.

This module contains tests that are specific to the Lagrange polynomials;
common functionalities and behavior of transformation instances
and the corresponding operators are tested in a separate module.

The transformation of polynomials in the Lagrange bases to the Newton
bases are available in two flavors: barycentric and naive versions.
These two should be tested.
"""
import numpy as np
import pytest

from numpy.linalg import LinAlgError

from conftest import build_rnd_coeffs

from minterpy import (
    MultiIndexSet,
    LagrangePolynomial,
    LagrangeToNewton,
    LagrangeToCanonical,
    LagrangeToChebyshev,
    NewtonPolynomial,
)
from minterpy.transformations.utils import (
    build_l2n_matrix_dds,
    _build_lagrange_to_newton_bary,
    _build_lagrange_to_newton_naive,
)


class TestDownwardClosed:
    """All transformation tests involving polys. with downward-closed sets."""
    def test_to_newton(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the default transformation to the Newton basis."""
        # Create a Lagrange polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        lag_coeffs_ref = build_rnd_coeffs(mi)
        lag_poly = LagrangePolynomial(mi, lag_coeffs_ref)

        # Transform to the Newton basis
        nwt_poly = LagrangeToNewton(lag_poly)()

        # Evaluate the Newton polynomial at the unisolvent nodes
        unisolvent_nodes = lag_poly.grid.unisolvent_nodes
        lag_coeffs = nwt_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_canonical(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the default transformation to the canonical basis.

        Notes
        -----
        - If the tested polynomial degrees defined in the conftest.py increase,
          the canonical polynomial may become unstable and the test fails.
        """
        # Create a Lagrange polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        lag_coeffs_ref = build_rnd_coeffs(mi)
        lag_poly = LagrangePolynomial(mi, lag_coeffs_ref)

        # Transform to the canonical basis
        can_poly = LagrangeToCanonical(lag_poly)()

        # Evaluate the canonical polynomial at the unisolvent nodes
        unisolvent_nodes = lag_poly.grid.unisolvent_nodes
        lag_coeffs = can_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_chebyshev(self, SpatialDimension, PolyDegree, LpDegree):
        """Test the transformation to the Chebyshev basis."""
        # Create a Lagrange polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        lag_coeffs_ref = build_rnd_coeffs(mi)
        lag_poly = LagrangePolynomial(mi, lag_coeffs_ref)

        # Transform to the Chebyshev basis
        cheb_poly = LagrangeToChebyshev(lag_poly)()

        # Evaluate the Chebyshev polynomial at the unisolvent nodes
        unisolvent_nodes = lag_poly.grid.unisolvent_nodes
        lag_coeffs = cheb_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)


class TestNonDownwardClosed:
    """All transformation tests involving polys. with non-downward-closed sets.

    Notes
    -----
    - Transformation of polynomials having a non-downward-close multi-index set
      is separately tested because Minterpy may use different methods
      for the transformation. Furthermore, certain transformations strictly
      require that the polynomials having a downward-close multi-index set.
    """
    def test_to_newton(self, multi_index_non_downward_closed):
        """Test the transformation to the Newton basis.

        Notes
        -----
        - If the multi-index set is not downward-closed, the naive
          transformation operator is automatically selected.
        """
        # Get the non-downward-closed set
        mi = multi_index_non_downward_closed
        assert not mi.is_downward_closed

        # Create a Lagrange polynomial
        lag_coeffs_ref = build_rnd_coeffs(mi)
        lag_poly = LagrangePolynomial(mi, lag_coeffs_ref)

        # Transform to the Newton basis
        nwt_poly = LagrangeToNewton(lag_poly)()

        # Evaluate the canonical polynomial at the unisolvent nodes
        unisolvent_nodes = lag_poly.grid.unisolvent_nodes
        lag_coeffs = nwt_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)

    def test_to_canonical(self, multi_index_non_downward_closed):
        """Test the transformation to the canonical basis.

        Notes
        -----
        - The transformation to the canonical basis strictly requires that
          the multi-index set is downward-closed.
        """
        # Get the non-downward-closed set
        mi = multi_index_non_downward_closed
        assert not mi.is_downward_closed

        # Create a Lagrange polynomial
        lag_coeffs_ref = build_rnd_coeffs(mi)
        lag_poly = LagrangePolynomial(mi, lag_coeffs_ref)

        # Transform to the canonical basis
        with pytest.raises(ValueError):
            LagrangeToCanonical(lag_poly)()

    def test_to_chebyshev(self, multi_index_non_downward_closed):
        """Test the transformation to the Chebyshev basis.

        Notes
        -----
        - Transformation of polynomials from the Lagrange basis to
          the Chebyshev basis may result in singular matrix for some cases of
          non-downward-closed multi-index sets. For such sets,
          the transformation is not guaranteed and the proper exception is
          caught below.
        """
        # Create a non-downward-closed set
        mi = multi_index_non_downward_closed
        assert not mi.is_downward_closed

        # Create a Lagrange polynomial
        lag_coeffs_ref = build_rnd_coeffs(mi)
        lag_poly = LagrangePolynomial(mi, lag_coeffs_ref)

        # Transform to the Chebyshev basis
        try:
            cheb_poly = LagrangeToChebyshev(lag_poly)()
        except LinAlgError as e_info:
            # Some cases may cause an inversion of a singular matrix
            # which is expected
            assert "singular matrix" in str(e_info).lower()
            return

        # Evaluate the chebyshev polynomial at the unisolvent nodes
        unisolvent_nodes = lag_poly.grid.unisolvent_nodes
        lag_coeffs = cheb_poly(unisolvent_nodes)

        # Assertion
        assert np.allclose(lag_coeffs_ref, lag_coeffs)


class TestLagrange2NewtonOperator:
    """Tests related to specific behaviors of Lag. to Newton transformation."""
    def test_to_newton_naive_vs_bary(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test the naive and barycentric Lagrange transformation."""
        # Create a Lagrange polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        lag_coeffs_ref = build_rnd_coeffs(mi)
        lag_poly = LagrangePolynomial(mi, lag_coeffs_ref)

        # Construct the transformer object
        transformer_l2n = LagrangeToNewton(lag_poly)

        # Naive transformation
        operator_l2n_naive = _build_lagrange_to_newton_naive(transformer_l2n)
        nwt_coeffs_naive = operator_l2n_naive @ lag_coeffs_ref
        nwt_poly_naive = NewtonPolynomial(mi, nwt_coeffs_naive)

        # Barycentric transformation
        operator_l2n_bary = _build_lagrange_to_newton_bary(transformer_l2n)
        nwt_coeffs_bary = operator_l2n_bary @ lag_coeffs_ref
        nwt_poly_bary = NewtonPolynomial(mi, nwt_coeffs_bary)

        # Evaluation at the unisolvent nodes yields the Lagrange coefficients
        unisolvent_nodes = lag_poly.grid.unisolvent_nodes
        lag_coeffs_naive = nwt_poly_naive(unisolvent_nodes)
        lag_coeffs_bary = nwt_poly_bary(unisolvent_nodes)

        # Assertions
        assert np.allclose(nwt_coeffs_naive, nwt_coeffs_bary)
        assert np.allclose(lag_coeffs_ref, lag_coeffs_naive)
        assert np.allclose(lag_coeffs_ref, lag_coeffs_bary)
        assert np.allclose(lag_coeffs_bary, lag_coeffs_naive)

    def test_to_newton_via_dds(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test the transformation via DDS."""
        # Create a Lagrange polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        lag_coeffs_ref = build_rnd_coeffs(mi)
        lag_poly = LagrangePolynomial(mi, lag_coeffs_ref)

        # Construct the transformer object
        transformer_l2n = LagrangeToNewton(lag_poly)

        # Naive transformation
        operator_l2n_naive = _build_lagrange_to_newton_naive(transformer_l2n)

        # Barycentric transformation
        operator_l2n_bary = _build_lagrange_to_newton_bary(transformer_l2n)

        # DDS transformation of the interpolation grid
        operator_l2n_dds = build_l2n_matrix_dds(lag_poly.grid)

        # Assertions
        assert np.allclose(
            operator_l2n_dds,
            operator_l2n_naive.array_repr_full,
        )
        assert np.allclose(
            operator_l2n_dds,
            operator_l2n_bary.array_repr_full,
        )
