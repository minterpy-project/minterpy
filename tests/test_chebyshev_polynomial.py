"""
testing module for chebyshev_polynomial.py

The subclassing is not tested here, see tesing module `test_polynomial.py`
"""

import numpy as np
import pytest
from conftest import (
    SEED,
    LpDegree,
    MultiIndices,
    NrPoints,
    NrSimilarPolynomials,
    PolyDegree,
    SpatialDimension,
    assert_polynomial_almost_equal,
    build_rnd_coeffs,
    build_rnd_points,
    build_random_newton_polynom,
)
from numpy.testing import assert_, assert_almost_equal

from minterpy import LagrangePolynomial, LagrangeToNewton, LagrangeToChebyshev

def test_eval(MultiIndices, NrPoints):
    # Create random polynomials in Lagrange base
    coeffs = build_rnd_coeffs(MultiIndices)
    lag_poly = LagrangePolynomial(MultiIndices, coeffs)
    pts = build_rnd_points(NrPoints, MultiIndices.spatial_dimension)

    # transform to Newton and Chebyshev basis to compare evaluations

    trafo_l2n = LagrangeToNewton(lag_poly)
    newt_poly = trafo_l2n()
    groundtruth = newt_poly(pts)

    trafo_l2cheb = LagrangeToChebyshev(lag_poly)
    cheb_poly = trafo_l2cheb()
    res = cheb_poly(pts)

    assert_almost_equal(res, groundtruth)
