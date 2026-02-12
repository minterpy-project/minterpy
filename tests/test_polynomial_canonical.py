"""
testing module for canonical_polynomial.py

The subclassing is not tested here, see tesing module `test_polynomial.py`
"""
import operator as op

import numpy as np
import pytest
from conftest import (
    SEED,
    LpDegree,
    NrPoints,
    NrSimilarPolynomials,
    PolyDegree,
    SpatialDimension,
    assert_polynomial_almost_equal,
    build_rnd_coeffs,
    build_rnd_points,
)
from numpy.testing import assert_, assert_almost_equal

from minterpy import (
    CanonicalPolynomial,
    CanonicalToLagrange,
    CanonicalToNewton,
    Grid,
    MultiIndexSet,
)

# tests with a single polynomial


def test_neg(multi_index_mnp, NrSimilarPolynomials):
    coeffs = build_rnd_coeffs(multi_index_mnp, NrSimilarPolynomials)
    poly = CanonicalPolynomial(multi_index_mnp, coeffs)
    res = -poly
    groundtruth_coeffs = (-1) * coeffs
    groundtruth = poly.__class__(multi_index_mnp, groundtruth_coeffs)
    assert_polynomial_almost_equal(res, groundtruth)


def test_eval(multi_index_mnp, NrPoints):
    coeffs = build_rnd_coeffs(multi_index_mnp)
    poly = CanonicalPolynomial(multi_index_mnp, coeffs)
    pts = build_rnd_points(NrPoints, multi_index_mnp.spatial_dimension)
    res = poly(pts)

    # navie impementation of canonical eval
    # related to issue #32
    groundtruth = np.zeros(NrPoints)
    for k,pt in enumerate(pts):
        single_groundtruth = 0.0
        for i,exponents in enumerate(multi_index_mnp.exponents):
            term = 1.0
            for j, expo in enumerate(exponents):
                term *= pt[j]**expo
            single_groundtruth+=coeffs[i]*term
        groundtruth[k] = single_groundtruth

    assert_almost_equal(res, groundtruth)


# tests with two polynomials
# todo:: find out if there are some more sophisticated tests for that
exps1 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
args1 = np.lexsort(exps1.T, axis=-1)
mi1 = MultiIndexSet(exps1[args1], lp_degree=1.0)
coeffs1 = np.array([1., 2., 3., 4.])

exps2 = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0], [0, 2, 0], [0, 0, 2]])
args2 = np.lexsort(exps2.T, axis=-1)
mi2 = MultiIndexSet(exps2[args2], lp_degree=1.0)
coeffs2 = np.array([1., 2., 3., 4., 5.])

polys = [
    CanonicalPolynomial(mi1, coeffs1[args1]),
    CanonicalPolynomial(mi2, coeffs2[args2]),
]


@pytest.fixture(params=polys)
def Poly(request):
    return request.param


def test_sub_same_poly(Poly):
    res = Poly - Poly
    groundtruth_coeffs = np.zeros(Poly.coeffs.shape)
    groundtruth = Poly.__class__(Poly.multi_index, groundtruth_coeffs)
    assert_polynomial_almost_equal(res, groundtruth)


def test_sub_different_poly():
    """
    .. todo::
        - make this a bit better
    """
    res = polys[0] - polys[1]
    groundtruth_coeffs = np.array([0, 2, -2, 3, 1, -4, -5])
    groundtruth_multi_index_exponents = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 2],
        ]
    )
    groundtruth_multi_index = MultiIndexSet(groundtruth_multi_index_exponents, lp_degree=1.0)
    groundtruth = polys[0].__class__(groundtruth_multi_index, groundtruth_coeffs)
    assert_polynomial_almost_equal(res, groundtruth)

