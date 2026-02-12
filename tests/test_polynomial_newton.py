"""
testing module for canonical_polynomial.py

The subclassing is not tested here, see tesing module `test_polynomial.py`
"""
import pytest

from conftest import build_rnd_coeffs, build_rnd_points
from numpy.testing import assert_almost_equal

from minterpy.utils.polynomials.newton import eval_newton_polynomials
from minterpy import Grid, NewtonPolynomial, NewtonToCanonical


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
