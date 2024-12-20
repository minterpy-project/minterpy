"""
Testing module for the Chebyshev polynomial implementation.
"""

import numpy as np

from scipy.special import eval_chebyt
from conftest import SEED, build_rnd_coeffs

from minterpy import ChebyshevPolynomial, MultiIndexSet
from minterpy.utils.multi_index import get_exponent_matrix


class TestEvaluation:
    """All tests related to the evaluation of Chebyshev polynomials."""
    def test_downward_closed(self, multi_index_mnp):
        """Test evaluating a poly. having a downward-closed multi-index."""
        # Get the complete multi-index set
        mi = multi_index_mnp

        # Create a random polynomial in the Chebyshev bases
        cheb_coeffs = build_rnd_coeffs(mi)
        cheb_poly = ChebyshevPolynomial(mi, cheb_coeffs)

        # Create random test points
        xx_test = -1 + 2 * np.random.rand(100, mi.spatial_dimension)
        yy_poly = cheb_poly(xx_test)

        # Compute reference (the same internal implementation)
        exponents = mi.exponents
        yy_ref = eval_chebyt(exponents[None, :, :], xx_test[:, None, :])
        yy_ref = np.prod(yy_ref, axis=-1) @ cheb_coeffs

        # Assertion
        assert np.array_equal(yy_ref, yy_poly)

    def test_nondownward_closed(self):
        """Test evaluating a Chebyshev poly. having an arbitrary multi-index.
        """
        # Create a non-downward-closed multi-index set
        rng = np.random.default_rng(SEED)
        m = 3  # spatial dimension
        n = 10  # polynomial degree
        p = 2.0  # lp-degree
        exponents = get_exponent_matrix(m, n, p)
        idx = rng.choice(len(exponents), 10, replace=False)
        exponents = exponents[idx]
        mi = MultiIndexSet(exponents, p)

        # Create a Chebyshev polynomial
        cheb_coeffs = build_rnd_coeffs(mi)
        cheb_poly = ChebyshevPolynomial(mi, cheb_coeffs)

        # Create random test points
        xx_test = -1 + 2 * np.random.rand(100, m)
        yy_poly = cheb_poly(xx_test)

        # Compute reference (the same internal implementation)
        exponents = mi.exponents  # lexsorted exponents
        yy_ref = eval_chebyt(exponents[None, :, :], xx_test[:, None, :])
        yy_ref = np.prod(yy_ref, axis=-1) @ cheb_coeffs

        # Assertion
        assert np.array_equal(yy_ref, yy_poly)
