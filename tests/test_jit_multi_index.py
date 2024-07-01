"""
Test suite for JIT-compiled utility functions related to the multi-index set.
"""
import numpy as np

from itertools import product
from minterpy.utils.multi_index import get_exponent_matrix, lex_sort

from minterpy.jit_compiled.multi_index import cross_and_sum, unique_indices


def test_cross_and_sum(SpatialDimension, PolyDegree, LpDegree):
    """Test the multiplication of two multi-indices."""
    # Create two multi-index sets
    mi_t = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    mi_2 = get_exponent_matrix(SpatialDimension+1, PolyDegree+1, LpDegree)
    # Make the set the same dimension
    mi_1 = np.zeros((mi_t.shape[0], mi_t.shape[1]+1), dtype=np.int_)
    mi_1[:, :SpatialDimension] = mi_t[:]

    # Create a reference by alternative method (they are not commutative)
    ref_1 = list(product(mi_1, mi_2))
    ref_1 = np.array([np.sum(np.array(i), axis=0) for i in ref_1])
    ref_2 = list(product(mi_2, mi_1))
    ref_2 = np.array([np.sum(np.array(i), axis=0) for i in ref_2])

    # Compute the cross and sum
    cross_sum_1 = cross_and_sum(mi_1, mi_2)
    cross_sum_2 = cross_and_sum(mi_2, mi_1)

    # Assertion
    assert np.allclose(cross_sum_1, ref_1)
    assert np.allclose(cross_sum_2, ref_2)


def test_unique_indices(SpatialDimension, PolyDegree, LpDegree):
    """Test the function to get unique indices."""
    # Create a multi-index set with non-unique elements
    mi = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    idx = np.random.choice(len(mi), len(mi), replace=True)
    mi = mi[idx]

    # Lex sort the array as reference
    mi_ref = lex_sort(mi)

    # Lex sort then compute the unique indices
    mi_sorted = mi[np.lexsort(mi.T)]
    unique_idx = unique_indices(mi_sorted)

    # Assertion
    assert np.all(mi_ref == mi_sorted[unique_idx])
