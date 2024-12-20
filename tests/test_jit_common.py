import numpy as np
import pytest

from itertools import combinations
from math import comb

from minterpy.jit_compiled.common import (
    n_choose_r,
    combinations_iter,
    get_max_columnwise,
)


@pytest.fixture(params=np.arange(15))
def num_total(request):
    return request.param


@pytest.fixture(params=np.arange(15))
def num_take(request):
    return request.param


class TestCombIter:
    """All tests related to the iterative n-choose-r."""

    def test_vs_comb(self, num_total, num_take):
        """Compare with comb() from the math module."""
        assert comb(num_total, num_take) == n_choose_r(num_total, num_take)


class TestCombinationsIter:
    """All tests related to the iterative getting all combinations."""

    def test_empty_take(self, num_total):
        """Test r=0 (take zero element from whatever)."""
        elements = np.arange(num_total)

        combs = combinations_iter(elements, r=0)

        # Assertions
        assert len(combs) == 0
        assert combs.size == 0

    def test_empty_from(self, num_take):
        """Test n=0 (take whatever from a zero-length array)."""
        elements = np.arange(0)

        combs = combinations_iter(elements, r=num_take)

        # Assertions
        assert len(combs) == 0
        assert combs.size == 0

    def test_vs_combinations(self, num_total, num_take):
        """Compare with combinations() from the itertools module."""
        elements = np.arange(num_total)
        combs_ref = combinations(elements, num_take)  # returns a generator
        combs_iter = combinations_iter(elements, num_take)  # returns an array

        # Assertion
        combs_ref = np.array(list(combs_ref), dtype=np.int_)
        if num_take == 0:
            # itertools.combinations returns an empty array of shape (1, 0)
            combs_ref = combs_ref.reshape((0, 0))
        if num_total < num_take:
            # itertools.combinations returns an empty array of shape (0, )
            combs_ref = combs_ref.reshape((0, num_take))

        assert np.array_equal(combs_ref, combs_iter)


def test_get_max_columnwise():
    """Test getting the column-wise max of a two-dimensional integer array."""
    num_rows = np.random.randint(low=100, high=1000)
    num_cols = np.random.randint(low=1, high=10)
    xx = np.random.randint(low=0, high=100, size=(num_rows, num_cols))

    # Maximum by NumPy
    max_ref = np.max(xx, axis=0)

    # Maximum by iteration
    max_iter = get_max_columnwise(xx)

    # Assertion
    assert np.array_equal(max_ref, max_iter)
