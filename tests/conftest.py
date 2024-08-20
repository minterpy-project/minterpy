"""
This is the conftest module of minterpy.

Within a pytest run, this module is loaded first. That means here all global fixutes shall be defined.
"""
import inspect
import itertools
import numpy as np
import pytest

from numpy.testing import assert_, assert_almost_equal, assert_equal

from minterpy import (
    MultiIndexSet,
    Grid,
    LagrangePolynomial,
    NewtonPolynomial,
    CanonicalPolynomial,
    ChebyshevPolynomial,
    LagrangeToNewton,
    LagrangeToCanonical,
    LagrangeToChebyshev,
    NewtonToLagrange,
    NewtonToCanonical,
    NewtonToChebyshev,
    CanonicalToLagrange,
    CanonicalToNewton,
    CanonicalToChebyshev,
    ChebyshevToLagrange,
    ChebyshevToNewton,
    ChebyshevToCanonical,
)
from minterpy.core.ABC.transformation_abstract import TransformationABC
from minterpy.utils.multi_index import get_exponent_matrix

# Global seed
SEED = 12345678

# Global settings
MIN_POLY_DEG = 0
MAX_POLY_DEG = 25

# --- Parameters to be tested
# Supported polynomial classes
POLY_CLASSES_NO_LAG = [
    NewtonPolynomial,
    CanonicalPolynomial,
    ChebyshevPolynomial,
]

# All
POLY_CLASSES = [LagrangePolynomial] + POLY_CLASSES_NO_LAG

# Supported polynomial transformation classes
TRANSFORMATION_CLASSES = [
    LagrangeToNewton,
    LagrangeToCanonical,
    LagrangeToChebyshev,
    NewtonToLagrange,
    NewtonToCanonical,
    NewtonToChebyshev,
    CanonicalToLagrange,
    CanonicalToNewton,
    CanonicalToChebyshev,
    ChebyshevToLagrange,
    ChebyshevToNewton,
    ChebyshevToCanonical,
]

# Primary parameters to create complete multi-index sets, grids, & polynomials
SPATIAL_DIMENSIONS = [1, 3]
POLY_DEGREES = [0, 1, 4]  # NOTE: Include test for poly_degree 0 (Issue #27)
LP_DEGREES = [0.5, 1.0, 2.0, np.inf]

# Number of coefficient sets in a single polynomial instance
NUM_POLYS = [1, 2, 5]


# asserts that a call runs as expected
def assert_call(fct, *args, **kwargs):
    try:
        fct(*args, **kwargs)
    except Exception as e:
        print(type(e))
        raise AssertionError(
            f"The function was not called properly. It raised the exception:\n\n {e.__class__.__name__}: {e}"
        )


# assert if multi_indices are equal
def assert_multi_index_equal(mi1, mi2):
    try:
        assert_(isinstance(mi1, type(mi2)))
        assert_equal(mi1.exponents, mi2.exponents)
        assert_equal(mi1.lp_degree, mi2.lp_degree)
        assert_equal(mi1.poly_degree, mi2.poly_degree)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of MultiIndexSet are not equal:\n\n {a}"
        )


# assert if multi_indices are almost equal
def assert_multi_index_almost_equal(mi1, mi2):
    try:
        assert_(isinstance(mi1, type(mi2)))
        assert_almost_equal(mi1.exponents, mi2.exponents)
        assert_almost_equal(mi1.lp_degree, mi2.lp_degree)
        assert_almost_equal(mi1.poly_degree, mi2.poly_degree)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of MultiIndexSet are not almost equal:\n\n {a}"
        )


# assert if two grids are equal
def assert_grid_equal(grid1, grid2):
    try:
        assert_(isinstance(grid1, type(grid2)))
        assert_equal(grid1.unisolvent_nodes, grid2.unisolvent_nodes)
        assert_equal(grid1.spatial_dimension, grid2.spatial_dimension)
        assert_equal(grid1.generating_points, grid2.generating_points)
        assert_multi_index_equal(grid1.multi_index, grid2.multi_index)
    except AssertionError as a:
        raise AssertionError(f"The two instances of Grid are not equal:\n\n {a}")


# assert if two grids are almost equal
def assert_grid_almost_equal(grid1, grid2):
    try:
        assert_(isinstance(grid1, type(grid2)))
        assert_almost_equal(grid1.unisolvent_nodes, grid2.unisolvent_nodes)
        assert_almost_equal(grid1.spatial_dimension, grid2.spatial_dimension)
        assert_almost_equal(grid1.generating_points, grid2.generating_points)
        assert_multi_index_almost_equal(grid1.multi_index, grid2.multi_index)
    except AssertionError as a:
        raise AssertionError(f"The two instances of Grid are not almost equal:\n\n {a}")


# assert if polynomials are almost equal
def assert_polynomial_equal(P1, P2):
    try:
        assert_(isinstance(P1, type(P2)))
        assert_multi_index_equal(P1.multi_index, P2.multi_index)
        assert_equal(P1.coeffs, P2.coeffs)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {P1.__class__.__name__} are not equal:\n\n {a}"
        )


# assert if polynomials are almost equal
def assert_polynomial_almost_equal(P1, P2):
    try:
        assert_(isinstance(P1, type(P2)))
        assert_multi_index_almost_equal(P1.multi_index, P2.multi_index)
        assert_almost_equal(P1.coeffs, P2.coeffs)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {P1.__class__.__name__} are not almost equal:\n\n {a}"
        )


# assert if two functions have the same object code
def assert_function_object_code_equal(fct1, fct2):
    try:
        assert_(fct1.__code__.co_code == fct2.__code__.co_code)
    except AssertionError as a:
        raise AssertionError(
            f"The object_code of {fct1} is not equal to the object_code of {fct2}:\n\n {inspect.getsource(fct1)}\n\n{inspect.getsource(fct2)}"
        )


# assert if interpolators are equal
def assert_interpolator_equal(interpolator1, interpolator2):
    try:
        assert_(isinstance(interpolator1, type(interpolator2)))
        assert_equal(interpolator1.spatial_dimension, interpolator2.spatial_dimension)
        assert_equal(interpolator1.poly_degree, interpolator2.poly_degree)
        assert_equal(interpolator1.lp_degree, interpolator2.lp_degree)
        assert_multi_index_equal(interpolator1.multi_index, interpolator2.multi_index)
        assert_grid_equal(interpolator1.grid, interpolator2.grid)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {interpolator1.__class__.__name__} are not equal:\n\n {a}"
        )


# assert if interpolators are almost equal
def assert_interpolator_almost_equal(interpolator1, interpolator2):
    try:
        assert_(isinstance(interpolator1, type(interpolator2)))
        assert_almost_equal(
            interpolator1.spatial_dimension, interpolator2.spatial_dimension
        )
        assert_almost_equal(interpolator1.poly_degree, interpolator2.poly_degree)
        assert_almost_equal(interpolator1.lp_degree, interpolator2.lp_degree)
        assert_multi_index_almost_equal(
            interpolator1.multi_index, interpolator2.multi_index
        )
        assert_grid_almost_equal(interpolator1.grid, interpolator2.grid)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {interpolator1.__class__.__name__} are not almost equal:\n\n {a}"
        )


# assert if interpolants are equal
def assert_interpolant_equal(interpolant1, interpolant2):
    try:
        assert_(isinstance(interpolant1, type(interpolant2)))
        assert_function_object_code_equal(interpolant1.fct, interpolant2.fct)
        assert_interpolator_equal(interpolant1.interpolator, interpolant2.interpolator)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {interpolant1.__class__.__name__} are not equal:\n\n {a}"
        )


# assert if interpolants are almost equal
def assert_interpolant_almost_equal(interpolant1, interpolant2):
    try:
        assert_(isinstance(interpolant1, type(interpolant2)))
        assert_function_object_code_equal(interpolant1.fct, interpolant2.fct)
        assert_interpolator_almost_equal(
            interpolant1.interpolator, interpolant2.interpolator
        )
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {interpolant1.__class__.__name__} are not almost equal:\n\n {a}"
        )


# --- Elementary fixtures
# Spatial dimension (`m`)
def _id_m(spatial_dimension):
    return f"m={spatial_dimension}"


@pytest.fixture(params=SPATIAL_DIMENSIONS, ids=_id_m)
def SpatialDimension(request):
    """Return spatial dimension (`m`) fixture."""
    return request.param


# Polynomial degree (`n`)
def _id_n(poly_degree):
    return f"n={poly_degree}"


@pytest.fixture(params=POLY_DEGREES, ids=_id_n)
def PolyDegree(request):
    """Return polynomial degree (`n`) fixture."""
    return request.param


# Lp-degree (`p`)
def _id_p(lp_degree):
    return f"p={lp_degree:<3}"


@pytest.fixture(params=LP_DEGREES, ids=_id_p)
def LpDegree(request):
    """Return lp-degree fixture."""
    return request.param


# Polynomial classes
def _id_poly_class(poly_class):
    return f"{poly_class.__name__:>19}"


@pytest.fixture(params=POLY_CLASSES, ids=_id_poly_class)
def poly_class_all(request):
    """Fixture for the supported concrete polynomial classes (all)."""
    return request.param


@pytest.fixture(params=POLY_CLASSES_NO_LAG, ids=_id_poly_class)
def poly_class_no_lag(request):
    """Fixture for the supported concrete polynomial classes (w/o Lagrange)."""
    return request.param


# Number of coefficient sets in a single polynomial instance
def _id_num_polys(num_polys):
    return f"num_polys={num_polys}"


@pytest.fixture(params=NUM_POLYS, ids=_id_num_polys)
def num_polynomials(request):
    """Fixture for the number of polynomials."""
    return request.param

# fixtures for number of similar polynomials

nr_similar_polynomials = [None, 1, 2]


@pytest.fixture(params=nr_similar_polynomials)
def NrSimilarPolynomials(request):
    return request.param


# fixture for the number of points for evaluations

nr_pts = [1, 2]


@pytest.fixture(params=nr_pts)
def NrPoints(request):
    return request.param





# Fixture for the number
batch_sizes = [1, 100, 1000]


@pytest.fixture(params=batch_sizes)
def BatchSizes(request):
    return request.param





@pytest.fixture(params=TRANSFORMATION_CLASSES)
def transformation_class(request) -> TransformationABC:
    """Fixture for the supported polynomial transformation classes."""
    return request.param


origin_type = poly_class_all
target_type = poly_class_all

# Fixture for pair
@pytest.fixture(
    params=[
        "equal",
        "dimensions",
        "poly-degrees",
        "lp-degrees",
        "empty",
        "all",
    ]
)
def param_diff(request):
    return request.param


# Fixture for polynomial domain selection
@pytest.fixture(params=["user", "internal"])
def poly_domain(request):
    return request.param


# --- Pair of elementary fixtures
def _pair_m_id(pair_m):
    return f"m=({pair_m[0]}, {pair_m[1]})"


@pytest.fixture(
    params=list(
        itertools.combinations_with_replacement(SPATIAL_DIMENSIONS, r=2)
    ),
    ids=_pair_m_id,
)
def spatial_dim_pair(request):
    """Return a pair of spatial dimension values."""
    return request.param


def _pair_n_id(pair_n):
    return f"n=({pair_n[0]}, {pair_n[1]})"


@pytest.fixture(
    params=list(
        itertools.combinations_with_replacement(POLY_DEGREES, r=2)
    ),
    ids=_pair_n_id,
)
def poly_degree_pair(request):
    """Return a pair of polynomial degree values."""
    return request.param


def _pair_p_id(pair_p):
    return f"p=({pair_p[0]:>3}, {pair_p[1]:>3})"


@pytest.fixture(
    params=list(
        itertools.combinations_with_replacement(LP_DEGREES, r=2)
    ),
    ids=_pair_p_id,
)
def lp_degree_pair(request):
    """Return a pair of lp-degree values."""
    return request.param


# --- Composite fixtures
@pytest.fixture
def multi_index_mnp(SpatialDimension, PolyDegree, LpDegree):
    """Create a complete multi-index set given spatial dimension (``m``),
    polynomial degree (``n``), and lp-degree (``p``).
    """
    return MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)


@pytest.fixture
def rand_poly_mnp_all(poly_class_all, multi_index_mnp, num_polynomials):
    """Create a random polynomial instance of each concrete class having
    a complete multi-index set.
    """
    # Generate random coefficients
    coefficients = np.random.rand(len(multi_index_mnp), num_polynomials)

    return poly_class_all(multi_index_mnp, coefficients)


@pytest.fixture
def rand_poly_mnp_no_lag(poly_class_no_lag, multi_index_mnp, num_polynomials):
    """Create a random polynomial instance of each concrete class (but not
    LagrangePolynomial) having a complete multi-index set.
    """
    # Generate random coefficients
    coefficients = np.random.rand(len(multi_index_mnp), num_polynomials)

    return poly_class_no_lag(multi_index_mnp, coefficients)


@pytest.fixture
def rand_poly_mnp_lag(multi_index_mnp, num_polynomials):
    """Create a random Lagrange polynomial instance having a complete
    multi-index set.
    """
    # Generate random coefficients
    coefficients = np.random.rand(len(multi_index_mnp), num_polynomials)

    return LagrangePolynomial(multi_index_mnp, coefficients)


@pytest.fixture
def rand_polys_mnp(poly_class_all, multi_index_mnp, num_polynomials):
    """Create a random polynomial instance of each concrete class having
    a complete multi-index set with possibly multiple sets of coefficients.
    """
    # Generate random coefficients
    coefficients = np.random.rand(len(multi_index_mnp), num_polynomials)

    return poly_class_all(multi_index_mnp, coefficients)


@pytest.fixture
def poly_mnp_uninit(poly_class_all, multi_index_mnp):
    """Create an uninitialized polynomial instance of each concrete class
    having a complete multi-index set.
    """
    return poly_class_all(multi_index_mnp)


@pytest.fixture
def poly_mnp_non_unif_domain(poly_class_all, multi_index_mnp, poly_domain):
    """Create an uninitialized polynomial instance of each concrete class
    having a complete multi-index set with non-uniform polynomial domain.
    """
    if multi_index_mnp.spatial_dimension == 1:
        pytest.skip(
            "Domain with spatial dimension 1 can always be extrapolated"
        )

    # Define the domain (non-uniform: cannot be extrapolated)
    domain = np.ones((2, multi_index_mnp.spatial_dimension))
    domain[0, :] = -1
    domain[0, 0] = -2
    domain[1, 0] = 2

    if poly_domain == "user":
        return poly_class_all(multi_index_mnp, user_domain=domain)
    elif poly_domain == "internal":
        return poly_class_all(multi_index_mnp, internal_domain=domain)
    else:
        raise ValueError


# --- Pair of composite fixtures
@pytest.fixture
def multi_index_mnp_pair(spatial_dim_pair, poly_degree_pair, lp_degree_pair):
    """Create a pair of complete multi-index sets with different combinations
    of spatial dimension, polynomial degree, and lp-degree.
    """
    # Get the pairs
    m_1, m_2 = spatial_dim_pair
    n_1, n_2 = poly_degree_pair
    p_1, p_2 = lp_degree_pair

    # Create a pair of complete multi-index sets
    mi_1 = MultiIndexSet.from_degree(m_1, n_1, p_1)
    mi_2 = MultiIndexSet.from_degree(m_2, n_2, p_2)

    return mi_1, mi_2


@pytest.fixture
def rand_poly_mnp_all_pair(
    poly_class_all,
    multi_index_mnp_pair,
    num_polynomials,
):
    """Create a random polynomial instances of each concrete class except
    Lagrange with a complete multi-index set.
    """
    # Get the multi-index sets
    mi_1, mi_2 = multi_index_mnp_pair

    # Generate random coefficients
    coeffs_1 = np.random.rand(len(mi_1), num_polynomials)
    coeffs_2 = np.random.rand(len(mi_2), num_polynomials)

    # Create a pair of polynomial instances
    poly_1 = poly_class_all(mi_1, coeffs_1)
    poly_2 = poly_class_all(mi_2, coeffs_2)

    return poly_1, poly_2


@pytest.fixture
def rand_poly_mnp_no_lag_pair(
    poly_class_no_lag,
    multi_index_mnp_pair,
    num_polynomials,
):
    """Create a random polynomial instances of each concrete class except
    Lagrange with a complete multi-index set.
    """
    # Get the multi-index sets
    mi_1, mi_2 = multi_index_mnp_pair

    # Generate random coefficients
    coeffs_1 = np.random.rand(len(mi_1), num_polynomials)
    coeffs_2 = np.random.rand(len(mi_2), num_polynomials)

    # Create a pair of polynomial instances
    poly_1 = poly_class_no_lag(mi_1, coeffs_1)
    poly_2 = poly_class_no_lag(mi_2, coeffs_2)

    return poly_1, poly_2


@pytest.fixture
def poly_mnp_pair_diff_dim(
    poly_class_all,
    SpatialDimension,
    PolyDegree,
    LpDegree,
):
    """Create a pair of uninitialized polynomial instances of each concrete
    class having a complete multi-index set but with different dimension.
    """
    # Create polynomial instances
    poly_1 = poly_class_all.from_degree(
        SpatialDimension,
        PolyDegree,
        LpDegree,
    )
    poly_2 = poly_class_all.from_degree(
        SpatialDimension + 1,
        PolyDegree,
        LpDegree,
    )

    return poly_1, poly_2


@pytest.fixture
def poly_mnp_pair_diff_domain(poly_class_all, multi_index_mnp, poly_domain):
    """Create a pair of un-initialized polynomial instances of each concrete
    class having a complete multi-index set but with a different domain.
    """
    # Define a non-default domain
    domain = np.ones((2, multi_index_mnp.spatial_dimension))
    domain[0, 0] = -2
    domain[1, 0] = 2

    # Create polynomial instances
    poly_1 = poly_class_all(multi_index_mnp)
    if poly_domain == "user":
        poly_2 = poly_class_all(multi_index_mnp, user_domain=domain)
    elif poly_domain == "internal":
        poly_2 = poly_class_all(multi_index_mnp, internal_domain=domain)
    else:
        raise ValueError

    return poly_1, poly_2


@pytest.fixture
def rand_poly_mnp_pair(poly_class_all, mi_pair):
    """Create a pair of randomly initialized polynomial instances of each
    concrete class having a complete multi-index sets.
    """
    # Get the multi-index sets
    mi_1, mi_2 = mi_pair
    if len(mi_1) == 0 or len(mi_2) == 0:
        pytest.skip("Polynomial can't have empty multi-index set.")

    # Create polynomial instances
    coeffs_1 = np.random.rand(len(mi_1))
    poly_1 = poly_class_all(mi_1, coeffs_1)

    coeffs_2 = np.random.rand(len(mi_2))
    poly_2 = poly_class_all(mi_2, coeffs_2)

    return poly_1, poly_2


@pytest.fixture
def rand_polys_mnp_pair(poly_class_all, mi_pair, num_polynomials):
    """Create a pair of randomly initialized polynomial instances of each
    concrete class having a complete multi-index sets and multiple sets of
    coefficients.
    """
    # Get the multi-index sets
    mi_1, mi_2 = mi_pair
    if len(mi_1) == 0 or len(mi_2) == 0:
        pytest.skip("Polynomial can't have empty multi-index set.")

    # Create polynomial instances
    coeffs_1 = np.random.rand(len(mi_1), num_polynomials)
    poly_1 = poly_class_all(mi_1, coeffs_1)

    coeffs_2 = np.random.rand(len(mi_2), num_polynomials)
    poly_2 = poly_class_all(mi_2, coeffs_2)

    return poly_1, poly_2


@pytest.fixture
def mi_pair(SpatialDimension, PolyDegree, LpDegree, param_diff):
    """Create a pair of MultiIndexSets with different parameters."""
    if param_diff == "equal":
        # A pair with equal parameter values
        m = SpatialDimension
        n = PolyDegree
        p = LpDegree
        mi_1 = MultiIndexSet.from_degree(m, n, p)
        mi_2 = MultiIndexSet.from_degree(m, n, p)
    elif param_diff == "dimensions":
        # A pair with different spatial dimensions.
        m_1 = SpatialDimension
        m_2 = SpatialDimension + 1
        n = PolyDegree
        p = LpDegree
        mi_1 = MultiIndexSet.from_degree(m_1, n, p)
        mi_2 = MultiIndexSet.from_degree(m_2, n, p)
    elif param_diff == "poly-degrees":
        # A pair with different polynomial degrees.
        m = SpatialDimension
        n_1 = PolyDegree
        n_2 = PolyDegree + np.random.randint(low=1, high=3)
        p = LpDegree
        mi_1 = MultiIndexSet.from_degree(m, n_1, p)
        mi_2 = MultiIndexSet.from_degree(m, n_2, p)
    elif param_diff == "lp-degrees":
        # A pair with different lp-degrees.
        m = SpatialDimension
        d = PolyDegree
        lp_degrees = [0.5, 1.0, 2.0, 3.0, np.inf]
        p_1, p_2 = np.random.choice(lp_degrees, size=2, replace=False)
        mi_1 = MultiIndexSet.from_degree(m, d, p_1)
        mi_2 = MultiIndexSet.from_degree(m, d, p_2)
    elif param_diff == "all":
        # A pair with all three parameters differ
        m_1 = np.random.randint(low=1, high=5)
        m_2 = np.random.randint(low=1, high=5)
        d_1 = np.random.randint(low=1, high=5)
        d_2 = np.random.randint(low=1, high=5)
        lp_degrees = [0.5, 1.0, 2.0, 3.0, np.inf]
        p_1, p_2 = np.random.choice(lp_degrees, 2)
        mi_1 = MultiIndexSet.from_degree(m_1, d_1, p_1)
        mi_2 = MultiIndexSet.from_degree(m_2, d_2, p_2)
        # mi_1, mi_2 = create_mi_pair_distinct()
    elif param_diff == "empty":
        # A pair with one of them is empty
        m_1 = np.random.randint(low=1, high=5)
        m_2 = np.random.randint(low=1, high=5)
        d_1 = np.random.randint(low=1, high=5)
        lp_degrees = [0.5, 1.0, 2.0, 3.0, np.inf]
        p_1, p_2 = np.random.choice(lp_degrees, 2)
        mi_1 = MultiIndexSet.from_degree(m_1, d_1, p_1)
        mi_2 = MultiIndexSet(np.empty((0, m_2)), p_2)
    else:
        return ValueError(f"'param-diff' = {param_diff} is not recognized!")

    return mi_1, mi_2


def create_mi_pair_distinct():
    """Create a pair of distinct multi-index sets."""
    # A pair with all three parameters differ
    m_1 = np.random.randint(low=1, high=3)
    m_2 = 2 * m_1
    n_1 = np.random.randint(low=1, high=5)
    n_2 = np.random.randint(low=1, high=5)
    lp_degrees = [0.5, 1.0, 2.0, 3.0, np.inf]
    p_1, p_2 = np.random.choice(lp_degrees, 2)
    mi_1 = MultiIndexSet.from_degree(m_1, n_1, p_1)
    mi_2 = MultiIndexSet.from_degree(m_2, n_2, p_2)

    return mi_1, mi_2

# some random builder


def build_rnd_exponents(dim, n, seed=None):
    """Build random exponents.

    For later use, if ``MultiIndexSet`` will accept arbitrary exponents again.

    :param dim: spatial dimension
    :param n: number of random monomials

    Notes
    -----
    Exponents are generated within the intervall ``[MIN_POLY_DEG,MAX_POLY_DEG]``

    """
    rng = np.random.default_rng(seed)

    exponents = rng.integers(MIN_POLY_DEG, MAX_POLY_DEG, (n, dim), dtype=int)

    return exponents


def build_rnd_coeffs(mi, nr_poly=None, seed=None):
    """Build random coefficients.

    For later use.

    :param mi: The :class:`MultiIndexSet` instance of the respective polynomial.
    :type mi: MultiIndexSet

    :param nr_poly: Number of similar polynomials. Default is 1
    :type nr_poly: int, optional

    :return: Random array of shape ``(nr of monomials,nr of similar polys)`` usable as coefficients.
    :rtype: np.ndarray

    """
    if nr_poly is None:
        additional_coeff_shape = tuple()
    else:
        additional_coeff_shape = (nr_poly,)
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    return np.random.random((len(mi),) + additional_coeff_shape)


def build_rnd_points(nr_points, spatial_dimension, nr_poly=None, seed=None):
    """Build random points in space.

    Return a batch of `nr_points` vectors in the real vector space of dimension ``spatial_dimension``.

    :param nr_points: Number of points
    :type nr_points: int

    :param spatial_dimension: Dimension of domain space.
    :type spatial_dimension: int

    :param nr_poly: Number of similar polynomials. Default is 1
    :type nr_poly: int, optional

    :param seed: Seed used for the random generation. Default is SEED.
    :type seed: int

    :return: Array of shape ``(nr_points,spatial_dimension[,nr_poly])`` containig random real values (distributed in the intervall :math:`[-1,1]`).
    :rtype: np.ndarray

    """
    if nr_poly is None:
        additional_coeff_shape = tuple()
    else:
        additional_coeff_shape = (nr_poly,)
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    return np.random.uniform(
        -1, 1, size=(nr_points, spatial_dimension) + additional_coeff_shape
    )


def build_random_newton_polynom(
    dim: int, deg: int, lp: int,  n_poly=1, seed=None
) -> NewtonPolynomial:
    """Build a random Newton polynomial.

    Return a :class:`NewtonPolynomial` with a lexicographically complete :class:`MultiIndex` (initiated from ``(dim,deg,lp)``) and randomly generated coefficients (uniformly distributes in the intervall :math:`[-1,1]`).

    :param dim: dimension of the domain space.
    :type dim: int
    :param deg: degree of the interpolation polynomials
    :type deg: int
    :param lp: degree of the :math:`l_p` norm used to determine the `poly_degree`.
    :type lp: int

    :return: Newton polynomial with random coefficients.
    :rtype: NewtonPolynomial

    """
    mi = MultiIndexSet.from_degree(dim, deg, lp)
    if seed is None:
        seed = SEED

    np.random.seed(seed)

    if n_poly == 1:
        rnd_coeffs = np.random.uniform(-1, 1, size=len(mi))
    else:
        rnd_coeffs = np.random.uniform(-1, 1, size=(len(mi), n_poly))

    return NewtonPolynomial(mi, rnd_coeffs)


def build_random_multi_index():
    """Build random complete multi-index set."""
    m = np.random.randint(1, 5)
    n = np.random.randint(1, 5)
    p = np.random.choice([1.0, 2.0, np.inf])

    mi = MultiIndexSet.from_degree(m, n, p)

    return mi


@pytest.fixture
def multi_index_non_downward_closed(SpatialDimension, PolyDegree, LpDegree):
    """Create a non-downward-closed multi-index set."""
    if PolyDegree == 0:
        # NOTE: Skip the test as degree 0 contains only one element
        pytest.skip("Poly. degree 0 cannot be made non-downward closed")
    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    if PolyDegree > 0:
        # NOTE: Only applies for poly_degree > 0 (== 0 has only 1 element)
        # Without the lexicographically smallest element
        exponents = np.delete(exponents, 0, axis=0)
    if PolyDegree > 1:
        # NOTE: Only applies for poly_degree > 1 (== 1 has at least 2 elements)
        # Without the 2nd lexicographically smallest element
        exponents = np.delete(exponents, 1, axis=0)

    mi = MultiIndexSet(exponents, LpDegree)

    return mi


@pytest.fixture
def multi_index_incomplete(SpatialDimension, PolyDegree, LpDegree):
    """Create an incomplete multi-index set."""
    if PolyDegree == 0:
        # NOTE: Skip the test as degree 0 contains only one element
        pytest.skip("Poly. degree 0 cannot be made incomplete")

    exponents = get_exponent_matrix(SpatialDimension, PolyDegree, LpDegree)
    if PolyDegree > 0:
        # Taking the first element make it not downward-closed
        exponents = np.delete(exponents, 0, axis=0)
    if PolyDegree > 1:
        # Taking out the largest element make the set incomplete
        exponents = np.delete(exponents, -1, axis=0)

    return MultiIndexSet(exponents, LpDegree)


@pytest.fixture
def grid_mnp(SpatialDimension, PolyDegree, LpDegree):
    """Create a Grid with a complete multi-index set."""
    return Grid.from_degree(SpatialDimension, PolyDegree, LpDegree)