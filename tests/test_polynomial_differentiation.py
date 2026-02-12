"""
Test suite for polynomial differentiation functionality.

This module verifies differentiation operations for all differentiable
polynomial bases. Tests are focused around  "obvious" mathematical properties
rather than validation against external reference data:

- Linearity: Differentiation distributes over addition and scalar
  multiplication
- Product rule: Differentiation of products follows the Leibniz rule
- Basis invariance: Differentiation commutes with basis transformations
- Domain awareness: Automatic chain rule scaling for custom domains
- Theorems: Schwarz's theorem (commutativity of mixed partials), identity
  operation, successive differentiation

The tests employ random polynomials across various domains, spatial dimensions,
and order of derivatives.
"""
import numpy as np
import pytest

from numpy.testing import assert_almost_equal

from minterpy import (
    Domain,
    CanonicalPolynomial,
    NewtonPolynomial,
    get_transformation,
)
from minterpy.utils.multi_index import find_match_between

##################################
# Module variables and fixtures  #
##################################

# Transformation between bases
TARGET_POLYS = {
    NewtonPolynomial: CanonicalPolynomial,
    CanonicalPolynomial: NewtonPolynomial,
}

# --- Differentiable polynomial classes
DIFFERENTIABLE_POLYS = [NewtonPolynomial, CanonicalPolynomial]


def _id_poly_class(poly_class):
    return f"{poly_class.__name__:>19}"


@pytest.fixture(params=DIFFERENTIABLE_POLYS, ids=_id_poly_class)
def differentiable_poly(request):
    return request.param


# --- Domain types
DOMAIN_TYPES = ["default", "random"]


def _id_domain_type(domain_type):
    return f"dom={domain_type:>7}"


@pytest.fixture(params=DOMAIN_TYPES, ids=_id_domain_type)
def domain_type(request):
    return request.param


# --- Domain instance
@pytest.fixture
def domain(SpatialDimension, domain_type):
    # Create a default domain
    if domain_type == "default":
        return Domain.normalized(SpatialDimension)

    # Create the same polynomial with custom domain
    lb = np.random.uniform(0, 5, size=SpatialDimension)
    # Ensure ub > lb
    ub = lb + np.random.uniform(5, 10, size=SpatialDimension)

    return Domain(np.c_[lb, ub])


# --- Random integrable polynomial instance
@pytest.fixture
def rand_poly(
    differentiable_poly,
    multi_index_mnp,
    domain,
    num_polynomials,
):
    """Create a random differentiable polynomial as a fixture."""
    # Create random coefficients
    if num_polynomials > 1:
        coeffs = np.random.rand(len(multi_index_mnp), num_polynomials)
    else:
        coeffs = np.random.rand(len(multi_index_mnp))

    # Create an instance of polynomial
    poly = differentiable_poly(multi_index_mnp, coeffs, domain=domain)

    return poly


# --- Differentiation backends (for Newton polynomial)
DIFF_BACKENDS = ["numpy", "numba", "numba-par"]


# --- Order of derivatives
deriv_order_types = [
    "single_1st",  # [1, 0, 0, ...], [0, 1, 0, ...], etc.
    "single_2nd",  # [2, 0, 0, ...], [0, 2, 0, ...], etc.
    "mixed_1st",   # [1, 1, 0, ...], [0, 1, 0, 1, ...], etc.
    "mixed_2nd",   # [1, 0, 2, ...], [0, 2, 1, 0, ...], etc.
]

def _id_deriv_order_type(deriv_order_type):
    return f"deriv_order={deriv_order_type:>10}"


@pytest.fixture(params=deriv_order_types, ids=_id_deriv_order_type)
def deriv_order_type(request):
    """The order of derivatives type as a test fixture."""
    return request.param


@pytest.fixture()
def deriv_order(deriv_order_type, multi_index_mnp):
    """The order of derivatives as a test fixture.

    Notes
    -----
    - The available order derivative depends on the spatial dimension and
      the degree of the polynomial.
    """
    # Get the dimension and degree
    m = multi_index_mnp.spatial_dimension
    n = multi_index_mnp.poly_degree

    deriv_order = np.zeros(m, dtype=int)
    if deriv_order_type == "single_1st":
        # Differentiate once with respect to a random dimension
        idx = np.random.choice(m)
        deriv_order[idx] = 1
    elif deriv_order_type == "single_2nd":
        if n < 2:
            pytest.skip(f"{deriv_order_type} requires n >= 2")
        # Differentiate twice with respect to a random dimension
        idx = np.random.choice(m)
        deriv_order[idx] = 2
    elif deriv_order_type == "mixed_1st":
        if m < 2 or n < 2:
            pytest.skip(f"{deriv_order_type} requires m >= 2 and n >= 2")
        # Differentiate twice with respect to two random dimensions
        idxs = np.random.choice(m, 2, replace=False)
        deriv_order[idxs[0]] = 1
        deriv_order[idxs[1]] = 1
    elif deriv_order_type == "mixed_2nd":
        if m < 2 or n < 3:
            pytest.skip(f"{deriv_order_type} requires m >= 2 and n >= 3")
        # Differentiate thrice with respect to two random dimensions
        idxs = np.random.choice(m, 2, replace=False)
        deriv_order[idxs[0]] = 1
        deriv_order[idxs[1]] = 2

    return deriv_order

#######################
# Internal functions  #
#######################

def assert_polynomial_almost_equal(poly_1, poly_2):
    """Assert that two polynomials are almost equal in value."""
    try:
        assert isinstance(poly_1, type(poly_2))
        assert poly_1.multi_index == poly_2.multi_index
        assert poly_1.grid == poly_2.grid
        assert_almost_equal(poly_1.coeffs, poly_2.coeffs)
    except AssertionError as a:
        raise AssertionError(
            f"The two instances of {poly_1.__class__.__name__} "
            f"are not almost equal:\n\n {a}"
        )


def diff_can_coeffs(
    coeffs: np.ndarray,
    exponents: np.ndarray,
    deriv_order: np.ndarray,
    diff_factor: float,
):
    """Compute differentiated canonical polynomial coefficients via power rule.

    This is an independent implementation of canonical polynomial
    differentiation for verification purposes. Applies the power rule directly
    to each monomial term.

    Parameters
    ----------
    coeffs : np.ndarray
        The coefficients of the original polynomial.
    exponents : np.ndarray
        The multi-index exponents of the original polynomial as an array
        of shape ``(N, m)``, where ``N`` is the number of monomials and ``m``
        is the number of spatial dimensions.
    deriv_order : np.ndarray
        The order of derivatives as an array of shape ``(m,)``, where ``m`` is
        the number of spatial dimensions. Each element of the array specifies
        the order of differentiation along each spatial dimension.
    diff_factor : float
        The domain scaling factor.

    Returns
    -------
    np.ndarray
        The coefficients of the differentiated polynomial.
    """
    # Get the non-negative exponents after differentiation
    nn_exponents = exponents[~np.any(exponents - deriv_order < 0, axis=1)]

    # Find correspondence between original and differentiated polynomials
    diff_exponents = nn_exponents - deriv_order
    indices_target = find_match_between(diff_exponents, exponents)
    indices_origin = find_match_between(nn_exponents, exponents)

    # Compute the coefficients
    coeffs_ = np.zeros(coeffs.shape)
    for i, (idx_1, idx_2) in enumerate(zip(indices_target, indices_origin)):
        # Compute the factor due to monomial exponents being knocked down
        factor = 1.0
        for j, deriv_order_dim in enumerate(deriv_order):
            for k in range(deriv_order_dim):
                factor *= nn_exponents[i, j] - k
        coeffs_[idx_1] = factor * coeffs[idx_2]

    return diff_factor * coeffs_

##########
# Tests  #
##########

def test_domain_scaling_factor(rand_poly, deriv_order):
    """Test the scaling factor of a differentiable polynomial (forward).

    Internal representation of polynomial does not change with respect to the
    user domain. The result between differentiation of polynomials in the user
    domain and in the default internal domain only differs by a scaling factor
    (which depends on the order of derivatives).
    """
    poly_1 = rand_poly
    # Create another instance with the default internal domain
    poly_2 = rand_poly.__class__(
        poly_1.multi_index,
        poly_1.coeffs,
    )

    # Differentiate the polynomial
    diff_poly_1 = poly_1.diff(deriv_order)
    diff_poly_2 = poly_2.diff(deriv_order)

    diff_factor = poly_1.domain.diff_factor(deriv_order)

    # Assertions
    # NOTE: Domains are by construction not equal, instances are not "close"!
    assert isinstance(diff_poly_1, type(diff_poly_2))
    assert diff_poly_1.multi_index == diff_poly_2.multi_index
    # Forward: user domain -> factor * internal domain
    assert np.allclose(diff_poly_1.coeffs, diff_factor * diff_poly_2.coeffs)
    assert np.allclose(diff_factor * diff_poly_2.coeffs, diff_poly_1.coeffs)
    # Backward: internal domain -> user domain / factor
    assert np.allclose(diff_poly_1.coeffs / diff_factor, diff_poly_2.coeffs)
    assert np.allclose(diff_poly_2.coeffs, diff_poly_1.coeffs / diff_factor)


def test_linearity_mul_add(rand_poly, deriv_order):
    """Test linearity of differentiation (scalar multiplication and addition).

    diff(a * poly_1 + b * poly_2) = a * diff(poly_1) + b * diff(poly_2)
    """
    poly_1 = rand_poly
    # Create another instance with different coefficients
    if len(poly_1) > 1:
        coeffs_2 = np.random.rand(len(poly_1.multi_index), len(poly_1))
    else:
        coeffs_2 = np.random.rand(len(poly_1.multi_index))
    poly_2 = poly_1.__class__.from_poly(poly_1, coeffs_2)

    # Generate random scalar factors
    a = np.random.uniform(1, 5)
    b = np.random.uniform(1, 5)

    # Differentiate the polynomials
    diff_poly_1 = (a * poly_1 + b * poly_2).diff(deriv_order)
    diff_poly_2 = a * poly_1.diff(deriv_order) + b * poly_2.diff(deriv_order)

    # Assertion
    assert_polynomial_almost_equal(diff_poly_1, diff_poly_2)


def test_linearity_div_sub(rand_poly, deriv_order):
    """Test linearity of differentiation (scalar division and subtraction).

    diff(poly_1 / a - poly_2 / b) = diff(poly_1) / a - diff(poly_2) / b
    """
    poly_1 = rand_poly
    # Create another instance with different coefficients
    if len(poly_1) > 1:
        coeffs_2 = np.random.rand(len(poly_1.multi_index), len(poly_1))
    else:
        coeffs_2 = np.random.rand(len(poly_1.multi_index))
    poly_2 = poly_1.__class__.from_poly(poly_1, coeffs_2)

    # Generate random scalar factors
    a = np.random.uniform(1, 5)
    b = np.random.uniform(1, 5)

    # Differentiate the polynomials
    diff_poly_1 = (poly_1 / a - poly_2 / b).diff(deriv_order)
    diff_poly_2 = poly_1.diff(deriv_order) / a - poly_2.diff(deriv_order) / b

    # Assertion
    assert_polynomial_almost_equal(diff_poly_1, diff_poly_2)


def test_product(rand_poly):
    """Test the product rule of differentiation.

    For first order differentiation:

    diff(poly_1 * poly_2) = poly_1 * diff(poly_2) + poly_2 * diff(poly_1)
    """
    poly_1 = rand_poly
    # Create another instance with different coefficients
    if len(poly_1) > 1:
        coeffs_2 = np.random.rand(len(poly_1.multi_index), len(poly_1))
    else:
        coeffs_2 = np.random.rand(len(poly_1.multi_index))
    poly_2 = poly_1.__class__.from_poly(poly_1, coeffs_2)

    # Product rule: diff(f*g) = f*diff(g) + g*diff(f)
    # Applied only for first order differentiation
    m = poly_1.spatial_dimension
    dim = np.random.choice(m)
    lhs = (poly_1 * poly_2).partial_diff(dim)
    rhs = poly_2 * poly_1.partial_diff(dim) + poly_1 * poly_2.partial_diff(dim)

    # Assertion
    assert_polynomial_almost_equal(lhs, rhs)


def test_constant_annihilation(
    differentiable_poly,
    multi_index_mnp,
    domain,
    num_polynomials,
    deriv_order,
):
    """Test differentiation of constant polynomial results in zero."""
    # Create a random constant polynomial (the first index is the constant)
    if num_polynomials > 1:
        coeffs = np.zeros((len(multi_index_mnp), num_polynomials))
        coeffs[0, :] = 1.0
    else:
        coeffs = np.zeros(len(multi_index_mnp))
        coeffs[0] = 1.0
    poly = differentiable_poly(multi_index_mnp, coeffs, domain=domain)

    # Differentiate the polynomial
    diff_poly = poly.diff(deriv_order)

    # Assertions: All coefficients are zero
    assert np.allclose(diff_poly.coeffs, 0)


def test_identity(rand_poly):
    """Test zero-order derivative is the identity operation.

    diff(poly, [0, 0, ...]) = poly (i.e., polynomial equal in value)
    """
    poly = rand_poly

    # Zero-order derivative
    deriv_order = np.zeros(poly.spatial_dimension, dtype=int)
    diff_poly = poly.diff(deriv_order)

    # Assertions
    # Polynomials are (strictly) equal in value
    assert poly == diff_poly
    # Polynomials are different object
    assert diff_poly is not poly


def test_partial_diff(rand_poly):
    """Test that partial_diff(dim, order) equals diff([0,...,order,...,0]).

    `partial_diff()` is a syntactic sugar for single dimension differentiation
    using `diff()`.
    """

    dim = np.random.choice(rand_poly.spatial_dimension)
    n = rand_poly.multi_index.poly_degree
    if n == 0:
        pytest.skip("Partial differentiation requires polynomial degree > 0")

    order = np.random.randint(low=1, high=n+1)
    # Create an equivalent order of derivatives array
    deriv_order = np.zeros(rand_poly.spatial_dimension, dtype=int)
    deriv_order[dim] = order

    # Differentiate the polynomial
    diff_poly_1 = rand_poly.partial_diff(dim, order)
    diff_poly_2 = rand_poly.diff(deriv_order)

    # Assertion: strict equality
    assert diff_poly_1 == diff_poly_2


def test_order_higher_than_degree(rand_poly):
    """Test that diff with order > degree results in a zero polynomial."""
    poly = rand_poly

    # Differentiate the polynomial with order > degree
    dim = np.random.choice(poly.spatial_dimension)
    order = poly.multi_index.poly_degree + 1

    diff_poly = poly.partial_diff(dim, order)

    # Assertions
    assert np.allclose(diff_poly.coeffs, 0)


def test_basis_invariance_eval(rand_poly, deriv_order):
    """Test that the evaluation of differentiated poly is basis invariant.

    A polynomial differentiated in one basis should evaluate identically to
    the same polynomial transformed to another basis and then differentiated.
    They represent the same mathematical object in different bases.

    Notes
    -----
    - This is a "weak form" of closeness between two polynomials.
      The underlying coefficients may not be strictly speaking "close" but
      as long as the evaluation over the domain is close, then the polynomials
      are close.
    """
    poly_origin = rand_poly

    # Transform to a different basis
    target_basis = TARGET_POLYS[type(poly_origin)]
    poly_target = get_transformation(poly_origin, target_basis)()

    # Differentiate the polynomials
    diff_poly_origin = poly_origin.diff(deriv_order)
    diff_poly_target = poly_target.diff(deriv_order)

    # Generate test points
    dom = poly_origin.grid.domain
    m = dom.spatial_dimension
    lb, ub = dom.lowers, dom.uppers
    xx_test = np.random.uniform(lb, ub, size=(1000, m))

    # Evaluate the polynomials
    yy_origin = diff_poly_origin(xx_test)
    yy_target = diff_poly_target(xx_test)

    # Assertion
    assert np.allclose(yy_origin, yy_target)


def test_basis_invariance_poly(rand_poly, deriv_order):
    """Test that differentiated polynomials are basis invariant.

    A polynomial differentiated in one basis should be equal or close to the
    same polynomial transformed to another basis, differentiated in that basis,
    then transformed back.

    Notes
    -----
    - This assumes a "strong form" of closeness between two polynomials where
      the coefficients must all be close.
    - This test may fail at sufficiently high polynomial degrees or spatial
      dimensions due to accumulated numerical errors in basis transformations,
      even when differentiation is correctly implemented.
    """
    poly_origin = rand_poly

    # Transform to a different basis
    target_basis = TARGET_POLYS[type(poly_origin)]
    poly_target = get_transformation(poly_origin, target_basis)()

    # Differentiate the origin polynomial
    diff_poly_origin_1 = poly_origin.diff(deriv_order)

    # Differentiate the target and transform back to origin basis
    diff_poly_target = poly_target.diff(deriv_order)
    diff_poly_origin_2 = get_transformation(
        diff_poly_target,
        type(poly_origin),
    )()

    # Assertion
    assert_polynomial_almost_equal(diff_poly_origin_1, diff_poly_origin_2)
    assert_polynomial_almost_equal(diff_poly_origin_2, diff_poly_origin_1)


def test_schwarz(rand_poly):
    """Test the Schwarz theorem: order of differentiation does not matter.

    Schwarz's theorem states that the order of differentiation does not matter
    for polynomials, i.e., mixed partial derivatives commute.
    """
    poly = rand_poly
    m = poly.spatial_dimension
    if poly.spatial_dimension < 2:
        pytest.skip("Schwarz theorem requires dimension >= 2")

    # Select two random dimensions
    dim_i, dim_j = np.random.choice(m, size=2, replace=False)

    # Differentiate the polynomial
    diff_poly_1 = poly.partial_diff(dim_i).partial_diff(dim_j)
    diff_poly_2 = poly.partial_diff(dim_j).partial_diff(dim_i)

    # Assertion
    assert_polynomial_almost_equal(diff_poly_1, diff_poly_2)


def test_successive(rand_poly):
    """Test the successive differentiation equals simultaneous differentiation.

    Applying `partial_diff()` once with respect to each dimension successively
    should be equivalent to calling `diff()` with order of derivatives
    [1, 1, ..., 1].
    """
    poly = rand_poly
    m = poly.spatial_dimension

    if poly.multi_index.poly_degree < 1:
        pytest.skip("Test requires polynomial degree >= 1")

    # Successive differentiation with randomly selected dimensions
    dims = np.random.choice(m, size=m, replace=False)
    diff_poly_1 = poly.partial_diff(dims[0])
    for dim in dims[1:]:
        diff_poly_1 = diff_poly_1.partial_diff(dim)
    # Simultaneous differentiation
    deriv_order = np.ones(m, dtype=int)
    diff_poly_2 = poly.diff(deriv_order)

    # Assertion
    assert_polynomial_almost_equal(diff_poly_1, diff_poly_2)


def test_newton_backend(rand_poly, deriv_order):
    """Test the different differentiation backend for Newton polynomial.

    All backends (e.g., numpy, numba, numba-par) should produce close results.
    """
    poly = rand_poly
    if not isinstance(poly, NewtonPolynomial):
        pytest.skip("Different backend is only defined for Newton polynomials")

    poly_diffs = []
    for backend in DIFF_BACKENDS:
        poly_diffs.append(poly.diff(deriv_order, backend=backend))

    # Assertions
    assert_polynomial_almost_equal(poly_diffs[0], poly_diffs[1])
    assert_polynomial_almost_equal(poly_diffs[1], poly_diffs[2])
    assert_polynomial_almost_equal(poly_diffs[2], poly_diffs[0])


def test_newton_backend_invalid(rand_poly, deriv_order):
    """Test invalid backend for Newton polynomial should raise an exception."""
    if not isinstance(rand_poly, NewtonPolynomial):
        pytest.skip("Newton backend is only defined for Newton polynomials")

    with pytest.raises(NotImplementedError, match="invalid"):
        _ = rand_poly.diff(deriv_order, backend="invalid")


def test_partial_diff_canonical(rand_poly):
    """Test the partial differentiation of a canonical polynomial.

    Notes
    -----
    - Because the differentiation of a polynomial in the canonical basis
      follows a simple(r) rule, the results can be compared with an alternative
      method.
    """
    poly = rand_poly
    if not isinstance(poly, CanonicalPolynomial):
        pytest.skip("Test is specific to CanonicalPolynomial")

    # (Partial) Differentiate the polynomial once
    m = poly.spatial_dimension
    for dim in range(m):
        deriv_order = np.zeros(m, dtype=int)
        deriv_order[dim] = 1
        diff_poly = poly.partial_diff(dim)

        coeffs, exps = poly.coeffs, poly.multi_index.exponents
        diff_factor = poly.domain.diff_factor(deriv_order)
        coeffs_ref = diff_can_coeffs(coeffs, exps, deriv_order, diff_factor)

        # Assertion
        assert np.allclose(coeffs_ref, diff_poly.coeffs)


def test_diff_canonical(rand_poly, deriv_order):
    """Test the differentiation of a canonical polynomial.

    Notes
    -----
    - Because the differentiation of a polynomial in the canonical basis
      follows a simple(r) rule, the results can be compared with an alternative
      method.
    """
    poly = rand_poly
    if not isinstance(poly, CanonicalPolynomial):
        pytest.skip("Test is specific to CanonicalPolynomial")

    # Differentiate the polynomial
    diff_poly = poly.diff(deriv_order)
    coeffs, exps = poly.coeffs, poly.multi_index.exponents
    diff_factor = poly.domain.diff_factor(deriv_order)
    coeffs_ref = diff_can_coeffs(coeffs, exps, deriv_order, diff_factor)

    # Assertion
    assert np.allclose(coeffs_ref, diff_poly.coeffs)
