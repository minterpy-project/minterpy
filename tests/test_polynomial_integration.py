"""
Test suite for polynomial integration functionality across different bases.

This module verifies the `integrate_over()` method for all "integrable"
polynomial representations. Tests cover integration over custom domains,
subdivisions, bound validations, linearity properties,
and invariance under basis transformations.
"""
import numpy as np
import pytest

from itertools import product

from minterpy import (
    Domain,
    LagrangePolynomial,
    NewtonPolynomial,
    CanonicalPolynomial,
    get_transformation,
)

##################################
# Module variables and fixtures  #
##################################

# Transformation between bases
TARGET_POLYS = {
    LagrangePolynomial: [NewtonPolynomial, CanonicalPolynomial],
    NewtonPolynomial: [LagrangePolynomial, CanonicalPolynomial],
    CanonicalPolynomial: [LagrangePolynomial, NewtonPolynomial],
}


# Integrable polynomial classes
INTEGRABLE_POLYS = [LagrangePolynomial, NewtonPolynomial, CanonicalPolynomial]


def _id_poly_class(poly_class):
    return f"{poly_class.__name__:>19}"


@pytest.fixture(params=INTEGRABLE_POLYS, ids=_id_poly_class)
def integrable_poly(request):
    return request.param


# Random integrable polynomial instance
@pytest.fixture
def rand_integrable_poly(
    integrable_poly,
    multi_index_mnp,
    domain,
    num_polynomials,
):
    """Create a random integrable polynomial as a fixture."""
    # Create random coefficients
    if num_polynomials > 1:
        coeffs = np.random.rand(len(multi_index_mnp), num_polynomials)
    else:
        coeffs = np.random.rand(len(multi_index_mnp))

    # Create an instance of polynomial
    poly = integrable_poly(multi_index_mnp, coeffs, domain=domain)

    return poly


def test_domain_scaling_factor(rand_integrable_poly):
    """Test the integration of polynomial over its domain.

    Notes
    -----
    - Minterpy polynomials are internally defined in the internal domain.
      The difference between integrating Minterpy polynomials with a custom
      domain and the ones with the default domain is just a factor (Jacobian).
    """
    poly_1 = rand_integrable_poly
    # Create another instance with the default internal domain
    poly_2 = rand_integrable_poly.__class__(
        poly_1.multi_index,
        poly_1.coeffs,
    )

    # Compute the integral over the whole domain
    int_1 = poly_1.integrate_over()
    int_2 = poly_2.integrate_over()

    # Assertions (commutative check)
    assert np.allclose(poly_1.domain.int_factor() * int_2, int_1)
    assert np.allclose(int_1, poly_1.domain.int_factor() * int_2)


def test_sum_over_subdivided_domain(rand_integrable_poly):
    """Test the sum of polynomial integration over subdomains."""
    poly = rand_integrable_poly

    # Divide the domain at a mid-point
    mid_point = np.mean(poly.domain.bounds, axis=1)
    interval_1 = np.stack([poly.domain.lowers, mid_point], axis=1)
    interval_2 = np.stack([mid_point, poly.domain.uppers], axis=1)
    # Divide the domains (in higher dimension -> hyper-rectangles)
    subdomains = [
        np.array(combi) for combi in product(*zip(interval_1, interval_2))
    ]

    # Compute the integral over the whole domain
    int_1 = poly.integrate_over()

    # Compute the integral over each subdomain
    int_2 = 0.0
    for subdomain in subdomains:
        int_2 += poly.integrate_over(subdomain)

    # Assertions (commutative check)
    assert np.allclose(int_1, int_2)
    assert np.allclose(int_2, int_1)


def test_bounds_equal(rand_integrable_poly):
    """Test that integral is zero when at least one integral bound collapses.
    """
    poly = rand_integrable_poly

    # Create bounds (one of them has lb == ub)
    dim = poly.domain.spatial_dimension
    bounds = poly.domain.bounds.copy()
    idx = np.random.choice(dim)
    bounds[idx, 0] = bounds[idx, 1]

    # Compute integral
    int_over = poly.integrate_over(bounds)

    # Assertion
    assert np.allclose(int_over, 0.0)


def test_bounds_flipped(rand_integrable_poly):
    """Test that integral flip sign when one of the bounds is flipped."""
    poly = rand_integrable_poly

    # Compute the integral
    int_over_1 = poly.integrate_over()

    # Flip one of the bounds
    dim = poly.domain.spatial_dimension
    bounds = poly.domain.bounds.copy()
    idx = np.random.choice(dim)
    bounds[idx, [0, 1]] = bounds[idx, [1, 0]]

    # Compute the integral with flipped bounds
    int_over_2 = poly.integrate_over(bounds)

    # Assertion
    assert np.allclose(int_over_1, -1 * int_over_2)


def test_bounds_invalid_shape(rand_integrable_poly):
    """Test polynomial integration with bounds of invalid shape."""
    poly = rand_integrable_poly

    # Create bounds (invalid shape)
    dim = poly.domain.spatial_dimension
    bounds = np.random.rand(dim + 3, 2)  # Dimension mismatch

    with pytest.raises(ValueError):
        _ = poly.integrate_over(bounds)


def test_list_as_bounds(rand_integrable_poly):
    """Test integrate over with bounds specified with lists.

    Notes
    -----
    - The method accepts bounds as lists (or any array-like structure),
      not just NumPy arrays, as they are internally converted using
      ``np.atleast_2d()``.
    """
    poly = rand_integrable_poly

    # Compute the integral
    value_1 = poly.integrate_over()

    # Create the bounds as a list
    bounds = [list(_) for _ in poly.domain.bounds]

    # Compute the integral with the bounds as a list
    value_2 = poly.integrate_over(bounds)

    # Assertion (equality should be exact)
    assert np.all(value_1 == value_2)


def test_linearity_mul_add(rand_integrable_poly):
    """Test linearity of integration from scalar multiplication and addition.

    Integ(a * poly_1 + b * poly_2) = a * integ(poly_1) + b * integ(poly_2)
    """
    if isinstance(rand_integrable_poly, LagrangePolynomial):
        pytest.skip(
            "Polynomial Arithmetic operations with Lagrange polynomials "
            "are not supported"
        )

    poly_1 = rand_integrable_poly
    # Create another instance with different coefficients
    if len(poly_1) > 1:
        coeffs_2 = np.random.rand(len(poly_1.multi_index), len(poly_1))
    else:
        coeffs_2 = np.random.rand(len(poly_1.multi_index))
    poly_2 = poly_1.__class__.from_poly(poly_1, coeffs_2)

    # Generate random scalar factors
    a = np.random.uniform(-5, 5)
    b = np.random.uniform(-5, 5)

    int_1 = (a * poly_1 + b * poly_2).integrate_over()
    int_2 = a * poly_1.integrate_over() + b * poly_2.integrate_over()

    # Assertions (commutative check)
    assert np.allclose(int_1, int_2)
    assert np.allclose(int_2, int_1)


def test_linearity_div_sub(rand_integrable_poly):
    """Test linearity of integration from scalar division and subtraction.

    Integ(poly_1 / a + poly_2 / b) = integ(poly_1) / a + integ(poly_2) / b
    """
    if isinstance(rand_integrable_poly, LagrangePolynomial):
        pytest.skip(
            "Polynomial Arithmetic operations with Lagrange polynomials "
            "are not supported"
        )

    poly_1 = rand_integrable_poly
    # Create another instance with different coefficients
    if len(poly_1) > 1:
        coeffs_2 = np.random.rand(len(poly_1.multi_index), len(poly_1))
    else:
        coeffs_2 = np.random.rand(len(poly_1.multi_index))
    poly_2 = poly_1.__class__.from_poly(poly_1, coeffs_2)

    # Generate random scalar factors
    a = np.random.uniform(-5, 5)
    b = np.random.uniform(-5, 5)

    int_1 = (poly_1 / a - poly_2 / b).integrate_over()
    int_2 = poly_1.integrate_over() / a - poly_2.integrate_over() / b

    # Assertions (commutative check)
    assert np.allclose(int_1, int_2)
    assert np.allclose(int_2, int_1)


def test_bases_transformation_invariance(rand_integrable_poly):
    """Test the integral of polynomial in different bases.

    Polynomials represented in different bases should have the same integral.
    In other words, polynomial integration is invariant under basis
    transformation.
    """
    poly = rand_integrable_poly
    int_origin = poly.integrate_over()

    # Convert to a different basis
    int_targets = []
    target_bases = TARGET_POLYS[poly.__class__]
    for target_basis in target_bases:
        poly_target = get_transformation(poly, target_basis)()
        int_targets.append(poly_target.integrate_over())

    # Assertions
    for int_target in int_targets:
        assert np.allclose(int_origin, int_target)
        assert np.allclose(int_target, int_origin)
