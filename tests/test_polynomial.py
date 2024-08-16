"""
Testing module for the common functionalities of the Minterpy polynomials.

Notes
-----
- Specific behaviors from different concrete polynomial implementations are
  tested in separate testing modules.
- Lagrange polynomials cannot be evaluated directly on a query point.
"""
import copy
import itertools
import numpy as np
import pytest

from conftest import (
    assert_call,
    num_polynomials,
    poly_class_all,
    create_mi_pair_distinct,
    POLY_CLASSES,
)
from minterpy import (
    Grid,
    LagrangePolynomial,
    MultiIndexSet,
)


class TestInitialization:
    """All tests related to the initialization of a polynomial instance."""

    def test_without_coeffs(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization without coefficients."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        assert_call(poly_class_all, mi)

    def test_with_coeffs(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization with coefficients."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)

        assert_call(poly_class_all, mi, coeffs)

    def test_with_grid(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization with a valid grid."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        grd = Grid(mi)
        coeffs = np.arange(len(mi), dtype=float)

        assert_call(poly_class_all, mi, grid=grd)
        assert_call(poly_class_all, mi, coeffs, grid=grd)

    def test_with_invalid_grid(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization with an invalid grid."""
        mi = MultiIndexSet.from_degree(
            SpatialDimension,
            PolyDegree + 1,
            LpDegree
        )
        # Create a "smaller" grid the the polynomial (invalid)
        mi_grid = MultiIndexSet.from_degree(
            SpatialDimension,
            PolyDegree,
            LpDegree
        )
        grd = Grid(mi_grid)
        coeffs = np.arange(len(mi), dtype=float)

        with pytest.raises(ValueError):
            poly_class_all(mi, coeffs, grid=grd)

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set(self, spatial_dimension, poly_class_all, LpDegree):
        """Test initialization with an empty multi-index set."""
        # Create an empty set
        mi = MultiIndexSet(np.empty((0, spatial_dimension)), LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            poly_class_all(mi)


class TestFrom:
    """All tests related to the different factory methods."""
    def test_from_grid_uninit(self, poly_class_all, grid_mnp):
        """Test creating an uninitialized polynomial from a Grid instance."""
        # Create a polynomial instance
        poly = poly_class_all.from_grid(grid_mnp)

        # Assertions
        assert poly.grid == grid_mnp
        assert poly.multi_index == grid_mnp.multi_index

    def test_from_grid_init(self, poly_class_all, grid_mnp):
        """Test creating an initialized polynomial from a Grid instance."""
        # Generate random coefficients to initialize the polynomial
        coeffs = np.random.rand(len(grid_mnp.multi_index))
        # Create a polynomial instance
        # From grid - grid the same as default
        poly_1 = poly_class_all.from_grid(grid_mnp, coeffs)
        # Default constructor - default grid
        poly_2 = poly_class_all(grid_mnp.multi_index, coeffs)
        # Default constructor - grid the same as default
        poly_3 = poly_class_all(grid_mnp.multi_index, coeffs, grid=grid_mnp)

        # Assertions
        assert poly_1 == poly_2
        assert poly_1 == poly_3


class TestGetSetCoeffs:
    """All tests related to setting the polynomial coefficients."""
    def test_uninit(self, poly_mnp_uninit):
        """Test accessing the coefficients of an uninitialized polynomial."""
        # Get the uninitialized polynomial
        poly = poly_mnp_uninit

        with pytest.raises(ValueError):
            print(poly.coeffs)

    def test_valid(self, poly_mnp_uninit):
        """Test setting with valid coefficients."""
        # Get the uninitialized polynomial
        poly = poly_mnp_uninit

        # Create valid coefficients
        coeffs = np.random.rand(len(poly.multi_index))

        # Set the coefficients
        poly.coeffs = coeffs

        # Assertion
        assert np.array_equal(poly.coeffs, coeffs)

    def test_invalid_type(self, poly_mnp_uninit):
        """Test setting an invalid type/value as the polynomial coefficients.
        """
        # Get the uninitialized polynomial
        poly = poly_mnp_uninit

        with pytest.raises(TypeError):
            poly.coeffs = {"a": 1}  # Dictionary can't be converted

    def test_invalid_value(self, poly_mnp_uninit):
        """Test setting an invalid value as the polynomial coefficients."""
        # Get the uninitialized polynomial
        poly = poly_mnp_uninit

        # Create the coefficients with invalid value
        coeffs = np.random.rand(len(poly.multi_index))
        coeffs[0] = np.nan

        with pytest.raises(ValueError):
            poly.coeffs = coeffs

    def test_invalid_length(self, poly_mnp_uninit):
        """Test setting polynomial coefficients with an invalid length."""
        # Get the uninitialized polynomial
        poly = poly_mnp_uninit

        # Create the coefficients with invalid length
        coeffs = np.random.rand(len(poly.multi_index) + 1)

        with pytest.raises(ValueError):
            poly.coeffs = coeffs


class TestLength:
    """All tests related to '__len__()' method of polynomial instances."""
    def test_init(self, poly_class_all, multi_index_mnp, num_polynomials):
        """Test getting the length of an initialized polynomial."""
        # Create a random coefficients
        coeffs = np.random.rand(len(multi_index_mnp), num_polynomials)

        # Create an instance of polynomial
        poly = poly_class_all(multi_index_mnp, coeffs)

        # Assertion
        assert len(poly) == num_polynomials

    def test_uninit(self, poly_mnp_uninit):
        """Test getting the length of an uninitialized polynomial."""
        with pytest.raises(ValueError):
            print(len(poly_mnp_uninit))


class TestExpandDim:
    """All tests related to the dimension expansion of polynomial instances."""
    def test_target_dim_higher_dim_uninit(self, poly_mnp_uninit):
        """Test dimension expansion of a un-initialized polynomial
        to a higher dimension.
        """
        # Get the current dimension
        origin_poly = poly_mnp_uninit
        origin_dim = origin_poly.spatial_dimension
        target_dim = origin_dim + 1

        # Expand the dimension to the same dimension
        target_poly = poly_mnp_uninit.expand_dim(target_dim)

        # Assertions
        assert target_poly != origin_poly
        assert target_poly.spatial_dimension == target_dim
        assert target_poly.multi_index == target_poly.multi_index.expand_dim(
            target_dim,
        )
        assert target_poly.grid == origin_poly.grid.expand_dim(target_dim)

    def test_target_dim_same_dim(self, rand_poly_mnp_all):
        """Test dimension expansion of a polynomial to the same dimension."""
        # Get the current dimension
        dim = rand_poly_mnp_all.spatial_dimension

        # Expand the dimension to the same dimension
        poly = rand_poly_mnp_all.expand_dim(dim)

        # Assertions
        assert poly == rand_poly_mnp_all
        assert poly.spatial_dimension == dim

    def test_target_dim_contraction(self, rand_poly_mnp_all):
        """Test dimension contraction; this should raise an exception."""
        # Get the current dimension
        dim = rand_poly_mnp_all.spatial_dimension

        # Contract the dimension
        with pytest.raises(ValueError):
            rand_poly_mnp_all.expand_dim(dim - 1)

    def test_target_dim_higher_dim(self, rand_poly_mnp_all):
        """Test dimension expansion of a polynomial to a higher dimension."""
        # Get the random polynomial
        poly_1 = rand_poly_mnp_all

        # Get the new dimension
        new_dim = poly_1.spatial_dimension + 1

        # Expand the dimension
        poly_2 = poly_1.expand_dim(new_dim)

        # Assertions
        assert poly_1 != poly_2
        assert poly_2.spatial_dimension == new_dim
        assert poly_2.multi_index == poly_1.multi_index.expand_dim(new_dim)
        assert poly_2.grid == poly_1.grid.expand_dim(new_dim)

    def test_target_dim_new_domains(self, rand_poly_mnp_all):
        """Test dimension expansion of a polynomial with specified domains."""
        # Get the random polynomial
        poly_1 = rand_poly_mnp_all

        # Get the current and the new dimension
        dim = poly_1.spatial_dimension
        new_dim = dim + 2

        # Define valid additional domains
        new_domains = np.array([[-2, -2], [2, 2]])

        # Expand the dimension
        poly_2 = poly_1.expand_dim(
            new_dim,
            extra_internal_domain=new_domains,
            extra_user_domain=new_domains,
        )

        # Assertions
        assert poly_1 != poly_2
        assert poly_2.spatial_dimension == new_dim
        assert poly_2.multi_index == poly_1.multi_index.expand_dim(new_dim)
        assert poly_2.grid == poly_1.grid.expand_dim(new_dim)
        assert np.array_equal(poly_2.user_domain[:, dim:], new_domains)
        assert np.array_equal(poly_2.internal_domain[:, dim:], new_domains)

    def test_target_dim_non_uniform_domain(self, poly_mnp_non_unif_domain):
        """Test dimension expansion in which the domain cannot be extrapolated.
        """
        origin_dim = poly_mnp_non_unif_domain.spatial_dimension
        target_dim = origin_dim + 1

        # Expansion of polynomials w/ a non-uniform domain raises an exception
        with pytest.raises(ValueError):
            poly_mnp_non_unif_domain.expand_dim(target_dim)

    def test_target_poly_same_dim(self, rand_poly_mnp_all):
        """Test dimension expansion of a polynomial to the dimension of
        another polynomial having the same dimension.
        """
        # Get the random polynomial
        poly_1 = rand_poly_mnp_all

        # Expand the dimension
        poly_2 = poly_1.expand_dim(poly_1)

        # Assertions
        assert poly_1 == poly_2
        assert poly_2 == poly_1

    def test_target_poly_higher_dim(self, poly_mnp_pair_diff_dim):
        """Test dimension expansion of a polynomial to the dimension of another
        polynomial having a higher dimension.
        """
        # Get the polynomial instances
        poly_1, poly_2 = poly_mnp_pair_diff_dim
        # The first polynomial must have smaller dimension
        if poly_1.spatial_dimension > poly_2.spatial_dimension:
            poly_1, poly_2 = poly_2, poly_1

        # Expand the dimension
        poly_1_exp = poly_1.expand_dim(poly_2)

        # Assertions
        assert poly_1_exp.has_matching_dimension(poly_2)
        assert poly_1_exp.has_matching_domain(poly_2)

    def test_target_poly_contraction(self, poly_mnp_pair_diff_dim):
        """Test dimension expansion of a polynomial to the dimension of another
        polynomial having a smaller dimension; this should raise an exception.
        """
        # Get the polynomial instances
        poly_1, poly_2 = poly_mnp_pair_diff_dim
        # The first polynomial must have larger dimension
        if poly_1.spatial_dimension < poly_2.spatial_dimension:
            poly_1, poly_2 = poly_2, poly_1

        # Expand (contract) the dimension
        with pytest.raises(ValueError):
            poly_1.expand_dim(poly_2)

    def test_target_poly_incompatible_domain(self, poly_mnp_pair_diff_domain):
        """Test dimension expansion of a polynomial to the dimension of another
        polynomial with incompatible internal domain.
        """
        # Get the polynomial instances
        poly_1, poly_2 = poly_mnp_pair_diff_domain

        # Expanding the dimension to a polynomial with incompatible domain
        with pytest.raises(ValueError):
            poly_1.expand_dim(poly_2)


class TestEquality:
    """All tests related to the equality check between polynomial instances."""

    def test_single(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test equality between two instances with one set of coefficients."""
        # Create a common multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Generate random coefficient values
        coeffs = np.random.rand(len(mi))

        # Create two equal polynomials
        poly_1 = poly_class_all(mi, coeffs)
        poly_2 = poly_class_all(mi, coeffs)

        # Assertions
        assert poly_1 is not poly_2  # Not identical instances
        assert poly_1 == poly_2  # But equal in values
        assert poly_2 == poly_1  # Symmetric property

    def test_multiple(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test equality between two instances with multiple sets of coeffs."""
        # Create a common multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Generate random coefficient values
        coeffs = np.random.rand(len(mi), num_polynomials)

        # Create two equal polynomials
        poly_1 = poly_class_all(mi, coeffs)
        poly_2 = poly_class_all(mi, coeffs)

        # Assertions
        assert poly_1 is not poly_2  # Not identical instances
        assert poly_1 == poly_2  # But equal in values
        assert poly_2 == poly_1  # Symmetric property


class TestInequality:
    """All tests related to the inequality check of two poly. instances."""

    def test_inconsistent_types(self, poly_class_all):
        """Test inequality due to inconsistent type in the comparison."""
        # Create a MultiIndexSet instance
        mi = MultiIndexSet.from_degree(
            spatial_dimension=3,
            poly_degree=4,
            lp_degree=2.0,
        )

        # Generate random coefficient values
        coeffs = np.random.rand(len(mi))

        # Create a polynomial instance
        poly = poly_class_all(mi, coeffs)

        # Assertions
        assert poly != mi
        assert poly != coeffs
        assert poly != "123"
        assert poly != 1
        assert poly != 10.0

    def test_different_multi_index(self, poly_class_all):
        """Test inequality due to different multi-index sets."""
        # Create two different multi-index sets
        mi_1, mi_2 = create_mi_pair_distinct()

        # Generate two sets of coefficients
        coeffs_1 = np.random.rand(len(mi_1))
        coeffs_2 = np.random.rand(len(mi_2))

        # Create two polynomials
        poly_1 = poly_class_all(mi_1, coeffs_1)
        poly_2 = poly_class_all(mi_2, coeffs_2)

        # Assertions
        assert poly_1 != poly_2
        assert poly_2 != poly_1

    def test_different_grids(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test inequality due to different underlying Grids."""
        # Create a common MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Generate two grids
        grd_1 = Grid(mi)
        grd_2 = Grid.from_value_set(
            mi,
            np.linspace(-0.99, 0.99, PolyDegree+1),
        )

        # Generate a set of random coefficients
        coeffs = np.random.rand(len(mi))

        # Create two polynomials
        poly_1 = poly_class_all(mi, coeffs, grid=grd_1)
        poly_2 = poly_class_all(mi, coeffs, grid=grd_2)

        # Assertions
        assert poly_1 != poly_2
        assert poly_2 != poly_1

    def test_different_coeffs(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test inequality due to different coefficient values."""
        # Create a common MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Generate two sets of random coefficients
        coeffs_1 = np.random.rand(len(mi))
        coeffs_2 = np.random.rand(len(mi))

        # Create two polynomials
        poly_1 = poly_class_all(mi, coeffs_1)
        poly_2 = poly_class_all(mi, coeffs_2)

        # Assertions
        assert poly_1 != poly_2
        assert poly_2 != poly_1

    def test_different_poly(self, SpatialDimension, PolyDegree, LpDegree):
        """Test inequality due to different concrete polynomial classes."""
        # Create a common MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Generate a set of random coefficients
        coeffs = np.random.rand(len(mi))

        # Create two polynomials of different classes
        poly_combinations = itertools.combinations(POLY_CLASSES, r=2)

        for poly_class_1, poly_class_2 in poly_combinations:
            poly_1 = poly_class_1(mi, coeffs)
            poly_2 = poly_class_2(mi, coeffs)

            # Assertions
            assert poly_1 != poly_2
            assert poly_2 != poly_1


class TestEvaluation:
    """All tests related to the evaluation of a polynomial instance."""

    def test_with_coeffs(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test the evaluation of an 'initialized' polynomial."""
        # Create a polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)
        poly = poly_class_all(mi, coeffs)

        xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
        if isinstance(poly, LagrangePolynomial):
            with pytest.raises(NotImplementedError):
                poly(xx_test)
        else:
            assert_call(poly, xx_test)

    def test_without_coeffs(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test the evaluation of an 'uninitialized' polynomial."""
        # Create a polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        poly = poly_class_all(mi)

        xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
        if isinstance(poly, LagrangePolynomial):
            with pytest.raises(NotImplementedError):
                poly(xx_test)
        else:
            with pytest.raises(ValueError):
                poly(xx_test)

    def test_multiple_coeffs(
        self,
        poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test the evaluation of multiple polys (multiple set of coeffs.)"""
        # Create polynomials (with multiple sets of coefficients)
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)
        coeffs = np.repeat(coeffs[:, None], num_polynomials, axis=1)
        poly = poly_class_all(mi, coeffs)

        xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
        if isinstance(poly, LagrangePolynomial):
            with pytest.raises(NotImplementedError):
                poly(xx_test)
        else:
            if num_polynomials == 1:
                pytest.skip("Only applies for multiple coefficient sets.")

            # Single evaluation point
            yy_test = poly(xx_test[0:1])
            assert yy_test.shape == (1, num_polynomials)

            # Multiple evaluation points
            yy_test = poly(xx_test)
            for i in range(num_polynomials):
                # Due to identical coefficients, results are identical
                assert np.all(yy_test[:, i] == yy_test[:, 0])


class TestNegation:
    """All tests related to the negation of a polynomial instance."""
    def test_neg_multi_poly(self, rand_poly_mnp_all):
        """Test the expected results from negating a polynomial."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Negate the polynomial
        poly_neg = -poly

        # Assertions
        assert poly_neg is not poly  # Must not be the same instance
        assert poly_neg != -poly_neg
        assert poly_neg.multi_index == poly.multi_index
        assert poly_neg.grid == poly.grid
        assert np.all(poly_neg.coeffs == -1 * poly.coeffs)

    def test_sanity_multiplication(self, rand_poly_mnp_all):
        """Test that negation is equivalent to multiplication with -1."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Negate the polynomial
        poly_neg_1 = -poly  # via negation
        poly_neg_2 = -1 * poly  # via negative multiplication

        # Assertions
        assert poly_neg_1 is not poly_neg_2
        assert poly_neg_1 == poly_neg_2

    def test_different_grid(
        self,
            poly_class_all,
        SpatialDimension,
        LpDegree,
        PolyDegree,
    ):
        """Test that grid remains the same as the one used for construction."""
        # Create a MultiIndexSet instance
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a random set of coefficients
        coeffs = np.random.rand(len(mi))

        # Create a grid of equidistant point
        grd = Grid.from_value_set(
            mi,
            np.linspace(-1, 1, PolyDegree+1),
        )

        # Create a polynomial instance
        poly = poly_class_all(mi, coeffs, grid=grd)

        # Negate the polynomial
        poly_neg = -poly

        # Assertions
        assert poly_neg is not poly  # Must not be the same instance
        assert poly_neg != -poly_neg
        assert poly_neg.multi_index == poly.multi_index
        assert poly_neg.grid == poly.grid
        assert np.all(poly_neg.coeffs == -1 * poly.coeffs)


class TestPos:
    """All tests related to the unary positive operator on a polynomial."""
    def test_pos_polys(self, rand_poly_mnp_all):
        """Test using the unary positive operator on a polynomial."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Assertions
        assert poly is (+poly)  # Object identity
        assert poly == (+poly)  # Equality in value


class TestHasMatchingDomain:
    """All tests related to method to check if polynomial domains match."""
    def test_sanity(self, rand_poly_mnp_all):
        """Test if a polynomial has a matching domain with itself."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Assertion
        assert poly.has_matching_domain(poly)

    def test_same_dim(self, rand_poly_mnp_all):
        """Test if poly. has a matching domain with another of the same dim."""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp_all
        poly_2 = copy.copy(rand_poly_mnp_all)

        # Assertions
        assert poly_1.has_matching_domain(poly_2)
        assert poly_2.has_matching_domain(poly_1)

    def test_diff_dim(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test if poly. has a matching domain with another of a diff. dim."""
        # Create a MultiIndexSet
        dim_1 = SpatialDimension
        mi_1 = MultiIndexSet.from_degree(dim_1, PolyDegree, LpDegree)
        dim_2 = SpatialDimension + 1
        mi_2 = MultiIndexSet.from_degree(dim_2, PolyDegree, LpDegree)

        # Create a polynomial instance
        poly_1 = poly_class_all(mi_1)
        poly_2 = poly_class_all(mi_2)

        # Assertion
        assert poly_1.has_matching_domain(poly_2)
        assert poly_2.has_matching_domain(poly_1)

    def test_user_domain(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test if poly. does not have a matching user domain."""
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a polynomial instance
        user_domain_1 = np.ones((2, SpatialDimension))
        user_domain_1[0, :] *= -2
        user_domain_1[1, :] *= 2
        poly_1 = poly_class_all(mi, user_domain=user_domain_1)
        user_domain_2 = np.ones((2, SpatialDimension))
        user_domain_2[0, :] *= -0.5
        user_domain_2[1, :] *= 0.5
        poly_2 = poly_class_all(mi, user_domain=user_domain_2)

        # Assertion
        assert not poly_1.has_matching_domain(poly_2)
        assert not poly_2.has_matching_domain(poly_1)

    def test_internal_domain(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test if poly. does not have a matching internal domain."""
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a polynomial instance
        internal_domain_1 = np.ones((2, SpatialDimension))
        internal_domain_1[0, :] *= -2
        internal_domain_1[1, :] *= 2
        poly_1 = poly_class_all(mi, internal_domain=internal_domain_1)
        internal_domain_2 = np.ones((2, SpatialDimension))
        internal_domain_2[0, :] *= -0.5
        internal_domain_2[1, :] *= 0.5
        poly_2 = poly_class_all(mi, internal_domain=internal_domain_2)

        # Assertion
        assert not poly_1.has_matching_domain(poly_2)
        assert not poly_2.has_matching_domain(poly_1)


class TestScalarMultiplication:
    """All tests related to the multiplication of a polynomial with scalars."""
    def test_mul_identity(self, rand_poly_mnp_all):
        """Left-sided multiplication identity"""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp_all

        # Left-sided multiplication
        poly_2 = poly_1 * 1.0

        # Assertions
        assert poly_1 is not poly_2  # The instances are not identical
        assert poly_1 == poly_2
        assert poly_2 == poly_1

    def test_rmul_identity(self, rand_poly_mnp_all):
        """Right-sided multiplication identity."""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp_all

        # Left-sided multiplication
        poly_2 = 1.0 * poly_1

        # Assertions
        assert poly_1 is not poly_2  # The instances are not identical
        assert poly_1 == poly_2
        assert poly_2 == poly_1

    def test_imul_identity(self, rand_poly_mnp_all):
        """Augmented multiplication identity."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all
        old_id = id(poly)
        old_poly = copy.copy(poly)

        # Left-sided multiplication
        poly *= 1.0

        # Assertions
        assert old_id == id(poly)  # The instances are not identical
        assert old_poly == poly

    def test_mul(self, rand_poly_mnp_all):
        """Multiplication of a polynomial with a valid scalar."""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp_all

        # Generate a random scalar
        scalar = np.random.random()

        # Multiply the polynomial both sides
        poly_2 = poly_1 * scalar  # Left-sided multiplication
        poly_3 = scalar * poly_1  # Right-sided multiplication

        # Assertions
        assert poly_2 is not poly_3
        assert poly_2 == poly_3
        assert poly_3 == poly_2
        assert np.all(poly_1.coeffs * scalar == poly_2.coeffs)
        assert np.all(poly_1.coeffs * scalar == poly_3.coeffs)

    def test_imul(self, rand_poly_mnp_all):
        """Augmented multiplication of polynomials with a valid scalar."""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp_all
        poly_2 = copy.copy(poly_1)

        # Generate a random scalar
        scalar = np.random.random()

        # Multiply the polynomials
        poly_1 *= scalar
        poly_2 = poly_2 * scalar  # not in-place

        # Assertions
        assert poly_1 is not poly_2
        assert poly_1 == poly_2
        assert poly_2 == poly_1


class TestPolyMultiplication:
    """All tests related to the polynomial-polynomial multiplication and for
    behaviors that are common to all instances of concrete polynomial classes.
    """
    def test_inconsistent_num_polys(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Multiplication of polynomials with an inconsistent shape of the
        respective coefficients raises error.
        """
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create random coefficient sets
        coeffs_1 = np.random.rand(len(mi), num_polynomials)
        coeffs_2 = np.random.rand(len(mi), num_polynomials+1)

        # Create polynomials
        poly_1 = poly_class_all(mi, coeffs_1)
        poly_2 = poly_class_all(mi, coeffs_2)

        # Multiplication
        with pytest.raises(ValueError):
            print(poly_1 * poly_2)

    def test_non_matching_domain(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Multiplication of polynomials with a non-matching domain raises
        and exception.
        """
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a random set of coefficients
        coeffs = np.random.rand(len(mi))

        # Create a polynomial instance
        domain_1 = np.ones((2, SpatialDimension))
        domain_1[0, :] *= -2
        domain_1[1, :] *= 2
        poly_1 = poly_class_all(mi, coeffs, user_domain=domain_1)
        domain_2 = np.ones((2, SpatialDimension))
        domain_2[0, :] *= -0.5
        domain_2[1, :] *= 0.5
        poly_2 = poly_class_all(mi, coeffs, user_domain=domain_2)

        # Perform multiplication
        with pytest.raises(ValueError):
            print(poly_1 * poly_2)

    @pytest.mark.parametrize("invalid_value", ["123", 1+1j, [1, 2, 3]])
    def test_invalid(self, rand_poly_mnp_all, invalid_value):
        """Multiplication of polynomials with an invalid type raises
        an exception.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Assertion
        with pytest.raises(TypeError):
            # Left-sided multiplication
            print(poly * invalid_value)

        with pytest.raises(TypeError):
            # Right-sided multiplication
            print(invalid_value * poly)

        with pytest.raises(TypeError):
            # Augmented multiplication
            poly *= invalid_value

    def test_imul(self, rand_poly_mnp_all):
        """Augmented multiplication of polynomials raises an exception."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Assertion
        with pytest.raises(NotImplementedError):
            # Perform multiplication
            poly *= poly

    def test_eval(self, rand_polys_mnp_pair):
        """Test the evaluation polynomials product."""
        # Get the polynomial pairs
        poly_1, poly_2 = rand_polys_mnp_pair

        # General Lagrange poly-poly multiplication is unsupported -> skip it
        if isinstance(poly_1, LagrangePolynomial):
            pytest.skip(f"Skipping evaluation of a product {type(poly_1)}.")

        # Get the maximum dimension
        dim_1 = poly_1.spatial_dimension
        dim_2 = poly_2.spatial_dimension
        dim = np.max([dim_1, dim_2])

        # Generate random test points
        xx_test = -1 + 2 * np.random.rand(1000, dim)

        # Compute reference results
        yy_1 = poly_1(xx_test[:, :dim_1])
        yy_2 = poly_2(xx_test[:, :dim_2])
        yy_ref = yy_1 * yy_2

        # Multiply polynomials and evaluate
        yy_prod_1 = (poly_1 * poly_2)(xx_test)
        yy_prod_2 = (poly_2 * poly_1)(xx_test)

        # Assertion
        assert np.allclose(yy_ref, yy_prod_1)
        assert np.allclose(yy_ref, yy_prod_2)

    def test_constant_poly(self, rand_poly_mnp_all):
        """Test the multiplication with an arbitrary constant polynomial.

        A polynomial multiplied with a constant polynomial should return
        a polynomial whose coefficients multiplied with the scalar coefficient.
        """
        # Get the polynomial
        poly = rand_poly_mnp_all

        # Create a constant polynomial
        exponents = np.zeros((1, poly.spatial_dimension), dtype=np.int_)
        mi = MultiIndexSet(exponents, poly.multi_index.lp_degree)
        coeffs = np.random.rand(1, len(poly))
        poly_constant = poly.__class__(mi, coeffs)

        # Multiplication
        poly_prod_1 = poly * poly_constant
        poly_prod_2 = poly_constant * poly

        # Assertions
        assert poly_prod_1 == poly_prod_2
        assert poly_prod_2 == poly_prod_1
        assert np.all(poly_prod_1.coeffs == poly.coeffs * coeffs)
        assert np.all(poly_prod_2.coeffs == poly.coeffs * coeffs)

    @pytest.mark.parametrize("lp_degree", [1.0, 2.0])
    def test_separate_indices(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        lp_degree,
    ):
        """Test the multiplication of polynomials with separate indices."""
        # Create multi-indices
        n_1 = PolyDegree
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, n_1, lp_degree)
        n_2 = PolyDegree + 1
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, n_2, lp_degree)

        # Create Grid instances
        # NOTE: All w.r.t lp-degree inf as it is the superset
        #       of the indices above
        grd_1 = Grid.from_degree(SpatialDimension, n_1, np.inf)
        grd_2 = Grid.from_degree(SpatialDimension, n_2, np.inf)

        # Create a random coefficient
        coeffs_1 = np.random.rand(len(mi_1))
        coeffs_2 = np.random.rand(len(mi_2))

        # Create polynomial_instances
        poly_1 = poly_class_all(mi_1, coeffs_1, grid=grd_1)
        poly_2 = poly_class_all(mi_2, coeffs_2, grid=grd_2)

        # LagrangePolynomial is a special case: only supports constant poly
        if poly_class_all is LagrangePolynomial:
           pytest.skip(
                "Skipping the multiplication test between non-constant "
                f"{poly_class_all} of separate indices."
            )

        # Multiply the polynomials
        poly_prod = poly_1 * poly_2

        # Evaluation
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension)
        yy_test = poly_1(xx_test) * poly_2(xx_test)
        yy_prod = (poly_1 * poly_2)(xx_test)

        # Assertions
        assert poly_prod.indices_are_separate
        assert np.allclose(yy_test, yy_prod)


class TestPolyAdditionSubtraction:
    """All tests related to polynomial-polynomial addition and subtraction
    and for behaviors that are common to instances of all concrete
    polynomial classes.
    """

    @pytest.mark.parametrize("invalid_value", ["123", 1 + 1j, [1, 2, 3]])
    def test_invalid_type(self, rand_poly_mnp_all, invalid_value):
        """Addition and subtraction of polynomials with an invalid type
        raises an exception.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Assertions
        with pytest.raises(TypeError):
            # Left-sided subtraction
            print(poly - invalid_value)

        with pytest.raises(TypeError):
            # Right-sided subtraction
            print(invalid_value - poly)

        with pytest.raises(TypeError):
            # Left-sided addition
            print(poly + invalid_value)

        with pytest.raises(TypeError):
            # Right-sided addition
            print(invalid_value + poly)

    def test_inconsistent_num_polys(
        self,
            poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Addition and subtraction of polynomials with an inconsistent shape
        of the respective coefficients raises an exception.
        """
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create random coefficient sets
        coeffs_1 = np.random.rand(len(mi), num_polynomials)
        coeffs_2 = np.random.rand(len(mi), num_polynomials+1)

        # Create polynomials
        poly_1 = poly_class_all(mi, coeffs_1)
        poly_2 = poly_class_all(mi, coeffs_2)

        # Assertions
        with pytest.raises(ValueError):
            # Addition
            print(poly_1 + poly_2)
        with pytest.raises(ValueError):
            # Subtraction
            print(poly_1 - poly_2)

    def test_non_matching_domain(
        self,
        poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Addition and subtraction of polynomials with a non-matching domain
        raises an exception.
        """
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a random set of coefficients
        coeffs = np.random.rand(len(mi))

        # Create a polynomial instance
        domain_1 = np.ones((2, SpatialDimension))
        domain_1[0, :] *= -2
        domain_1[1, :] *= 2
        poly_1 = poly_class_all(mi, coeffs, user_domain=domain_1)
        domain_2 = np.ones((2, SpatialDimension))
        domain_2[0, :] *= -0.5
        domain_2[1, :] *= 0.5
        poly_2 = poly_class_all(mi, coeffs, user_domain=domain_2)

        # Assertions
        with pytest.raises(ValueError):
            # Addition
            print(poly_1 + poly_2)
        with pytest.raises(ValueError):
            # Subtraction
            print(poly_1 - poly_2)


class TestPolyAddition:
    """All tests related to polynomial-polynomial addition for all concrete
    polynomial classes.
    """
    def test_self_once(self, rand_poly_mnp_no_lag):
        """Test adding a polynomial with itself, once.

        Notes
        -----
        - Instances of `LagrangePolynomial` are excluded from this test;
          they are tested from its dedicated test module.
        """
        # Get the polynomial
        poly = rand_poly_mnp_no_lag

        # Self addition
        poly_sum = poly + poly

        # Assertion
        assert poly_sum == 2 * poly

    def test_self_twice(self, rand_poly_mnp_no_lag):
        """Test adding a polynomial with itself, twice.

        Notes
        -----
        - Instances of `LagrangePolynomial` are excluded from this test;
          they are tested from its dedicated test module.
        """
        # Get the polynomial
        poly = rand_poly_mnp_no_lag

        # Self addition
        poly_sum = poly + poly + poly

        # Assertion
        assert poly_sum == 3 * poly

    def test_self_thrice(self, rand_poly_mnp_no_lag):
        """Test adding a polynomial with itself, thrice.

        Notes
        -----
        - Instances of `LagrangePolynomial` are excluded from this test;
          they are tested from its dedicated test module.
        """
        # Get the polynomial
        poly = rand_poly_mnp_no_lag

        # Self addition
        poly_sum = poly + poly + poly + poly

        # Assertion
        assert poly_sum == 4 * poly

    def test_scalar_poly_same_dim(self, rand_poly_mnp_no_lag):
        """Test adding a scalar polynomial of the same dimension.

        Notes
        -----
        - Scalar polynomials have one element multi-index of
          :math:`(0, \ldots, 0)` that both defines the polynomial and the
          underlying grid.
        - Instances of `LagrangePolynomial` are excluded from this test.
        """
        # Get the pair of random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Create a scalar polynomial with the same dimension
        m = poly.multi_index.spatial_dimension
        p = poly.multi_index.lp_degree
        poly_scalar = poly.__class__.from_degree(m, 0, p)
        scalar = np.random.rand(1)[0]
        # Repeat the scalar column-wise to match the length of the polynomial
        poly_scalar.coeffs = np.repeat(scalar, len(poly))[np.newaxis, :]

        # Add the polynomial
        poly_sum = poly + poly_scalar

        # Assertion
        assert poly_sum == poly + scalar

    def test_scalar_poly_diff_dim(self, rand_poly_mnp_no_lag):
        """Test adding a scalar polynomial of the higher dimension

        Notes
        -----
        - Scalar polynomials have one element multi-index of
          :math:`(0, \ldots, 0)` that both defines the polynomial and the
          underlying grid.
        - Instances of `LagrangePolynomial` are excluded from this test.
        """
        # Get the pair of random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Create a scalar polynomial with the higher dimension
        m = poly.multi_index.spatial_dimension + 1
        p = poly.multi_index.lp_degree
        poly_scalar = poly.__class__.from_degree(m, 0, p)
        scalar = np.random.rand(1)[0]
        # Repeat the scalar column-wise to match the length of the polynomial
        poly_scalar.coeffs = np.repeat(scalar, len(poly))[np.newaxis, :]

        # Add the polynomial
        poly_sum = poly + poly_scalar

        # Assertion
        assert poly_sum != poly + scalar
        assert poly_sum == (poly + scalar).expand_dim(m)

    def test_eval(self, rand_poly_mnp_no_lag_pair):
        """Test the evaluation of summed polynomial.

        Notes
        -----
        - Instances of `LagrangePolynomial` are excluded.
        """
        # Get the polynomial pairs
        poly_1, poly_2 = rand_poly_mnp_no_lag_pair

        # Get the maximum dimension
        dim_1 = poly_1.spatial_dimension
        dim_2 = poly_2.spatial_dimension
        dim = np.max([dim_1, dim_2])

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, dim)

        # Compute the reference results
        yy_r1 = poly_1(xx_test[:, :dim_1])
        yy_r2 = poly_2(xx_test[:, :dim_2])
        yy_ref = yy_r1 + yy_r2

        # Summed a polynomial
        yy_1 = (poly_1 + poly_2)(xx_test)
        yy_2 = (poly_2 + poly_1)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)

    @pytest.mark.parametrize("lp_degree", [1.0, 2.0])
    def test_separate_indices(
        self,
        poly_class_no_lag,
        SpatialDimension,
        PolyDegree,
        lp_degree,
    ):
        """Test the addition of polynomials with separate indices.

        Notes
        -----
        - Instances of `LagrangePolynomials` are excluded.
        """
        # Create multi-indices
        n_1 = PolyDegree
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, n_1, lp_degree)
        n_2 = PolyDegree + 1
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, n_2, lp_degree)

        # Create Grid instances
        # NOTE: All w.r.t lp-degree inf as it is the superset
        #       of the indices above
        grd_1 = Grid.from_degree(SpatialDimension, n_1, np.inf)
        grd_2 = Grid.from_degree(SpatialDimension, n_2, np.inf)

        # Create a random coefficient
        coeffs_1 = np.random.rand(len(mi_1))
        coeffs_2 = np.random.rand(len(mi_2))

        # Create polynomial_instances
        poly_1 = poly_class_no_lag(mi_1, coeffs_1, grid=grd_1)
        poly_2 = poly_class_no_lag(mi_2, coeffs_2, grid=grd_2)

        # Add the polynomials
        poly_add = poly_1 + poly_2

        # Evaluation
        xx_test = -1 + 2 * np.random.rand(5, SpatialDimension)
        yy_test = poly_1(xx_test) + poly_2(xx_test)
        yy_add = (poly_1 + poly_2)(xx_test)

        # Assertions
        assert poly_add.indices_are_separate
        assert np.allclose(yy_test, yy_add)


class TestScalarAddition:
    """All tests related to polynomial-scalar addition."""
    def test_sanity(self, rand_poly_mnp_all):
        """Test adding additive identity."""
        # Get the polynomial
        poly = rand_poly_mnp_all

        # Assertions (addition with additive identity)
        assert poly == poly + 0
        assert poly == 0 + poly  # Commutativity must hold

    def test_add(self, rand_poly_mnp_no_lag):
        """Test adding a polynomial with an arbitrary real scalar.

        Notes
        -----
        - Polynomials in the Lagrange basis are excluded from this test as
          it follows different logic that results in a different outcome.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Generate a random scalar
        scalar = np.random.rand(1)[0]

        # Add the scalar
        poly_sum_1 = poly + scalar
        poly_sum_2 = scalar + poly  # right-sided addition (commutative)

        # Compute the reference
        coeffs_ref = poly.coeffs.copy()
        coeffs_ref[0] += scalar  # only apply to the first coefficient

        # Assertion
        assert np.all(coeffs_ref == poly_sum_1.coeffs)
        assert np.all(coeffs_ref == poly_sum_2.coeffs)

    def test_inplace(self, rand_poly_mnp_all):
        """Test in-place scalar addition: ``poly += scalar``.

        Notes
        -----
        - This operation is not explicitly supported via `__iadd__()` so
          it will fall back to `__add__()`.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_all
        poly_init = poly

        # In-place addition
        poly += 5

        #  Assertions
        assert poly is not poly_init   # A new instance has been created
        assert poly == poly_init + 5

    def test_inplace_right(self, rand_poly_mnp_all):
        """Test in-place scalar addition from the right side.

        This is to verify: ``scalar += poly``.

        Notes
        -----
        - This operation will fall back to `__radd__()`.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_all
        poly_init = poly
        scalar = 5

        # In-place addition
        scalar += poly

        #  Assertions
        assert scalar is not poly_init   # The scalar becomes a polynomial
        assert scalar == poly_init + 5

    def test_eval(self, rand_poly_mnp_no_lag):
        """Test the evaluation of a polynomial summed with a scalar.

        Notes
        -----
        - Due to evaluation, instances of `LagrangePolynomial` are excluded.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, poly.spatial_dimension)

        # Compute reference results
        yy_ref = poly(xx_test) + 5

        # Summed a polynomial
        yy_1 = (poly + 5)(xx_test)
        yy_2 = (5 + poly)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)


class TestPolySubtraction:
    """All tests related to polynomial-polynomial subtraction for all concrete
    polynomial classes.
    """
    def test_self_once(self, rand_poly_mnp_no_lag):
        """Test subtracting a polynomial with itself, once.

        Notes
        -----
        - Instances of `LagrangePolynomial` are excluded from this test;
          they are tested from its dedicated test module.
        """
        # Get the polynomial
        poly = rand_poly_mnp_no_lag

        # Self subtraction
        poly_sub_1 = poly - poly
        poly_sub_2 = poly + (-poly)

        # Assertions
        assert poly_sub_1 == poly_sub_2
        assert poly_sub_1 == 0 * poly
        assert poly_sub_2 == 0 * poly

    def test_self_twice(self, rand_poly_mnp_no_lag):
        """Test subtracting a polynomial with itself, twice.

        Notes
        -----
        - Instances of `LagrangePolynomial` are excluded from this test;
          they are tested from its dedicated test module.
        """
        # Get the polynomial
        poly = rand_poly_mnp_no_lag

        # Self subtraction
        poly_sub = poly - poly - poly

        # Assertions
        assert poly_sub == -1 * poly
        assert poly_sub == -poly

    def test_self_thrice(self, rand_poly_mnp_no_lag):
        """Test subtracting a polynomial with itself, thrice.

        Notes
        -----
        - Instances of `LagrangePolynomial` are excluded from this test;
          they are tested from its dedicated test module.
        """
        # Get the polynomial
        poly = rand_poly_mnp_no_lag

        # Self subtraction
        poly_sub = poly - poly - poly - poly

        # Assertions
        assert poly_sub == -2 * poly

    def test_scalar_poly_same_dim(self, rand_poly_mnp_no_lag):
        """Test subtracting a scalar polynomial of the same dimension.

        Notes
        -----
        - Scalar polynomials have one element multi-index of
          :math:`(0, \ldots, 0)` that both defines the polynomial and the
          underlying grid.
        - Instances of `LagrangePolynomial` are excluded from this test.
        """
        # Get the pair of random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Create a scalar polynomial with the same dimension
        m = poly.multi_index.spatial_dimension
        p = poly.multi_index.lp_degree
        poly_scalar = poly.__class__.from_degree(m, 0, p)
        scalar = np.random.rand(1)[0]
        # Repeat the scalar column-wise to match the length of the polynomial
        poly_scalar.coeffs = np.repeat(scalar, len(poly))[np.newaxis, :]

        # Add the polynomial
        poly_sum = poly - poly_scalar

        # Assertion
        assert poly_sum == poly - scalar

    def test_scalar_poly_diff_dim(self, rand_poly_mnp_no_lag):
        """Test subtracting a scalar polynomial of the higher dimension

        Notes
        -----
        - Scalar polynomials have only one element in the multi-index set
          (:math:`(0, \ldots, 0)`) that both defines the polynomial and
          the underlying grid.
        - Instances of `LagrangePolynomial` are excluded from this test.
        """
        # Get the pair of random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Create a scalar polynomial with the higher dimension
        m = poly.multi_index.spatial_dimension + 1
        p = poly.multi_index.lp_degree
        poly_scalar = poly.__class__.from_degree(m, 0, p)
        scalar = np.random.rand(1)[0]
        # Repeat the scalar column-wise to match the length of the polynomial
        poly_scalar.coeffs = np.repeat(scalar, len(poly))[np.newaxis, :]

        # Add the polynomial
        poly_sum = poly - poly_scalar

        # Assertion
        assert poly_sum != poly - scalar
        assert poly_sum == (poly - scalar).expand_dim(m)

    def test_eval(self, rand_poly_mnp_no_lag_pair):
        """Test the evaluation of subtracted polynomial.

        Notes
        -----
        - Instances of `LagrangePolynomial` are excluded from this test.
        """
        # Get the polynomial pairs
        poly_1, poly_2 = rand_poly_mnp_no_lag_pair

        # Get the maximum dimension
        dim_1 = poly_1.spatial_dimension
        dim_2 = poly_2.spatial_dimension
        dim = np.max([dim_1, dim_2])

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, dim)

        # Compute the reference results
        yy_r1 = poly_1(xx_test[:, :dim_1])
        yy_r2 = poly_2(xx_test[:, :dim_2])
        yy_ref = yy_r1 - yy_r2

        # Summed a polynomial
        yy_1 = (poly_1 - poly_2)(xx_test)
        yy_2 = (-poly_2 + poly_1)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)

    @pytest.mark.parametrize("lp_degree", [1.0, 2.0])
    def test_separate_indices(
        self,
        poly_class_no_lag,
        SpatialDimension,
        PolyDegree,
        lp_degree,
    ):
        """Test the subtraction of polynomials with separate indices."""
        # Create multi-indices
        n_1 = PolyDegree
        mi_1 = MultiIndexSet.from_degree(SpatialDimension, n_1, lp_degree)
        n_2 = PolyDegree + 1
        mi_2 = MultiIndexSet.from_degree(SpatialDimension, n_2, lp_degree)

        # Create Grid instances
        # NOTE: w.r.t lp-degree inf as it is the superset of the index sets
        #       defined above
        grd_1 = Grid.from_degree(SpatialDimension, n_1, np.inf)
        grd_2 = Grid.from_degree(SpatialDimension, n_2, np.inf)

        # Create a random coefficient
        coeffs_1 = np.random.rand(len(mi_1))
        coeffs_2 = np.random.rand(len(mi_2))

        # Create polynomial_instances
        poly_1 = poly_class_no_lag(mi_1, coeffs_1, grid=grd_1)
        poly_2 = poly_class_no_lag(mi_2, coeffs_2, grid=grd_2)

        # Multiply the polynomials
        poly_sub = poly_1 - poly_2

        # Evaluation
        xx_test = -1 + 2 * np.random.rand(1000, SpatialDimension)
        yy_test = poly_1(xx_test) - poly_2(xx_test)
        yy_sub = (poly_1 - poly_2)(xx_test)

        # Assertions
        assert poly_sub.indices_are_separate
        assert np.allclose(yy_test, yy_sub)


class TestPolyAdditionSubtractionAugmented:
    """All tests related to polynomial-polynomial augmented addition and
    subtraction and for behaviors that are common to instances
    of all concrete polynomial classes.
    """

    @pytest.mark.parametrize("invalid_value", ["123", 1 + 1j, [1, 2, 3]])
    def test_invalid_type(self, rand_poly_mnp_all, invalid_value):
        """Augmented addition and subtraction of polynomials with an invalid
        type raises and exception.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Assertions
        with pytest.raises(TypeError):
            # Addition
            poly += invalid_value
        with pytest.raises(TypeError):
            # Subtraction
            poly -= invalid_value

    def test_inconsistent_num_polys(
        self,
        poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Augmented addition and subtraction of polynomials with a wrong shape
        of the respective coefficients raises an exception.
        """
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create random coefficient sets
        coeffs_1 = np.random.rand(len(mi), num_polynomials)
        coeffs_2 = np.random.rand(len(mi), num_polynomials+1)

        # Create polynomials
        poly_1 = poly_class_all(mi, coeffs_1)
        poly_2 = poly_class_all(mi, coeffs_2)

        # Assertions
        with pytest.raises(ValueError):
            # Addition
            poly_1 += poly_2
        with pytest.raises(ValueError):
            # Subtraction
            poly_1 -= poly_2

    def test_non_matching_domain(
        self,
        poly_class_all,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Augmented addition and subtraction of polynomials with a
        non-matching domain raises an exception.
        """
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a random set of coefficients
        coeffs = np.random.rand(len(mi))

        # Create a polynomial instance with different domains
        domain_1 = np.ones((2, SpatialDimension))
        domain_1[0, :] *= -2
        domain_1[1, :] *= 2
        poly_1 = poly_class_all(mi, coeffs, user_domain=domain_1)
        domain_2 = np.ones((2, SpatialDimension))
        domain_2[0, :] *= -0.5
        domain_2[1, :] *= 0.5
        poly_2 = poly_class_all(mi, coeffs, user_domain=domain_2)

        # Assertions
        with pytest.raises(ValueError):
            # Addition
            poly_1 += poly_2
        with pytest.raises(ValueError):
            # Subtraction
            poly_1 -= poly_2


class TestScalarSubtraction:
    """All tests related to polynomial-scalar subtraction."""
    def test_sanity(self, rand_poly_mnp_all):
        """Test adding additive identity."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Assertions (subtraction with additive identity)
        assert poly == poly - 0
        assert poly == -0 + poly  # commutativity must hold

    def test_sub(self, rand_poly_mnp_no_lag):
        """Test subtracting a polynomial with an arbitrary real scalar.

        This tests the expression: ``poly - scalar``
        Notes
        -----
        - Polynomials in the Lagrange basis are excluded from this test as
          it follows different logic that results in a different outcome.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Generate a random scalar
        scalar = np.random.rand(1)[0]

        # Subtract with the scalar
        poly_sum_1 = poly - scalar
        poly_sum_2 = -scalar + poly  # Commutativity must hold

        # Compute the reference
        coeffs_ref = poly.coeffs.copy()
        coeffs_ref[0] -= scalar  # only apply to the first coefficient

        # Assertion
        assert np.all(coeffs_ref == poly_sum_1.coeffs)
        assert np.all(coeffs_ref == poly_sum_2.coeffs)

    def test_rsub(self, rand_poly_mnp_no_lag):
        """Test right-sided subtraction of a scalar with a polynomial.

        Notes
        -----
        - The test verifies the expression: ``scalar - poly``.
        - Polynomials in the Lagrange basis are excluded from this test as
          it follows different logic that results in a different outcome.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Generate a random scalar
        scalar = np.random.rand(1)[0]

        # Subtract with the scalar
        poly_sum_1 = scalar - poly
        poly_sum_2 = -poly + scalar  # Commutativity must hold

        # Compute the reference
        coeffs_ref = -1 * poly.coeffs.copy()
        coeffs_ref[0] += scalar  # only apply to the first coefficient

        # Assertion
        assert np.all(coeffs_ref == poly_sum_1.coeffs)
        assert np.all(coeffs_ref == poly_sum_2.coeffs)

    def test_inplace(self, rand_poly_mnp_all):
        """Test in-place scalar subtraction.

         Notes
        -----
        - This operation is not explicitly supported via `__isub__()` so
          it will fall back to `__add__()` with negated scalar operand.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_all
        poly_init = poly

        # In-place subtraction
        poly -= 5

        # Assertions
        assert poly is not poly_init  # A new instance has been created
        assert poly == poly_init - 5

    def test_inplace_right(self, rand_poly_mnp_all):
        """Test in-place scalar subtraction from the right side.

        Notes
        -----
        - This test verifies the statement: ``scalar -= poly``.
        - This operation will fall back to `__radd__()` with a negated
          scalar operand.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_all
        poly_init = poly
        scalar = 5

        # In-place addition
        scalar -= poly

        #  Assertions
        assert scalar is not poly_init   # The scalar becomes a polynomial
        assert scalar == -poly_init + 5

    def test_eval(self, rand_poly_mnp_no_lag):
        """Test the evaluation of a polynomial subtracted by a scalar.

        Notes
        -----
        - Due to evaluation, instances of `LagrangePolynomial` are excluded.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Generate a random set of test points
        xx_test = -1 + 2 * np.random.rand(1000, poly.spatial_dimension)

        # Compute reference results
        yy_ref = poly(xx_test) - 5

        # Summed a polynomial
        yy_1 = (poly - 5)(xx_test)
        yy_2 = (-5 + poly)(xx_test)

        # Assertions
        assert np.allclose(yy_ref, yy_1)
        assert np.allclose(yy_ref, yy_2)


class TestExponentiation:
    """All tests related to the exponentiation of a polynomial."""

    @pytest.mark.parametrize("invalid_value", [-1, 1.1, 0.5])
    def test_by_invalid_value(self, rand_poly_mnp_all, invalid_value):
        """Test polynomial exponentiation by an invalid value."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Exponentiation
        with pytest.raises(ValueError):
            _ = poly**invalid_value

    @pytest.mark.parametrize("invalid_type", ["ab", 1+1j, np.array([1, 2, 3])])
    def test_by_invalid_type(self, rand_poly_mnp_all, invalid_type):
        """Test polynomial exponentiation by an invalid type."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Exponentiation
        with pytest.raises(TypeError):
            _ = poly ** invalid_type

    def test_by_one(self, rand_poly_mnp_all):
        """Test polynomial exponentiation by one."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Exponentation
        poly_exp = poly**1

        # Assertion
        assert poly_exp == poly

    def test_by_zero(self, rand_poly_mnp_all):
        """Test polynomial exponentation by zero."""
        # Get a random polynomial instance
        poly = rand_poly_mnp_all

        # Exponentiation
        poly_exp = poly**0

        # Assertions
        assert poly_exp == poly * 0 + 1

    def test_by_three(self, rand_poly_mnp_no_lag):
        """Test polynomial exponentation by three.

        Notes
        -----
        - Instances of `LagrangePolynomial` are excluded because the class
          does not support general polynomial-polynomial multiplication.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Exponentation
        poly_exp = poly ** 3.0

        # Assertion
        assert poly_exp == poly * poly * poly

    def test_by_two_eval(self, rand_poly_mnp_no_lag):
        """Test the evaluation of an exponentiated polynomial (by two).

         Notes
        -----
        - Instances of `LagrangePolynomial` are excluded because the class
          does not support general polynomial-polynomial multiplication.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp_no_lag

        # Exponentiation
        poly_exp = poly**2

        # Generate random test points
        xx_test = -1 + 2 * np.random.rand(5, poly.spatial_dimension)
        yy_test = poly_exp(xx_test)
        yy_ref = poly(xx_test) * poly(xx_test)

        # Assertion
        assert np.allclose(yy_ref, yy_test)

    def test_non_downward_closed(
        self,
        poly_class_all,
        multi_index_non_downward_closed,
    ):
        """Test exponentiation by 0 and 1 of non-downward-closed polynomial.

        Notes
        -----
        - Regardless of the downward-closedness of the underlying multi-index
          set, all polynomials may be exponentiated by 0 or 1. Beyond that,
          it depends on whether the basis allows it.
        """
        # Get a non-downward closed multi-index set
        mi = multi_index_non_downward_closed

        # Create random coefficients
        coeffs = np.random.rand(len(mi))

        # Create a random polynomial
        poly = poly_class_all(mi, coeffs)

        # Exponentiation by 0
        poly_exp = poly**0

        # Assertion
        assert poly_exp == poly * 0 + 1

        # Exponentiation by 1
        poly_exp = poly**1

        # Assertion
        assert poly_exp == poly
