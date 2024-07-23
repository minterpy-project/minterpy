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
    polynomial_class,
    create_mi_pair_distinct,
    POLY_CLASSES,
)

from minterpy import (
    LagrangePolynomial,
    MultiIndexSet,
    Grid,
)


class TestInitialization:
    """All tests related to the initialization of a polynomial instance."""

    def test_without_coeffs(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization without coefficients."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        assert_call(polynomial_class, mi)

    def test_with_coeffs(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization with coefficients."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)

        assert_call(polynomial_class, mi, coeffs)

    def test_with_grid(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test initialization with a valid grid."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        grd = Grid(mi)
        coeffs = np.arange(len(mi), dtype=float)

        assert_call(polynomial_class, mi, grid=grd)
        assert_call(polynomial_class, mi, coeffs, grid=grd)

    def test_with_invalid_grid(
        self,
        polynomial_class,
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
            polynomial_class(mi, coeffs, grid=grd)

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_set(self, spatial_dimension, polynomial_class, LpDegree):
        """Test initialization with an empty multi-index set."""
        # Create an empty set
        mi = MultiIndexSet(np.empty((0, spatial_dimension)), LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            polynomial_class(mi)


class TestFrom:
    """All tests related to the different factory methods."""
    def test_from_grid_uninit(self, polynomial_class, grid_mnp):
        """Test creating an uninitialized polynomial from a Grid instance."""
        # Create a polynomial instance
        poly = polynomial_class.from_grid(grid_mnp)

        # Assertions
        assert poly.grid == grid_mnp
        assert poly.multi_index == grid_mnp.multi_index

    def test_from_grid_init(self, polynomial_class, grid_mnp):
        """Test creating an initialized polynomial from a Grid instance."""
        # Generate random coefficients to initialize the polynomial
        coeffs = np.random.rand(len(grid_mnp.multi_index))
        # Create a polynomial instance
        # From grid - grid the same as default
        poly_1 = polynomial_class.from_grid(grid_mnp, coeffs)
        # Default constructor - default grid
        poly_2 = polynomial_class(grid_mnp.multi_index, coeffs)
        # Default constructor - grid the same as default
        poly_3 = polynomial_class(grid_mnp.multi_index, coeffs, grid=grid_mnp)

        # Assertions
        assert poly_1 == poly_2
        assert poly_1 == poly_3


class TestEquality:
    """All tests related to the equality check between polynomial instances."""

    def test_single(
        self,
        polynomial_class,
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
        poly_1 = polynomial_class(mi, coeffs)
        poly_2 = polynomial_class(mi, coeffs)

        # Assertions
        assert poly_1 is not poly_2  # Not identical instances
        assert poly_1 == poly_2  # But equal in values
        assert poly_2 == poly_1  # Symmetric property

    def test_multiple(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        num_polynomials,
    ):
        """Test equality between two instances with multiple sets of coeffs."""
        # Create a common multi-index set
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)\

        # Generate random coefficient values
        coeffs = np.random.rand(len(mi), num_polynomials)

        # Create two equal polynomials
        poly_1 = polynomial_class(mi, coeffs)
        poly_2 = polynomial_class(mi, coeffs)

        # Assertions
        assert poly_1 is not poly_2  # Not identical instances
        assert poly_1 == poly_2  # But equal in values
        assert poly_2 == poly_1  # Symmetric property


class TestInequality:
    """All tests related to the inequality check of two poly. instances."""

    def test_inconsistent_types(self, polynomial_class):
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
        poly = polynomial_class(mi, coeffs)

        # Assertions
        assert poly != mi
        assert poly != coeffs
        assert poly != "123"
        assert poly != 1
        assert poly != 10.0

    def test_different_multi_index(self, polynomial_class):
        """Test inequality due to different multi-index sets."""
        # Create two different multi-index sets
        mi_1, mi_2 = create_mi_pair_distinct()

        # Generate two sets of coefficients
        coeffs_1 = np.random.rand(len(mi_1))
        coeffs_2 = np.random.rand(len(mi_2))

        # Create two polynomials
        poly_1 = polynomial_class(mi_1, coeffs_1)
        poly_2 = polynomial_class(mi_2, coeffs_2)

        # Assertions
        assert poly_1 != poly_2
        assert poly_2 != poly_1

    def test_different_grids(
        self,
        polynomial_class,
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
        poly_1 = polynomial_class(mi, coeffs, grid=grd_1)
        poly_2 = polynomial_class(mi, coeffs, grid=grd_2)

        # Assertions
        assert poly_1 != poly_2
        assert poly_2 != poly_1

    def test_different_coeffs(
        self,
        polynomial_class,
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
        poly_1 = polynomial_class(mi, coeffs_1)
        poly_2 = polynomial_class(mi, coeffs_2)

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
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test the evaluation of an 'initialized' polynomial."""
        # Create a polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)
        poly = polynomial_class(mi, coeffs)

        xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
        if isinstance(poly, LagrangePolynomial):
            with pytest.raises(NotImplementedError):
                poly(xx_test)
        else:
            assert_call(poly, xx_test)

    def test_without_coeffs(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test the evaluation of an 'uninitialized' polynomial."""
        # Create a polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        poly = polynomial_class(mi)

        xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
        if isinstance(poly, LagrangePolynomial):
            with pytest.raises(NotImplementedError):
                poly(xx_test)
        else:
            with pytest.raises(ValueError):
                poly(xx_test)

    def test_multiple_coeffs(
        self,
        polynomial_class,
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
        poly = polynomial_class(mi, coeffs)

        xx_test = -1 + 2 * np.random.rand(10, SpatialDimension)
        if isinstance(poly, LagrangePolynomial):
            with pytest.raises(NotImplementedError):
                poly(xx_test)
        else:
            yy_test = poly(xx_test)
            for i in range(num_polynomials):
                # Due to identical coefficients, results are identical
                np.all(yy_test[:, i] == yy_test[:, 0])


class TestNegation:
    """All tests related to the negation of a polynomial instance."""
    def test_neg(self, rand_poly_mnp):
        """Test the expected results from negating a polynomial."""
        # Get a random polynomial instance
        poly = rand_poly_mnp

        # Negate the polynomial
        poly_neg = -poly

        # Assertions
        assert poly_neg is not poly  # Must not be the same instance
        assert poly_neg != -poly_neg
        assert poly_neg.multi_index == poly.multi_index
        assert poly_neg.grid == poly.grid
        assert np.all(poly_neg.coeffs == -1 * poly.coeffs)

    def test_neg_multi_poly(self, rand_polys_mnp):
        """Test the expected results from negating multiple polynomials."""
        # Get a random polynomial instance
        poly = rand_polys_mnp

        # Negate the polynomial
        poly_neg = -poly

        # Assertions
        assert poly_neg is not poly  # Must not be the same instance
        assert poly_neg != -poly_neg
        assert poly_neg.multi_index == poly.multi_index
        assert poly_neg.grid == poly.grid
        assert np.all(poly_neg.coeffs == -1 * poly.coeffs)

    def test_sanity_multiplication(self, rand_poly_mnp):
        """Test that negation is equivalent to multiplication with -1."""
        # Get a random polynomial instance
        poly = rand_poly_mnp

        # Negate the polynomial
        poly_neg_1 = -poly  # via negation
        poly_neg_2 = -1 * poly  # via negative multiplication

        # Assertions
        assert poly_neg_1 is not poly_neg_2
        assert poly_neg_1 == poly_neg_2

    def test_different_grid(
        self,
        polynomial_class,
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
        poly = polynomial_class(mi, coeffs, grid=grd)

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
    def test_single_poly(self, rand_poly_mnp):
        """Test using the unary positive operator on a polynomial with a single
        set of coefficients.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp

        # Assertions
        assert poly is (+poly)  # Object identity
        assert poly == (+poly)  # Equality in value

    def test_multiple_polys(self, rand_polys_mnp):
        """Test using the unary positive operator on a polynomial with multiple
        sets of coefficients.
        """
        # Get a random polynomial instance
        poly = rand_polys_mnp

        # Assertions
        assert poly is (+poly)  # Object identity
        assert poly == (+poly)  # Equality in value


class TestHasMatchingDomain:
    """All tests related to method to check if polynomial domains match."""
    def test_sanity(self, rand_poly_mnp):
        """Test if a polynomial has a matching domain with itself."""
        # Get a random polynomial instance
        poly = rand_poly_mnp

        # Assertion
        assert poly.has_matching_domain(poly)

    def test_same_dim(self, rand_poly_mnp):
        """Test if poly. has a matching domain with another of the same dim."""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp
        poly_2 = copy.copy(rand_poly_mnp)

        # Assertions
        assert poly_1.has_matching_domain(poly_2)
        assert poly_2.has_matching_domain(poly_1)

    def test_diff_dim(
        self,
        polynomial_class,
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
        poly_1 = polynomial_class(mi_1)
        poly_2 = polynomial_class(mi_2)

        # Assertion
        assert poly_1.has_matching_domain(poly_2)
        assert poly_2.has_matching_domain(poly_1)

    def test_user_domain(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test if poly. does not have a matching user domain."""
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a polynomial instance
        user_domain_1 = np.ones((SpatialDimension, 2))
        user_domain_1[:, 0] *= -2
        user_domain_1[:, 1] *= 2
        poly_1 = polynomial_class(mi, user_domain=user_domain_1)
        user_domain_2 = np.ones((SpatialDimension, 2))
        user_domain_2[:, 0] *= -0.5
        user_domain_2[:, 1] *= 0.5
        poly_2 = polynomial_class(mi, user_domain=user_domain_2)

        # Assertion
        assert not poly_1.has_matching_domain(poly_2)
        assert not poly_2.has_matching_domain(poly_1)

    def test_internal_domain(
        self,
        polynomial_class,
        SpatialDimension,
        PolyDegree,
        LpDegree
    ):
        """Test if poly. does not have a matching internal domain."""
        # Create a MultiIndexSet
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)

        # Create a polynomial instance
        internal_domain_1 = np.ones((SpatialDimension, 2))
        internal_domain_1[:, 0] *= -2
        internal_domain_1[:, 1] *= 2
        poly_1 = polynomial_class(mi, internal_domain=internal_domain_1)
        internal_domain_2 = np.ones((SpatialDimension, 2))
        internal_domain_2[:, 0] *= -0.5
        internal_domain_2[:, 1] *= 0.5
        poly_2 = polynomial_class(mi, internal_domain=internal_domain_2)

        # Assertion
        assert not poly_1.has_matching_domain(poly_2)
        assert not poly_2.has_matching_domain(poly_1)


class TestScalarMultiplication:
    """All tests related to the multiplication of a polynomial with scalars."""
    def test_mul_identity(self, rand_poly_mnp):
        """Left-sided multiplication identity"""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp

        # Left-sided multiplication
        poly_2 = poly_1 * 1.0

        # Assertions
        assert poly_1 is not poly_2  # The instances are not identical
        assert poly_1 == poly_2
        assert poly_2 == poly_1

    def test_rmul_identity(self, rand_poly_mnp):
        """Right-sided multiplication identity."""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp

        # Left-sided multiplication
        poly_2 = 1.0 * poly_1

        # Assertions
        assert poly_1 is not poly_2  # The instances are not identical
        assert poly_1 == poly_2
        assert poly_2 == poly_1

    def test_imul_identity(self, rand_poly_mnp):
        """Augmented multiplication identity."""
        # Get a random polynomial instance
        poly = rand_poly_mnp
        old_id = id(poly)
        old_poly = copy.copy(poly)

        # Left-sided multiplication
        poly *= 1.0

        # Assertions
        assert old_id == id(poly)  # The instances are not identical
        assert old_poly == poly

    def test_mul(self, rand_poly_mnp):
        """Multiplication of polynomials with a valid scalar."""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp

        # Generate a random scalar
        scalar = np.random.random()

        # Multiply the polynomial both sides
        poly_2 = poly_1 * scalar  # Left-sided multiplication
        poly_3 = scalar * poly_1  # Right-sided multiplication

        # Assertions
        assert poly_2 is not poly_3
        assert poly_2 == poly_3
        assert poly_3 == poly_2

    def test_imul(self, rand_poly_mnp):
        """Augmented multiplication of polynomials with a valid scalar."""
        # Get a random polynomial instance
        poly_1 = rand_poly_mnp
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

    def test_multi_poly(self, rand_polys_mnp):
        """Multiplication of polynomials with multiple sets of coefficients
        with a scalar.
        """
        # Get a random polynomial instance
        poly_1 = rand_polys_mnp

        # Generate a random scalar
        scalar = np.random.random()

        # Multiply with a scalar
        poly_2 = poly_1 * scalar
        poly_3 = scalar * poly_1

        # Assertions
        assert poly_2 is not poly_3
        assert poly_2 == poly_3
        assert poly_3 == poly_2
        assert np.all(poly_1.coeffs * scalar == poly_2.coeffs)
        assert np.all(poly_1.coeffs * scalar == poly_3.coeffs)


class TestPolyMultiplication:
    """All tests related to the polynomial-polynomial multiplication and for
    behaviors that are common to all instances of concrete polynomial classes.
    """
    def test_inconsistent_num_polys(
        self,
        polynomial_class,
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
        poly_1 = polynomial_class(mi, coeffs_1)
        poly_2 = polynomial_class(mi, coeffs_2)

        # Multiplication
        with pytest.raises(ValueError):
            print(poly_1 * poly_2)

    def test_non_matching_domain(
        self,
        polynomial_class,
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
        domain_1 = np.ones((SpatialDimension, 2))
        domain_1[:, 0] *= -2
        domain_1[:, 1] *= 2
        poly_1 = polynomial_class(mi, coeffs, user_domain=domain_1)
        domain_2 = np.ones((SpatialDimension, 2))
        domain_2[:, 0] *= -0.5
        domain_2[:, 1] *= 0.5
        poly_2 = polynomial_class(mi, coeffs, user_domain=domain_2)

        # Perform multiplication
        with pytest.raises(ValueError):
            print(poly_1 * poly_2)

    @pytest.mark.parametrize("invalid_value", ["123", 1+1j, [1, 2, 3]])
    def test_invalid(self, rand_poly_mnp, invalid_value):
        """Multiplication of polynomials with an invalid type raises
        an exception.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp

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

    def test_imul(self, rand_poly_mnp):
        """Augmented multiplication of polynomials raises an exception."""
        # Get a random polynomial instance
        poly = rand_poly_mnp

        # Assertion
        with pytest.raises(NotImplementedError):
            # Perform multiplication
            poly *= poly


class TestPolyAdditionSubtraction:
    """All tests related to polynomial-polynomial addition and subtraction
    and for behaviors that are common to instances of all concrete
    polynomial classes.
    """

    @pytest.mark.parametrize("invalid_value", ["123", 1 + 1j, [1, 2, 3]])
    def test_invalid_type(self, rand_poly_mnp, invalid_value):
        """Addition and subtraction of polynomials with an invalid type
        raises an exception.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp

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
        polynomial_class,
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
        poly_1 = polynomial_class(mi, coeffs_1)
        poly_2 = polynomial_class(mi, coeffs_2)

        # Assertions
        with pytest.raises(ValueError):
            # Addition
            print(poly_1 + poly_2)
        with pytest.raises(ValueError):
            # Subtraction
            print(poly_1 - poly_2)

    def test_non_matching_domain(
        self,
        polynomial_class,
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
        domain_1 = np.ones((SpatialDimension, 2))
        domain_1[:, 0] *= -2
        domain_1[:, 1] *= 2
        poly_1 = polynomial_class(mi, coeffs, user_domain=domain_1)
        domain_2 = np.ones((SpatialDimension, 2))
        domain_2[:, 0] *= -0.5
        domain_2[:, 1] *= 0.5
        poly_2 = polynomial_class(mi, coeffs, user_domain=domain_2)

        # Assertions
        with pytest.raises(ValueError):
            # Addition
            print(poly_1 + poly_2)
        with pytest.raises(ValueError):
            # Subtraction
            print(poly_1 - poly_2)


class TestPolyAdditionSubtractionAugmented:
    """All tests related to polynomial-polynomial augmented addition and
    subtraction and for behaviors that are common to instances
    of all concrete polynomial classes.
    """

    @pytest.mark.parametrize("invalid_value", ["123", 1 + 1j, [1, 2, 3]])
    def test_invalid_type(self, rand_poly_mnp, invalid_value):
        """Augmented addition and subtraction of polynomials with an invalid
        type raises and exception.
        """
        # Get a random polynomial instance
        poly = rand_poly_mnp

        # Assertions
        with pytest.raises(TypeError):
            # Addition
            poly += invalid_value
        with pytest.raises(TypeError):
            # Subtraction
            poly -= invalid_value

    def test_inconsistent_num_polys(
        self,
        polynomial_class,
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
        poly_1 = polynomial_class(mi, coeffs_1)
        poly_2 = polynomial_class(mi, coeffs_2)

        # Assertions
        with pytest.raises(ValueError):
            # Addition
            poly_1 += poly_2
        with pytest.raises(ValueError):
            # Subtraction
            poly_1 -= poly_2

    def test_non_matching_domain(
        self,
        polynomial_class,
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
        domain_1 = np.ones((SpatialDimension, 2))
        domain_1[:, 0] *= -2
        domain_1[:, 1] *= 2
        poly_1 = polynomial_class(mi, coeffs, user_domain=domain_1)
        domain_2 = np.ones((SpatialDimension, 2))
        domain_2[:, 0] *= -0.5
        domain_2[:, 1] *= 0.5
        poly_2 = polynomial_class(mi, coeffs, user_domain=domain_2)

        # Assertions
        with pytest.raises(ValueError):
            # Addition
            poly_1 += poly_2
        with pytest.raises(ValueError):
            # Subtraction
            poly_1 -= poly_2
