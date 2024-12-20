"""
Testing module for Transformation classes.
"""
import numpy as np
import pytest

from conftest import (
    assert_polynomial_almost_equal,
    build_rnd_coeffs,
    POLY_CLASSES,
)

from minterpy import MultiIndexSet
from minterpy.core.ABC import OperatorABC, TransformationABC
from minterpy.transformations import (
    Identity,
    get_transformation,
    get_transformation_class,
)


class TestInitialization:
    """All tests related to the initialization of a transformation class."""

    def test_success(
        self,
        transformation_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test successful initialization."""
        # Create a polynomial of the origin type
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        origin_poly = transformation_class.origin_type(mi)

        # Create a transformation instance
        transform = transformation_class(origin_poly)

        # Assertions
        # The type of the origin polynomial is consistent
        assert isinstance(transform.origin_poly, transform.origin_type)
        # Transformation operator instance exists
        assert isinstance(transform.transformation_operator, OperatorABC)

    def test_failure(
        self,
        transformation_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test a failed initialization."""
        # Create a polynomial of the origin type
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        # Intentionally pick another type of origin poly
        origin_polys = [
            _ for _ in POLY_CLASSES if _ != transformation_class.origin_type
        ]
        idx = np.random.choice(len(origin_polys), 1)[0]
        origin_poly = origin_polys[idx](mi)

        # Assertion
        with pytest.raises(TypeError):
            # Create a transformation instance with inconsistent origin poly.
            transformation_class(origin_poly)


class TestTransformation:
    """All tests related to carrying out the polynomial transformation."""

    def test_call_success(
        self,
        transformation_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test successful calling of an instance to transform the origin poly.
        """
        # Create a polynomial of the origin type
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)
        origin_poly = transformation_class.origin_type(mi, coeffs)

        # Create a transformation instance
        transform = transformation_class(origin_poly)

        # Call the instance
        target_poly = transform()

        # Assertion
        assert isinstance(target_poly, transform.target_type)

    def test_call_failure(
        self,
        transformation_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test failed calling of an instance to transform the origin poly."""
        # Create a polynomial of the origin type
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        # Polynomial without coefficients
        origin_poly = transformation_class.origin_type(mi)

        # Create a transformation instance
        transform = transformation_class(origin_poly)

        # Call the instance
        with pytest.raises(ValueError):
            # Polynomial without coefficients can't be transformed
            transform()

    def test_transform(
        self,
        transformation_class,
        SpatialDimension,
        PolyDegree,
        LpDegree,
    ):
        """Test an alternative approach to transformation."""
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)
        origin_poly = transformation_class.origin_type(mi, coeffs)

        # Create a transformation instance
        transform = transformation_class(origin_poly)

        # Call the instance
        target_poly = transform()

        # Transform the coefficients of the origin poly. manually
        transformed_coeffs = transform.transformation_operator @ coeffs

        # Assertion
        assert np.allclose(transformed_coeffs, target_poly.coeffs)

    def test_identity(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
            poly_class_all,
    ):
        """Test the identity transformation."""
        # Create a polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        origin_poly = poly_class_all(mi, coeffs)

        # Create an identity transformer
        transformer = Identity(origin_poly)
        operator = transformer.transformation_operator.array_repr_full
        target_poly = transformer()

        # Assertions
        # no changes to the polynomial
        assert_polynomial_almost_equal(origin_poly, target_poly)
        # the operator is the identity matrix
        assert np.allclose(operator, np.eye(len(mi)))

    def test_back_n_forth(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        origin_type,
        target_type
):
        """Test the forward and backward transformation."""
        # Create a polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = build_rnd_coeffs(mi)
        origin_poly = origin_type(mi, coeffs)

        # Forward transformer
        fwd_transformer = get_transformation(origin_poly, target_type)
        target_poly = fwd_transformer()

        # Backward transformer
        bwd_transformer = get_transformation(target_poly, origin_type)
        origin_poly_recovered = bwd_transformer()

        # Assertions
        assert isinstance(target_poly, target_type)
        assert_polynomial_almost_equal(origin_poly, origin_poly_recovered)


class TestGetTransformation:
    """All tests related to the helper function get_transformation()."""

    def test_get_transformation(
        self,
        SpatialDimension,
        PolyDegree,
        LpDegree,
        origin_type,
        target_type,
    ):
        """Test getting the transformation instance."""
        # Create an origin polynomial
        mi = MultiIndexSet.from_degree(SpatialDimension, PolyDegree, LpDegree)
        coeffs = np.arange(len(mi), dtype=float)
        origin_poly = origin_type(mi, coeffs)

        # Get the relevant transformation
        transform = get_transformation(origin_poly, target_type)

        # Assertions
        if origin_type == target_type:
            assert isinstance(transform, Identity)
        else:
            assert isinstance(transform, TransformationABC)
            assert transform.target_type == target_type

    def test_get_transformation_class_success(self, origin_type, target_type):
        """Test successfully getting the transformation class."""
        trans_class = get_transformation_class(origin_type, target_type)

        # Assertions
        if origin_type == target_type:
            assert trans_class == Identity
        else:
            assert trans_class.origin_type == origin_type
            assert trans_class.target_type == target_type

    def test_get_transformation_class_no_target(self, origin_type):
        """Test unsuccessfully getting the transformation class."""
        target_type = None
        with pytest.raises(NotImplementedError):
            get_transformation_class(origin_type, target_type)

    def test_get_transformation_class_no_origin(self, target_type):
        """Test unsuccessfully getting the transformation class."""
        origin_type = None
        with pytest.raises(NotImplementedError):
            get_transformation_class(origin_type, target_type)
