"""
Test suite for ordinary polynomial regression.

This module verifies the behavior of the `OrdinaryRegression` class instances.
Tests cover initialization, fitting, and prediction.
"""
import math
import numpy as np
import pytest

from scipy.sparse.linalg import gmres

from minterpy import Domain, Grid, MultiIndexSet
from minterpy.utils.exceptions import DomainMismatchError
from minterpy.extras.regression import OrdinaryRegression
from minterpy.polynomials import (
    LagrangePolynomial,
    NewtonPolynomial,
    CanonicalPolynomial,
    ChebyshevPolynomial,
)

from conftest import assert_call, build_random_newton_polynom

#######################
# Internal functions  #
#######################

def _solve_gmres(rr: np.ndarray, yy: np.ndarray, ww: np.ndarray, **kwargs):
    """Solve a least-squares problem using a Generalized Min Resid solver."""
    if ww is None:
        coeffs, _ = gmres(rr, yy, **kwargs)
    else:
        coeffs, _ = gmres(
            rr.T @ ww @ rr, rr.T @ ww @ yy, **kwargs
        )

    return coeffs


class _EmptyClass:
    def __init__(self, multi_index, grid):
        self.multi_index = multi_index
        self.grid = grid
        self.exponents = None
        self.generating_points = None

##################################
# Module variables and fixtures  #
##################################

# --- Least-squares solver keyword arguments (These are the defaults)
LSQ_SOLVER_ARGS = {
    "lstsq": {"cond": None, "check_finite": True},
    "pinv": {"rcond": 1e-15},
    "dgesv": {"overwrite_a": False},
    "dsysv": {"lower": False},
    "dposv": {"overwrite_b": False},
}

# --- Fixtures for least-squares solver
lst_sqr_solvers = [
    "lstsq",
    "inv",
    "pinv",
    "dgesv",
    "dsysv",
    "dposv",
    "qr",
    "svd",
    _solve_gmres,
]


@pytest.fixture(params=lst_sqr_solvers)
def lst_sqr_solver(request):
    return request.param


# --- Fixtures for an origin polynomial basis
origin_polys = [
    LagrangePolynomial,
    NewtonPolynomial,
    CanonicalPolynomial,
    ChebyshevPolynomial,
]


@pytest.fixture(params=origin_polys)
def origin_poly(request):
    return request.param


# --- Fixture for random Newton polynomial
@pytest.fixture
def rand_newton_poly(grid_mnp):
    """Create a random Newton polynomial."""
    mi = grid_mnp.multi_index
    m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree

    return build_random_newton_polynom(m, n, p, domain=grid_mnp.domain)

##########
# Tests  #
##########

class TestInit:
    """All tests related to the initialization of the class."""

    def test_with_multi_index(self, multi_index_mnp, domain):
        """Test the initialization with a multi-index set."""

        # Create a grid from the multi-index set
        grd = Grid(multi_index_mnp, domain=domain)

        # Create an instance of ordinary regression
        reg = OrdinaryRegression(multi_index_mnp, domain=domain)

        # Assertions
        assert reg.multi_index == multi_index_mnp
        assert reg.grid == grd
        assert reg.domain == grd.domain
        # Assert uninitialized attributes prior to fitting
        assert reg.loocv_error is None
        assert reg.regfit_linf_error is None
        assert reg.regfit_l2_error is None
        assert reg.coeffs is None
        assert reg.eval_poly is None
        # show() method can be called
        assert_call(reg.show)

    def test_with_grid(self, grid_mnp):
        """Test the initialization with a grid instance."""
        # Create an instance of ordinary regression
        reg = OrdinaryRegression(grid=grid_mnp)

        # Assertions
        assert reg.multi_index == grid_mnp.multi_index
        assert reg.grid == grid_mnp
        assert reg.domain == grid_mnp.domain
        # Assert uninitialized attributes prior to fitting
        assert reg.loocv_error is None
        assert reg.regfit_linf_error is None
        assert reg.regfit_l2_error is None
        assert reg.coeffs is None
        assert reg.eval_poly is None
        # show() method can be called
        assert_call(reg.show)

    def test_with_multi_index_and_grid(self, multi_index_mnp, grid_mnp):
        """Test the initialization with a multi-index set and grid instance."""
        # Create an instance of ordinary regression
        reg = OrdinaryRegression(multi_index_mnp, grid=grid_mnp)

        # Assertions
        assert reg.multi_index == multi_index_mnp
        assert reg.grid == grid_mnp
        assert reg.domain == grid_mnp.domain
        # Assert uninitialized attributes prior to fitting
        assert reg.loocv_error is None
        assert reg.regfit_linf_error is None
        assert reg.regfit_l2_error is None
        assert reg.coeffs is None
        assert reg.eval_poly is None

    def test_with_insufficient_inputs(self):
        """Test the initialization without sufficient inputs."""
        with pytest.raises(ValueError):
            _ = OrdinaryRegression()

    def test_with_invalid_multi_index(self):
        """Test the initialization with an invalid multi-index set."""
        with pytest.raises(TypeError):
            _ = OrdinaryRegression(multi_index=1)

    def test_with_invalid_grid_type(self):
        """Test the initialization with an invalid grid type."""
        with pytest.raises(TypeError):
            _ = OrdinaryRegression(grid=1)

    def test_with_invalid_grid_subset(self, multi_index_mnp):
        """Test if using a lower degree grid than multi-index raises an error.
        """
        mi = multi_index_mnp
        m, n, p = mi.spatial_dimension, mi.poly_degree, mi.lp_degree

        if n == 0:
            pytest.skip("Cannot be further reduced to make lower grid")

        grd = Grid.from_degree(m, n-1, p)
        with pytest.raises(ValueError):
            _ = OrdinaryRegression(multi_index=mi, grid=grd)

    def test_with_invalid_domain_type(self):
        """Test the initialization with an invalid domain type."""
        mi = MultiIndexSet.from_degree(1, 3, 1)
        with pytest.raises(TypeError):
            _ = OrdinaryRegression(multi_index=mi, domain=1)

    def test_with_invalid_domain_instance(self, grid_mnp):
        """Test the initialization with an invalid domain instance."""
        # Create an inconsistent domain
        a, b = np.sort(np.random.rand(2))
        dom = Domain.uniform(grid_mnp.spatial_dimension, a, b)

        with pytest.raises(DomainMismatchError):
            _ = OrdinaryRegression(grid=grid_mnp, domain=dom)

    @pytest.mark.parametrize("spatial_dimension", [0, 1, 5])
    def test_empty_multi_index_set(self, spatial_dimension, LpDegree):
        """Test construction with an empty set."""
        # Create an empty set
        mi = MultiIndexSet(np.empty((0, spatial_dimension)), LpDegree)

        # Assertion
        with pytest.raises(ValueError):
            _ = OrdinaryRegression(multi_index=mi)


class TestFit:
    """All tests related to the fitting of the model."""

    def test_unisolvent(self, rand_newton_poly, lst_sqr_solver):
        """Test fitting on the unisolvent nodes is interpolation."""
        # Create a regression
        grd = rand_newton_poly.grid
        reg = OrdinaryRegression(grid=grd)

        # Generate training data
        xx_train = (rand_newton_poly.domain.map_from_internal(
            rand_newton_poly.grid.unisolvent_nodes
        ))
        yy_train = rand_newton_poly(xx_train)

        # fit() can be called with additional fitting arguments
        kwargs = LSQ_SOLVER_ARGS.get(lst_sqr_solver, {})
        reg.fit(
            xx_train,
            yy_train,
            lstsq_solver=lst_sqr_solver,
            **kwargs,
        )

        # Assertions
        assert np.allclose(yy_train, reg(xx_train))
        assert reg.loocv_error is not None
        assert reg.regfit_linf_error is not None
        assert reg.regfit_l2_error is not None
        assert reg.coeffs is not None
        assert reg.eval_poly is not None
        # show() method can be called
        assert_call(reg.show)

    @pytest.mark.parametrize(
        "weight",
        ["vector", "matrix"],
        ids=[f"weight={i:<6}" for i in ["vector","matrix"]],
    )
    def test_with_weights(self, rand_newton_poly, lst_sqr_solver, weight):
        """Test fitting with weights vector."""
        # Create a regression
        grd = rand_newton_poly.grid
        reg = OrdinaryRegression(grid=grd)

        # Generate training data
        xx_train = rand_newton_poly.domain.map_from_internal(
            rand_newton_poly.grid.unisolvent_nodes
        )
        yy_train = rand_newton_poly(xx_train)
        if weight == "vector":
            weights = np.ones(yy_train.shape)
        else:
            weights = np.eye(yy_train.shape[0])

        # fit() can be called with additional fitting arguments
        kwargs = LSQ_SOLVER_ARGS.get(lst_sqr_solver, {})
        reg.fit(
            xx_train,
            yy_train,
            weights=weights,
            lstsq_solver=lst_sqr_solver,
            **kwargs,
        )

        # Assertions
        assert np.allclose(yy_train, reg(xx_train))
        assert reg.loocv_error is not None
        assert reg.regfit_linf_error is not None
        assert reg.regfit_l2_error is not None
        assert reg.coeffs is not None
        assert reg.eval_poly is not None
        # show() method can be called
        assert_call(reg.show)

    def test_invalid_solver(self, rand_newton_poly):
        """Test fitting with an invalid solver."""
        # Create a regression
        grd = rand_newton_poly.grid
        reg = OrdinaryRegression(grid=grd)

        # Generate training data
        xx_train = rand_newton_poly.domain.map_from_internal(
            rand_newton_poly.grid.unisolvent_nodes
        )
        yy_train = rand_newton_poly(xx_train)

        # Fit with an invalid solver
        with pytest.raises(NotImplementedError):
            reg.fit(xx_train, yy_train, lstsq_solver="rand123")

    def test_underdetermined_system(self, rand_newton_poly):
        """Test fitting an under-determined system."""
        # Create a regression
        grd = rand_newton_poly.grid
        reg = OrdinaryRegression(grid=grd)

        # Generate training data
        sample_size = max(1, math.floor(0.8 * len(grd.multi_index)))
        xx_train = rand_newton_poly.domain.map_from_internal(
            -1 + 2 * np.random.rand(sample_size, grd.spatial_dimension)
        )
        yy_train = rand_newton_poly(xx_train)

        # Fit the ordinary regression model
        reg.fit(xx_train, yy_train, lstsq_solver="qr")

        assert reg.loocv_error == (np.inf, np.inf)

    def test_unsupported_polynomial_basis(self, rand_newton_poly):
        """Test fitting on an unsupported polynomial basis."""
        # Create a regression
        grd = rand_newton_poly.grid
        reg = OrdinaryRegression(grid=grd, origin_poly=_EmptyClass)

        # Generate training data
        xx_train = rand_newton_poly.domain.map_from_internal(
            rand_newton_poly.grid.unisolvent_nodes
        )
        yy_train = rand_newton_poly(xx_train)

        # Fit an OrdinaryRegression instance
        with pytest.raises(TypeError):
            reg.fit(xx_train, yy_train)


    def test_loocv(self, rand_newton_poly):
        """Test fitting with a different loo-cv option."""
        # Create a regression
        grd = rand_newton_poly.grid
        reg = OrdinaryRegression(grid=grd)

        # Generate training data
        xx_train = rand_newton_poly.domain.map_from_internal(
            rand_newton_poly.grid.unisolvent_nodes
        )
        yy_train = rand_newton_poly(xx_train)

        # Fit and compute LOO-CV
        reg.fit(xx_train, yy_train, compute_loocv=True)
        assert reg.loocv_error is not None

        # Fit without computing LOO-CV
        reg.fit(xx_train, yy_train, compute_loocv=False)
        assert reg.loocv_error is None

        # Fit with an invalid LOO-CV option
        with pytest.raises(ValueError):
            reg.fit(xx_train, yy_train, compute_loocv=123)


class TestPredict:
    """All tests related to the prediction of the model."""

    def test_no_fitting(self, rand_newton_poly):
        """Test predict without fitting raises an error."""
        # Create a regression
        grd = rand_newton_poly.grid
        reg = OrdinaryRegression(grid=grd)

        # Generate training data
        xx_train = rand_newton_poly.domain.map_from_internal(
            rand_newton_poly.grid.unisolvent_nodes
        )

        # Predict without fitting
        with pytest.raises(TypeError):
            reg.predict(xx_train)
        with pytest.raises(TypeError):
            reg(xx_train)

    def test_interpolation(self, rand_newton_poly, origin_poly):
        """Test predict with fitting on the unisolvent nodes."""
        # Create a regression
        grd = rand_newton_poly.grid
        reg = OrdinaryRegression(grid=grd, origin_poly=origin_poly)

        # Generate training data
        xx_train = rand_newton_poly.domain.map_from_internal(
            rand_newton_poly.grid.unisolvent_nodes
        )
        yy_train = rand_newton_poly(xx_train)

        # Fit the regression model
        reg.fit(xx_train, yy_train)

        # Generate test data
        xx_test = -1 + 2 * np.random.rand(10, grd.spatial_dimension)
        yy_test = rand_newton_poly(xx_test)

        # Assertion
        assert np.allclose(yy_test, reg(xx_test))
