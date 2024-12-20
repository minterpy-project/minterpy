"""
Utility functions for computing matrices for transformation between canonical, lagrange, and newton basis.

.. todo::
   - Consider if simplifying the names makes any sense
   - make sure that the transformations are tested
   - find solution for the case that the multi-indices are separated
     from the grid indices
   - Consider if making separate modules for some relevant functions
     makes any sense
"""
import numpy as np

from typing import no_type_check

from minterpy.core.ABC import OperatorABC, TransformationABC
from minterpy.dds import dds
from minterpy.global_settings import ARRAY, DEBUG, FLOAT_DTYPE
from minterpy.jit_compiled.transformations import compute_vandermonde_n2c
from minterpy.schemes.barycentric.precomp import (
    _build_lagrange_to_newton_bary,
    _build_newton_to_lagrange_bary,
)
from minterpy.schemes.matrix_operator import MatrixOperator
from minterpy.utils.polynomials.newton import eval_newton_monomials

from minterpy.utils.polynomials.chebyshev import (
    evaluate_monomials as evaluate_monomials_chebyshev,
)


# NOTE: avoid looping over a numpy array! e.g. for j in np.arange(num_monomials):
# see: # https://stackoverflow.com/questions/10698858/built-in-range-or-numpy-arange-which-is-more-efficient


def invert_triangular(triangular_matrix: np.ndarray) -> np.ndarray:
    # FIXME: triangular inversion is not working! required when using barycentric transforms?
    # i, j = triangular_matrix.shape  # square matrix
    # inverted_matrix = solve_triangular(triangular_matrix, np.identity(i))
    inverted_matrix = np.linalg.inv(triangular_matrix)
    return inverted_matrix


def _build_n2l_array(grid, multi_index=None, require_invertible: bool = False) -> ARRAY:
    # NOTE: the indices might be different from the ones used in the grid!
    # -> just some "active" Lagrange polynomials
    if multi_index is None or require_invertible:
        # NOTE: the transformation matrix needs to be square for the inversion
        # even if some of the multi indices are "inactive",
        # the unisolvent nodes (grid) need to correspond to the multi indices!
        multi_index = grid.multi_index

    exponents = multi_index.exponents
    unisolvent_nodes = grid.unisolvent_nodes
    generating_points = grid.generating_points
    # NOTE: the shape of unisolvent_nodes and exponents might be different! -> non square transformation matrix
    transformation_matrix = eval_newton_monomials(
        unisolvent_nodes,
        exponents,
        generating_points,
        verify_input=DEBUG,
        triangular=True,
    )
    return transformation_matrix


def _build_newton_to_lagrange_naive(
    transformation: TransformationABC,
) -> MatrixOperator:
    """computes the Newton to Lagrange transformation given by an array

    SPECIAL PROPERTY: the evaluation of any polynomial on unisolvent nodes yields
        the Lagrange coefficients of this polynomial
        (for the Lagrange basis defined implicitly by these unisolvent nodes)
    -> evaluating every Newton polynomial (<-> the monomials forming the Newton basis) on all interpolation points,
        naturally yields the operator transforming the Newton coefficients into Lagrange coefficients

     NOTE: it is inefficient to compute this by inversion:
         newton_to_lagrange = inv(lagrange_to_newton) # based on DDS

    special property: half of the values will always be 0 (lower triangular matrix).

    :param require_invertible: weather or not the output matrix should be square
    """
    grid = transformation.grid
    transformation_matrix = _build_n2l_array(
        grid, transformation.origin_poly.multi_index
    )
    transformation_operator = MatrixOperator(transformation, transformation_matrix)
    return transformation_operator


def build_l2n_matrix_dds(grid):
    num_monomials = len(grid.multi_index)
    lagr_coeff_matrix = np.eye(num_monomials, dtype=FLOAT_DTYPE)
    tree = grid.tree
    lagrange_to_newton = dds(lagr_coeff_matrix, tree)
    return lagrange_to_newton


def _build_lagrange_to_newton_naive(
    transformation: TransformationABC,
) -> MatrixOperator:
    """computes the Lagrange to Newton transformation given by an array

    NOTE: each column of the L2N transformation matrix
        corresponds to the Newton coefficients of the respective Lagrange polynomial.

    NOTE: computing the L2N matrix could be done with the DDS scheme, but only if the multi indices are complete.
        in this case however it is more efficient to use the barycentric transformations right away.
    for the case that the indices are not complete the matrix must be computed by inverting the N2L transformation.
    """
    newton_to_lagrange = _build_n2l_array(transformation.grid, require_invertible=True)
    transformation_matrix = invert_triangular(newton_to_lagrange)
    transformation_operator = MatrixOperator(transformation, transformation_matrix)
    return transformation_operator


def _build_c2n_array(transformation) -> ARRAY:
    multi_index = transformation.grid.multi_index
    num_monomials = len(multi_index)
    V_n2c = np.ones((num_monomials, num_monomials), dtype=FLOAT_DTYPE)
    compute_vandermonde_n2c(
        V_n2c, transformation.grid.unisolvent_nodes, multi_index.exponents
    )  # computes the result "in place"
    tree = transformation.grid.tree
    c2n = dds(V_n2c, tree)
    return c2n


def _build_n2c_array(transformation: TransformationABC) -> ARRAY:
    # TODO achieve by calling inverse(n2c_operator)
    return invert_triangular(_build_c2n_array(transformation))


# --- From LagrangePolynomial
@no_type_check
def build_lagrange_to_newton_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """Construct the Lagrange-to-Newton transformation operator.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of LagrangePolynomial) and
        the target type (NewtonPolynomial).

    Returns
    -------
    OperatorABC
        The Lagrange-to-Newton transformation operator.

    Notes
    -----
    - The barycentric transformation operator is employed if the multi-indices
      are downward-closed.
    - The naive transformation operator is inefficient due to the following
      inversion: ``inv(nwt2lag_matrix)``.
    """
    grid = transformation.grid
    is_downward_closed = grid.multi_index.is_downward_closed
    identical_indices = not transformation.origin_poly.indices_are_separate
    if is_downward_closed and identical_indices:
        # use barycentric transformation
        transformation_operator = _build_lagrange_to_newton_bary(transformation)
    else:  # use "naive" matrix transformation format
        transformation_operator = _build_lagrange_to_newton_naive(transformation)
    return transformation_operator


def build_lagrange_to_canonical_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """Construct the Lagrange-to-Canonical transformation operator.

    The transformation is the chain: Lagrange-to-Newton then
    Newton-to-Canonical.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of LagrangePolynomial) and
        the target type (CanonicalPolynomial).

    Returns
    -------
    OperatorABC
        The Lagrange-to-Canonical transformation operator.

    Notes
    -----
    - Barycentric transformation operator is employed if the multi-indices
      are downward-closed.
    """
    lag2nwt_operator = build_lagrange_to_newton_operator(transformation)
    nwt2can_operator = build_newton_to_canonical_operator(transformation)

    return nwt2can_operator @ lag2nwt_operator


def build_lagrange_to_chebyshev_operator(
    transformation: TransformationABC
) -> OperatorABC:
    """Compute the Lagrange-to-Chebyshev transformation.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of LagrangePolynomial) and
        the target type (ChebyshevPolynomial).

    Returns
    -------
    OperatorABC
        The Lagrange-to-Chebyshev transformation operator.

    Notes
    -----
    - The transformation is carried out by inverting the evaluation
      at the unisolvent nodes; this is not a sparse transformation operator.
    """
    # Get the relevant data from the transformer
    mi_poly = transformation.multi_index
    mi_grid = transformation.grid.multi_index
    if mi_poly != mi_grid:
        raise ValueError(
            "Inconsistent behavior for different multi-indices of "
            "the polynomial and the grid!"
        )
    else:
        unisolvent_nodes = transformation.grid.unisolvent_nodes
        exponents = transformation.grid.multi_index.exponents

    # Compute the Chebyshev monomials at the unisolvent nodes
    cheb2lag_matrix = evaluate_monomials_chebyshev(
        unisolvent_nodes,
        exponents,
    )

    # Compute the inverse transformation
    lag2cheb_matrix = np.linalg.inv(cheb2lag_matrix)

    # Create the operator
    lag2cheb_operator = MatrixOperator(transformation, lag2cheb_matrix)

    return lag2cheb_operator


# --- From NewtonPolynomial
@no_type_check
def build_newton_to_lagrange_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """Construct the Newton-to-Lagrange transformation operator.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of NewtonPolynomial) and
        the target type (LagrangePolynomial).

    Returns
    -------
    OperatorABC
        The Newton-to-Lagrange transformation operator.

    Notes
    -----
    - The barycentric transformation operator is employed if the multi-indices
      are downward-closed.
    """
    grid = transformation.grid
    is_downward_closed = grid.multi_index.is_downward_closed
    identical_indices = not transformation.origin_poly.indices_are_separate
    if is_downward_closed and identical_indices:
        # use barycentric transformation
        transformation_operator = _build_newton_to_lagrange_bary(transformation)
    else:  # use "naive" matrix transformation format
        transformation_operator = _build_newton_to_lagrange_naive(transformation)

    return transformation_operator


def build_newton_to_canonical_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """Construct the Newton-to-Canonical transformation operator.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of NewtonPolynomial) and
        the target type (CanonicalPolynomial).

    Returns
    -------
    OperatorABC
        The Newton-to-Canonical transformation operator.
    """
    nwt2can_matrix = _build_n2c_array(transformation)
    nwt2can_operator = MatrixOperator(transformation, nwt2can_matrix)

    return nwt2can_operator


def build_newton_to_chebyshev_operator(
    transformation: TransformationABC
) -> OperatorABC:
    """Construct the Newton-to-Chebyshev transformation operator.

    The transformation is the chain: Newton-to-Lagrange then
    Lagrange-to-Chebyshev.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of NewtonPolynomial) and
        the target type (ChebyshevPolynomial).

    Returns
    -------
    OperatorABC
        The Newton-to-Chebyshev transformation operator.
    """
    nwt2lag_operator = build_newton_to_lagrange_operator(transformation)
    lag2cheb_operator = build_lagrange_to_chebyshev_operator(transformation)

    return lag2cheb_operator @ nwt2lag_operator


# --- From CanonicalPolynomial
def build_canonical_to_newton_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """Construct the Canonical-to-Newton transformation operator.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of CanonicalPolynomial) and
        the target type (NewtonPolynomial).

    Returns
    -------
    OperatorABC
        The Canonical-to-Newton transformation operator.
    """
    can2nwt_matrix = _build_c2n_array(transformation)
    can2nwt_operator = MatrixOperator(transformation, can2nwt_matrix)

    return can2nwt_operator


def build_canonical_to_lagrange_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """Construct the Canonical-to-Lagrange transformation operator.

    The transformation is the chain: Canonical-to-Newton then
    Newton-to-Lagrange.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of CanonicalPolynomial) and
        the target type (LagrangePolynomial).

    Returns
    -------
    OperatorABC
        The Canonical-to-Lagrange transformation operator.
    """
    nwt2lag_operator = build_newton_to_lagrange_operator(transformation)
    can2nwt_operator = build_canonical_to_newton_operator(transformation)

    return nwt2lag_operator @ can2nwt_operator


def build_canonical_to_chebyshev_operator(
    transformation: TransformationABC
) -> OperatorABC:
    """Construct the Canonical-to-Chebyshev transformation operator.

    The transformation is the chain: Canonical-to-Newton then
    Newton-to-Lagrange then Lagrange-to-Chebyshev.

    The Canonical-to-Lagrange operator is implemented as a single operator.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of CanonicalPolynomial) and
        the target type (ChebyshevPolynomial).

    Returns
    -------
    OperatorABC
        The canonical-to-Chebyshev transformation operator.
    """
    can2lag_operator = build_canonical_to_lagrange_operator(transformation)
    lag2cheb_operator = build_lagrange_to_chebyshev_operator(transformation)

    return lag2cheb_operator @ can2lag_operator


# --- From ChebyshevPolynomial
def build_chebyshev_to_lagrange_operator(
    transformation: TransformationABC
) -> OperatorABC:
    """Construct the Chebyshev to Lagrange transformation operator.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of ChebyshevPolynomial) and
        the target type (LagrangePolynomial).

    Returns
    -------
    OperatorABC
        The Chebyshev-to-Lagrange transformation operator.

    Notes
    -----
    - The transformation is carried out by evaluation at the unisolvent nodes;
      this is not a sparse transformation operator.
    """
    # Get relevant data from the transformer
    mi_poly = transformation.multi_index
    mi_grid = transformation.grid.multi_index
    if mi_poly != mi_grid:
        raise ValueError(
            "Inconsistent behavior for different multi-indices of "
            "the polynomial and the grid!"
        )
    else:
        unisolvent_nodes = transformation.grid.unisolvent_nodes
        exponents = transformation.grid.multi_index.exponents

    # Compute the Chebyshev monomials at the unisolvent nodes
    chebyshev_monomials = evaluate_monomials_chebyshev(
        unisolvent_nodes,
        exponents,
    )

    # The monomials are the (full) transformation operator
    cheb2lag_operator = MatrixOperator(transformation, chebyshev_monomials)

    return cheb2lag_operator


def build_chebyshev_to_newton_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """Construct the Chebyshev-to-Newton transformation operator.

    The transformation is the chain: Chebyshev-to-Lagrange then
    Lagrange-to-Newton.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of ChebyshevPolynomial) and
        the target type (NewtonPolynomial).

    Returns
    -------
    OperatorABC
        The Chebyshev-to-Newton transformation operator.
    """
    cheb2lag_operator = build_chebyshev_to_lagrange_operator(transformation)
    lag2nwt_operator = build_lagrange_to_newton_operator(transformation)

    return lag2nwt_operator @ cheb2lag_operator


def build_chebyshev_to_canonical_operator(
    transformation: TransformationABC,
) -> OperatorABC:
    """Construct the Chebyshev-to-Canonical transformation operator.

    The transformation is the chain: Chebyshev-to-Lagrange then
    Lagrange-to-Newton then Newton-to-Canonical.

    The Lagrange-to-Canonical is implemented as a single operator.

    Parameters
    ----------
    transformation : TransformationABC
        The transformer instance with information about the origin polynomial
        (an instance of ChebyshevPolynomial) and
        the target type (NewtonPolynomial).

    Returns
    -------
    OperatorABC
        The Chebyshev-to-Newton transformation operator.
    """
    cheb2lag_operator = build_chebyshev_to_lagrange_operator(transformation)
    lag2can_operator = build_lagrange_to_canonical_operator(transformation)

    return lag2can_operator @ cheb2lag_operator
