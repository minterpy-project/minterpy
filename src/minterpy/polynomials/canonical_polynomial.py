"""
Base class for polynomials in the canonical base.

"""
import numpy as np

from copy import deepcopy
from scipy.special import factorial

from minterpy.global_settings import INT_DTYPE
from minterpy.core.ABC import MultivariatePolynomialSingleABC
from minterpy.utils.polynomials.canonical import integrate_monomials_canonical
from minterpy.utils.verification import (
    convert_eval_output,
    dummy,
    verify_domain,
)
from minterpy.utils.arrays import make_coeffs_2d
from minterpy.utils.multi_index import find_match_between
from minterpy.jit_compiled.multi_index import all_indices_are_contained

__all__ = ["CanonicalPolynomial"]


# Arithmetics
def _match_dims(poly1, poly2, copy=None):
    """Dimensional expansion of two polynomial in order to match their spatial_dimensions.

    Parameters
    ----------
    poly1 : CanonicalPolynomial
        First polynomial in canonical basis
    poly2 : CanonicalPolynomial
        Second polynomial in canonical basis
    copy : bool
        If True, work on deepcopies of the passed polynomials (doesn't change the input).
        If False, inplace expansion of the passed polynomials

    Returns
    -------
    (poly1,poly2) : (CanonicalPolynomial,CanonicalPolynomial)
        Dimensionally matched polynomials in the same order as input.

    Notes
    -----
    - Maybe move this to the MultivariatePolynomialSingleABC since it shall be avialable for all poly bases
    """
    if copy is None:
        copy = True

    if copy:
        p1 = deepcopy(poly1)
        p2 = deepcopy(poly2)
    else:
        p1 = poly1
        p2 = poly2

    dim1 = p1.multi_index.spatial_dimension
    dim2 = p2.multi_index.spatial_dimension
    if dim1 >= dim2:
        p2 = p2.expand_dim(dim1)
    else:
        p1 = p1.expand_dim(dim2)
    return p1, p2


def _shape_coeffs(poly_1, poly_2):
    """Shape the polynomial coefficients before processing."""
    assert len(poly_1) == len(poly_2)

    num_poly = len(poly_1)
    if num_poly > 1:
        return poly_1.coeffs, poly_2.coeffs

    coeffs_1 = poly_1.coeffs[:, np.newaxis]
    coeffs_2 = poly_2.coeffs[:, np.newaxis]

    return coeffs_1, coeffs_2


def _canonical_add(
    poly_1: "CanonicalPolynomial",
    poly_2: "CanonicalPolynomial",
) -> "CanonicalPolynomial":
    """Add two polynomial instances in the canonical basis.

    This is the concrete implementation of ``_add()`` method in the
    ``MultivariatePolynomialSingleABC`` abstract class.

    Parameters
    ----------
    poly_1 : CanonicalPolynomial
        Left operand of the addition expression.
    poly_2 : CanonicalPolynomial
        Right operand of the addition expression.

    Returns
    -------
    CanonicalPolynomial
        The sum of two polynomials in the canonical basis; a new instance
        of polynomial.

    Notes
    -----
    - This function assumes: both polynomials must be in canonical basis,
      they must be initialized, have the same dimension and their domains
      are matching, and the number of polynomials per instance are the same.
      These conditions are not explicitly checked in this function.
    """
    # --- Compute the union of the grid instances
    grd_add = poly_1.grid | poly_2.grid

    # --- Compute union of the multi-index sets if they are separate
    if poly_1.indices_are_separate or poly_2.indices_are_separate:
        mi_add = poly_1.multi_index | poly_2.multi_index
    else:
        # Otherwise use the one attached to the grid instance
        mi_add = grd_add.multi_index
    num_monomials = len(mi_add)

    # --- Process the coefficients
    idx_1 = find_match_between(poly_1.multi_index.exponents, mi_add.exponents)
    idx_2 = find_match_between(poly_2.multi_index.exponents, mi_add.exponents)

    # Shape the coefficients before addition
    coeffs_1, coeffs_2 = _shape_coeffs(poly_1, poly_2)

    # Add the coefficients column-wise
    num_polys = len(poly_1)
    coeffs_add = np.zeros((num_monomials, num_polys))
    coeffs_add[idx_1, :] += coeffs_1[:, :]
    coeffs_add[idx_2, :] += coeffs_2[:, :]

    # Squeeze the resulting coefficients if there's only one polynomial
    if num_polys == 1:
        coeffs_add = coeffs_add.reshape(-1)

    # --- Return a new instance
    return CanonicalPolynomial(
        mi_add,
        coeffs=coeffs_add,
        user_domain=poly_1.user_domain,
        internal_domain=poly_1.internal_domain,
        grid=grd_add,
    )


def _canonical_eval(pts: np.ndarray,exponents: np.ndarray, coeffs: np.ndarray):
    """
    Unsafe version of naive canonical evaluation

    :param pts: List of points, the polynomial must be evaluated on. Assumed shape: `(number_of_points,spatial_dimension)`.
    :type pts: np.ndarray

    :param exponents: Exponents from a multi-index set. Assumed shape:` (number_of_monomials, spatial_dimension)`.
    :type exponents: np.ndarray

    :param coeffs: List of coefficients. Assumed shape: `(number_of_monomials,)`
    :type coeffs: np.ndarray

    :return: result of the canonical evaluation.
    :rtype: np.ndarray
    
    """
    yy = np.dot(np.prod(np.power(pts[:, None, :], exponents[None, :, :]), axis=-1), coeffs)

    return convert_eval_output(yy)


def _verify_eval_input(pts, spatial_dimension):
    """
    verification of the input of the canonical evaluation. 
    """
    assert(isinstance(pts,np.ndarray))
    assert(pts.ndim==2)
    assert(pts.shape[-1]==spatial_dimension)


def canonical_eval(canonical_poly, pts: np.ndarray):
    """
    Navie canonical evaluation

    :param canonical_poly: Polynomial in canonical form to be evaluated.
    :type canonical_poly: CanonicalPolynomial

    :param pts: List of points, the polynomial must be evaluated on. Assumed shape: `(number_of_points,spatial_dimension)`.
    :type pts: np.ndarray

    :return: result of the canonical evaluation. 
    :rtype: np.ndarray

    """
    _verify_eval_input(pts,canonical_poly.spatial_dimension)
    return _canonical_eval(pts,canonical_poly.multi_index.exponents,canonical_poly.coeffs)


# TODO redundant
canonical_generate_internal_domain = verify_domain
canonical_generate_user_domain = verify_domain


def _canonical_partial_diff(poly: "CanonicalPolynomial", dim: int, order: int) -> "CanonicalPolynomial":
    """ Partial differentiation in Canonical basis.
    """
    spatial_dim = poly.multi_index.spatial_dimension
    deriv_order_along = np.zeros(spatial_dim, dtype=INT_DTYPE)
    deriv_order_along[dim] = order
    return _canonical_diff(poly, deriv_order_along)


def _canonical_diff(poly: "CanonicalPolynomial", order: np.ndarray) -> "CanonicalPolynomial":
    """ Partial differentiation in Canonical basis.
    """

    coeffs = make_coeffs_2d(poly.coeffs)
    exponents = poly.multi_index.exponents

    # Guard rails in ABC ensures that the len(order) == poly.spatial_dimension
    subtracted_exponents = exponents - order

    # compute mask for non-negative multi index entries
    diff_exp_mask = np.all(exponents >= order, axis = 1)

    # multi index entries in the differentiated polynomial
    diff_exponents = subtracted_exponents[diff_exp_mask]

    # Checking if the necessary multi index entries are present
    # Zero size check is needed here as all_indices_are_contained throws error otherwise
    if diff_exponents.size != 0 and not all_indices_are_contained(diff_exponents, exponents):
        raise ValueError(f"Cannot differentiate as some of the required multi indices are not present.")

    # coefficients of the differentiated polynomial
    diff_coeffs = coeffs[diff_exp_mask] * np.prod(factorial(exponents[diff_exp_mask]) / factorial(diff_exponents),
                                                  axis=1)[:,None]

    # The differentiated polynomial being expressed wrt multi indices of the given poly
    # NOTE: 'find_match_between' assumes 'exponents' is lexicographically ordered
    map_pos = find_match_between(diff_exponents, exponents)
    new_coeffs = np.zeros_like(coeffs)
    new_coeffs[map_pos] = diff_coeffs

    # Squeezing the last dimension to handle single polynomial
    return CanonicalPolynomial.from_poly(poly, new_coeffs.reshape(poly.coeffs.shape))


def _canonical_integrate_over(
    poly: "CanonicalPolynomial",
    bounds: np.ndarray,
) -> np.ndarray:
    """Compute the definite integral of a polynomial in the canonical basis.

    Parameters
    ----------
    poly : CanonicalPolynomial
        The polynomial of which the integration is carried out.
    bounds : :class:`numpy:numpy.ndarray`
        The bounds (lower and upper) of the definite integration, an ``(M, 2)``
        array, where ``M`` is the number of spatial dimensions.

    Returns
    -------
    :class:`numpy:numpy.ndarray`
        The integral value of the polynomial over the given domain.
    """
    # --- Compute the integral of the canonical monomials (quadrature weights)
    exponents = poly.multi_index.exponents

    quad_weights = integrate_monomials_canonical(exponents, bounds)

    return quad_weights @ poly.coeffs


class CanonicalPolynomial(MultivariatePolynomialSingleABC):
    """
    Polynomial type in the canonical base.
    """

    __doc__ = """
    This is the docstring of the canonical base class.
    """

    # __doc__ += MultivariatePolynomialSingleABC.__doc_attrs__
    # Virtual Functions
    _add = staticmethod(_canonical_add)
    _sub = staticmethod(dummy)
    _mul = staticmethod(dummy)
    _div = staticmethod(dummy)
    _pow = staticmethod(dummy)
    _eval = staticmethod(canonical_eval)
    _iadd = staticmethod(dummy)

    _partial_diff = staticmethod(_canonical_partial_diff)
    _diff = staticmethod(_canonical_diff)

    _integrate_over = staticmethod(_canonical_integrate_over)

    generate_internal_domain = staticmethod(canonical_generate_internal_domain)
    generate_user_domain = staticmethod(canonical_generate_user_domain)
