# Changelog

## Unreleased

### Added

- Support for the Chebyshev polynomials of the first kind as a polynomial basis
  via `ChebyshevPolynomial`. Differentiation and integration of polynomials in
  this basis are not yet supported.
- Faster differentiation of polynomials in the Newton basis due to a Numba 
  implementation; the methods `diff()` and `partial_diff()` now support
  a keyword argument `backend` to select the numerical routine for
  differentiation. Supported values are: `"numpy"` (NumPy-based implementation
  used by default in v0.2.0-alpha), `"numba"` (Numba-based implementation,
  now the default), `"numba-par"` (CPU parallelization in the Numba-based
  implementation; may accelerates computation for larger problems).

# Version 0.2.0-alpha
This is the next alpha release of `minterpy`, which adds several
new functionalities and enhances code quality and performance.

## new features

- partial derivatives for canonical and Newton polynomials
- support of arbitrary positive `lp_degree` 
- ordinary regression based on multivariate polynomials
  as the first extra feature

## maintenance

- bug fixes
- adding API documentation
- improvement of user documentations
- clean-up: deletion of code/comments, which are no longer used
- introduction of a corporate design including logo and banner to docs, 
  repository, README etc.

This code is still marked as experimental and there is no assurance,
that neither everything works as expected,
nor if further releases will break the current API.


# Version 0.1.0-alpha

This is the initial alpha release of `minterpy`.
It contains general structures to perform the polynomial interpolation task
in multiple dimensions:
 
- Multivariate polynomial bases (ABC + concrete implementations)
- Base transformations
- Interpolation schemes

This code is still highly experimental and there is no assurance,
that neither everything works as expected,
nor if further releases will break the current API.
