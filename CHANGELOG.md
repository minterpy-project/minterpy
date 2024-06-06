# Changelog

## Unreleased

### Added

- Support for the Chebyshev polynomials of the first kind as a polynomial basis
  via `ChebyshevPolynomial`. Differentiation and integration of polynomials in
  this basis are not yet supported.
- Exact equality check via the `==` operator has been implemented for instances
  of the `Grid` class. Two instances of `Grid` are equal (in values) if and
  only if both the underlying multi-index sets and generating points are
  equal. Note that as the generating points are of floating types, the equality
  check via `==` is exact without any tolerance specifications.
- Exact equality check via the `==` operator has been implemented for all
  instances of concrete polynomial classes. Two polynomials are equal in values
  if and only if the concrete class is the same, the multi-index sets are
  equal, the grids are equal, and the coefficient values are all equal.
  As coefficients are of floating type, the comparison is carried out exactly
  without any tolerance specifications.

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
