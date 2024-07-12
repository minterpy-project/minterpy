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
- Polynomial-(real)scalar multiplication is now supported for polynomial
  instances of all concrete classes. The implementation includes left-side,
  right-side, and in-place multiplication.
- All polynomial instances now has the method `has_matching_domain()` method
  to check if a given instance has matching internal and user domains with
  another instance. The two polynomials do not have to be on the same basis.
- Polynomial-polynomial multiplication in the Newton basis is now supported.
  The implementation includes the left-side multiplication via `__mul__()`.
  Multiplication with a constant polynomial returns a consistent result with
  multiplication with a scalar.
- Polynomial-polynomial addition/subtraction as well as polynomial-(real)scalar
  addition/subtraction are now supported for polynomials in the Newton basis.
  The implementation includes the left-sided addition via `__add__()` and
  subtraction via `__sub__()` (for both Newton polynomial and real scalar 
  number) as well as right-sided addition via `__radd__()` and subtraction via 
  `__rsub__()` (for real scalar numbers).
- Static abstract method `_iadd()` is now included in the
  `MultivariatePolynomialSingleABC` as a placeholder for the concrete
  implementation of augmented addition operation.
- Instances of `MultiIndexSet` may now be multiplied with each other;
  the result is an instance of `MultiIndexSet` whose exponents are
  the cross-product-and-sum of the two operands exponents.
- A method `expand_dim()` is introduced to instances  of the `Grid` class
  to encapsulate the procedure of expanding the dimension of a `Grid` instance.
- Add two new factory methods for the `Grid` class: `from_degree()` to
  create a `Grid` instance with a complete multi-index set and `from_points()`
  to create an instance with a given array of generating points.
- Instances of `Grid` has now `has_generating_function` property that returns
  `True` if a generating function is defined on the grid and `False` otherwise.
- Instances of `MultiIndexSet` has now `max_exponent` and `max_exponents`
  properties. The former is the maximum exponent across all dimensions in the
  multi-index set, while the latter is the maximum exponents per dimension.

### Fixed

- Negating a polynomial with a non-default Grid instance returns a polynomial
  with the default grid.

### Changed

- The utility modules that were scattered across the codebase are now
  collected inside the sub-package `minterpy.utils`. The residing functions
  are supposed to deal with low-level computational details that support
  higher-level constructs of Minterpy such as polynomials, multi-index set,
  grid, etc. Vice versa, higher-level constructs should avoid directly
  deal with low-level array manipulations without interfacing functions.
  This reorganization has no effect on the overall code functionality.
- The property `generating_values` has been removed from the `Grid` class.
  Furthermore, `generating_values` does not appear in the default constructor
  of `Grid` and therefore is no longer required to construct an instance of
  the class.
- Multi-index set is now a read-only property of `Grid` instances instead
  of an instance attribute.
- `generating_function`` is now stored as a read-only  property of a `Grid`
  instances and also a factor in determining instances equality in value.
- `generating_function` is now stored as a read-only property of a `Grid`
  instances and also a factor in determining instances equality in value.
- The default constructor of the `Grid` class now accepts as optional arguments
  `generating_function` (instead of `generating_values` which is deprecated)
  and `generating_points` both are defaulted to `None`.
  If `generating_function` is not specified, then the default of Leja-ordered
  Chebyshev-Lobatto generating function is selected. This preserves the 
  previous behavior of calling the constructor without any optional arguments.
- The factory method `from_generator()` of the `Grid` class has been renamed
  to `from_function()` to avoid confusion with the Python's term.
- The generating points provided or created by a generating function for
  an instance of the `Grid` class must now have unique values per column,
  otherwise an exception is raised.

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
