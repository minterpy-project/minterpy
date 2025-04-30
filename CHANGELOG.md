# Changelog

All notable changes to the Minterpy project is documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [Version 0.3.1] - 2025-04-30

This minor release incorporates feedback from the review process of
the Minterpy submission to the Journal of Open Source Software (JOSS).
There are no changes to the package's functionality.

### Added

- Included the source code and manuscript for the JOSS paper in the repository.

### Changed

- Revised and updated several documentation components, including `README.md`,
  `CONTRIBUTING.md`, and the Getting Started guides.

## Version 0.3.0 - 2024-12-20

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
  implementation; may accelerate computation for larger problems).
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
  instances of all concrete classes. The implementation includes left-side and
  right-side multiplication
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
- Static abstract method `_scalar_add()` is now included in the
  `MultivariatePolynomialSingleABC` as a placeholder for the concrete
  implementation of scalar addition for polynomials.
- Instances of `MultiIndexSet` may now be multiplied with each other;
  the result is an instance of `MultiIndexSet` whose exponents are
  the cross-product-and-sum of the two operands exponents.
- A method `expand_dim()` is introduced to instances of the `Grid` class
  to encapsulate the procedure of expanding the dimension of a `Grid` instance.
- Add two new factory methods for the `Grid` class: `from_degree()` to
  create a `Grid` instance with a complete multi-index set and `from_points()`
  to create an instance with a given array of generating points.
- Instances of `Grid` has now `has_generating_function` property that returns
  `True` if a generating function is defined on the grid and `False` otherwise.
- Instances of `MultiIndexSet` has now `max_exponent` and `max_exponents`
  properties. The former is the maximum exponent across all dimensions in the
  multi-index set, while the latter is the maximum exponents per dimension.
- An instance of `Grid` is now a callable; when it is called with a callable
  or a function, the function will be evaluated on the unisolvent nodes of the
  grid and the corresponding function values are returned.
- Instances of `Grid` may now be multiplied with each other via the `*`
  operator; this operation returns a new instance of `Grid` whose multi-index
  set is the product of the multi-index sets of the operands. Procedures are
  implemented such that only instances that are compatible (w.r.t generating
  function or points) with each other can be multiplied.
- Instances of `Grid` may now be unionized with each other via the `|`
  operator; this operation returns a new instance of `Grid` whose multi-index
  set is the union of the multi-index sets of the operands. Procedures are
  implemented such that only instances that are compatible (w.r.t generating
  function or points) with each other can be unionized.
- The method `make_downward_closed()` is now available for instances of
  the `Grid` class. Calling the method results in a new instance of `Grid`
  whose underlying multi-index set is downward-closed.
- A new factory method `from_grid()` is available for all concrete polynomial
  classes. Calling the method with an instance of `Grid` creates a polynomial
  with the given grid and with the multi-index of the given grid.
- All concrete polynomial classes now inherit `__len__()` from the abstract
  base class. Calling `len()` on a polynomial instance returns the number
  of coefficient sets the instance has.
- Polynomial-polynomial multiplication is now supported for polynomials in the
  canonical basis.
- Instances of all polynomial bases may now be exponentiated by a non-negative
  scalar whole number. Exponentiation by zero returns a constant polynomial
  whose constant term is `1`. Because the underlying implementation relies on
  the polynomial-polynomial multiplication, polynomials in the Lagrange basis
  may only be exponentiated with either `0` or `1`.
- A new public method `is_compatible()` is introduced in the `Grid`
  class to verify compatibility between two grid instances
  based on their generating functions and points.
- Instances of all polynomial bases may now be divided by a real scalar number.
  Both the `/` (true division) and `//` (floor division) operators are
  supported.
- Overhauled the "Getting Started" section in the documentation to include
  a quickstart guide for Minterpy and six in-depth tutorials.
  The tutorials cater to both beginner and advanced levels,
  starting from scratch and gradually introducing sophisticated
  Minterpy features.
- Added support for exponentiation of polynomials, which extents the support for
  arithmetic manipulations of polynomials.
- Added polynomial bases transformations to the Interpolant class.
- Major update to the contributors guide section of the documentation.
- Added `sphinx_design` as new dependency for the documentation.
- Added support for polynomial-scalar division.

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
- The method `add_points()` of the `Grid` class has been renamed to
  `add_exponents()` to conform with a method of the same name of the
  `MultiIndexSet` class that handle the process of adding exponents
  to the underlying multi-index set of a `Grid` instance.
- Polynomial coefficients will now be stored as an array of `numpy.float64`
  as expected by Numba. Conversion will always be attempted.
- Polynomial-polynomial addition and multiplication of polynomials in the
  Chebyshev basis are now supported.
- Support for scalar addition and subtraction are moved to the abstract polynomial base class,
  which allows distinct implementations for the different polynomial bases.
- Polynomial-polynomial addition/subtraction of Newton polynomials with a common grid
  add the coefficients of their matching multi-index set elements instead of using base
  transformations.
- Revision and reorganization of the fundamentals section of the documentation.
- Reorganization of the API reference section of the documentation.
- Refactoring of the multiplication of all polynomial instances with a constant scalar
  polynomial, to avoid the usage of standard polynomial-polynomial multiplication and its
  related issues.
- Refactoring of the multiplication of all polynomial instances with a constant scalar
  polynomial, to avoid the usage of standard polynomial-polynomial multiplication and its
  related issues.

### Removed

- The method `apply_func()` of the `Grid` class previously not implemented
  (calling it raises an exception) is now removed completely
  in favor of `__call__()`.
- General polynomial-polynomial addition/subtraction and multiplication
  for polynomials in the Lagrange basis is now removed to avoid unnecessary
  coupling with the Newton polynomial class and the corresponding
  transformation class.

# Version 0.2.0-alpha - 2023-01-06

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

[Unreleased]: https://github.com/minterpy-project/minterpy/compare/main...dev
[0.3.1]: https://github.com/minterpy-project/minterpy/compare/v0.3.0...v0.3.1
