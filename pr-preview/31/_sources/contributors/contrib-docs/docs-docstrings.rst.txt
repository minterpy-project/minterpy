========================================
Writing Docstrings for API Documentation
========================================

Docstrings ("documentation strings") are the main form of
the in-code documentation in Minterpy.
These are multi-line comments beneath a module, function, class, or method
that details their functionality.
All public components of Minterpy must have their own docstrings.

Below is a snippet of a Minterpy function that shows its docstring.

.. code-block:: python

   def get_poly_degree(exponents: np.ndarray, lp_degree: float) -> int:
       """Get the polynomial degree of a multi-index set for a given lp-degree.

       Parameters
       ----------
       exponents : np.ndarray
           An array of exponents of a multi-index set.

       lp_degree : float
           The lp-degree of the multi-index set.

       Returns
       -------
       int
           The polynomial degree from the multi-index set
           for the given lp-degree.
       """
       ...

Docstrings may be written with reStructuredText (reST) markup language
and thus support various formatting options like lists, mathematics, and
cross-references.

.. tip::

   If a docstring contains numerous reST formatting, then it may be a good idea
   to precede the opening ``"""`` with the character ``r`` such that Python
   will take the docstring literally and does not interpret backslashes as
   escape characters.

A docstring consists of two main components enclosed between two
sets of ``"""``:

- **One-line summary**: The first line of the docstrings written right after
  the opening ``"""``.
- **Body**: The rest of the docstrings until the closing ``"""``.

General convention
==================

Minterpy adheres to the  `Numpy Style Python Docstrings`_ format and
American English for its docstrings.

One-line summary
================

The one-line summary of a docstring should provide a concise and clear
explanation of what a module, class, method, or function does.
It should be written up to the 79-character mark.

For **functions** and **methods**, the docstring should be written
in the imperative mood.

For example:

- **Write**: *Get the polynomial degree of a multi-index set for a given lp-degree.*
  **Instead of**: *The function computes the polynomial degree of a multi-index set.*
- **Write**: *Build the cartesian product of any number of 1D arrays.*
  **Instead of**: *The cartesian product of any number of 1D arrays.*

For **modules** and **classes**, the docstring should be written
in indicative mood.

For example:

- **Write**: *The concrete implementation of polynomials in the Lagrange basis.*
  Instead of: *Implement the Lagrange basis.*
- **Write**: *A class to represent the set of multi-indices.*
  Instead of: *Represent the set of multi-indices.*

Body
====

The body of a docstring shall be separated by a single line from the start of
the body. The body can span multiple lines; be sure to limit each line up to
the 79-character mark.

The body may include consist of several sections, such as:

- **Parameters**: Described the parameters accepted by the function or method.
  Note that for instance methods, the parameter ``self`` does not need to be
  documented.
- **Returns**: Details what the function or method returns.
- **Detailed description**: Provides a more detail explanation of the module,
  class, method, or function focusing on its implementation and behavior.
  There is no need to rewrite the whole theory section about it.
- **Raises**: List any exceptions that the function or method might raise.
- **Examples**: Show examples of how to use the function or method.
- **Notes**: Provides additional information or content.

Functions and methods
=====================

For functions and methods, the following sections are mandatory (in the
following order):

- **Parameters**
- **Returns**

Optional sections that may also be included (in the following order):

- **Detailed description**
- **Raises**
- **Examples**
- **Notes**

With the exception of **Detailed description**, all sections must start with
the respective keyword (e.g., ``Parameters``) which should be underlined
with a series of hyphens (``-``) that matches the length of the keyword.
In the example above the sections **Parameters** and **Returns** are opened
by the keyword ``Parameters`` and ``Returns``, respectively each of which
is underlined with hyphens.

Parameters
----------

Here is an example of how the parameters section of a method is written:

.. code-block:: python

   """Create an instance from given spatial dim., poly., and lp-degrees.

        Parameters
        ----------
        spatial_dimension : int
            Spatial dimension of the multi-index set (:math:`m`); the value of
            ``spatial_dimension`` must be a positive integer (:math:`m > 0`).
        poly_degree : int
            Polynomial degree of the multi-index set (:math:`n`); the value of
            ``poly_degree`` must be a non-negative integer (:math:`n \geq 0`).
        lp_degree : float, optional
            :math:`p` of the :math:`l_p`-norm (i.e., the :math:`l_p`-degree)
            that is used to define the multi-index set. The value of
            ``lp_degree`` must be a positive float (:math:`p > 0`).
            If not specified, ``lp_degree`` is assigned with the value of
            :math:`2.0`.
   ...

Each parameter entry consists of three components:

- **Name**: The name of the parameter as it appears in the function
  or method definition.
- **Type**: The type of the parameter separated by a colon ":" from the name.
- **Description**: A multi-line description of the parameter.
  If the parameter is an array, include the shape of the array if possible.

All parameters in the function or method signature
(except for ``self`` in instance methods) should have its own entry
in the **Parameters** section.
If the function or method has no parameter, then the section is excluded.

There is no blank line separating each parameter entry.

Returns
-------

The **Returns** section is similar to the **Parameters** section, but it does
not include the **Name**.
Instead, it directly provides the type and description of the return value.

For instance:

.. code-block:: python

   def add_exponents(
       self,
       exponents: np.ndarray,
       inplace=False,
   ) -> Optional["MultiIndexSet"]:
       r"""

       Returns
       -------
       `MultiIndexSet`, optional
           The multi-index set with an updated set of exponents.
           If ``inplace`` is set to ``True``, then the modification
           is carried out in-place without an explicit output
           (it returns ``None``).
    ...
    """
    ...

Detailed description
--------------------

The Detailed Description of a function or method should appear after
the one-line summary and before the **Parameters** section.
It is written as multi-line comments and provides an in-depth explanation
of the function’s behavior, purpose, and any important details.

This section can be written rather freely, but should be focused and concise,
providing additional context or details that are not covered
in the one-line summary.

Below is an example of a **Detailed description** section taken from the
:py:func:`make_complete() <minterpy.utils.multi_index.make_complete>`
function.

.. code-block:: python

   def make_complete(exponents: np.ndarray, lp_degree: float) -> np.ndarray:
       """Create a complete exponents from a given array of exponents.

       A complete set of exponents contains all monomials, whose :math:`l_p`-norm
       of the exponents are smaller or equal to the polynomial degree of the set.

       Parameters
       ----------
       ...

       """
       ...

Raises
------

The **Raises** section contains any exceptions that a function or a method
might raise.

Below is an example of a **Raises** section taken from the
:py:func:`verify_spatial_dimension() <minterpy.utils.verification.verify_spatial_dimension>`
function.

.. code-block:: python

   def verify_spatial_dimension(spatial_dimension: int) -> int:
       """Verify if the value of a given spatial dimension is valid.

       ...
       Raises
       ------
       TypeError
           If ``spatial_dimension`` is not of a correct type, i.e., its
           strict-positiveness cannot be verified or the conversion to `int`
           cannot be carried out.
       ValueError
           If ``spatial_dimension`` is, for example, not a positive
           or a whole number.
       ...
       """
       ...

Each raised exception entry consists of:

- **Exception Type**: The specific type of exception that the function or
  method might raise.
- **Condition**: A description of the condition under which the exception is
  raised.

Examples
--------

The **Examples** section provides practical demonstrations of how to use the
function, method, or class.
Each example should be followed by the expected outcome,
illustrating what the code will return
or how it will behave given specific inputs.

Below is an example of a **Examples** section taken from the
:py:func:`lex_sort() <minterpy.utils.multi_index.lex_sort>` function.

.. code-block:: python

   def lex_sort(indices: np.ndarray) -> np.ndarray:
       """Lexicographically sort an array of multi-indices.

       Examples
       --------
       >>> xx = np.array([
       ... [0, 1, 2, 3],
       ... [0, 1, 0, 1],
       ... [0, 0, 0, 0],
       ... [0, 0, 1, 1],
       ... [0, 0, 0, 0],
       ... ])
       >>> lex_sort(xx)  # Sort and remove duplicates
       array([[0, 0, 0, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 1],
              [0, 1, 2, 3]])
       ...
       """
       ...

In the Examples section, each example should be structured by first defining
a specific input, passing that input through the function or method,
and then documenting the resulting output.
This clear format helps developers understand exactly how the function
behaves and what results to expect.

The Examples section can be particularly useful for small but important
functions, as it helps to clearly illustrate the input and output behavior.
Providing examples can convince developers of the function's correctness
and expected behavior, making it easier for them to integrate
and use the function effectively.

Notes
-----

The **Notes** section of a docstring is used to provide additional information
that doesn’t fit neatly into other sections.

This section can be useful for conveying:

- Important assumptions
- Implementation details
- Algorithmic details
- Performance considerations
- Particular behaviors
- Compatibility
- Limitations
- Caveats and potential side effects

For instance, the :py:meth:`is_disjoint() <minterpy.core.multi_index.MultiIndexSet.is_disjoint>`
method of the :py:class:`minterpy.core.multi_index.MultiIndexSet` has the following notes

.. code-block:: python

   def is_disjoint(
       self,
       other: "MultiIndexSet",
       expand_dim: bool = False,
   ) -> bool:
       """Return ``True`` if this instance is disjoint with another.

       ...

       Notes
       -----
       - The spatial dimension of the sets is irrelevant if one of the sets is
         empty.
       ...
       """
       ...

The notes above indicates a particular behavior when an empty set is involved.

Classes
=======

The docstring for class should be placed directly after the class definition,
not after the ``__init__()`` method (default constructor).
The structure of a class docstring is similar to that of a function or method,
but excludes the **Returns** section.
Furthermore, the **Parameters** section contains the list of parameters
used to create an instance via the default constructor.

Below is an example of a docstring for
the :py:class:`Grid <minterpy.core.grid.Grid>` class

.. code-block:: python

   class Grid:
       """A class representing the nodes on which interpolating polynomials live.

       Instances of this class provide the data structure for the unisolvent
       nodes, i.e., points in a hypercube that uniquely determine
       a multi-dimensional interpolating polynomial
       (of a specified multi-index set).

       Parameters
       ----------
       multi_index : MultiIndexSet
           The multi-index set of exponents of multi-dimensional polynomials
           that the Grid should support.
       generating_function : Union[GEN_FUNCTION, str], optional
           The generating function to construct an array of generating points.
           One of the built-in generating functions may be selected via
           a string as a key to a dictionary.
           This parameter is optional; if neither this parameter nor
           ``generating_points`` is specified, the default generating function
           based on the Leja-ordered Chebyshev-Lobatto nodes is selected.
       generating_points : :class:`numpy:numpy.ndarray`, optional
           The generating points of the interpolation grid, a two-dimensional
           array of floats whose columns are the generating points
           per spatial dimension. The shape of the array is ``(n + 1, m)``
           where ``n`` is the maximum degree of all one-dimensional polynomials
           (i.e., the maximum exponent) and ``m`` is the spatial dimension.
           This parameter is optional. If not specified, the generating points
           are created from the default generating function. If specified,
           then the points must be consistent with any non-``None`` generating
           function.

       Notes
       -----
       - The ``Callable`` as a ``generating_function`` must accept as its
         arguments two integers, namely, the maximum exponent (``n``) of all
         of the multi-index set of polynomial exponents and the spatial dimension
         (``m``). Furthermore, it must return an array of shape ``(n + 1, m)``
         whose values are unique per column.
       - The multi-index set to construct a :class:`Grid` instance may not be
         downward-closed. However, building a :class:`.MultiIndexTree` used
         in the transformation between polynomials in the Newton and Lagrange
         bases requires a downward-closed multi-index set.
       - The notion of unisolvent nodes, strictly speaking, relies on the
         downward-closedness of the multi-index set. If the set is not
         downward-closed then unisolvency cannot be guaranteed.
       """
       ...

A class may have attributes or properties; a property should be documented via
their getter.

For example, the :py:meth:`generating_points <minterpy.core.grid.Grid.generating_points>`
property of the :py:class:`Grid <minterpy.core.grid.Grid>` class:

.. code-block:: python

   @property
   def generating_points(self) -> np.ndarray:
       """The generating points of the interpolation Grid.

       The generating points of the interpolation grid are one two main
       ingredients of constructing unisolvent nodes (the other being the
       multi-index set of exponents).

       Returns
       -------
       :class:`numpy:numpy.ndarray`
           A two-dimensional array of floats whose columns are the
           generating points per spatial dimension. The shape of the array
           is ``(n + 1, m)`` where ``n`` is the maximum exponent of the
           multi-index set of exponents and ``m`` is the spatial dimension.
       """
       ...

Modules
=======

The docstrings for modules may be written rather freely, allowing for a
broad range of content.

Keep in mind that module-level docstrings will be integrated
nto the documentation page of the modules listed in the :doc:api/index`.
This means that whatever you include in the module-level docstring will
directly appear on the documentation page for that module.

For instance the docstring written for
the :py:mod:`minterpy.polynomials.lagrange_polynomial` is directly rendered
into the built documentation :doc:`/api/polynomials/lagrange`. You can verify
that the docstrings of the module is the source of the rendered documentation.

.. _Numpy Style Python Docstrings: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
