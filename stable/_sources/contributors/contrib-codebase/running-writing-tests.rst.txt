=========================
Running and Writing Tests
=========================

The Minterpy project strongly advocates for testing, and we promote the practice
of Test-Driven Development (TDD) *where practical*.
This methodology essentially involves writing (initially) failing tests before
starting with the developments, and then incrementally refining the implementation
until all the test pass by the end of the development cycle.

Staying strictly aligned with this practice can sometimes be challenging, especially
when you already have concrete ideas for implementing a feature or fixing an issue.
However, ensuring that any modifications you make are fully tested by the end
of the process is of critical importance.
Therefore, it's useful to consider "*how to test these changes*"
as you continue to modify and update your code.

Maintaining a comprehensive test suite--such that it is current and thorough--is
essential to ensure the stability and robustness of the Minterpy project.
As long as you prevent and rectify any regressions, a robust test suite acts
like a safety net; it gives you the confidence to make the necessary changes,
without constant worry that your modifications might break something unexpectedly.

.. note::

   If you're developing a new feature, improving an existing feature, or
   fixing a bug, the corresponding issue is the first location you should look
   for new test scenarios. You may need to expand that to cover more use
   cases, though.

Running the test suite
======================

The Minterpy project uses `pytest`_ to run the test suite of Minterpy.
The test suite is located inside the ``tests`` directory within the source
directory.

Once you've completed the :ref:`installation of Minterpy for development
<contributors/development-environment:Installing Minterpy for development>`,
you can run the complete test suite.
To do this, navigate to the Minterpy source directory and execute the following
command:

.. code-block::

   pytest

If you want the whole test suite to stop as soon as a test fails, execute:

.. code-block::

   pytest -x

And to have a verbose display, showing each individual test per line, execute:

.. code-block::

   pytest -vvv

----

During development, you might find it handy to run a specific test module instead
of running the whole test suite.
To do that, execute:

.. code-block::

   pytest tests/test_<test-module-name>.py [-k <regex-matching-name]

where the optional ``-k <regex-matching-name>`` further specifies particular
tests within the test module that have a name that matches the given regex.

Alternatively, you can precisely specify the particular tests to run with
one of the following constructs:

.. code-block::

   pytest tests/<test-module-name>.py::<test-class-name>
   pytest tests/<test-module-name>.py::<test-class-name>::<test-method-name>

The first construct executes all the tests in the ``<test-module-name>.py``
that are in the ``<test-class-name>`` class, while the second executes
one particular test within the same file and class.

.. note::

   You can combine the ``-x``, ``-vvv``, and a test selection options.

----

When you run pytest, a coverage test is automatically performed.
A summary of the coverage results will be printed in the terminal,
and you can find a detailed HTML version of the coverage report in
``htmlcov/index.html``.

.. note::

   Running the test suite requires a specific set of dependencies.
   Be sure to install Minterpy with the ``[dev]`` or ``[all]`` options.

   See :ref:`Installing Minterpy for development <contributors/development-environment:Installing Minterpy for development>`
   for details.

Organization of tests
=====================

The Minterpy test suite is located inside the ``tests`` directory within
the source directory; all future tests must be placed in this category.

When writing new tests, you should look browse into the existing tests to
have a feeling how things are currently done.
We encourage you to use an integrated development environment (IDE) to browse
the tests.

The tests are organized based the main components of Minterpy.
Here are some examples:

+----------------------------------+---------------------------------------------------------------------------------------------+
| Test module                      | Tested components                                                                           |
+==================================+=============================================================================================+
| ``test_multi_index.py``          | :py:class:`MultiIndexSet <.core.multi_index.MultiIndexSet>`                                 |
+----------------------------------+---------------------------------------------------------------------------------------------+
| ``test_grid.py``                 | :py:class:`Grid <.core.grid.Grid>`                                                          |
+----------------------------------+---------------------------------------------------------------------------------------------+
| ``test_ordinary_regression.py``  | :py:class:`OrdinaryRegression <.extras.regression.ordinary_regression.OrdinaryRegression>`  |
+----------------------------------+---------------------------------------------------------------------------------------------+
| ``test_polynomial.py``           | :py:class:`LagrangePolynomial <.polynomials.lagrange_polynomial.LagrangePolynomial>`,       |
|                                  | :py:class:`NewtonPolynomial <.polynomials.newton_polynomial.NewtonPolynomial>`,             |
|                                  | :py:class:`CanonicalPolynomial <.polynomials.canonical_polynomial.CanonicalPolynomial>`,    |
|                                  | :py:class:`ChebyshevPolynomial <.polynomials.chebyshev_polynomial.ChebyshevPolynomial>`     |
+----------------------------------+---------------------------------------------------------------------------------------------+
| ``test_polynomial_lagrange.py``  | :py:class:`LagrangePolynomial <.polynomials.lagrange_polynomial.LagrangePolynomial>`        |
+----------------------------------+---------------------------------------------------------------------------------------------+

A test module like ``test_polynomial.py`` tests the behavior of multiple
Minterpy components because they are expected to share common behaviors.
In this case, these components are concrete polynomial implementations.

On the other hand, a test modules like ``test_polynomial_lagrange.py``
is intended to specifically test the behaviors unique to the
:py:class:`LagrangePolynomial <.polynomials.lagrange_polynomial.LagrangePolynomial>` class
that are not shared with the other concrete polynomial classes.

Some tests are also organized by the low-level (numerical) operations.
For instance:

- ``test_verification.py`` for all the verification functions.
- ``test_multi_index_utils.py`` for all the underlying utility functions that support
  the :py:class:`MultiIndexSet <.core.multi_index.MultiIndexSet>` class.
- ``test_jit_multi_index.py`` for all the underlying Just-in-time-compiled numerical
  routines that support the :py:class:`MultiIndexSet <.core.multi_index.MultiIndexSet>` class.

These tests primarily verify that the functions perform as expected
by comparing their outputs with reference values.
Unlike the main components of Minterpy (e.g.,
:py:class:`MultiIndexSet <.core.multi_index.MultiIndexSet>`,
:py:class:`Grid <.core.grid.Grid>`),
these functions are typically smaller in size
and have a more narrowly defined scope.

----

Ideally, each test should be have a single, clear location where it belongs;
in practice, please rely on your common sense.

Writing new tests
=================

Once you know what tests to write and where to put them, write your test
in either *functional* style (single standing function) or *class-based*
(multiple tests wrapped in a class).
Minterpy uses *class-based* simply for organization purposes, i.e.,
to collect similar tests together and nothing more.

So for example, consider the following snippet:

.. code-block:: python

   class TestEquality:
       """All tests related to the equality check between polynomial instances."""

       def test_single(self, poly_class_all, multi_index_mnp):
           """Test equality between two instances with one set of coefficients."""
           # Generate random coefficient values
           coeffs = np.random.rand(len(multi_index_mnp))

           # Create two equal polynomials
           poly_1 = poly_class_all(multi_index_mnp, coeffs)
           poly_2 = poly_class_all(multi_index_mnp, coeffs)

           # Assertions
           assert poly_1 is not poly_2  # Not identical instances
           assert poly_1 == poly_2  # But equal in values
           assert poly_2 == poly_1  # Symmetric property

       def test_multiple(self, poly_class_all, multi_index_mnp, num_polynomials):
           """Test equality between two instances with multiple sets of coeffs."""
           # Generate random coefficient values
           coeffs = np.random.rand(len(multi_index_mnp), num_polynomials)

           # Create two equal polynomials
           poly_1 = poly_class_all(multi_index_mnp, coeffs)
           poly_2 = poly_class_all(multi_index_mnp, coeffs)

           # Assertions
           assert poly_1 is not poly_2  # Not identical instances
           assert poly_1 == poly_2  # But equal in values
           assert poly_2 == poly_1  # Symmetric property

The test class ``TestEquality`` inside ``test_polynomial.py`` module
contains two tests: testing the equality of a single polynomial and the equality
of a polynomial with multiple sets of coefficients. This organization
avoids having a very long name for a test to be specific.

On the other hand, if there is only a single behavior being tested, then
there is not much sense organizing the test inside a class.

Test fixtures are located in the ``conftest.py`` inside the ``tests`` directory.
Many of the fixtures are parameterized, meaning that using them in a test
would run through every combination of fixture values.

Additional tips
===============

We highly recommend that you leverage the features and conventions of pytest
for writing tests.

Please keep these important guidelines in mind:

- Initial tests should be written by the developers who are responsible for
  a feature; those tests may be modified and extended by later developers.
- Focus on testing the expected behavior of your code, not just the failure points.
- The tests aren't just to see if the code works, they're also to make sure
  that the code *continues* to work.
- Always aim for a highers code coverage possible.
- HOWEVER, be mindful that even 100% coverage doesn't guarantee every scenario
  has been covered (watch out for those edge cases!)

For additional reference on how to write tests, have a look at the following resources:

- `Pytest--Examples and customization tricks`_
- `Effective Python Testing with Pytest`_


.. _pytest: https://docs.pytest.org/en/stable/
.. _Pytest--Examples and customization tricks: https://docs.pytest.org/en/6.2.x/example/index.html
.. _Effective Python Testing with Pytest: https://realpython.com/pytest-python-testing/
