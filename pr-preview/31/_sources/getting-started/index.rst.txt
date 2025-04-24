#############################
Getting Started with Minterpy
#############################

Have you installed Minterpy?
Read on how to install it before moving on.

Installation
############

.. tab-set::

    .. tab-item:: release

       The public release of Minterpy (i.e., the `main branch`_) can be obtained
       directly from `PyPI`_ with `pip`_:

        .. code-block:: bash

           pip install minterpy

    .. tab-item:: dev

       The latest `development branch`_ of Minterpy can be obtained from
       its GitHub `repository`_:

       .. code-block:: bash

          git clone -b dev https://github.com/casus/minterpy

       After moving inside the cloned directory, the package can be installed
       from source with ``pip``:

       .. code-block:: bash

          pip install [-e] .

A best practice is to create a virtual environment so as not to install
external package to the your base Python environment.
You can do this with the help of, among others:

- `venv`_
- `virtualenv`_
- `mamba`_
- `conda`_

What's next?
############

If you're brand new to Minterpy and simply want to approximate a function using
polynomial interpolations, start with:

:doc:`functions-approximation`

While approximating functions using polynomials is a main feature of Minterpy,
it also offers multi-dimensional polynomials in Python.
These polynomials have a consistent interface that allows for advanced
manipulation such arithmetic and calculus operations.

To learn more about these features of Minterpy, follow the series of tutorials
below. We recommend that you go through these tutorials in sequence.

.. list-table:: Available Getting Started Guides (In-Depth Tutorials)
   :header-rows: 1

   * - If you want to...
     - Go to...
   * - understand Minterpy polynomials through approximating a 1D function
     - :doc:`1d-polynomial-interpolation`
   * - learn how to approximate mD function with polynomial interpolation
     - :doc:`md-polynomial-interpolation`
   * - know more about the supported *arithmetic operations*
       with Minterpy polynomials
     - :doc:`arithmetic-operations-with-polynomials`
   * - know more more the supported *calculus operations*
       with Minterpy polynomials
     - :doc:`calculus-operations-with-polynomials`
   * - understand the available polynomial bases and how to transform between
       them
     - :doc:`polynomial-bases-and-transformations`
   * - learn how to construct a polynomial from scattered data
     - :doc:`Polynomial Regression <polynomial-regression>`

Once you've become more familiar with Minterpy and need help to achieve
a particular task, be sure to check out the :doc:`/how-to/index`!

.. toctree::
   :maxdepth: 1
   :hidden:

   Functions approximation <functions-approximation>
   1D Polynomial Interpolation <1d-polynomial-interpolation>
   mD Polynomial Interpolation <md-polynomial-interpolation>
   Arithmetic with Polynomials <arithmetic-operations-with-polynomials>
   Calculus with Polynomials <calculus-operations-with-polynomials>
   Change of Basis <polynomial-bases-and-transformations>
   Polynomial Regression <polynomial-regression>

.. _main branch: https://github.com/casus/minterpy
.. _development branch: https://github.com/casus/minterpy/tree/dev
.. _conda: https://docs.conda.io/
.. _pip: https://pip.pypa.io/en/stable/
.. _pytest: https://docs.pytest.org/en/6.2.x/
.. _PyPI: https://pypi.org/project/minterpy/
.. _repository: https://github.com/casus/minterpy
.. _mamba: https://mamba.readthedocs.io/en/latest/
.. _virtualenv: https://virtualenv.pypa.io/en/latest/index.html
.. _venv: https://docs.python.org/3/library/venv.html