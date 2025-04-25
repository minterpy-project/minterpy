==================================
Building the Documentation Locally
==================================

If you're making changes directly to the documentation source code,
it's a good idea to preview your changes locally before submitting them.
To do this, you'll need to build the documentation.

.. note::

   Building the documentation on your local system requires a specific set
   of dependencies.
   Be sure to install Minterpy with the ``[docs]`` or ``[all]`` options.

   See :ref:`Installing Minterpy for development <contributors/development-environment:Installing Minterpy for development>`
   for details.

Building the documentation
==========================

The Minterpy project uses `sphinx`_ to build its documentation.

To build the docs in the HTML format,
run the following command from within the Minterpy source directory:

.. code-block::

   sphinx-build -M html docs docs/build

Alternatively, you can use the supplied Makefile (if present).
To do this, navigate to the ``docs`` directory and run the appropriate command:

.. tab-set::

    .. tab-item:: Linux / Mac OS

       .. code-block::

          make html

    .. tab-item:: Windows

       .. code-block::

          make.bat html

This command builds the documentation and stores it in ``docs/build``.
You can view the documentation in a web browser by opening
``docs/build/html/index.html``.

.. rubric:: Live-reload

While you're working with the documentation,
you might prefer to have a live-reload.
By using  `sphinx-autobuild`_ (which is part of the docs requirements),
you can have the HTML documentation recompiled automatically every time
the sources changes.

You need to invoke the following command in the terminal
from the main Minterpy source directory
(take note -- it's *not* the ``docs`` directory; it's one level above it):


.. tab-set::

    .. tab-item:: Linux / Mac OS

       .. code-block:: bash

          sphinx-autobuild --ignore "*.ipynb" docs ./docs/build/html

    .. tab-item:: Windows

       .. code-block::

          sphinx-autobuild --ignore "*.ipynb" docs .\docs\build\html

.. note::

   The option ``--ignore "*.ipynb"`` is to exclude all Jupyter-notebook-based
   documentation from live reload as it creates an issue for sphinx-autobuild
   by keep detecting changes. If you modify the
   notebooks, you need to rebuild the documentation yourself.

Once you run the command, a local server will start and an address will be
displayed; open the address in your web browser.
sphinx-autobuild watches for any changes in the ``docs`` directory.
As soon as it detects them, it rebuilds the documentation automatically.

.. note::

   To shut down the local server and live reload,
   press ``Ctrl-C`` in the terminal.

.. rubric:: Generating LaTeX documentation

To generate the documentation in PDF format using ``pdflatex``
(which requires a LaTeX distribution installed on your system), run:

.. tab-set::

    .. tab-item:: Linux / Mac OS

       .. code-block::

          make latexpdf

    .. tab-item:: Windows

       .. code-block::

          make.bat latexpdf

This command builds the documentation as a PDF and stores it,
along with all the LaTeX source files, in ``docs/build/latex``.

Testing the Jupyter-notebook-based documentation
================================================

Jupyter-notebook-based documentation in Minterpy contains Python code that is
meant to be executed during the build process.
If the execution fails, then the documentation cannot be built properly.
Consequently, if you make changes to these documents or add new ones,
you should ensure that they can (still) be executed.

To test the Jupyter notebooks in the documentation,
execute the following command from the source directory:

.. tab-set::

    .. tab-item:: Linux / Mac OS

       .. code-block::

          pytest -vvv --nbmake "./docs/getting-started" "./docs/how-to"

    .. tab-item:: Windows

       .. code-block::

          pytest -vvv --nbmake ".\docs\getting-started" ".\docs\how-to"

The two directories represents the default locations for Jupyter-notebook-based
documentation.

.. important::

   These tests do not verify the correctness of the results presented in the
   notebooks but simply check whether the notebooks can be successfully executed
   from start to finish.

.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _sphinx-autobuild: https://github.com/executablebooks/sphinx-autobuild
