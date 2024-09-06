===========
Pre-Commits
===========

To adhere to the code standards, you should run numerous checks before
committing and submitting the changes for a merge request.
To assist you in enforcing code standards, we recommend you to use `pre-commit`_
before commiting some changes to your branch.

.. note::

   `pre-commit`_ is installed if you follow the instructions on
   :ref:`Installing Minterpy for development <contributors/development-environment:Installing Minterpy for development>`.

Running and activating pre-commit
=================================

To run all the hooks defined in the ``.pre-commit-config.yaml`` file
against all relevant files, run:

.. code-block::

   pre-commit run --all-files

If you want to enable pre-commit, execute:

.. code-block::

   pre-commit install

This command will allow ``pre-commit`` to run against all files
when ``git commit`` is invoked.

If you want to disable the pre-commit script, execute:

.. code-block::

   pre-commit uninstall

.. note::

   To temporarily disable pre-commit checks when committing for a particular
   commit, use:

   .. code-block::

      git commit --no-verify

.. warning::

   The Minterpy project utilizes pre-commit hooks, with the required scripts
   listed in the ``.pre-commit-config.yaml`` file.
   These scripts carry out various checks in your code.

   While some hooks may alter the codebase, we restrict the modifications
   to less-opinionated (and definitely harmless) operations such as
   eliminating trailing whitespaces. Many of the pre-commit checks do not
   change the codebase. However, if the checks fail, it is your responsibility
   to address these issues yourself.

   You are encouraged to consider the suggestions provided by the hooks,
   but always do so with caution. Incorrect implementation may potentially
   cause the current code to fail.

   Even if some changes fail the pre-commit checks, they may still be committed
   and possibly merged later, provided the test suite passes.
   In fact, running some pre-commit hooks on the current codebase would
   definitely fail.

   In the future, we aim to comply with a comprehensive set of pre-commit hooks
   to further enhance the quality and consistency of the codebase.

Currently defined hooks
=======================

The table below summarizes the pre-commit hooks in the Minterpy projects.
Unless stated otherwise, the checks do not modify the codebase directly.

+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| Hook                         | Description                                                                      | Remarks                                                              |
+==============================+==================================================================================+======================================================================+
| `check-added-large-files`_   | Prevent giant files from being committed                                         |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `check-ast`_                 | Simply check whether files parse as valid python                                 |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `check-builtin-literals`_    | Require literal syntax when initializing empty or zero Python builtin types      |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `check-case-conflict`_       | Check for files with names that would conflict on a case-insensitive filesystem  |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `check-docstring-first`_     | Checks for a common error of placing code before the docstring                   |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `check-merge-conflict`_      | Check for files that contain merge conflict strings                              |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `check-toml`_                | Attempts to load all TOML files to verify syntax                                 |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `check-vsc-permalinks`_      | Ensures that links to vcs websites are permalinks                                |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `check-yaml`_                | Attempts to load all yaml files to verify syntax                                 |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `debug-statements`_          | Check for debugger imports and py37+ breakpoint() calls in python source         |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `detect-private-key`_        | Checks for the existence of private keys                                         |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `end-of-file-fixer`_         | Makes sure files end in a newline and only a newline                             | **Modifies the codes**                                               |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `fix-byte-order-marker`_     | Removes UTF-8 byte order marker                                                  | **Modifies the codes**                                               |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `name-tests-test`_           | Verifies that test files are named correctly                                     | Minterpy adopts ``test_*.py`` filename convention                    |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `no-commit-to-branch`_       | Protect specific branches from direct commits                                    | ``dev`` and ``main`` are protected                                   |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `trailing-whitespace`_       | Trims trailing whitespaces                                                       | **Modifies the codes**                                               |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `flake8`_                    | A Python linter (static code analyzer)                                           | See the configurations in the ``setup.cfg`` under ``[flake8]``       |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `black`_                     | A code formatter                                                                 | Only checks ``src`` and ``tests``                                    |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `black-jupyter`_             | A Jupyter notebook formatter                                                     | Only checks ``docs/how-to`` and ``docs/getting-started``             |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `mypy`_                      | A static type checker                                                            | See the configurations in the ``setup.cfg`` under ``[mypy*]``        |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `check-manifest`_            | A ``MANIFEST.in`` state checker                                                  |                                                                      |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+
| `isort`_                     | An import statement sorter                                                       | Make sure changes according to the suggestion do not break the code  |
+------------------------------+----------------------------------------------------------------------------------+----------------------------------------------------------------------+

.. _pre-commit: https://pre-commit.com
.. _check-added-large-files: https://github.com/pre-commit/pre-commit-hooks
.. _check-ast: https://github.com/pre-commit/pre-commit-hooks
.. _check-builtin-literals: https://github.com/pre-commit/pre-commit-hooks
.. _check-case-conflict: https://github.com/pre-commit/pre-commit-hooks
.. _check-docstring-first: https://github.com/pre-commit/pre-commit-hooks
.. _check-merge-conflict: https://github.com/pre-commit/pre-commit-hooks
.. _check-toml: https://github.com/pre-commit/pre-commit-hooks
.. _check-vsc-permalinks: https://github.com/pre-commit/pre-commit-hooks
.. _check-yaml: https://github.com/pre-commit/pre-commit-hooks
.. _debug-statements: https://github.com/pre-commit/pre-commit-hooks
.. _detect-private-key: https://github.com/pre-commit/pre-commit-hooks
.. _end-of-file-fixer: https://github.com/pre-commit/pre-commit-hooks
.. _fix-byte-order-marker: https://github.com/pre-commit/pre-commit-hooks
.. _name-tests-test: https://github.com/pre-commit/pre-commit-hooks
.. _no-commit-to-branch: https://github.com/pre-commit/pre-commit-hooks
.. _trailing-whitespace: https://github.com/pre-commit/pre-commit-hooks
.. _flake8: https://flake8.pycqa.org/en/latest/user/using-hooks.html
.. _black: https://black.readthedocs.io/en/stable/integrations/source_version_control.html
.. _black-jupyter: https://black.readthedocs.io/en/stable/integrations/source_version_control.html
.. _mypy: https://github.com/pre-commit/mirrors-mypy
.. _check-manifest: https://github.com/mgedmin/check-manifest
.. _isort: https://pycqa.github.io/isort/index.html
