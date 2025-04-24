==========
Code Style
==========

This document summarizes code conventions used in Minterpy and intended
as a reference for Minterpy contributors.

.. note::

   The current codebase of Minterpy may not fully adhere to the rules specified
   in this document. The effort to bring the current codebase up to standard
   is on going. Just make sure that your upcoming changes adhere to the rules,
   though.

Style guide
===========

For the overall code style, Minterpy follows the `Google Python Style Guide`_
with some exceptions.
Notably, docstrings adheres to the  `Numpy Style Python Docstrings`_ format.

Formatting
==========

Minterpy adopts formatting ruleset according to `Black`_ which is a stricter
subset of `PEP8`_.

`Black`_  is a tool that enforces this formatting ruleset by automatically
modifying the codebase. However, we discourage direct automatic changes
and instead recommend running the following command
from the project root directory:

.. code-block::

   black --check --diff src tests

This command will highlight the lines where the rules are not followed so that
you can fix them manually.
The directories ``src`` and ``tests`` contain the codebase.

During the development process, you can check the format of your changes
via a predefined :doc:`pre-commit hook <pre-commits>` as follows:

.. code-block::

   pre-commit run black --all-files

This check will be automatically executed every time you commit a change
if the pre-commit is activated.

English spelling
================

Minterpy codebase and its documentation (including docstrings) are written in
American English. This means:

- "favor" instead of "favour"
- "color" instead of "colour"
- "center" instead of "centre"
- "modeling" instead of "modelling"
- "chips" instead of "crisps"
- etc.

.. _Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
.. _Numpy Style Python Docstrings: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
.. _Black: https://github.com/psf/black
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
