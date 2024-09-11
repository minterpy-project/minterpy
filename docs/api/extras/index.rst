===============
minterpy.extras
===============

The extras sub-packages provide additional capabilities for Minterpy
that extend beyond its core functionalities, but are not implemented as
separate independent packages.
While these extras are part of Minterpy, they rely on the Minterpy core
for their implementation,
but the core itself does not depend on the extras.

.. tip::

   If you're developing an extra sub-package, please ensure that its
   functionalities do not create dependencies for the core components of
   Minterpy (i.e., everything else outside extras).
   In other words, removing any extra sub-package should not break Minterpy
   core code base.

+-------------------------------+--------------------------------------------------------------------------------------+
| Sub-package                   | Description                                                                          |
+===============================+======================================================================================+
| :py:mod:`.extras.regression`  | The set of multi-indices representing the exponents of multidimensional polynomials  |
+-------------------------------+--------------------------------------------------------------------------------------+

.. toctree::
   :maxdepth: 1
   :hidden:

   Polynomial Regression <regression/index>
