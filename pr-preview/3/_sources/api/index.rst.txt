=============
API Reference
=============

:Release: |version|
:Date: |today|

This section of the documentation provides a comprehensive overview of the available Minterpy
public modules, classes, methods, and functions.

+--------------------------------------------------------+-----------------------------------------------------------------+
| Module / Sub-package                                   | Description                                                     |
+========================================================+=================================================================+
| :py:mod:`Interpolation <minterpy.interpolation>`       | A high-level interface to conveniently create interpolants      |
+--------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`Core <minterpy.core>`                         | Top domain-specific classes and abstract classes                |
+--------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`Polynomial Bases <minterpy.polynomials>`      | Concrete classes representing polynomial bases                  |
+--------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`Transformations <minterpy.transformations>`   | Concrete classes for polynomial basis transformations           |
+--------------------------------------------------------+-----------------------------------------------------------------+
| :py:mod:`Transformation Operators <minterpy.schemes>`  | Concrete classes for polynomial basis transformation operators  |
+--------------------------------------------------------+-----------------------------------------------------------------+
| :doc:`Extras </api/extras/index>`                      | Features that are not part of the core package                  |
+--------------------------------------------------------+-----------------------------------------------------------------+
| :doc:`Internal </api/internal/index>`                  | Utility modules and low-level numerical routines                |
+--------------------------------------------------------+-----------------------------------------------------------------+

.. toctree::
   :maxdepth: 4
   :hidden:

   Interpolation <interpolation>
   Core <core/index>
   Polynomials Bases <polynomials/index>
   Basis Transformations <transformations/index>
   Transformation Operators <transformation-operators/index>
   Extras <extras/index>
   internal/index
