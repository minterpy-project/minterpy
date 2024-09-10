#################################
Polynomial Transformation Classes
#################################

An :doc:`abstract base class </api/core/ABC/abc-transformation>` is provided
as the blueprint from which every implementation of a polynomial basis
transformation class must be derived.

Below are the available transformation classes between the built-in
polynomial bases.

For most common use cases, the high-level :doc:`helper functions <interface>`
provides a convenient way to do basis transformations.

.. toctree::
   :maxdepth: 2

   transformation-lagrange
   transformation-newton
   transformation-canonical
   transformation-chebyshev
   transformation-identity
   interface
