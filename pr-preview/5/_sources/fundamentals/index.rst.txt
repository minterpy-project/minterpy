========================
Fundamentals of Minterpy
========================

This section provides a detailed explanation of the mathematical foundation
behind Minterpy.

.. grid:: auto
   :margin: 0
   :padding: 0
   :gutter: 3

   .. grid-item-card:: Multidimensional Polynomial Interpolation
      :link: interpolation-problem
      :link-type: doc
      :columns: 12 8 8 6
      :class-card: sd-border-0, sd-card-hover-1

      *The problem*
      ^^^
      The polynomial interpolation problem is central to Minterpy.

      Revisit the polynomial interpolation problem and
      its connection to approximation on this page.

   .. grid-item-card:: Polynomial Bases
      :link: polynomial-bases
      :link-type: doc
      :columns: 12 8 8 6
      :class-card: sd-border-0, sd-card-hover-1

      *Representing polynomials*
      ^^^
      Polynomials can be expressed as a linear combination of a set of
      polynomials called the basis polynomials.

      Review the polynomial bases supported by Minterpy on this page.

   .. grid-item-card:: Interpolation at Unisolvent Nodes
      :link: interpolation-at-unisolvent-nodes
      :link-type: doc
      :columns: 12 8 8 6
      :class-card: sd-border-0, sd-card-hover-1

      *Getting interpolating polynomials*
      ^^^
      With a careful consideration, a polynomial that interpolate
      a given function can be constructed.

      Review the essential components required to build
      an interpolating polynomial on this page.

   .. grid-item-card:: Evaluation of Polynomials
      :link: polynomial-evaluation
      :link-type: doc
      :columns: 12 8 8 6
      :class-card: sd-border-0, sd-card-hover-1

      *Evaluating polynomials*
      ^^^
      Once a polynomial is obtained, how can it be evaluated at an arbitrary
      query point?

      The answer depends on the basis in which the polynomial is represented.

   .. grid-item-card:: Transformation between Bases
      :link: transformation
      :link-type: doc
      :columns: 12 8 8 6
      :class-card: sd-border-0, sd-card-hover-1

      *Changing polynomial basis*
      ^^^
      A polynomial expressed in one basis may be expressed in another.
      Depending on the purpose, some representations may be better
      than the others.

      On this page, the theorems behind such a transformation are presented.

   .. grid-item-card:: Polynomial Regression
      :link: polynomial-regression
      :link-type: doc
      :columns: 12 8 8 6
      :class-card: sd-border-0, sd-card-hover-1

      *Polynomial based on scattered data*
      ^^^
      Stable polynomial interpolation requires a careful selection of
      interpolation points; but what if scattered data is provided instead?

      Using the least squares method, you can still construct a polynomial
      from the data.

.. rubric:: Advanced topics

.. grid:: auto
   :margin: 0
   :padding: 0
   :gutter: 2

   .. grid-item-card:: Divided Difference Scheme (DDS)
      :link: dds
      :link-type: doc
      :columns: 12 6 6 4
      :class-card: sd-border-0, sd-card-hover-1

   .. grid-item-card:: Barycentric Transformation
      :link: barycentric-transformation
      :link-type: doc
      :columns: 12 6 6 4
      :class-card: sd-border-0, sd-card-hover-1

   .. grid-item-card:: The Notion of Unisolvence
      :link: unisolvence
      :link-type: doc
      :columns: 12 6 6 4
      :class-card: sd-border-0, sd-card-hover-1

.. toctree::
   :maxdepth: 3
   :hidden:

   mD Polynomial Interpolation <interpolation-problem>
   mD Polynomial Bases <polynomial-bases>
   interpolation-at-unisolvent-nodes
   Evaluation of mD Polynomials <polynomial-evaluation>
   transformation
   polynomial-regression
   mD Divided Difference Scheme <dds>
   barycentric-transformation
   unisolvence
