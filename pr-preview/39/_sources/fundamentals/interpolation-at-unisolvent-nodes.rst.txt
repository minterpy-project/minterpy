=================================
Interpolation at Unisolvent Nodes
=================================

The construction of a multi-dimensional interpolating polynomial relies on
a set of carefully selected points in the domain and a chosen polynomial basis.
The points must be selected such that the polynomial of a given degree can
be uniquely determined, that is, the points are unisolvent.

This page introduces the required points for this construction and the selected
polynomial basis for interpolation.

Generating points
=================

The generating points of degree :math:`n_j` are selected by choosing arbitrary
set of points :math:`P_j \subseteq [-1,1]`
with :math:`|P_j| = n_j + 1 \in \mathbb{N}` points
for each dimension :math:`m \in \mathbb{N}`,
such that for a fixed
:ref:`multi-index set
<fundamentals/polynomial-bases:Multi-index sets and polynomial degree>`
:math:`A \subseteq \mathbb{N}`, it holds that
:math:`n_j \geq \max \{\alpha_j : \alpha \in A\}`.

.. rubric:: Chebyshev-Lobatto points

When :math:`A = A_{m, n, p}` is defined by a multi-index set of
specified polynomial degree :math:`n` and math:`l_p`-degree :math:`p`,
the points are typically chosen to tbe the *Chebyshev-Lobatto* points:

.. math::
  :label: eq_Cheb

  P_i =\{p_{0,i},\dots,p_{n,i}\} = (-1)^m \mathrm{Cheb}_n = \left\{ \cos(k\pi/n) :  0 \leq k \leq n\right\}

Other prominent choices of generating points are the `Gauss-Legendre points`_,
`Chebyshev points`_ of first \& second kind, equidistant points,
see for example\ :footcite:`stoer2002,trefethen2019,gautschi2012`.

.. rubric:: Matrix of generating points

The generating points are stored as a matrix where each column represents
the chosen set of points for a particular dimension.
Specifically, the matrix is constructed by stacking the chosen set of points
for each dimension column-wise.
That is,

.. math::

  \mathrm{GP} = \oplus_{j=1}^m P_j\,.

.. rubric:: Leja ordering

The choice of the ordering of the generating points is crucial for the
stability of the interpolating polynomial.
We assume that the generating points :math:`P_j` are
**Leja-ordered**\ :footcite:`Leja1957`, which means that they satisfy the
following conditions:

.. math::
  |p_0| = \max_{p \in P}|p|\,, \quad \prod_{i=0}^{j-1}|p_j-p_i| = \max_{j\leq k\leq m} \prod_{i=0}^{j-1}|p_k-p_i|\,,\quad 1 \leq j \leq n\,.

The Leja ordering of the generating points not only ensures the numerical
stability of the :ref:`Newton interpolation
<fundamentals/interpolation-at-unisolvent-nodes:Newton interpolation>`,
but also results in :ref:`unisolvent nodes
<fundamentals/interpolation-at-unisolvent-nodes:Unisolvent nodes>` forming
a (sparse, non-tensorial) grid with high approximation power.

Unisolvent nodes
================

For a :ref:`multi-index set
<fundamentals/polynomial-bases:Multi-index sets and polynomial degree>`
:math:`A \subseteq \mathbb{N}^m` and chosen :ref:`generating points
<fundamentals/interpolation-at-unisolvent-nodes:Generating points>`
:math:`\mathrm{GP} = \oplus_{j = 1}^m P_j` the unisolvent nodes :math:`P_A`
are given as the sub-grid

.. math::

  P_A = \left\{ p_\alpha = (p_{\alpha_1, 1}, \ldots, p_{\alpha_m,m})
  \in \Omega\subseteq \mathbb{R}^m : \alpha \in A\right\}\,, \quad p_{\alpha_j, j}
  \in P_j\,.

In addition to the possibly different choices of the :math:`P_j`
in each dimension, the impact of different orderings can be observed
in the example below (:numref:`un_1`).

.. _un_1:
.. figure:: ./images/UN1.png
  :align: center

  Unisolvent nodes for :math:`A= A_{2,3,1}` (left, middle) and :math:`A_{2,3,2}` (right). Orderings in :math:`x,y`--directions are indicated by numbers and non-tensorial nodes  :math:`p=(p_x,p_y) \in P_A` in red with missing symmetric
  blank counter parts :math:`(p_y,p_x)\not \in P_A`.

In the figure, the sets :math:`P_j` differ only in their orderings,
as indicated by the enumerations along the dimensions (:math:`x, y`).
The resulting unisolvent nodes may form *non-tensorial* or
*non-symmetric grids*, where nodes exist
with :math:`p=(p_x, p_y) \in P_A`, but :math:`(p_y, p_x) \not \in P_A`.

Examples of unisolvent nodes in two dimensions and three dimensions
for the default choice of the :doc:`Leja-ordered Chebyshev-Lobatto nodes
</fundamentals/interpolation-at-unisolvent-nodes>` are visualized below.

.. figure:: ./images/Nodes.png
  :align: center

  Leja ordered Chebyshev-Lobatto nodes for Euclidian :math:`l_2`-degree
  :math:`n = 5`.

From a general perspective, a more detailed discussion of their construction
and resulting properties is available\ :footcite:`Hecht2020`.
Crucially, for :ref:`downward-closed <fundamentals/polynomial-bases:Downward-closedness>`
multi-index sets :math:`A \subseteq \mathbb{N}^m`, the interpolating polynomial
:math:`Q_{f,A}` is uniquely determined in the :ref:`polynomial space
<fundamentals/polynomial-bases:Polynomial spaces>`

.. math::

   \Pi_A =\left<x^\alpha = x_1^{\alpha_1}\cdots x_m^{\alpha_m} : \alpha \in A\right>

spanned by all :ref:`canonical basis polynomials <fundamentals/polynomial-bases:Canonical basis>`.

Lagrange interpolation
======================

Given:

- a :ref:`multi-index set
  <fundamentals/polynomial-bases:Multi-index sets and polynomial degree>`
  :math:`A \subseteq \mathbb{N}^m` for :math:`m \in \mathbb{N}`,
- :ref:`unisolvent nodes
  <fundamentals/interpolation-at-unisolvent-nodes:Unisolvent nodes>`
  :math:`P_A \subseteq \Omega = [-1,1]^m`, and
- a function :math:`f: \Omega\longrightarrow \mathbb{R}`,

the uniquely determined interpolating polynomial in the
:ref:`fundamentals/polynomial-bases:Lagrange basis`
:math:`Q_{f,A}` of :math:`f` is given by

.. math::

  Q_{f,A}(x) = \sum_{\alpha \in A}f(p_{\alpha})L_{\alpha}(x)\,, \quad p_{\alpha} \in P_A\,,

where :math:`L_\alpha` denote the Lagrange basis polynomial
that satisfy :math:`L_{\alpha}(p_\beta) = \delta_{\alpha,\beta}` with
:math:`\delta_{\cdot,\cdot}` denoting the **Kronecker delta**.

In fact, deriving the interpolating polynomial in the Lagrange basis
:math:`Q_{f,A}` of a function :math:`f` is straightforward
and can be obtained with linear :math:`\mathcal{O}(|A|)`
storage amount and runtime complexity provided that
the unisolvent nodes :math:`P_A` are given.

However, interpolation polynomial in the Lagrange basis is rather a theoretical
construct in Minterpy because the computational scheme for evaluating
the polynomial at a query point is non-trivial.
In particular, the closed-form formula for the
Lagrange monomials is difficult to derive for non-tensorial multi-index sets.

Newton interpolation
====================

As an alternative to the interpolating polynomial in the Lagrange basis,
the same polynomial in the Newton basis is given by

.. math::

  Q_{f,A}(x) = \sum_{\alpha \in A} c_{\mathrm{nwt}, \alpha} \, N_{\alpha}(x),

where :math:`A \subseteq \mathbb{N}^m` is a set of multi-indices,
:math:`N_\alpha` are the :ref:`Newton basis polynomial
<fundamentals/polynomial-bases:Newton basis>` with respect to
the generating points :math:`\mathrm{GP}`,
and :math:`c_{\mathrm{nwt}, \alpha}` are the corresponding Newton coefficients.

The coefficients :math:`c_{\mathrm{nwt}, \alpha}` can be derived by the
:doc:`multi-dimensional divided difference scheme (DDS)
</fundamentals/dds>`
for a given generating points :math:`\mathrm{GP}`.
The scheme requires quadratic :math:`\mathcal{O}(|A|^2)`
runtime complexity and linear :math:`\mathcal{O}(|A|)` storage.

.. rubric:: References

.. footbibliography::

.. _Gauss-Legendre points: https://en.wikipedia.org/wiki/Gauss–Legendre_quadrature
.. _Chebyshev points: https://en.wikipedia.org/wiki/Chebyshev_nodes