=================================
Multidimensional Polynomial Bases
=================================

This page introduces the polynomial bases supported by Minterpy.
To extend the notion of polynomial degree and exponents to multiple dimensions,
the concept of multi-indices is introduced first.

Multi-index sets and polynomial degree
======================================

Multi-index sets :math:`A \subseteq \mathbb{N}^m` generalize the notion
of polynomial degree to multiple dimensions :math:`m \in \mathbb{N}`.

Downward-closedness
-------------------

We call a multi-index set *downward closed* if and only if there is no
:math:`\beta = (\beta_1, \dots, \beta_m) \in \mathbb{N}^m \setminus A`
with :math:`b_i \leq a_i`,  for all :math:`i=1, \dots, m`
and some :math:`\alpha = (\alpha_1, \dots, \alpha_m) \in A`.
This follows the classic terminology introduced for instance
by Cohen and Migliorati\ :footcite:`Cohen2018`.

Lexicographical ordering
------------------------

Any (not necessarily downward-closed) multi-index set
:math:`A\subseteq \mathbb{N}^m` is assumed to be ordered *lexicographically*
:math:`\preceq` from the last entry to the first, for example,
:math:`(5, 3, 1) \preceq (1, 0, 3) \preceq(1, 1, 3)`.

Complete multi-index set
------------------------

For :math:`\alpha=(\alpha_1, \ldots, \alpha_m) \in \mathbb{N}^m`,
we consider the :math:`l_p`-norm

.. math::

  \|\alpha\|_p  = (\alpha_1^p + \cdots +\alpha_m^p)^{1/p}

and denote the multi-index sets of bounded :math:`l_p`-norm by

.. math::

  A_{m,n,p} = \{\alpha \in \mathbb{N}^m :  \|\alpha\|_p \leq n \}\,, \quad p>1 \,.

Indeed, the sets :math:`A_{m,n,p}` are downward closed and called
*complete multi-index sets*.
The sets yield the relevant multi-indices when considering polynomials of
:math:`l_p`-degree :math:`n \in \mathbb{N}` in dimension
:math:`m \in \mathbb{N}`.

.. note::

   For a given :math:`n`, all :math:`A_{1, n, p}` are identical
   for all :math:`p > 0`.

Polynomial spaces
=================

Given a multi-index set :math:`A\subseteq \mathbb{N}^m`
in dimension :math:`m \in \mathbb{N}`,
consider the *polynomial spaces*

.. math::
   :label: eq_Pi_A

   \Pi_A =\langle \Psi_{\boldsymbol{\alpha}} : \alpha \in A \rangle

spanned by the *basis polynomials* :math:`\Psi_{\boldsymbol{\alpha}}`.

A *polynomial basis* consists of a distinct set of basis polynomials.
Many polynomial bases exist in the literature, and Minterpy supports a couple
of them, which are reviewed in the subsequent sections.

.. tip::

   A *polynomial basis* is an entire set of *basis polynomials*.

Given a polynomial basis and a multi-index set :math:`A`,
a multidimensional polynomial :math:`Q` can then be written as

.. math::
   :label: eq_poly_arbitrary_basis

   Q(\boldsymbol{x}) = \sum_{\boldsymbol{\alpha} \in A}
   c_{\cdot, \boldsymbol{\alpha}} \, \Psi_{\cdot, \boldsymbol{\alpha}}
   (\boldsymbol{x}) \in \Pi_A,

where :math:`\Psi_{\cdot, \boldsymbol{\alpha}}` and
:math:`c_{\cdot, \boldsymbol{\alpha}}` are the chosen basis polynomials and
the corresponding coefficients, respectively.

A polynomial in the form of :eq:`eq_poly_arbitrary_basis` is determined
by the chosen basis, the multi-index set :math:`A`, and the corresponding
coefficients
:math:`\left( c_{\cdot, \boldsymbol{\alpha}} \right)_{\boldsymbol{\alpha} \in A}
\in \mathbb{R}^{\lvert A \rvert}`. The coefficients are stored as an array
ordered according to the :ref:`lexicographical ordering
<fundamentals/polynomial-bases:Lexicographical ordering>`
:math:`\preceq` of the corresponding multi-index set.

.. note::

   The Lagrange and Newton polynomial basis, as will explained below,
   require an additionally set of generating points to be defined.

Canonical basis
===============

The basis polynomial of the canonical basis associated with a multi-index
element :math:`\boldsymbol{\alpha} \in A \subseteq \mathbb{N}^m` is defined as

.. math::

   \Psi_{\mathrm{can}, \boldsymbol{\alpha}} (\boldsymbol{x}) =
   x_1^{\alpha_1} \cdots x_m^{\alpha_m} =
   \prod_{j = 1}^m x_j^{\alpha_j},

where :math:`m` is the spatial dimension of the polynomial.

.. note::

   The canonical basis in Minterpy is synonymous with the *monomial basis*.

----

The crucial point of our general setup of multi-index sets
:math:`A \subseteq \mathbb{N}^m` can be observed by, for instance, realizing
that the multi-index element :math:`\lVert (2,2) \rVert_1 = 4  > 3`,
but :math:`\lVert (2,2) \rVert_2 = \sqrt{8} < 3` implies

.. math::

  x^2y^2 \not \in \Pi_{A_{2,3,1}}\,, \quad \text{but}\quad x^2y^2  \in \Pi_{A_{2,3,2}}\,.

In other words, the choice of :math:`l_p`-degree constrains the combinations
of monomials considered.
This fact is crucial for the approximation power of polynomials,
as asserted in the :doc:`Introduction </fundamentals/interpolation-problem>`.

Lagrange basis
==============

Given a multi-index set :math:`A \subseteq \mathbb{N}^m` in dimension
:math:`m \in \mathbb{N}` and a set of :ref:`unisolvent nodes
<fundamentals/interpolation-at-unisolvent-nodes:Unisolvent nodes>` defined
by the sub-grid

.. math::

  P_A = \left\{ p_\alpha = (p_{\alpha_1, 1}, \ldots, p_{\alpha_m,m})
  \in \Omega\subseteq \mathbb{R}^m : \alpha \in A\right\}\,, \quad p_{\alpha_j, j}
  \in P_j \subseteq [-1,1]\,.

which is specified by the chosen :ref:`generating points
<fundamentals/interpolation-at-unisolvent-nodes:Generating points>`
:math:`\mathrm{GP} = \oplus_{j=1}^m P_j`,
the Lagrange basis polynomial :math:`L_{\boldsymbol{\alpha}}`
are uniquely determined by their requirement to satisfy

.. math::

  L_{\boldsymbol{\alpha}}(p_{\boldsymbol{\beta}}) =
  \delta_{\boldsymbol{\alpha}, \boldsymbol{\beta}}, \quad
  p_{\boldsymbol{\beta}} \in P_A,

where :math:`\delta_{\cdot, \cdot}` denotes the *Kronecker delta*.

To be fully determined, polynomials in the Lagrange basis also require
the set of unisolvent nodes :math:`P_A` to be defined.

Newton basis
============

Given a multi-index set :math:`A \subseteq \mathbb{N}^m` in dimension
:math:`m \in \mathbb{N}` and a set of :ref:`unisolvent nodes
<fundamentals/interpolation-at-unisolvent-nodes:Unisolvent nodes>` defined
by the sub-grid

.. math::

  P_A = \left\{ p_\alpha = (p_{\alpha_1, 1}, \ldots, p_{\alpha_m, m})
  \in \Omega\subseteq \mathbb{R}^m : \alpha \in A\right\}\,, \quad p_{\alpha_j, j}
  \in P_j\,.

which is specified by the chosen :ref:`generating points
<fundamentals/interpolation-at-unisolvent-nodes:Generating points>`
:math:`\mathrm{GP} = \oplus_{j = 1}^m P_j`,
the Newton monomials :math:`N_\alpha` are defined by

.. math::
   :label: eq_newton_basis

   N_\alpha(x) = \prod_{j = 1}^m \prod_{i=0}^{\alpha_j - 1}(x_j - p_{i, j})\,,\quad  p_{i, j} \in P_i

which generalizes their classic form from one dimension to multiple
dimensions\ :footcite:`stoer2002,gautschi2012`.

To be fully determined, and as noted in :eq:`eq_newton_basis`,
polynomials in the Newton basis require
the set of generating points :math:`\mathrm{GP}` to be defined.

Chebyshev basis
===============

The basis polynomial of the Chebyshev basis (of the first kind) associated with
a multi-index element :math:`\boldsymbol{\alpha} \in A \subseteq \mathbb{N}^m`
is defined as

.. math::

   \Psi_{\mathrm{cheb}, \boldsymbol{\alpha}} (\boldsymbol{x}) =
   T_{\alpha_1} (x_1) \cdots T_{\alpha_m} (x_m) =
   \prod_{j = 1}^m T_{\alpha_j} (x_j),

where :math:`T_{\alpha_j} (x_j)` is the :math:`\alpha_j`th-degree
(one-dimensional) Chebyshev polynomial of the first kind associated
with the :math:`j`-th-dimension.
In other words, the multidimensional basis polynomial is constructed by
taking the tensor product of one-dimensional basis polynomial

The one-dimensional Chebyshev basis polynomial of the first kind satisfies
the following three-term recurrence (TTR) relation:

.. math::

   \begin{aligned}
      T_0 (x) & = 1 \\
      T_1 (x) & = x \\
      T_{n + 1} (x) & = 2 x T_n (x) - T_{n - 1} (x)
   \end{aligned}

.. rubric:: References

.. footbibliography::
