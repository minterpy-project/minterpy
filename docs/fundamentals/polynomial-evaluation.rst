==========================================
Evaluation of Multidimensional Polynomials
==========================================

Once a multidimensional polynomial :math:`Q` is obtained,
its value at an arbitrary query point :math:`\boldsymbol{x}_o \in \Omega`
can be computed.

In this page, we sketch how this value can be computed depending on the
:doc:`representation of the polynomial </fundamentals/polynomial-bases>`
:math:`Q`.

Canonical basis
===============

Consider an :math:`m`-dimensional polynomial :math:`Q \in \Pi_A` with
:math:`A \subseteq \mathbb{N}^m`
in the :ref:`fundamentals/polynomial-bases:Canonical basis`

.. math::

   Q(\boldsymbol{x}) = \sum_{\boldsymbol{\alpha} \in A}
   c_{\mathrm{can}, \boldsymbol{\alpha}} \, \prod_{j = 1}^m x_j^{\alpha_j},
   \quad c_{\mathrm{can}, \boldsymbol{\alpha}} \in \mathbb{R}.

Minterpy evaluates such a polynomial at a query point :math:`x_o` naively
by computing the value
:math:`x_o^{\boldsymbol{\alpha}} = \prod_{j = 1}^m x_{o, j}^{\alpha_j}`
for each basis polynomial (itself is a product),
multiplying these values by the corresponding
coefficients, and then summing them up.

.. warning::

   Such an approach, while straightforward to implement, is computationally
   expensive and prone to numerical instability and round-off errors,
   as it potentially involves handling very larger or very small intermediate
   values in the computing each term separately, especially when the polynomial
   degree is high.

The `Horner's method`_ is a well-known algorithm to stably and efficiently
evaluate a polynomial in the canonical basis.
Multidimensional generalizations of this method is available, such as the one
presented by Michelfeit (2020)\ :footcite:`Michelfeit2020`.
While this method is significantly faster and more stable than the current
Minterpy implementation, these advantages
come at the cost of requiring the factorization of the multi-dimensional
polynomial.

Newton basis
============

For the evaluation of polynomials in the :ref:`fundamentals/polynomial-bases:Newton basis`
the following important theorem\ :footcite:`Hecht2020` applies.

.. rubric:: Theorem: Newton evaluation

Let :math:`A \subseteq \mathbb{N}^m`, :math:`m\in \mathbb{N}`
be a multi-index set and
:math:`Q \in \Pi_A` be a polynomial in the Newton basis

.. math::

  Q(\boldsymbol{x}) = \sum_{\boldsymbol{\alpha} \in A}
  c_{\mathrm{nwt}, \boldsymbol{\alpha}} \, N_{\boldsymbol{\alpha}} (\boldsymbol{x})
  \in \Pi_A\,, \quad c_{\mathrm{nwt}, \boldsymbol{\alpha}} \in \mathbb{R}.

It then requires :math:`\mathcal{O}(m|A|)` operations and
:math:`\mathcal{O}(|A|)` storage to evaluate the value
:math:`Q(\boldsymbol{x}_o)` at :math:`\boldsymbol{x}_o \in \Omega`.

----

Minterpy realizes a generalized version of the classic
**Aitken-Neville algorithm**\ :footcite:`neville` upon which the theorem above
is based.
Additionally, with the :ref:`Leja ordering
<fundamentals/interpolation-at-unisolvent-nodes:Generating points>` of the
:ref:`unisolvent nodes
<fundamentals/interpolation-at-unisolvent-nodes:Unisolvent nodes>`,
the implementation is numerically stable, accurate, and efficient.

Lagrange basis
==============

Consider an :math:`m`-dimensional polynomial :math:`Q \in \Pi_A` with
:math:`A \subseteq \mathbb{N}^m`
in the :ref:`fundamentals/polynomial-bases:Lagrange basis`

.. math::

   Q(\boldsymbol{x}) = \sum_{\boldsymbol{\alpha} \in A}
   c_{\mathrm{lag}, \boldsymbol{\alpha}} \, L_\boldsymbol{\alpha} (\boldsymbol{x}),
   \quad c_{\mathrm{lag}, \boldsymbol{\alpha}} \in \mathbb{R},

where :math:`L_{\boldsymbol{\alpha}}` are the Lagrange basis polynomials.

Because there is no convenient way to derive a formula for the Lagrange
basis polynomials for a non-tensorial multi-index set, Minterpy cannot evaluate
the polynomial represented in the Lagrange basis directly.

Instead, the polynomial must first be transformed into either the
:ref:`fundamentals/polynomial-bases:Newton basis` or
:ref:`fundamentals/polynomial-bases:Canonical basis`,
after which the corresponding evaluation schemes can be applied
(as discussed above).

.. note::

   The transformation into the Newton basis, in particular, is efficiently
   implemented in Minterpy with machine precision (for reasonable instance
   sizes). The Newton basis, is therefore, currently the preferred basis
   for polynomial evaluation in Minterpy.

.. rubric:: References

.. footbibliography::

.. _Horner's method: https://en.wikipedia.org/wiki/Horner%27s_method
