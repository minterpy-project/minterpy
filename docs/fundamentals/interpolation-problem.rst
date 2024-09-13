=========================================
Multidimensional Polynomial Interpolation
=========================================

Polynomial interpolation dates back to Newton, Lagrange,
and others\ :footcite:`meijering2002`,
and its fundamental importance in both mathematics and computing is widely
recognized.

Interpolation is rooted in the principle that, in one dimension,
there exists exactly one polynomial :math:`Q_{f, A}` of degree
:math:`n \in \mathbb{N}` that *interpolates* a function
:math:`f : \mathbb{R} \longrightarrow \mathbb{R}` on :math:`n+1` distinct
*interpolation nodes* :math:`P_A = \{p_i\}_{i \in A} \subseteq \mathbb{R}`
where :math:`A=\{0, \ldots, n\}`.
Specifically, this means

.. math::

  Q_{f,A}(p_i) = f(p_i)\,, \quad \text{for all} \quad  p_i \in P_A \,, i \in A\,.

This makes interpolation fundamentally different from approximation,
see :numref:`1d_interpol`.

.. _1d_interpol:
.. figure:: ./images/1D_interpol.png
   :align: center

   Interpolation and approximation in one dimension.
   While the interpolant :math:`Q_{f, A}` must match the function :math:`f`
   at the interpolation nodes :math:`p_0, \ldots, p_n`,
   an approximation :math:`Q^*_n` is not required to coincide with :math:`f`
   at all.

Weierstrass Approximation Theorem
=================================

The famous *Weierstrass Approximation Theorem*\ :footcite:`weierstrass1885`
asserts that any continuous function :math:`f : \Omega \longrightarrow \mathbb{R}`,
defined on a compact domain, such as :math:`\Omega = [-1,1]^m`,
can be uniformly approximated by polynomials\ :footcite:`debranges1959`.
However, the theorem does not require the polynomials to coincide with
:math:`f` at any given points.
In fact, it is possible to have a sequence of multidimensional polynomials
:math:`Q_n^*`, :math:`n \in \mathbb{N}`, where :math:`Q_n^*(x) \neq f(x)`
for all :math:`x \in \Omega`, but still achieve uniform approximation,
that is

.. math::

  Q_{n}^* \xrightarrow[n \rightarrow \infty]{} f \quad \text{uniformly on} \quad \Omega\,.

There are several constructive proofs of the Weierstrass approximation theorem,
including the well-known version by Serge Bernstein\ :footcite:`bernstein1912`.
The resulting *Bernstein approximation scheme* is universal and has been shown
to reach the optimal (inverse-linear)
approximation rate for the absolute value function :math:`f(x) = |x|`
\ :footcite:`bernstein1914`.
However, despite its generality, the Bernstein scheme achieves only slow
convergence rates for analytic functions, leading to high computational costs
in practical applications.

As a result, extensive research has focused on extending one-dimensional
**Newton** or **Lagrange interpolation schemes** to multiple dimensions (mD)
while preserving their computational efficiency.
Any method addressing this challenge must avoid Runge's
phenomenon\ :footcite:`runge1901` (overfitting) by ensuring uniform
approximation of the interpolation target
:math:`f : \Omega \longrightarrow \mathbb{R}`.
Additionally, it resist the *curse of dimensionality*,
achieving highly accurate polynomial approximations
for general multidimensional functions with a sub-exponential demand
for data samples :math:`f(p)\,, p \in P_A`, where :math:`|P_A| \in o(n^m)`.

Lifting the curse of dimensionality
===================================

To address the problem we consider
the :math:`l_p`-norm
:math:`\|\alpha\|_p = (\alpha_1^p + \cdots +\alpha_m^p)^{1/p}`,
:math:`\alpha = (\alpha_1,\dots,\alpha_m) \in\mathbb{N}^m`,
:math:`m \in \mathbb{N}` and the
:ref:`lexicographically-ordered <fundamentals/polynomial-bases:Lexicographical ordering>`
and :ref:`complete multi-index set <fundamentals/polynomial-bases:Complete multi-index set>`

.. math::
  :label: eq_A

  A_{m,n,p} = \left\{\alpha \in \mathbb{N}^m : \|\alpha\|_p \leq n \right\}\,, \quad m,n \in \mathbb{N}\,, p \geq 1\,.

This concept generalizes the one-dimensional notion of polynomial degree
to multi-dimensional :math:`l_p`-degree.
Specifically, we consider the polynomial spaces spanned
by all monomials with a bounded :math:`l_p`-degree, that is

.. math::

   \Pi_A =  \langle x^\alpha = x^{\alpha_1}\cdots x^{\alpha_m} : \alpha \in A \rangle \,, A =A_{m,n,p}\,.

Given :math:`A=A_{m,n,p}`, we ask for:

1. Unisolvent interpolation nodes :math:`P_A` that uniquely determine
   the interpolant :math:`Q_{f,A} \in \Pi_A` by satisfying
   :math:`Q_{f,A}(p_{\alpha}) = f(p_{\alpha})`,
   :math:`\forall p_{\alpha} \in P_A`, :math:`\alpha \in A`.

2. An interpolation scheme that computes the uniquely determined interpolant
   :math:`Q_{f,A} \in \Pi_A` efficiently and with numerical accuracy
   (machine precision).

3. The unisolvent nodes :math:`P_A` that scale sub-exponentially with
   the spatial dimension :math:`m \in \mathbb{N}`,
   :math:`|P_A| \in o(n^m)` and guarantee uniform approximation of even strongly
   varying functions (avoiding over fitting), such as as the Runge function
   :math:`f_R(x) = 1/(1+\|x\|^2)`, by achieving fast (ideally exponential)
   approximation rates.

In fact, the results of\ :footcite:`Hecht2020` suggest that the
:doc:`multidimensional DDS </fundamentals/dds>`
resolves issues 1--3 for the so called *Trefethen functions*
by selecting the Euclidian :math:`l_2`-degree
and :doc:`Leja ordered Chebyshev-Lobatto unisolvent interpolation nodes
</fundamentals/interpolation-at-unisolvent-nodes>`.
Thus,

.. math::

  |P_A| \approx \frac{(n+1)^m }{\sqrt{\pi m}} (\frac{\pi \mathrm{e}}{2m})^{m/2} \in o(n^m)\,, \quad  A=A_{m,n,2}\,,

scales sub-exponentially with space dimension :math:`m` and

.. math::

  Q_{f,A_{m,n,2}} \xrightarrow[n\rightarrow \infty]{} f

converges uniformly and fast (exponentially) on :math:`\Omega = [-1,1]^m`.

:numref:`mip_approximation` shows the approximation rates for the classic
Runge function\ :footcite:`runge1901` in dimension :math:`m = 4`,
which is known to exhibit Runge's phenomenon (over-fitting)
when interpolated naively.

.. _mip_approximation:
.. figure:: ./images/mip_approximation.png
   :align: center

   Approximation errors rates for interpolating the Runge function
   in dimension :math:`m = 4`.

----

There is an optimal (upper bound) approximation rate

.. math::
  \|Q_{f,A} - f\| \in \mathcal{O}_{\varepsilon}(\rho^{-n})

known\ :footcite:`trefethen2017`, which we call the *Trefethen rate*.

In fact, the :doc:`the DDS </fundamentals/dds>` scheme numerically
reaches the optimal Trefethen rate.
In contrast, spline-type interpolation is based on works of Carl de Boor
et al.\ :footcite:`deboor1972, deboor1977, deboor1978, deboor2010` and limited
to reach only polynomial approximation rates\ :footcite:`deboor1988`.
Similarly, interpolation using rational functions, such as
Floater-Hormann interpolation\ :footcite:`cirillo2017, floater2007`
and tensorial Chebyshev interpolation,
relying on :math:`l_{\infty}`-degree\ :footcite:`gaure2018`,
do not achieve optimality.

By combining sub-exponential growth of node (data) counts with exponential
approximation rates, the :doc:`DDS algorithm </fundamentals/dds>`
has the potential to *lift the curse of dimensionality* for interpolation
problems involving regular (Trefethen) functions\ :footcite:`Hecht2020`.

.. rubric:: References

.. footbibliography::
