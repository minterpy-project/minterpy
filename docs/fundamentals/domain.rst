=========================
User and Internal Domains
=========================

The main goal of Minterpy is to approximate a function

.. math::

   f: \Omega \subset \mathbb{R}^m \rightarrow \mathbb{R}

defined on some domain :math:`\Omega` (called *user domain*)
via a polynomial approximant :math:`Q_f`.

Rather than constructing :math:`Q_f` directly on :math:`\Omega`, Minterpy
builds a polynomial :math:`Q` on the internal domain :math:`[-1, 1]^m`
such that

.. math::

   Q_f(\boldsymbol{x}) = Q \circ \mathcal{T} (\boldsymbol{x})
   = Q(\mathcal{T}(\boldsymbol{x})), \quad \boldsymbol{x} \in \Omega,

where :math:`\mathcal{T}: \Omega \rightarrow [-1, 1]^m` is a coordinate
transformation from the user domain to the internal domain.

The approximation goal is then

.. math::

   f(\boldsymbol{x}) \approx Q_f(\boldsymbol{x})
   = Q(\mathcal{T}(\boldsymbol{x})), \quad \boldsymbol{x} \in \Omega.

Working internally on :math:`[-1, 1]^m` is not an arbitrary choice.
Classical polynomial approximation theory,
including the properties of Chebyshev polynomials and the analysis
of interpolation error, is naturally developed on this interval.
From a numerical standpoint, rescaling to :math:`[-1, 1]^m` prevents
large growth of monomials and maintains values
in a numerically well-behaved range.
Minterpy inherits these theoretical results and best practices
by anchoring the underlying computations to :math:`[-1, 1]^m`.

In Minterpy, :math:`Q` is a multivariate polynomial defined
on the internal domain :math:`[-1, 1]^m`.
The polynomial :math:`Q_f = Q \circ \mathcal{T}` is therefore the object
that approximates :math:`f` defined on :math:`\Omega`. :math:`Q_f` is what users
evaluate, differentiate, and integrate.

The remainder of this page examines the consequences of this composition,
which depend on the structure of :math:`\Omega`, for each of these operations.

Rectangular domains and affine separable transformations
========================================================

We focus on the case where :math:`\Omega` is a *rectangular domain*

.. math::

   \Omega = [a_1, b_1] \times \cdots \times [a_m, b_m],

where :math:`a_i < b_i` for each dimension :math:`i = 1, \ldots, m`.

For such domains, the transformation
:math:`\mathcal{T}: \Omega \rightarrow [-1, 1]^m`
can be constructed as an affine map [#affine]_ applied
*independently per dimension*.
Specifically, :math:`\mathcal{T}` decomposes into separable components

.. math::

   \mathcal{T}(\boldsymbol{x}) =
   \left( \mathcal{T}_1(x_1), \ldots, \mathcal{T}_m(x_m) \right)

where each component map :math:`\mathcal{T}_i: [a_i, b_i] \rightarrow [-1, 1]`
is given by

.. math::

   \mathcal{T}_i(x_i) = -1 + \frac{2}{b_i - a_i} (x_i - a_i),
   \quad x_i \in [a_i, b_i].

Given a point :math:`\boldsymbol{x} \in \Omega`, evaluating :math:`Q_f`
thus means first transforming :math:`x`
to :math:`\mathcal{T}(\boldsymbol{x}) \in [-1, 1]^m`
and then evaluating :math:`Q` there.

The inverse transformation
:math:`\mathcal{T}^{-1}_i: [-1, 1] \rightarrow [a_i, b_i]` is given by:

.. math::

   \mathcal{T}^{-1}_i(x_{t, i}) = a_i + \frac{b_i - a_i}{2} (x_{t, i} + 1),
   \quad x_{t, i} \in [-1, 1].

In practice, the inverse transformation is useful in transforming
the unisolvent nodes given in :math:`[-1, 1]^m` to the corresponding values
in the function domain.

Integration
===========

The definite integral of :math:`Q_f` over :math:`\Omega` can be written as

.. math::

   \int_\Omega Q_f(\boldsymbol{x}) \, d\boldsymbol{x} =
   \int_\Omega Q(\mathcal{T}(\boldsymbol{x})) \, d\boldsymbol{x}.

Applying the change of variables :math:`\boldsymbol{x}_t = \mathcal{T}(\boldsymbol{x})`
turns the integral over :math:`\Omega` into an integral over :math:`[-1, 1]^m`,

.. math::

   \int_\Omega Q_f(\boldsymbol{x}) \, d\boldsymbol{x} =
   \int_{[-1, 1]^m} Q(\boldsymbol{x}_t)
   \lvert \frac{\partial \boldsymbol{x}}{\partial \boldsymbol{x}_t} \rvert
   \, d\boldsymbol{x}_t.

where :math:`\lvert \frac{\partial \boldsymbol{x}}{\partial \boldsymbol{x}_T} \rvert`
is the Jacobian determinant of the inverse transformation :math:`\mathcal{T}^{-1}`.

Due to the separable structure of :math:`\mathcal{T}`,
the Jacobian matrix is diagonal,
and its determinant reduces to the product of the diagonal entries,

.. math::

   \lvert \frac{\partial \boldsymbol{x}}{\partial \boldsymbol{x}_t} \rvert =
   \prod_{i = 1}^m \frac{\partial x_i}{\partial x_{t, i}} =
   \prod_{i = 1}^m \frac{b_i - a_i}{2}.

The integral of :math:`Q_f` over :math:`\Omega` therefore reduces
to the integral of :math:`Q` over :math:`[-1, 1]^m` scaled
by a constant factor,

.. math::

   \int_\Omega Q_f(\boldsymbol{x}) \, d\boldsymbol{x} =
   \left( \prod_{i = 1}^m \frac{b_i - a_i}{2} \right)
   \int_{[-1, 1]^m} Q(\boldsymbol{x}_t) \, d\boldsymbol{x}_t.

Differentiation
===============

Differentiating :math:`Q_f = Q \circ \mathcal{T}` with respect to :math:`x_i`
by the chain rule gives

.. math::

   \frac{\partial Q_f}{\partial x_i} =
   \frac{\partial Q}{\partial x_{t,i}} \frac{\partial x_{t,i}}{\partial x_i}.

Since :math:`\mathcal{T}_i` is affine, its derivative with respect to
:math:`x_i` is a constant:

.. math::

   \frac{\partial \mathcal{T}_i}{\partial x_i} = \frac{2}{b_i - a_i},

so that the derivative of :math:`Q_f` in the user domain is the derivative of
:math:`Q` in :math:`[-1, 1]^m` scaled by this constant factor.

For mixed partial derivatives of order
:math:`\boldsymbol{k} = \left(k_1, \ldots, k_m \right)`
with :math:`k_i \geq 0`, the separable structure of :math:`\mathcal{T}` means
that the chain rule factors apply independently per dimension, giving

.. math::

   \frac{\partial^{k_1 + \cdots + k_m} Q_f}{\partial x_1^{k_1} \cdots
   \partial x_m^{k_m}} =
   \frac{\partial^{k_1 + \cdots + k_m} Q}{\partial x_{t,1}^{k_1} \cdots
   \partial x_{t,m}^{k_m}} \cdot \prod_{i=1}^{m}
   \left(\frac{2}{b_i - a_i}\right)^{k_i}.

Summary
=======

The object that approximates $f$ on the rectangular user domain :math:`\Omega`
is :math:`Q_f = Q \circ \mathcal{T}`. It is what the user evaluates,
integrates, and differentiates in their own coordinates.
In Minterpy, :math:`Q` is specifically a multivariate polynomial defined
on the internal reference domain :math:`[-1, 1]^m`, and the coefficients stored
internally belong to :math:`Q`. All user-facing operations, however,
act on :math:`Q_f`. This means:

- evaluation on :math:`\Omega` transparently applies :math:`\mathcal{T}`
  before passing the points to :math:`Q`,
- integration takes into account the Jacobian factor arising
  from the change of variables, and
- differentiation takes into account the chain rule scaling factors.

In each case, the separable affine structure of :math:`\mathcal{T}`
keeps the computation of these factors relatively simple:
closed-form and dimension-wise separable.

.. rubric:: Footnotes

.. [#affine] A map of the form :math:`\boldsymbol{x} \mapsto \boldsymbol{A} \boldsymbol{x} + \boldsymbol{b}`, a linear map with translation.
