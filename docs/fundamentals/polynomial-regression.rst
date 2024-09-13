=====================
Polynomial Regression
=====================

Interpolating polynomials are constructed based on a carefully selected points
in the domain (i.e.,
the :ref:`unisolvent nodes <fundamentals/interpolation-at-unisolvent-nodes:Unisolvent nodes>`).
However, in many practical applications, the data is often provided as either
randomly scattered or arranged on a uniform grid.
In these situations, a polynomial---albeit not strictly interpolatory---can still be
constructed via least squares, which minimizes the error between the given data
and the polynomial.

Given a polynomial with respect to a multi-index set :math:`A` of the form

.. math::
   :label: eq_ols

   Q(\boldsymbol{x}) = \sum_{\boldsymbol{\alpha} \in A}
   c_{\boldsymbol{\alpha}} \, \Psi_{\boldsymbol{\alpha}}
   (\boldsymbol{x}),

represented in any basis
and a set of data :math:`\mathcal{D} = \{ (\boldsymbol{x}^{(i)}, y^{(i)}) \}_{i = 1}^N`,
the coefficients :math:`c_{\boldsymbol{\alpha}}` can be obtained by solving
the least squares problem

.. math::

   \hat{\boldsymbol{c}} =  \underset{\boldsymbol{c} \in \mathbb{R}^{\lvert A \rvert}}{\mathrm{argmax}}
   \; \lVert \mathbf{R}_A \boldsymbol{c} - \boldsymbol{y} \rVert^2_2,

where :math:`\boldsymbol{y}` is the vector of dependent variable :math:`(y^{(i)})_{i = 1}^N`,
and :math:`R_A` is the regression matrix.
The regression matrix, in turn, is obtained by evaluation the basis polynomials
at the data points :math:`(\boldsymbol{x}^{(i)})_{i = 1}^N`

.. math::

   \mathbf{R}_A = \left( \Psi_{\boldsymbol{\alpha}} \left( \boldsymbol{x}^{(i)} \right)
   \right)_{i = 1, \ldots, N, \boldsymbol{\alpha} \in A}
   \in \mathbb{R}^{N \times \lvert A \rvert}.


:math:`\Psi_{\boldsymbol{\alpha}}` are the basis polynomials that correspond
to the chosen :doc:`polynomial basis </fundamentals/polynomial-bases>`.

To ensure that the optimization problem in :eq:`eq_ols` is well-posed,
the regression matrix :math:`\mathbf{R}_A` must be of full rank, meaning that
the number of data points :math:`N` should be greater than or equal to
the number of basis polynomials :math:`\lvert A \rvert`.
This condition guarantees that there is enough data to uniquely determine
the coefficients of the polynomial.
