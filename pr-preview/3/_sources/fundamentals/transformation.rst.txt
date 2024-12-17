============================
Transformation Between Bases
============================

If an :math:`m`-dimensional polynomial :math:`Q \in \Pi_A`,
where :math:`A \subseteq \mathbb{N}^m`, is given in one of the
:doc:`supported bases </fundamentals/polynomial-bases>`, you might want
to transform it into another representation to leverage specific properties.
For instance, canonical polynomials are familiar while Newton polynomials
are more stable and accurate to evaluate.

On this page, we outline how the underlying theorem how such a transformation
is carried out.

Theorem: Basis transformations
==============================

Let :math:`A= A_{m,n,p}\,, m,n \in \mathbb{N}\,, p > 0`
be a :ref:`complete multi-index set
<fundamentals/polynomial-bases:Complete multi-index set>`,
:math:`P_A \subseteq \Omega` be :ref:`unisolvent nodes
<fundamentals/interpolation-at-unisolvent-nodes:Unisolvent nodes>`,
and :math:`Q \in \Pi_A` be a polynomial.
Then:

- Lower triangular matrices
  :math:`\mathrm{NL}_A, \mathrm{LN}_A \in \mathbb{R}^{|A|\times |A|}`
  can be computed in :math:`\mathcal{O}(|A|^3)` operations, such that

  .. math::
     \mathrm{LN}_A \cdot\mathrm{NL}_A = \mathrm{I} \,, \quad
     \mathrm{NL}_A \cdot \boldsymbol{c}_{\mathrm{nwt}} =
     \boldsymbol{c}_{\mathrm{lag}}\,, \quad
     \mathrm{LN}_A\cdot \boldsymbol{c}_{\mathrm{lag}} =
     \boldsymbol{c}_{\mathrm{nwt}} \,,

  where :math:`\boldsymbol{c}_{\mathrm{lag}} =
  (c_{\mathrm{lag}, \boldsymbol{\alpha}})_{\alpha \in A} \in \mathbb{R}^{\lvert A \rvert}`
  and :math:`\boldsymbol{c}_{\mathrm{nwt}} =
  (c_{\mathrm{lag}, \boldsymbol{\alpha}})_{\alpha \in A} \in \mathbb{R}^{\lvert A \rvert}`
  are the Lagrange coefficients and the Newton coefficients of :math:`Q`,
  respectively.

- Upper triangular matrices
  :math:`\mathrm{CL}_A, \mathrm{CN}_A \in \mathbb{R}^{\lvert A \rvert \times \lvert A \rvert}`
  can be computed in :math:`\mathcal{O}(\lvert A \rvert^3)` operations,
  such that

  .. math::
     \mathrm{CL}_A \cdot \boldsymbol{c}_{\mathrm{can}} =
     \boldsymbol{c}_{\mathrm{lag}}\,, \quad \mathrm{CN}_A \cdot
     \boldsymbol{c}_{\mathrm{can}} = \boldsymbol{c}_{\mathrm{nwt}}\,,

  where :math:`\boldsymbol{c}_{\mathrm{can}} =
  (c_{\mathrm{can}, \boldsymbol{\alpha}})_{\alpha \in A} \in \mathbb{R}^{\lvert A \rvert}`
  denotes the canonical coefficients of :math:`Q`.

Due to their triangular structure, the inverse matrices
:math:`\mathrm{NC}_A =\mathrm{CN}_A^{-1}`,
:math:`\mathrm{LC}_A =\mathrm{CL}_A^{-1}` can be computed efficiently
and accurately in :math:`\mathcal{O}(\lvert A \rvert^2)` time,
providing realizations of the inverse transformations.

Further relationships
=====================

If the :ref:`unisolvent nodes <fundamentals/interpolation-at-unisolvent-nodes:Unisolvent nodes>`
:math:`P_A` are fixed, all matrices can be precomputed.
The columns of :math:`\mathrm{NL}_A` are given
by :ref:`evaluating the Newton polynomials <fundamentals/polynomial-evaluation:Newton basis>`
on the unisolvent nodes, that is,

.. math::

  \mathrm{NL}_A = (N_{\boldsymbol{\alpha}}
  (\boldsymbol{p}_\beta))_{\boldsymbol{\beta},\boldsymbol{\alpha} \in A}
  \in \mathbb{R}^{\lvert A \rvert \times \lvert A \rvert}\,.

Thus, the above theorem enables both efficient and numerically accurate
computations. Conversely, the :doc:`/fundamentals/dds` can be used to
interpolate the polynomial in the :ref:`fundamentals/polynomial-bases:Lagrange basis`
:math:`L_{\boldsymbol{\alpha}} =
\sum_{\boldsymbol{\beta} \in A} d_{\boldsymbol{\alpha}, \boldsymbol{\beta}}
N_{\boldsymbol{\beta}}`, :math:`\alpha \in A` expressed in
the :ref:`fundamentals/polynomial-bases:Newton basis`.
This process yields the columns of :math:`\mathrm{LN}_A`, that is,

.. math::

  \mathrm{LN}_A =
  (d_{\boldsymbol{\alpha},\boldsymbol{\beta}})_{\boldsymbol{\beta}, \boldsymbol{\alpha} \in A}
  \in \mathbb{R}^{\lvert A \rvert \times \lvert A \rvert}\,.

With respect to the canonical basis,

.. math::

  \mathrm{CL}_A =
  (\Psi_{\mathrm{can}, \boldsymbol{\alpha}}(p_{\beta}))_{\boldsymbol{\alpha}, \boldsymbol{\beta} \in A}
  \in \mathbb{R}^{\lvert A \rvert \times \lvert A \rvert}

coincides with the classic *Vandermonde matrix*\ :footcite:`gautschi2012`.

Moreover, the columns of :math:`\mathrm{CN}_A` are given by applying
:doc:`DDS </fundamentals/dds>` to the canonical basis polynomial
:math:`\Psi_{\mathrm{can}, \boldsymbol{\alpha}}` evaluated at the unisolvent
nodes.

.. rubric:: Triangular matrices

All matrices presented above are of recursive triangular sparse structure,
which enables numerically accurate precomputation of the sub-matrices
and helps avoid storage issues.

Consequently, the explicit structure of the matrices can be condensed
into :doc:`barycentric transformations </fundamentals/barycentric-transformation>`,
which perform much faster than classic matrix multiplication.
This results in more efficient polynomial interpolation and evaluation.
A preliminary implementation of these fast transformations is already
utilized in Minterpy.

.. rubric:: References

.. footbibliography::
