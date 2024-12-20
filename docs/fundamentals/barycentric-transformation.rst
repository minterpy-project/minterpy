==========================
Barycentric Transformation
==========================

In one dimension, barycentric Lagrange interpolation is the most efficient
scheme for fixed interpolation nodes\ :footcite:`berrut2004`.
Both determining the interpolating polynomial :math:`\left( Q_{f,A} \right)`
for :math:`\left( A = \{0, \ldots, n\} \right)`
and evaluating :math:`\left( Q_{f,A} \right)` at any point
:math:`\left( \boldsymbol{x} \in \Omega \right)`
require linear runtime :math:`\left( \mathcal{O}(n) \right)`.
This efficiency is achieved by precomputing the constant *barycentric weights*,
which depend only on the interpolation node locations,
not on the function :math:`\left( f: \Omega \to \mathbb{R} \right)`
and its values.

Minterpy has partially extended the classic barycentric Lagrange interpolation
to the multidimensional case.
This extension leverages the fact that the transformation
from the :doc:`Lagrange to Newton basis </fundamentals/transformation>`
involves structured sparse triangular matrices.
Exploiting this structure, as indicated by our preliminary results
for the case of :math:`\left( l_1 \right)`-degree\ :footcite:`sivkin`,
allows for faster inversion and multiplication of the corresponding matrices
compared to the general case,
as seen in similar approaches\ :footcite:`struct2, struct1`.

In summary, we aim to reduce the runtime complexity from
:math:`\left( \mathcal{O}(|A|^2) \right)` for deriving and executing the transformations
to :math:`\left( \mathcal{O}(mn|A| ) \right)`, and further to
:math:`\left( \mathcal{O}(\log(|A|)|A|) \right)`
for :math:`\left( A = A_{m,n,p}, p = 1, \infty \right)`.
Research and implementation enhancements are ongoing to achieve these goals.

.. rubric:: References

.. footbibliography::
