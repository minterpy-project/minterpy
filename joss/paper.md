---
title: 'Minterpy: multivariate polynomial interpolation in Python'
tags:
  - Python
  - numerical computing
  - function approximation
  - polynomial interpolation
  - polynomial regression
authors:
  - name: Damar Wicaksono
    orcid: 0000-0001-8587-7730
    equal-contrib: false
    affiliation: "1"
  - name: Uwe Hernandez Acosta
    orcid: 0000-0002-6182-1481
    equal-contrib: false
    affiliation: "1"
  - name: Sachin Krishnan Thekke Veettil
    orcid: 0000-0003-4852-2839
    equal-contrib: false
    affiliation: "3"
  - name: Jannik Kissinger
    orcid: 0000-0002-1819-6975
    equal-contrib: false
    affiliation: "3, 4"
  - name: Michael Hecht
    orcid: 0000-0001-9214-8253
    equal-contrib: false
    affiliation: "1, 2"
affiliations:
 - name: Center for Advanced Systems Understanding (CASUS) - Helmholtz-Zentrum Dresden-Rossendorf (HZDR), Germany
   index: 1
 - name: University of Wrocław, Poland
   index: 2
 - name: Max Planck Institute of Molecular Cell Biology and Genetics, Dresden, Germany
   index: 3
 - name: Technische Universität Dresden, Germany
   index: 4
date: 30 September 2024
bibliography: paper.bib
---

# Summary

Interpolation is essential in various computational tasks,
including function approximation, curve fitting, numerical integration,
differential geometry,
spectral methods, optimization,
and uncertainty quantification.

Minterpy is an open-source Python package designed
for multivariate polynomial interpolation.
It provides stable and accurate interpolating polynomials for approximating a wide range of functions.
Key features include:

- Polynomial interpolation on properly selected nodes and
  regression on arbitrary nodes.
- Differentiation and integration operations on the polynomials.
- Addition, subtraction, and multiplication operations
  on the polynomials.

Minterpy's long-term vision is to provide researchers and engineers
a software solution that mitigates the curse of dimensionality
commonly associated with interpolation tasks.

# Statement of need

Polynomial interpolation, rooted in the works of Newton and Lagrange [@Meijering2002],
plays a vital role in mathematics and computing [@Cools2002;@Hecht2018;@Xiu2009].
Global polynomials---defined over the entire domain---enable efficient numerical operations
like differentiation, integration, addition, subtraction, or multiplication [@Trefethen2019].

In one dimension, a unique polynomial $Q_{f, n}$ of degree $n$ interpolates
a function $f: \Omega \mapsto \mathbb{R}$ in a bounded domain $\Omega$
at $n + 1$ distinct _unisolvent interpolation nodes_ $P_n$ satisfying
$$
Q_{f, n} (p_i) = f(p_i),\; \forall p_i \in P_n \subset \Omega,\; i = 0, \ldots, n.
$$
The Stone-Weierstrass Approximation Theorem further states that,
in multiple dimensions, any continuous function on a bounded domain
can be approximated by multivariate global polynomials [@Branges1959].
The theorem, however, does not provide construction method;
it does not even require the sequence of polynomials $Q_{f, n}$
to coincide with $f$ _anywhere_ in the domain,
while still having uniform convergence on the domain.
Thus, approximation and interpolation
using global polynomials are distinct challenges.

Global polynomials can achieve a better global accuracy
for a wide class of regular functions
(differentiable, smooth, analytic),
and are amenable to further numerical computations.
However, they are often misperceived as unstable,
susceptible to Runge's phenomenon,
prone to round-off errors,
and subject to the curse of dimensionality [@Trefethen2011;@Trefethen2017].

Minterpy addresses these challenges by constructing multivariate polynomial interpolants
using appropriate interpolation nodes,
representing them in the Newton basis for stable evaluation,
and providing options to tailor the underlying multi-index sets
to mitigate the curse of dimensionality.
While not a universal tool for function approximation,
Minterpy enables the accurate and stable approximation
of Lipschitz continuous functions with high convergence rates,
reaching geometric rates for analytic functions [@Chkifa2013;@Trefethen2019].

Minterpy shares similar objectives and functionality with Chebfun [@Driscoll2014],
a popular MATLAB package[^chebfun-ports] designed for numerical computations using interpolating polynomials,
specifically Chebyshev polynomials.
Chebfun provides features such as root finding, differentiation, and integration 
for function approximation in up to three dimensions.
In contrast, Minterpy supports higher dimensions but with fewer features.

Several Python packages, such as Chaospy [@Feinberg2015], equadratures [@Seshadri2017], PyGPC [@Weise2020], PyThia [@Hegemann2023],
and UncertainSci [@Tate2023], provide polynomial-based function approximations,
primarily for uncertainty quantification (UQ) using generalized polynomial chaos expansion [@Xiu2002].
These tools often require framing problems as UQ tasks, where inputs are modeled probabilistically.
In contrast, Minterpy offers a straightforward, UQ-free approach to polynomial approximation.

Several other Python packages construct polynomial approximations from data.
SciPy [@Virtanen2020] provides multivariate interpolation methods (e.g., linear, nearest, pchip[^pchip]) for rectilinear grids.
ndsplines [@Margolis2019] efficiently implements tensor-product multivariate B-splines. 
Unlike Minterpy, these tools rely on piecewise local polynomials
and are tailored for input/output pairs data pairs.
Familiar and widely used, piecewise polynomials (especially splines) remain established tools for polynomial interpolation tasks.

# Package overview

Minterpy interpolates $m$-dimensional functions defined on a hypercube using a polynomial expansion in the Lagrange basis:
\begin{equation}
\label{eq:interpolating-polynomial}
f (\boldsymbol{x}) \approx Q (\boldsymbol{x}) = \sum_{\boldsymbol{\alpha} \in A} f(\boldsymbol{p}_{\boldsymbol{\alpha}}) \, L_{\boldsymbol{\alpha}} (\boldsymbol{x}),
\end{equation}
where where $A \subseteq \mathbb{N}^m$,
$L_{\boldsymbol{\alpha}}$, and $\boldsymbol{p}_{\boldsymbol{\alpha}}$
are the multi-index set,
the Lagrange basis polynomial,
and the unisolvent node that correspond to the index element $\boldsymbol{\alpha}$,
respectively.
The set $\{ \boldsymbol{p}_{\boldsymbol{\alpha}} \}_{\boldsymbol{\alpha} \in A}$
forms the interpolation grid.

Each basis polynomial satisfies the Kronecker delta condition
$$
L_{\boldsymbol{\alpha}} (p_{\boldsymbol{\beta}}) = \delta_{\boldsymbol{\alpha}, \boldsymbol{\beta}},\;\; p_{\boldsymbol{\beta}} \in \{ \boldsymbol{p}_{\boldsymbol{\alpha}} \}_{\boldsymbol{\alpha} \in A},\;\; \boldsymbol{\alpha} \in A,
$$
ensuring $Q(\boldsymbol{x})$ is an interpolating polynomial.

The multi-index set $A$ determines polynomial coefficients, unisolvent nodes, and function evaluations.
Minterpy defaults to a downward-closed set $A_{m, n, p}$
of spatial dimension $m \in \mathbb{N}_{> 0}$,
polynomial degree $n \in \mathbb{N}$,
and $l_p$-degree $p \in \mathbb{R}_{> 0}$,
with $A = A_{m, n, p}$
$$
A_{m, n, p} = \{ \boldsymbol{\alpha} \in \mathbb{N}^m: \lVert \boldsymbol{\alpha} \rVert_p = (\alpha_1^p + \cdots + \alpha_m^p)^{1/p} \leq n \}.
$$
Typical choices for $p$ are $1.0$, $2.0$, and $\infty$, representing the total, Euclidian, and tensor-product degree, respectively.
The Euclidean degree, in particular, demonstrates a greater resistance to the curse of dimensionality [@Trefethen2017a;@Hecht2020].

Deriving multidimensional Lagrange basis expressions for general non-tensorial grids is challenging.
For operations like evaluation and differentiation, Minterpy converts polynomials into the Newton basis
$$
Q (\boldsymbol{x}) = \sum_{\boldsymbol{\alpha} \in A} c_{\boldsymbol{\alpha}} \, N_{\boldsymbol{\alpha}} (\boldsymbol{x}),\;\; N_{\boldsymbol{\alpha}} (\boldsymbol{x}) = \prod_{i = 1}^m \prod_{j = 0}^{\alpha_i - 1} (x_i - q_j),\;\; q_j \in P_i,
$$
where $c_{\boldsymbol{\alpha}}$ and $N_{\boldsymbol{\alpha}}$ are
the Newton coefficient and Newton polynomial
that correspond to the index element $\boldsymbol{\alpha}$, respectively;
$P_i$ is a set of interpolation nodes in each dimension.
Using Leja-ordered Chebyshev-Lobatto interpolation nodes by default, Newton basis offers numerical stability.
Computing Newton coefficients, based on the Lagrange coefficients and interpolation grid,
via a multidimensional divided-difference scheme (DDS) is a key step in Minterpy [@Hecht2020].

Minterpy also supports other polynomial bases, including the canonical (monomial) and Chebyshev (first kind) bases,
along with transformations between them.

## Minterpy polynomials for function approximation

Minterpy prioritizes stable and accurate function approximation, even for high-degree polynomials.
Consider, the Runge function:
$$
f(\boldsymbol{x}) = \frac{1}{1 + \lVert \boldsymbol{x} \rVert^2},\; \boldsymbol{x} \in [-1, 1]^m,
$$
commonly used to demonstrate Runge's phenomenon, a pitfall in high-degree interpolation with equispaced points.

![The comparison of Minterpy interpolants, approximating the Runge function in dimension $m = 3, 4$, with alternative methods from designated packages.\label{fig:convergence}](convergence.png)

\autoref{fig:convergence} shows the accuracy of Minterpy interpolating polynomials
for three different $l_p$-degrees in dimension $m = 3, 4$[^machine].
The horizontal axis shows the number of coefficients (and function evaluations),
directly linked to the polynomial degree, to enable comparisons with other methods.
The infinity norm of the difference between the function and its approximation,
$$
\lVert f - Q_f \rVert_{\infty} = \max_{x \in \square^m} \lvert f - Q_f \rvert
$$
is measured at $1'000'000$ random points.

The figure compares data-driven methods (SciPy v1.13.1, ndsplines v0.2.0) and pseudo-spectral methods (Chaospy v4.3.17, Equadratures v10).
In the data-driven methods, approximation complexity is fixed as data increases.
While ndsplines supports higher degrees, splines above degree 5 are rare in practice.
The pseudo-spectral methods approximate functions using Legendre polynomial expansions on tensor-product grids,
with coefficients computed via numerical integration.
The coefficient count matches Minterpy interpolants with $l_\infty$-degree.
Equadratures, whose results are comparable to Minterpy, (softly) limits multi-index cardinality to $5 \times 10^4$ due to computational expense,
while Chaospy struggles with tensor-product grids[^sparse].

The results show Minterpy polynomials provide accurate function approximation with stability and convergence up to $10^{-14}$,
outperforming competing tools.

## Operations on the Minterpy polynomials

Minterpy polynomials support arithmetic operations (addition, subtraction, multiplication) 
and calculus operations (differentiation, definite integration).
Except for definite integration (yielding a numerical value),
these operations produce another polynomial, ensuring closure.
Among compared tools, only Chaospy offers similar capabilities.

## Polynomial regression

Minterpy interpolation uses Leja-ordered Chebyshev-Lobatto nodes by default;
for scattered or equispaced data, it supports construction
via well-conditioned least-squares schemes [@Veettil2022].
The resulting polynomials can then be processed like any other Minterpy polynomials.

## Applications

Minterpy has been applied in various research fields,
including data fitting in physics [@Dornheim2023],
serving as a surrogate model in blackbox optimization [@Schreiber2023],
and representing level sets in differential geometry [@Veettil2023].

# Author contributions

The contributions to this paper are listed according
to [CRediT](https://credit.niso.org).
**Damar Wicaksono**: Conceptualization, software, validation, visualization, writing--original draft.
**Uwe Hernandez Acosta**: Conceptualization, project administration, software, writing--review and editing.
**Sachin Krishnan Thekke Veettil**: Conceptualization, software.
**Jannik Kissinger**: Conceptualization, software.
**Michael Hecht**: Conceptualization, supervision, funding acquisition, writing--review and editing.

# Acknowledgments

The authors express their gratitude to
Michael Bussmann for his support and invaluable suggestions;
Michał Bajda for designing the Minterpy logo;
Janina Schreiber for the thorough code review
during the development process.

The work is partly funded by the Center for Advanced Systems Understanding ([CASUS](https://www.casus.science))
which is financed by Germany's Federal Ministry of Education and Research (BMBF)
and by the Saxony Ministry for Science, Culture and Tourism (SMWK).
Funding is provided through taxfunds based on the budget approved the Saxony State Parliament.

# References

[^chebfun-ports]: Packages in other languages based on or similar to Chebfun
include ApproxFun [@Olver2023] (Julia),
and ChebPy [@Richardson2024] and pychebfun [@Swierczewski2024] (Python).
[^machine]: The numerical experiment was conducted on a machine equipped with a 32-core AMD EPYC processor, 256 GB of RAM,
running Python v3.9.19 on Debian 12 Linux.
[^sparse]: Both Chaospy and equadratures support sparse polynomial construction, which can help reduce the number of coefficients.
Comparing these approaches, however, is beyond the scope of this work.
[^pchip]: Piecewise cubic Hermite interpolating polynomial.
[^non-interpolatory]: The polynomials are, however, not strictly interpolatory.
