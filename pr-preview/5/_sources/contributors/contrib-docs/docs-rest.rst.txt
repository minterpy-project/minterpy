==================================
Writing reStructuredText Documents
==================================

reStructuredText (reST) is the default markup language used by Sphinx for
generating the documentation of the Minterpy project.
The majority of content within the Minterpy documentation is composed of reST.

This page delves into the specifics of working with reST-based documents
within the Minterpy project.
It covers various features of the Sphinx documentation generator employed
in the documentation and demonstrates their proper usage.

.. seealso::

   - Documentation rich with code--akin to what is found in
     :doc:`Getting Started </getting-started/index>`
     and :doc:`/how-to/index` is written as `Jupyter notebooks`_.
     The particularities of creating such documentation for the Minterpy
     project are covered in :doc:`/contributors/contrib-docs/docs-ipynb`.
   - Minterpy API documentation is authored directly into the Python source
     code using docstrings. Although docstrings employ the reST markup
     language, they possess distinct trains and stylistic conventions.
     These specific aspects are discussed in detain in
     :doc:`/contributors/contrib-docs/docs-docstrings`.
   - This page does not exhaustively cover reST syntax and usage.
     If you don't find what you need here,
     please refer to the `reStructuredText Primer`_.

80-Character width rule
=======================

While it may appear updated to have a rule related to line length,
given that modern computer screens can effectively display well over 80
characters in a single line, there are still valid reasons for this guideline.
Indeed, lines of text beyond 80 characters can become more challenging to read,
and enforcing such a limit can enhance overall readability.
Furthermore, many developers prefer to work within the confines of half-screen
windows, thereby reinforcing the utility of this constraint.

----

The following are some best-practice recommendations to consider.

.. tip::

   - Aim to comply with the 80-character rule when contributing to
     the documentation.
     Utilize the internal ruler feature at the 80-character mark in your
     preferred text editor--this option is widely available in most editors.
   - Remember, rules have exceptions. Code blocks, URLs, and Sphinx roles and
     directives may occasionally extend beyond 80 characters for Sphinx
     to properly parse them. When in doubt, use common sense.

.. seealso::

   For additional rational on the 80-character width as well as
   where to break a line in the documentation source, see:

   - `Is the 80 character line limit still relevant`_ by Richard Dingwall
   - `Semantic Linefeeds`_ by Brandon Rhodes

Admonitions
============

Sphinx provides support for a whole class of built-in `admonitions`_
as a set of directives to render text inside a highlighted box.

.. rubric:: Available admonitions

There are several types of admonitions that may be used in the Minterpy
documentation:

.. note::

    Add an additional information that the reader may need to be aware of.

    Use the ``.. note::`` directive to create a note block.

.. important::

   Use an important block to make sure that the reader is aware of some key steps
   and what might go wrong if the reader doesn't have the provided information.

   Use the ``.. important::`` directive to create an important block.

.. warning::

   Add a warning to indicate irreversible (possibly detrimental) actions and
   known longstanding issues or limitations.

   Use the ``.. warning::`` directive to create a warning block.

.. tip::

   Use a tip block to offer best-practice or alternative workflow
   with respect to he current instructions.

   Use the ``..tip::`` directive to create a tip-block.

.. seealso::

   Use a see-also block to provide a list of cross-references
   (internal or external) if you think these cross-references must be listed
   separately to attract more attention.

   Use the ``.. seealso::`` directive to create a see-also block.

----

The following are some best-practice recommendations to consider.

.. tip::

   - Use admonitions sparingly and judiciously in the Minterpy documentation
     as they tend to obstruct the reading flow.
     Besides, if used too often, readers may become immune to notes
     and warnings and would simply ignore them.

Bibliographic citations
=======================

A bibliographic citation is a special case of
:ref:`cross-referencing <contributors/contrib-docs/docs-rest:Cross-references>`,
aimed at cross-referencing external academic resources such as articles,
books, or reports.
It is crucial to incorporate relevant scientific works into the Minterpy
documentation whenever suitable.
This is predominantly applicable when writing the
the :doc:`Fundamentals </fundamentals/index>`
section of the documentation.

Bibliography file
-----------------

The bibliographic entries are located in the bibliography file, a `BibTeX`_ file
named ``refs.bib`` in the root ``docs`` directory.
An entry in the file is written in the standard BibTeX format.

For example, an article entry is written as follows:

.. code-block:: bibtex

   @article{Dyn2014,
        title={Multivariate polynomial interpolation on lower sets},
        author={Dyn, Nira and Floater, Michael S.},
        journal={Journal of Approximation Theory},
        volume={177},
        pages={34--42},
        year={2014},
        doi={10.1016/j.jat.2013.09.008}
    }

Citations
---------

To cite an entry in a page, use ``:footcite:`` role followed by the entry key.
For example:

.. code-block::

   Earlier versions of this statement were limited to the case
   where :math:`P_A` is given by a (sparse) tensorial grid\ :footcite:`Dyn2014`.

.. note::

   Notice that the backslash that precedes the space
   before ``:footcite:`` directive; it suppresses the space when rendered.

will be rendered as:

   Earlier versions of this statement were limited to the case
   where :math:`P_A` is given by a (sparse) tensorial grid\ :footcite:`Dyn2014`.

Multiple citation keys can be specified in the ``:footcite:`` role.
For example:

.. code-block::

   Spline-type interpolation is based on works of by Carl de Boor et al.\ :footcite:`DeBoor1972, DeBoor1977, DeBoor1978, DeBoor2010`.

will be rendered as:

   Spline-type interpolation is based on works of by Carl de Boor et al.\ :footcite:`DeBoor1972, DeBoor1977, DeBoor1978, DeBoor2010`.

Displaying a list of references
-------------------------------

In the minterpy documentation, each page that contains bibliographic citations
should display its own list of references, rather than having a single page
listing all references.
If a page includes bibliographic citations, the list of references should be
placed at the end of the document using the ``.. footbibliography::``
directive.
Use "References" as the second-level heading.

For example:

.. code-block:: rest

   ...

   References
   ==========

   .. footbibliography::


which will be rendered as (``References`` heading is intentionally
not displayed):

   .. footbibliography::

----

The following are some best-practice recommendations to consider.

.. tip::

   - When possible, always include the digital object identifier (`DOI`_)
     for each entry in the bibliography file.
   - Don't forget the backslash that precedes the space before
     ``:footcite:`` role; it will suppress the space when rendered.
   - Display the list of references at the very end of each page that contains
     bibliographic citations.
   - Use ``References`` as the heading title of the list of references.

Implementation notes
--------------------

- Bibliographic citations in the Minterpy documentation uses
  the `bibtex extension`_ for Sphinx.

- The `bibtex extension documentation`_ recommends using ``footcite`` and
  ``footbibliography`` to create a *local* bibliography.
  The Minterpy documentation follows this recommendation.

  .. important::

     Doing this saves us a lot of trouble customizing the ``bibtex`` extension
     to avoid duplication issues.

Code examples
=============

Use code examples to illustrate how Minterpy programming elements can be used
to achieve specific goals.
Depending on their length, these examples can be categorized as follows:

- **In-line code examples**: Simple one-liners integrated into the text.
- **Code example blocks**: Short to longer, self-contained examples used to
  demonstrate a concept or solution.

In-line code examples
---------------------

Use the ``:code:`` role to put a code examples.
For example:

.. code-block:: rest

   Load ``minterpy`` using :code:`import minterpy as mp`

will be rendered as:

    Load ``minterpy`` using :code:`import minterpy as mp`

Code example blocks
-------------------

Code example blocks are written using the ``.. code-block::`` directive.
For example:

.. code-block:: rest

   .. code-block::

       import minterpy as mp

       mi = mp.MultiIndexSet.from_degree(3, 2, 1)

will be rendered as:

    .. code-block::

       import minterpy as mp

       mi = mp.MultiIndexSet.from_degree(3, 2, 1)

.. rubric:: Syntax highlighting

Sphinx also supports syntax highlighting for various programming languages.
Specify the language after the ``.. code-block::`` directive.
Use the proper syntax highlighting when it is appropriate.
Python code in the Minterpy docs should be syntax-highlighted.

For example, the same code above should have been written:

.. code-block:: python

   import minterpy as mp

   mi = mp.MultiIndexSet.from_degree(3, 2, 1)

Code examples involving interactive Python session should be written
using the ``pycon`` (python console) language specification.

For example:

.. code-block:: rest

    .. code-block:: pycon

        >>> import minterpy as mp
        >>> mi = mp.MultiIndexSet.from_degree(3, 2, 1)
        >>> mi
        MultiIndexSet
        [[0 0 0]
         [1 0 0]
         [2 0 0]
         [0 1 0]
         [1 1 0]
         [0 2 0]
         [0 0 1]
         [1 0 1]
         [0 1 1]
         [0 0 2]]

will be rendered as:

    .. code-block:: pycon

        >>> import minterpy as mp
        >>> mi = mp.MultiIndexSet.from_degree(3, 2, 1)
        >>> mi
        MultiIndexSet
        [[0 0 0]
         [1 0 0]
         [2 0 0]
         [0 1 0]
         [1 1 0]
         [0 2 0]
         [0 0 1]
         [1 0 1]
         [0 1 1]
         [0 0 2]]

Cross-referencing code blocks
-----------------------------

Cross-referencing a code example block may be done via custom anchor (label).
For instance, create an anchor for a code example to be cross-referenced later:

.. code-block:: rest

   .. _code-example:

   .. code-block:: python

      fx = lambda x: np.sin(x)
      fx_interpolator = mp.interpolate(fx, 1, 3)

this will be rendered as:

   .. _code-example:

   .. code-block:: python

      fx = lambda x: np.sin(x)
      fx_interpolator = mp.interpolate(fx, 1, 3)

and can be cross-referenced using the ``:ref:`` role.
For example:

.. code-block:: rest

   See the code example :ref:`code example <code-example>`.

which will be rendered as:

   See the :ref:`code example <code-example>`.

.. important::

   Cross-referencing a code example block always requires a custom label.

----

The following are some best-practice recommendations to consider.

.. tip::

   - Although double backticks and ``:code:`` role render text in a fixed-width
     font, always use ``:code:`` role for displaying inline code example
     for clarity.
   - When possible, always specify the programming language in the code example
     blocks to enable syntax highlighting. Python code examples in the Minterpy
     documentation should always be syntax-highlighted.
   - If you need to cross-reference a code example block, define a unique
     custom label for it. Ensure that the label is unique across
     the documentation, and check for "duplicate labels" warnings when building
     the documentation.
   - Keep in mind that users may copy and paste code blocks, potentially with
     minor modifications. Make sure code examples are meaningful.
   - Use common sense regarding the length of code blocks. Overly long code
     blocks without accompanying narrative are difficult to read and understand
     in the documentation.

Cross-references
================

The Minterpy documentation uses various types of cross-references, including
external and internal links, bibliographic citations, and more.

.. seealso::

   The Minterpy documentation uses various types of internal cross-references
   specific to documentation elements, such as pages, section headings, images,
   equations, and API elements.

   This guideline focuses on cross-references for pages, section headings, and
   API elements; other types of internal cross-referencing are covered in
   separate guidelines.

External resources
------------------

External cross-references link to external resources, typically other
web pages.
The Minterpy documentation uses the `link-target`_ approach for external
cross-references.
In this approach, the visible text link is separated from its target URL,
resulting in cleaner source code and allowing for target reuse,
at least within the same page.

As an example:

.. code-block:: rest

   The problem is well explained in this `Wikipedia article`_
   and also in a `DeepAI article`_.

   .. _Wikipedia article: https://en.wikipedia.org/wiki/Curse_of_dimensionality
   .. _DeepAI article: https://deepai.org/machine-learning-glossary-and-terms/curse-of-dimensionality

which will be rendered as:

    The problem is well explained in this `Wikipedia article`_
    and also in a `DeepAI article`_.

Page
----

A whole documentation page (a single reST or Jupyter notebook file)
may be cross-referenced using the ``:doc:`` role.
The default syntax is:

.. code-block:: rest

   :doc:`<target>`

For example, to cross-reference the main page of the Developers guide, type:

.. code-block:: rest

   See the :doc:`/contributors/index` for details.

which will be rendered as:

    See the :doc:`/contributors/index` for details.

.. important::

    Don't include the ``.rst`` extension when specifying the target in
    the ``:doc:`` role.

By default, the displayed link title is the title of the page.
You can replace the default title using the following syntax:

.. code-block:: rest

   :doc:`custom_link_title <target>`

Replace ``custom_link_title`` accordingly.
For example:

.. code-block:: rest

   For details, see the Developers guide :doc:`here </contributors/index>`.

which will be rendered as:

    For details, see the Developers guide :doc:`here </contributors/index>`.

The target specification may be written in two different ways:

- **Relative to the current document**: For example, ``:doc:docs-ipynb``
  refers to the :doc:`docs-ipynb` section of the contribution guidelines.
- **Full path** relative to the root ``docs`` directory: The same example above
  can be specified using its full path relative to the ``docs`` directory.

.. important::

    Don't forget to include the backslash in front of the directory name
    if it's specified in full path (relative to the root ``docs`` directory).

Section headings
----------------

Section headings within a page may be cross-referenced using
the ``:ref:`` role.
The Mintepry documentation uses the `autosectionlabel`_ extension for Sphinx,
which means that you don't need to manually label a heading before
cross-referencing it.
Additionally, all section heading labels are automatically ensured
to be unique.

The syntax to cross-reference a section heading is:

.. code-block:: rest

   :ref:`path/to/document:Heading title`

By default, the heading title in the page will be rendered.
To display a custom title, use:

.. code-block:: rest

   :ref:`custom_link_title <path/to/document:Heading title>`

For example, to cross-reference the math blocks section
of the documentation contribution guidelines, type:

.. code-block:: rest

   To write math blocks in the Minterpy documentation,
   refer to :ref:`contributors/contrib-docs/docs-rest:Mathematics blocks`.

which will be rendered as:

   To write math blocks in the Minterpy documentation,
   refer to :ref:`contributors/contrib-docs/docs-rest:Mathematics blocks`.

To replace the default title, type:

.. code-block:: rest

   To write math blocks in the Minterpy documentation,
   refer to the :ref:`relevant section <contributors/contrib-docs/docs-rest:Mathematics blocks>`
   in the docs contribution guidelines.

which will be rendered as:

   To write math blocks in the Minterpy documentation,
   refer to the :ref:`relevant section <contributors/contrib-docs/docs-rest:Mathematics blocks>`
   in the docs contribution guidelines.

.. important::

    Don't *include* the backslash in front of the directory name for target
    specified using ``:ref:`` role. The path is always relative
    to the root ``docs`` directory.

Minterpy API elements
---------------------

Elements of the documented Minterpy API, including modules, functions, classes,
methods, attributes or properties, may be cross-referenced within
the documentation.
The `Python domain`_ allows for cross-referencing most documented objects.
However, before an API element can be cross-referenced, its documentation
must be available in the :doc:`/api/index`.

Refer to the table below for usage examples.

=========  ==================  =========================================  =====================================
Element    Role                Example                                    Rendered as
=========  ==================  =========================================  =====================================
Module     :code:`:py:mod:`    ``:py:mod:`.transformations.lagrange```    :py:mod:`.transformations.lagrange`
Function   :code:`:py:func:`   ``:py:func:`.interpolate```                :py:func:`.interpolate`
Class      :code:`:py:class:`  ``:py:class:`.core.grid.Grid```            :py:class:`.core.grid.Grid`
Method     :code:`:py:meth:`   ``:py:meth:`.MultiIndexSet.from_degree```  :py:meth:`.MultiIndexSet.from_degree`
Attribute  :code:`py:attr:`    ``:py:attr:`.MultiIndexSet.exponents```    :py:attr:`.MultiIndexSet.exponents`
=========  ==================  =========================================  =====================================

.. important::

    Precede the object identifier with a dot indicating that it is relative
    to the ``minterpy`` package.

Other projects' documentation
-----------------------------

Documentation from other projects (say, NumPy, SciPy, or Matplotlib)
may be cross-referenced in the Minterpy documentation.

To cross-reference a part or an API element from another project's docs,
use the following syntax:

.. code-block:: rest

   :py:<type>:`<mapping_key>.<ref>`

replace ``<type>`` with one of the types listed in the table above,
``<mapping_key>`` with the key listed in the ``intersphinx_mapping`` variable
inside the ``conf.py`` file, and ``ref`` with the actual documentation element.

For example, to refer to the docs for ``ndarray`` in the ``NumPy`` docs, write:

.. code-block:: rest

   :class:`numpy:numpy.ndarray`

which will be rendered as:

   :class:`numpy:numpy.ndarray`

This functionality is provided by the `intersphinx`_ extension for Sphinx.

.. note::

   Check the variable ``intersphinx_mapping`` inside the ``conf.py`` file
   of the Sphinx documentation for updated list of mappings.

----

The following are some best-practice recommendations to consider.

.. tip::

   - For external cross-references, use the `link-target`_ approach to define
     an external cross-reference and put the list of targets at the very bottom
     of a page source. See the source of this page for example.
   - Try to be descriptive with what being cross-referenced; use custom link title
     if necessary.
   - Limit the cross-references to the API elements from
     the :doc:`Fundamentals </fundamentals/index>` section.

Images
======

To add images to a reStructuredText document, use the ``.. image::`` directive.

For example:

.. code-block:: rest

   .. image:: /assets/minterpy-logo.png
      :width: 200
      :alt: Minterpy Logo

will be rendered as:

.. image:: /assets/minterpy-logo.png
   :width: 200
   :alt: Minterpy Logo

The path to the file is by default relative to the root source directory of
the documentation (i.e., ``docs``).

Notice also the two options used in the snippet above:

- ``:width:``: This option is used to define image width in pixels.
- ``:alt:``: This option is used to assign an alternative text
  for screen readers.

Mathematics
===========

In the Minterpy documentation,
Sphinx is configured to display mathematical notations using `MathJax`_.
The MathJax library offers comprehensive support for LaTeX,
*the* markup language for writing mathematics.

Inline mathematics
------------------

Inline mathematics can be written using the ``:math:`` role.

For example:

.. code-block:: rest

   :math:`A_{m,n,p} = \left\{\boldsymbol{\alpha} \in \mathbb{N}^m | \|\boldsymbol{\alpha}\|_p \leq n, m,n \in \mathbb{N}, p \geq 1 \right\}` is the multi-index set.

will be rendered as:

    :math:`A_{m,n,p} = \left\{\boldsymbol{\alpha} \in \mathbb{N}^m | \|\boldsymbol{\alpha}\|_p \leq n, m,n \in \mathbb{N}, p \geq 1 \right\}` is the multi-index set.

Mathematics blocks
------------------

Mathematics blocks can be written using the ``.. math::`` directive.

For example:

.. code-block:: rest

   .. math::

      N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A


will be rendered as:

    .. math::

       N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A

Numbering and cross-referencing
-------------------------------

A math block in a page may be numbered if they are labelled using
the ``:label:`` option within the ``.. math::`` directive.

For example:

.. code-block:: rest

    .. math::
       :label: eq:newton_polynomial_basis

        N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A


will be rendered in the page as:

    .. math::
       :label: eq:newton_polynomial_basis

        N_{\boldsymbol{\alpha}}(\boldsymbol{x}) = \prod_{i=1}^{M} \prod_{j=0}^{\alpha_i - 1} (x_i - p_{j,i}), \; \boldsymbol{\alpha} \in A

The equation can then be cross-referenced *within the same page* using
the ``:eq:`` role followed by the equation name previously assigned.

For example:

.. code-block:: rest

   The multivariate Newton polynomial is defined in :eq:`eq:newton_polynomial_basis`.

The rendered page will display the equation number as a hyperlink:

    The multivariate Newton polynomial is defined in :eq:`eq:newton_polynomial_basis`.

.. note::

   Equations are numbered consecutively within the same page.
   The equation numbering will be reset to 1 in another page as ``minterpy``
   docs doesn't use numbered table of contents.
   Therefore, it is not straightforward to cross-reference an equation defined
   in another page.
   Use instead the nearest or the most relevant heading to the equation
   as an anchor.

----

The following are some best-practice recommendations to consider.

.. tip::

   - Use the following syntax to label an equation:

     .. code-block:: rest

        :label: `eq:equation_name`

     and replace the ``equation_name`` part with the actual name of the equation
     but keep the preceding ``eq:``.


   - Avoid cross-referencing an equation in one page from another.
     Use, instead, the nearest or the most relevant heading to the equation
     as an anchor.
     See the guidelines of
     :ref:`section heading cross-references <contributors/contrib-docs/docs-rest:Section headings>`
     for details.

.. important::

   The ``equation_name`` for the label must be unique across the documentation.
   Make sure there's no "duplicate warning" when building the docs.

   If such warnings arise, use common sense to rename the equation.

.. _Jupyter notebooks: https://jupyter-notebook.readthedocs.io/en/stable/
.. _reStructuredText Primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _Is the 80 character line limit still relevant: https://www.richarddingwall.name/2008/05/31/is-the-80-character-line-limit-still-relevant
.. _Semantic Linefeeds: https://rhodesmill.org/brandon/2012/one-sentence-per-line
.. _bibtex extension: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/index.html
.. _BibTeX: http://www.bibtex.org
.. _DOI: https://en.wikipedia.org/wiki/Digital_object_identifier
.. _bibtex extension documentation: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#local-bibliographies
.. _link-target: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#hyperlinks
.. _Wikipedia article: https://en.wikipedia.org/wiki/Curse_of_dimensionality
.. _DeepAI article: https://deepai.org/machine-learning-glossary-and-terms/curse-of-dimensionality
.. _autosectionlabel: https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
.. _Python domain: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects
.. _intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
.. _built-in Glossary: https://www.sphinx-doc.org/en/master/glossary.html
.. _Wikipedia: https://www.wikipedia.org
.. _MathJax: https://www.mathjax.org/
