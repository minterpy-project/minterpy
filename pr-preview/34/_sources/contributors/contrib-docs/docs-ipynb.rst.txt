#########################
Writing Jupyter Notebooks
#########################

The functionalities of Minterpy are demonstrated using documents
written as Jupyter notebooks, which are executed and integrated into
the official documentation.
These built documents can be accessed in the :doc:`/how-to/index`
and :doc:`/getting-started/index` sections of the Minterpy documentation.

This page outlines the specific guidelines and considerations
for writing a Jupyter notebook that will be included
as part of the Minterpy documentation.

Source organization
===================

Jupyter-notebook-based documents for the Minterpy documentation are currently
restricted to the following sections:

- :doc:`/getting-started/index`
- :doc:`/how-to/index`

The source files for these sections are located in the following directories:

- ``/docs/getting-started`` for the :doc:`/getting-started/index`
- ``/docs/how-to`` for the :doc:`/how-to/index`

Additionally, the How-To Guides are generally organized into a further level
of hierarchy, based on either:

- **Main Components of Minterpy**, for example, :doc:`/how-to/grid/index`, :doc:`/how-to/multi-index-set/index`.
- **Numerical Tasks**, for example, :doc:`/how-to/integration/index`.

New tutorials should be put in ``/docs/getting-started`` directory and
new How-To Guides should be put in a relevant sub-directory of
``/docs/how-to`` .

Adding a notebook to the source tree
====================================

If you're adding a new notebook file to the documentation source tree,
please use one of these locations:

.. rubric:: Getting Started Guides / Tutorials

Place new Tutorials in the ``/docs/getting-started`` directory.

.. rubric:: How-To Guides

Place new How-To Guides in the appropriate subdirectory within ``/docs/how-to``,
depending on the topic.

---

By adhering to this structure, the Jupyter-notebook-based documents will be
correctly organized and accessible within the Minterpy documentation.

Adding a notebook to documentation
==================================

If you add a new Jupyter notebook to the Minterpy documentation,
ensure that the relevant ``index.rst`` files are updated accordingly.

.. rubric:: Updating the index.rst File

In the appropriate ``index.rst`` file, you must add a new entry to
the ``toctree`` directive to include the new notebook.
This ensures the new document is accessible in the built documentation.

.. code-block:: rest

   .. toctree::
      :maxdepth: 1
      :hidden:

      ...

.. rubric:: Getting Started Guides / Tutorials

For new tutorials, update the ``/docs/getting-started/index.rst`` file
with the new notebook entry under the ``toctree`` directive.

.. rubric:: How-To Guides:

If the new How-To Guide belongs to an existing sub-directory,
update the corresponding ``/docs/how-to/<relevant-directory>/index.rst`` file.

If it does not belong to an existing sub-directory, follow these steps:

- Create a new directory for the guide with its own ``index.rst`` file.
- Put the new notebook file in the new directory.
- Add a ``toctree`` directive inside the new ``index.rst`` to list the new file.
- Update the parent ``/docs/how-to/index.rst`` to include the new directory
  and its index file.

----

By following these steps, you ensure that the new Jupyter notebooks are properly linked and accessible within the Minterpy documentation structure.

Setup
=====

We recommend creating and editing Jupyter notebooks using JupyterLab,
which is included as part of the extras requirement
for the documentation development environment.

If you follow the installation steps in
:ref:`Installing Minterpy for development <contributors/development-environment:Installing Minterpy for development>`
with ``[docs]`` or ``[all]`` as option, then you launch invoke JupyterLab with
the following command:

.. code-block::

   jupyter lab

This command will start a local server, which you can access via a web browser.

Structure
=========

Although How-To Guides and Getting-Started Guides both focus on the practical
aspects of using Minterpy, there are notable differences as shown below.

+---------------------+-----------------------------------------------+------------------------------------------------+
| Perspective         | Tutorial                                      | Remarks                                        |
+=====================+===============================================+================================================+
| Book analogy        | The book "*How to cook potatoes*"             | The book "*100 Potato Recipes*"                |
+---------------------+-----------------------------------------------+------------------------------------------------+
| Goal                | Acquire basic competence                      | Perform a particular task                      |
+---------------------+-----------------------------------------------+------------------------------------------------+
| Audience            | Limited familiarity                           | Has familiarity; able to formulate question    |
+---------------------+-----------------------------------------------+------------------------------------------------+
| Purpose             | Provides a lesson                             | Provides a direction                           |
+---------------------+-----------------------------------------------+------------------------------------------------+
| Responsibility      | Author assumes responsibility                 | Author delegates responsibility                |
+---------------------+-----------------------------------------------+------------------------------------------------+
| Motivating Example  | Somewhat contrived                            | Very contrived                                 |
+---------------------+-----------------------------------------------+------------------------------------------------+
| Narrative           | A single line of carefully managed narrative  | May branch out ("If you want this, try that")  |
+---------------------+-----------------------------------------------+------------------------------------------------+

Unlike tutorials, which must be carefully designed and curated
by the project maintainers, How-To Guides are relatively straightforward
to write due to their more limited scope.
Below, we focus on the structure of How-To Guides.

----

Both tutorials and How-To Guides start with a motivating example,
but How-To Guides often feature a more contrived example.

For instance, the How-To Guide on
:doc:`/how-to/multi-index-set/multi-index-set-downward-closed` starts
with a non-downward closed set, but:

- **Why this particular set?** It doesn’t matter---you have to start somewhere.
- **What’s the importance of a downward-closed set?** Out of scope.
  You should probably already know that. If not, refer to other resources.

.. rubric:: Structure of a How-To Guide

Here is a suggested structure for creating a new How-To Guide:

- State the purpose of the guide.
- Provide a motivating example to set the context albeit contrively.
- Create relevant instances using Minterpy.
- Perform an action with the instances (function or method call).
- Demonstrate that the goal has been achieved.
- Remark on particularities and highlight other possibilities.

----

By following this structure, you ensure that the How-To Guides
are clear, purposeful, and easy to follow for users.

Best practices
==============

Below are some stylistic conventions we encourage you to follow when
writing a Jupyter notebook for Minterpy documentation.

.. rubric:: Import statements

You can place the import statements either after the first (opening) paragraph
or before it:

- In How-To Guides, import statements are typically placed at the beginning,
  **before** the opening paragraph.
- In Tutorials, they are usually placed **after** the opening paragraph.

.. rubric:: Minterpy abbrevation in import

Minterpy should always be imported with ``mp`` abbreviation in
the documentation.

.. rubric:: Dependencies

Limit the dependencies in the documentation,
rely on NumPy, Scipy, and Matplotlib to get your point across.

.. rubric:: Plotting

Plotting is allowed via Matplotlib, if the statements related to plotting
becomes very long and disturb the flow of the guide, the input cell must be
hidden.

Place a cell tag "hide-input" in the relevant cell.

For more details on hiding cell contents, see `Hide cell contents`_.

.. rubric:: Clear all cell output

Before submitting, all cell output must be cleared.

.. rubric:: Markup

Jupyter notebooks support markup language via Markdown enriched with `MyST-NB`_
which allows you to write mathematics, use admonitions, or cross reference
other part of Minterpy documentation.

.. rubric:: Admonitions

Here's an example of creating a note box in a markdown cell of a Jupyter
notebook:

.. code-block::

   ```{note}
   The set constructed via `from_degree()` constructor is _complete_
   in the sense that it contains _all_
   the exponents $\boldsymbol{\alpha} = \left( \alpha_1, \ldots, \alpha_m \right) \in \mathbb{N}^m$
   such that the $l_p$-norm $\lVert \boldsymbol{\alpha} \rVert_{p} = (\alpha_1^{p} + \ldots + \alpha_m^{p})^{\frac{1}{p}} \leq n$
   holds.
   ```

.. rubric:: Cross-references

Below are a couple of examples cross-referencing other part of Minterpy documentation
from a Jupyter notebook:

- Minterpy API element: ``{py:meth}`.Grid.from_points```
- Minterpy documentation page: ``{doc}`example <grid-create-from-points>`)``

Testing
=======

If you modify an existing Jupyter notebook or add a new one to the Minterpy
documentation, please ensure that it can still be executed.

Assuming that you have placed everything in
:ref:`the correct location <contributors/contrib-docs/docs-ipynb:Adding a notebook to the source tree>`
you can test whether all the available notebooks are still (successfully)
executable by following this :ref:`instruction <contributors/contrib-docs/docs-build:Testing the Jupyter-notebook-based documentation>`.

.. _JupyterLab: https://jupyter.org
.. _Hide cell contents: https://myst-nb.readthedocs.io/en/latest/render/hiding.html
.. _MyST-NB: https://myst-nb.readthedocs.io/en/latest/index.html