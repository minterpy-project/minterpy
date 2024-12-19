=================================
Contributing to the Documentation
=================================

We welcome contributions to the Minterpy documentation.
Like the codebase, the documentation is a perpetual work in progress,
with always something missing, incomplete, or unclear.

If you encounter anything in the documentation that doesn't make sense,
please raise an issue.
If you can't find what you need, please raise an issue.
If you spot a mistake, well... you get the idea.

That said, before contributing to the Minterpy documentation,
please review the guidelines and conventions outlined in the following sections.

.. note::

   For information on contributing to the code base,
   see :doc:`/contributors/contrib-codebase/index` instead.

About the documentation
=======================

We put some thoughts on how we designed the documentation and the selection
of tools to build and serve the documentation; so please take a quick look
at :doc:`docs-design` and :doc:`docs-tools`.

How to contribute
=================

We care deeply about a useful, readable, and consistent documentation;
that's why we value your contributions, no matter how small or large.

Here's how you can help us maintain and improve the Minterpy documentation:

- **General language improvements**: Stumbled upon a typo or grammatical error?
  Found a phrase that's imprecise or misleading? We appreciate your sharp eye!
  Please let us know so we can correct these instances right away.
- **Correcting technical errors**: Should you encounter any technical errors
  such as incorrect statements, broken links, inaccurate code examples, or
  missing parameters in the API reference, let us know about them.
- **Adding new how-to guides**: Is there a how-to guide you wish existed?
  Would you like to create one yourself? We welcome your input and contributions
  in this area too!
- **Documenting new features**: If you're developing a new feature for Minterpy,
  we strongly encourage you to provide the accompanying documentation.
  This would include how-to guides. In fact, any new feature additions to the
  main branch of Minterpy **must** feature sufficient documentation.
  We weigh the quality of the code and the documentation equally during
  review stages.

.. rubric:: Opening an issue

If you have an idea or suggestion to improve and fix the Minterpy documentation,
please open an issue in the Minterpy repository and share your thoughts.
This way we can track of all documentation-related issues and your valuable
contributions.

Afterward, you have a couple of options:

- You can sit back and let another team member review your suggestion.
  If it's a go, they will implement it for you.
- If you're eager, feel free to make the necessary changes yourself.
  More about that below.

.. rubric:: Contributing directly to the source

As proponents of the :doc:`docs-like-code framework <docs-tools>`,
we maintain all of our documentation source files alongside the Minterpy codebase,
all stored within the same repository.

This setup allows for more seamless contributions.
If you're inclined to do so, feel free to directly modify the source files
to improve the documentation.

Before you start, we recommend checking out the following documents to get
familiar with the recommended workflow:

- :doc:`../development-environment`: So that you get everything set up on your system.
- :doc:`docs-source-structure`: So that you can add new files in the right places
  by knowing how the documentation source files are organized in the repository.
- :doc:`docs-build`: So that you can preview the changes you make.

You may also find the following references helpful:

- :doc:`docs-rest`
- :doc:`docs-ipynb`
- :doc:`docs-docstrings`

Once you ready to put your changes into the Minterpy codebase,
be sure to :doc:`create a merge (pull) request <../merge-request>`.

.. toctree::
   :maxdepth: 2
   :hidden:

   docs-design
   docs-tools
   docs-source-structure
   docs-build
   docs-rest
   docs-ipynb
   docs-docstrings
