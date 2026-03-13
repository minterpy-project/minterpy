==============================
Documentation Source Structure
==============================

The Minterpy documentation, located in the ``docs`` directory of the Minterpy
repository, is comprised of five key sections. Each of these sections aligns
with a sub-directory within the ``docs`` directory.

======================================================  =============================
Main Section                                            Sub-directory within ``docs``
======================================================  =============================
:doc:`Getting Started Guides </getting-started/index>`  ``getting-started``
:doc:`How-to Guides </how-to/index>`                    ``how-to``
:doc:`Fundamentals </fundamentals/index>`               ``fundamentals``
:doc:`API Reference </api/index>`                       ``api``
:doc:`Contributors Guide </contributors/index>`         ``contributors``
======================================================  =============================

.. important::

   The main sections are meant to be stable.
   Changes at this level could significantly alter the organization and presentation
   of our documentation, potentially requiring changes to the layout as well.
   As such, the addition of new top-level directories within the ``docs``
   directory should not be carried out without consultation with
   the Minterpy's project maintainers.

.. rubric:: Subsections and Pages

Within each main section, you'll find information nested into **Subsections**
and **Pages**:

- **Pages** are individual reStructuredText (reST) files recognizable
  by their ``.rst`` extension or Jupyter notebook file recognizable
  by their ``.ipynb`` extension.
- **Subsections** are directories with collections of related **Pages**;
  **Subsections** form subdivisions within each of the five top-level sections.

For example, the :doc:`/contributors/index` has subsections such as
:doc:`/contributors/contrib-codebase/index` and :doc:`/contributors/contrib-docs/index`,
and individual pages such as :doc:`/contributors/about-us`
and :doc:`/contributors/code-of-conduct`.

.. rubric:: Directory structure

The documentation source files are organized into the following directory
structure:

.. code-block::

   docs
   |--- api
   |--- contributors
   |    |--- contrib-codebase
   |    |--- contrib-docs
   |    |--- about-us.rst
   |    |--- code-of-conduct.rst
   |    |--- index.rst
   ...
   |--- index.rst
   ...

Navigating the documentation architecture, you'll find that ``index.rst``
in the top ``docs`` directory acts as the main index or root file of the
documentation.
This file defines what you see when you navigate
to the Minterpy :doc:`docs homepage </index>`.

Each of the main sections also has its own index file that serves
as the main page of the section;
it lists all the pages that belong to that section.
Some of the subsections inside the main sections may contain
their own index file as well, like :doc:`/contributors/contrib-codebase/index`
and :doc:`/contributors/contrib-docs/index`.
