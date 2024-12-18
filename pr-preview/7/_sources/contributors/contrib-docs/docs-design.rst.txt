=====================
Documentation: Design
=====================

We believe that documentation is essential to the on-going development of
any software.
Good documentation helps users and contributors--including ourselves--navigate
the steep learning curve of using and contribution to Minterpy.
That's why we invest significant time in developing
and improving the documentation.

On this page, you'll learn more about the underlying design of the Minterpy
documentation.

Design principles
=================

There isn't just *one* documentation but *four*,
each serving a distinct purpose.
We follow this documentation design principles from `Diátaxis`_
to structure the Minterpy documentation.
The four different *categories* of documentation are:

- :ref:`contributors/contrib-docs/docs-design:Tutorials`
- :ref:`contributors/contrib-docs/docs-design:How-to guides`
- :ref:`contributors/contrib-docs/docs-design:Explanation (fundamentals, theory)`
- :ref:`(API) reference <contributors/contrib-docs/docs-design:API reference>`

The distinction between these categories is summarized in the table below.

===============  ===============  =============== ===================== ===========
Category         Orientation      Main content    Most useful when...   Think of...
===============  ===============  =============== ===================== ===========
Tutorials        Learning         Practical steps learning the code     Lessons
How-to           Problem-solving  Practical steps working with the code Recipes
Explanation      Understanding    Exposition      learning the code     Textbooks
(API) reference  Information      Exposition      working with the code Dictionary
===============  ===============  =============== ===================== ===========

Or if you're more visually-oriented...

.. figure:: /assets/images/contributors/documentation-system.png
  :align: center

  Four distinct categories of documentation (adapted from the `Diátaxis`_).

.. seealso::

   We encourage you to read through the design as outlined in
   the `Diátaxis`_ and watch a talk by the author Daniele Procida
   in `Pycon 2017`_.

The :doc:`main sections <docs-source-structure>`
of the Minterpy documentation directly reflect all these four categories.
Additionally, we have a special section for contributors called
the :doc:`/contributors/index`, which primarily contains how-to guides tailored
for contributors rather than users.
Each addition to the documentation should be clearly assigned to a single
category.

Tutorials
=========

*Tutorials* are *learning-oriented* documents; think of them as *lessons*
for Minterpy users.
In a tutorial, the tasks should be meaningful and the outcomes should be immediate,
while avoiding unecessary explanations, jargon, and technical details--especially
in the beginning.
They should have a minimal, but non-trivial, context.
Writing tutorials is challenging; you need to select a problem that is
relatively easy yet non-trivial
and carefully introduce important and more advanced topics as the tutorial
progresses.

Well-curated tutorials are a crucial, if not the most important,
element of the documentation for onboarding both users and contributors.
The offer a glimpse of what Minterpy can do and how to use it as intended.

We organize all tutorials inside the :doc:`/getting-started/index` and list
them in order of difficulty. This section includes a quickstart guide
and a series of in-depth tutorials.

How-to guides
=============

*How-to Guides* are *problem-solving-oriented* documents;
think of them as *recipes* for performing specific tasks with Minterpy.
Each guide provides step-by-step instructions for solving a clearly defined problem.
These guides are largely context-free, assuming that users who have some
proficiency with Minterpy are seeking solutions to common, specific issues.

Writing how-to guides is somewhat easier than writing tutorials, as the problems
are more well-defined, context-free, and tailored for users with assumed
proficiency.

We organize all how-to guides inside the :doc:`/how-to/index`, grouping them
by common tasks involving Minterpy objects or numerical tasks
(e.g., interpolation, regression, differentiation, integration).

Explanation (fundamentals, theory)
==================================

*Explanation or fundamentals* are *understanding-oriented* documents;
think of them as the *theoretical expositions* (like those found in textbooks)
of the mathematics underlying Minterpy.
A fundamentals section provides the context and background of Minterpy,
detailing the different layers of abstraction from the top down and explaining
why things work as they do.
They avoid instructional content and minimize descriptions related to the
code implementation.

We write the fundamentals section of the documentation
to help users and contributors *understand* Minterpy and its underlying concepts
more deeply.
The fundamentals section is crucial for advancing users and contributors:
a user might become a contributor, and a contributor might improve
their contributions by understanding the theory behind Minterpy better.

Writing a fundamental section is challenging because the topics are more open-ended.
You need to decide what to explain, how deep is your explanation, and where to conclude.

We organize all theoretical topics related to Minterpy within
the :doc:`/fundamentals/index`.

API reference
=============

*API reference* is an *information-oriented* document;
think of it as a *dictionary* or an *encyclopedia* [#]_
that describes all the exposed components and machinery of Minterpy.
The API reference avoids explaining basic concepts or providing extensive
usage examples; its main focus is *to describe*.
This type of documentation is particularly important for advanced users
and contributors.

The API reference documentation tends to be terse and follow a well-defined,
consistent structure.
It often has a nearly one-to-one correspondence with the codebase itself.
If you're a developer, you're likely already familiar with creating the
API reference (e.g., for modules, classes, functions).

We organize all references for the exposed Minterpy components within
the :doc:`/api/index`.

Contributors guides
===================

*Contributors guides* are documents intended for, well, contributors.
They primarily consists of how-to guides for contributing to the Minterpy project,
either to its codebase or documentation.
They also include meta-information about the project's organization,
history, and the people behind it.

We organize all guides for contributors within the :doc:`/contributors/index`.

Summary
=======

The Minterpy documentation is currently organized into five main sections:

- :doc:`/getting-started/index` (including a Quickstart Guide and a series of in-depth Tutorials)
- :doc:`/how-to/index`
- :doc:`/fundamentals/index`
- :doc:`/api/index`
- :doc:`/contributors/index`

These sections align with the documentation design principles from
from `Diátaxis`_, which identifies four distinct types of documentations:

- :ref:`contributors/contrib-docs/docs-design:Tutorials` (:doc:`/getting-started/index`)
- :ref:`contributors/contrib-docs/docs-design:How-to guides` (:doc:`/how-to/index` and :doc:`/contributors/index`)
- :ref:`Explanation <contributors/contrib-docs/docs-design:Explanation (fundamentals, theory)>` (:doc:`/fundamentals/index`)
- :ref:`Reference <contributors/contrib-docs/docs-design:API reference>` (:doc:`/api/index`)

----

Now that you understand how we've designed the documentation,
you might be interested in learning more about how we build it.

.. rubric:: Footnotes
.. [#] If you don't know what that is, here is `a Wikipedia article`_ about it.

.. _Diátaxis: https://diataxis.fr
.. _Documentation System: https://documentation.divio.com/
.. _Pycon 2017: https://www.youtube.com/watch?v=azf6yzuJt54
.. _a Wikipedia article: https://en.wikipedia.org/wiki/Encyclopedia
