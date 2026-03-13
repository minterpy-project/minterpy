===============================
Creating a Merge (Pull) Request
===============================

Once you finished your code changes,
be it for the :doc:`codebase <contrib-codebase/index>`
or the :doc:`documentation <contrib-docs/index>`,
please make sure that you follow the respective guidelines.

If you're happy with your contribution, you're ready to submit a merge (pull) request.
Such a request is how code from your repository becomes available in the
Minterpy repository so that the code becomes available in the upcoming release
of Minterpy.

To submit a merge (pull) request:

- Navigate to your forked Minterpy repository on GitHub.
- Click on **Pull Requests**.
- Click on **New pull request**.
- In the **Comparing changes** page, the **base repository** should point
  to the main Minterpy repository and the **base ref** to the ``dev`` branch.
  The **head repository** should point to your forked repository.
  For the **head ref** (i.e., *compare* branch drop-down menu),
  select the (feature) branch you want to merge into the Minterpy ``dev`` branch.
  You'll then see a list commits and files changed.
- Click on **Create pull request**.
- Write a descriptive title (**Add a title**)
  and explain what you've changed, referencing any relevant issue (**Add a description**).
- Once you're happy with the title and the description, click on **Create pull request**.

Your submission will then go to the main Minterpy repository for review by the
:ref:`project maintainers <contributors/about-us:Maintainers and contributors>`.
They will review the code or assign someone else to do so.
There may be discussions, and you might be asked to modify your code.

Once all discussion threads are resolved, the project maintainers will merge
your request into the ``dev`` branch of the main Minterpy repository.

.. rubric:: Updating your request

Based on the review you get, you might need to make some changes to your code.
Follow the code
:ref:`committing <contributors/development-environment:Committing changes>` and
:ref:`pushing <contributors/development-environment:Pushing changes>` steps
in your repository again to address any feedback.
By pushing, your pull request will be automatically updated;
simply continue the discussion you've already had on the discussion threads.

.. rubric:: Synchronizing with main Minterpy repository

The current state of the ``dev`` branch in the main Minterpy repository should
always be reflected in your pull request.

To update your feature branch with the latest changes in the ``dev`` branch of
the main Minterpy repository, run the following commands (replacing
the placeholder ``<your-feature-branch>`` with your actual branch name):

.. code-block::

   git checkout <your-feature-branch>
   git fetch upstream
   git merge upstream/dev

If there are no conflicts (or they could be fixed automatically),
a file with a default commit message will open.
Simply save and quite to complete the merge

If there are merge conflicts, you need to resolve those conflicts.
For an example of how to do this, refer to the following `GitHub resources`_.

Once the conflicts are resolved, run the following commands:

.. code-block::

   git add -u
   git commit

These commands will add all tracked files to the staging area and commit them
to finalize the merge.

.. _GitHub resources: https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/
