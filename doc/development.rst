Development
===========

Versioning
----------

We use `Semantic Versioning <http://semver.org/>`_ with the modifications
described under :ref:`deprecation_policy`.

.. _deprecation_policy:

Deprecation policy
------------------

petab aims to provide a stable API for users. However, not all features can be
maintained indefinitely. We will deprecate features in minor releases and
where possible, issue a warning when they are used. We will keep deprecated
features for at least six months after the release that includes the
respective deprecation warning and then remove them earliest in the next minor
or major release. If a deprecated feature is the source of a major bug, we may
remove it earlier.

Python compatibility
--------------------

We follow `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_
for the minimum supported versions of Python and core dependencies
(NumPy, SciPy, Matplotlib, pandas, ...),
but may keep supporting a Python version
as long as current NumPy releases support it.

Release process
---------------

1. Update the version number in ``petab/version.py``.

2. Collect the changelog entries from ``changelog.d/`` into
   ``CHANGELOG.md`` by running
   `towncrier <https://towncrier.readthedocs.io/>`__
   (included in the ``test`` dependency group) from the repository root:

   .. code-block:: shell

      towncrier build --version X.Y.Z

   This adds a new section for the release to ``CHANGELOG.md``, and removes
   the collected fragments (``git rm``). Run with ``--draft`` to only preview
   the result. Review the new section and edit it as necessary:
   group related entries (e.g., by PEtab version or module),
   reorder entries (they are sorted alphabetically within each category),
   credit new contributors, and add the ``**Full Changelog**`` link.
   For a new minor version, add a ``## X.Y series`` heading and move the
   ``<!-- towncrier release notes start -->`` marker below it.

3. Create a pull request with the to-be-released changes to the main branch
   (usually from ``develop``).

4. Once the pull request is merged, create a new release on GitHub.
   Make sure to set the tag to the version number prefixed with 'v'
   (e.g., ``v1.0.0``), and the release title to ``libpetab-python $RELEASE_TAG``
   (e.g., ``libpetab-python v1.0.0``).

5. Check that the release is now available on PyPI.
   The upload to PyPI is performed automatically by a GitHub Actions workflow,
   which may take a few minutes to complete.

6. Merge the main branch back into the `develop` branch.

Style guide
-----------

Code style
~~~~~~~~~~

We use pre-commit with ruff to enforce code style. To install pre-commit and
the pre-commit hooks, run:

.. code-block:: bash

    uv tool install pre-commit
    pre-commit install

To run the pre-commit checks manually on all, not just the modified files, run:

.. code-block:: bash

    pre-commit run --all-files

Documentation style
~~~~~~~~~~~~~~~~~~~

We use `Sphinx <https://www.sphinx-doc.org/>`_ to generate the documentation.
The documentation is written in `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_.

We use the `sphinx docstring-style <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`__ for new code.
The ``:param [ParamName]:`` and ``:return:`` statements are important when
applicable.
Manual type annotations (``:type  [ParamName]:``) are redundant and should be
avoided.

To build the documentation, run:

.. code-block:: bash

    uv sync --group doc
    cd doc
    make html
    # then open `build/html/index.html` in a browser
