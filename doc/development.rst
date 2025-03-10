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

We follow `numpy's Python support policy <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_.

Release process
---------------

1. Update the version number in ``petab/version.py``.

2. Update the changelog in ``doc/CHANGELOG.md``.

3. Create a pull request with the changes to the main branch.

4. Once the pull request is merged, create a new release on GitHub.
   Make sure to set the tag to the version number prefixed with 'v'
   (e.g., ``v1.0.0``), and the release title to ``libpetab-python $RELEASE_TAG``
   (e.g., ``libpetab-python v1.0.0``).

5. The release will be automatically uploaded to PyPI through a GitHub actions
   workflow.


Style guide
-----------

Code style
~~~~~~~~~~

We use pre-commit with ruff to enforce code style. To install pre-commit and
the pre-commit hooks, run:

.. code-block:: bash

    pip install pre-commit
    pre-commit install

To run the pre-commit checks manually on all, not just the modified files, run:

.. code-block:: bash

    pre-commit run --all-files

Documentation style
~~~~~~~~~~~~~~~~~~~

We use `Sphinx <https://www.sphinx-doc.org/>`_ to generate the documentation.
The documentation is written in `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_.

We use the `sphinx docstring-style <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`__ for new code.

To build the documentation, run:

.. code-block:: bash

    cd doc
    make html
    # then open `build/html/index.html` in a browser
