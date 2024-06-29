"""Deprecated module for linting PEtab visualization files.

Use petab.v1.visualize.lint instead."""
from petab import _deprecated_import_v1
from petab.v1.visualize.lint import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
