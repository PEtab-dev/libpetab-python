"""Deprecated module for simplifying PEtab problems.

Use petab.simplify instead."""
from petab import _deprecated_import_v1
from petab.v1.simplify import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
