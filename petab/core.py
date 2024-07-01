"""Deprecated module for PEtab core classes and functions.

Use petab.v1.core instead."""
from petab import _deprecated_import_v1
from petab.v1.core import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
