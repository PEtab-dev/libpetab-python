"""Deprecated module for mapping tables.

Use petab.v1.mapping instead."""
from petab import _deprecated_import_v1
from petab.v1.mapping import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
