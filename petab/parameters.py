"""Deprecated module for parameter table handling.

Use petab.v1.parameters instead."""
from petab import _deprecated_import_v1
from petab.v1.parameters import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
