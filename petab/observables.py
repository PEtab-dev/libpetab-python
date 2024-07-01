"""Deprecated module for observable tables.

Use petab.v1.observables instead.
"""
from petab import _deprecated_import_v1
from petab.v1.observables import *  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
