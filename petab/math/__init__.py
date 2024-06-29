"""Deprecated module for math handling.

Use petab.v1.math instead."""
from petab import _deprecated_import_v1
from petab.v1.math import *  # noqa: F403, F401, E402

from .sympify import sympify_petab  # noqa: F401

_deprecated_import_v1(__name__)
