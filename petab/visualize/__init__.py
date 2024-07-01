"""Deprecated module for visualization of PEtab problems.

Use petab.v1.visualize instead."""

from petab import _deprecated_import_v1
from petab.v1.visualize import *  # noqa: F403, F401, E402

from .plotting import DataProvider, Figure  # noqa: F403, F401, E402

_deprecated_import_v1(__name__)
