"""Deprecated module. Use petab.v1.visualize.plotting instead."""
from petab import _deprecated_import_v1
from petab.v1.visualize.plotting import *  # noqa: F403, F401, E402
from petab.v1.visualize.plotting import DEFAULT_FIGSIZE  # noqa: F401

_deprecated_import_v1(__name__)
