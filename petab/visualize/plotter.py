"""Deprecated module. Use petab.v1.visualize.plotter instead."""
from petab import _deprecated_import_v1
from petab.v1.visualize.plotter import *  # noqa: F403, F401, E402
from petab.v1.visualize.plotter import (  # noqa: F401
    measurement_line_kwargs,
    simulation_line_kwargs,
)

_deprecated_import_v1(__name__)
