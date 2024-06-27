"""
Visualize
=========

PEtab comes with visualization functionality. Those need to be imported via
``import petab.visualize``.

"""
# ruff: noqa: F401
import importlib.util

from .plotting import DataProvider, Figure

__all__ = ["DataProvider", "Figure"]

if importlib.util.find_spec("matplotlib") is not None:
    from .plot_data_and_simulation import (
        plot_problem,
        plot_with_vis_spec,
        plot_without_vis_spec,
    )
    from .plot_residuals import (
        plot_goodness_of_fit,
        plot_residuals_vs_simulation,
    )
    from .plotter import MPLPlotter

    __all__.extend(
        [
            "plot_without_vis_spec",
            "plot_with_vis_spec",
            "plot_problem",
            "plot_goodness_of_fit",
            "plot_residuals_vs_simulation",
            "MPLPlotter",
        ]
    )
