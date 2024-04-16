"""
Visualize
=========

PEtab comes with visualization functionality. Those need to be imported via
``import petab.visualize``.

"""
import importlib.util

mpl_spec = importlib.util.find_spec("matplotlib")

from .plotting import DataProvider, Figure

__all__ = ["DataProvider", "Figure"]

if mpl_spec is not None:
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
