"""
Functions for plotting residuals.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats

from ..calculate import calculate_residuals
from ..core import get_simulation_df
from ..problem import Problem
from ..C import *

__all__ = ['plot_residuals_vs_simulation']


def plot_residuals_vs_simulation(
        petab_problem: Problem,
        simulations_df: Union[str, Path, pd.DataFrame],
        size: Optional[Tuple] = (10, 7),
        axes: Optional[Tuple[plt.Axes, plt.Axes]] = None
) -> matplotlib.axes.Axes:
    """
    Plot residuals versus simulation values for measurements with normal noise
    assumption.

    Parameters
    ----------
    petab_problem:
        A PEtab problem.
    simulations_df:
        A simulation DataFrame in the PEtab format or path to the simulation
        output data file.
    size:
        Figure size.
    axes:
        Axis object.

    Returns
    -------
        ax: Axis object of the created plot.
    """
    if isinstance(simulations_df, (str, Path)):
        simulations_df = get_simulation_df(simulations_df)

    if NOISE_DISTRIBUTION in petab_problem.observable_df:
        if OBSERVABLE_TRANSFORMATION in petab_problem.observable_df:
            observable_ids = petab_problem.observable_df[
                (petab_problem.observable_df[NOISE_DISTRIBUTION] == NORMAL) &
                (petab_problem.observable_df[OBSERVABLE_TRANSFORMATION] == LIN)
            ].index

        else:
            observable_ids = petab_problem.observable_df[
                petab_problem.observable_df[NOISE_DISTRIBUTION] == NORMAL
            ].index
    else:
        observable_ids = petab_problem.observable_df.index

    if observable_ids.empty:
        raise ValueError("Residuals plot is only applicable for normal "
                         "additive noise assumption")

    if axes is None:
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=size,
                                 width_ratios=[2, 1])
        fig.set_layout_engine("tight")
        fig.suptitle("Residuals")

    residual_df = calculate_residuals(
        measurement_dfs=petab_problem.measurement_df,
        simulation_dfs=simulations_df,
        observable_dfs=petab_problem.observable_df,
        parameter_dfs=petab_problem.parameter_df)[0]

    normal_residuals = residual_df[residual_df[OBSERVABLE_ID].isin(
        observable_ids)]
    simulations_normal = simulations_df[
        simulations_df[OBSERVABLE_ID].isin(observable_ids)]

    # compare to standard normal distribution
    ks_result = stats.kstest(normal_residuals[RESIDUAL], stats.norm.cdf)

    # plot the residuals plot
    axes[0].hlines(y=0, xmin=min(simulations_normal[SIMULATION]),
                   xmax=max(simulations_normal[SIMULATION]), ls='--',
                   color='gray')
    axes[0].scatter(simulations_normal[SIMULATION],
                    normal_residuals[RESIDUAL])
    axes[0].text(0.15, 0.85,
                 f'Kolmogorov-Smirnov test results:\n'
                 f'statistic: {ks_result[0]:.2f}\n'
                 f'pvalue: {ks_result[1]:.2e} ', transform=axes[0].transAxes)
    axes[0].set_xlabel('simulated values')
    axes[0].set_ylabel('residuals')

    # plot histogram
    axes[1].hist(normal_residuals[RESIDUAL], density=True,
                 orientation='horizontal')
    axes[1].set_xlabel('distribution')

    ymin, ymax = axes[0].get_ylim()
    ylim = max(abs(ymin), abs(ymax))
    axes[0].set_ylim(-ylim, ylim)
    axes[1].tick_params(left=False, labelleft=False, right=True,
                        labelright=True)

    return axes
