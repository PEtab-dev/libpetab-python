"""
Functions for plotting residuals.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

from ..calculate import calculate_residuals
from ..core import get_simulation_df
from ..problem import Problem
from ..C import *

__all__ = ['plot_goodness_of_fit', 'plot_residuals_vs_simulation']


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


def plot_goodness_of_fit(
        petab_problem: Problem,
        simulations_df: Union[str, Path, pd.DataFrame],
        size: Tuple = (10, 7),
        ax: Optional[plt.Axes] = None
) -> matplotlib.axes.Axes:
    """
    Plot goodness of fit.

    Parameters
    ----------
    petab_problem:
        A PEtab problem.
    simulations_df:
        A simulation DataFrame in the PEtab format or path to the simulation
        output data file.
    size:
        Figure size.
    ax:
        Axis object.

    Returns
    -------
        ax: Axis object of the created plot.
    """

    if isinstance(simulations_df, (str, Path)):
        simulations_df = get_simulation_df(simulations_df)

    if simulations_df is None or petab_problem.measurement_df is None:
        raise NotImplementedError('Both measurements and simulation data '
                                  'are needed for goodness_of_fit')

    residual_df = calculate_residuals(
        measurement_dfs=petab_problem.measurement_df,
        simulation_dfs=simulations_df,
        observable_dfs=petab_problem.observable_df,
        parameter_dfs=petab_problem.parameter_df)[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        petab_problem.measurement_df['measurement'],
        simulations_df['simulation'])  # x, y

    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        fig.set_layout_engine("tight")

    ax.scatter(petab_problem.measurement_df['measurement'],
               simulations_df['simulation'])

    ax.axis('square')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim = [min([xlim[0], ylim[0]]),
           max([xlim[1], ylim[1]])]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    x = np.linspace(lim, 100)
    ax.plot(x, x, linestyle='--',
            color='gray')
    ax.plot(x,
            intercept + slope*x, 'r',
            label='fitted line')

    mse = np.mean(np.abs(residual_df['residual']))
    ax.text(0.1, 0.70,
            f'$R^2$: {r_value**2:.2f}\n'
            f'slope: {slope:.2f}\n'
            f'intercept: {intercept:.2f}\n'
            f'pvalue: {std_err:.2e}\n'
            f'mean squared error: {mse:.2e}\n',
            transform=ax.transAxes)

    ax.set_title("Goodness of fit")
    ax.set_xlabel('simulated values')
    ax.set_ylabel('measurements')
    return ax
