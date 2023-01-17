"""
Functions for plotting residuals.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats

from .. import calculate, core, problem
from ..C import *


def plot_residuals(
        petab_problem: problem.Problem,
        simulations_df: Union[str, pd.DataFrame],
        size: Tuple = (10, 7),
        ax: Optional[plt.Axes] = None
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

    Returns
    -------
        ax: Axis object of the created plot.
    """
    if isinstance(simulations_df, (str, Path)):
        simulations_df = core.get_simulation_df(simulations_df)

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

    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        fig.set_layout_engine("tight")

    residual_df = calculate.calculate_residuals(
        measurement_dfs=petab_problem.measurement_df,
        simulation_dfs=simulations_df,
        observable_dfs=petab_problem.observable_df,
        parameter_dfs=petab_problem.parameter_df)[0]

    normal_resuduals = residual_df[residual_df[OBSERVABLE_ID].isin(
        observable_ids)]
    simulations_normal = simulations_df[
        simulations_df[OBSERVABLE_ID].isin(observable_ids)]

    # compare to standard normal distribution
    ks_result = stats.kstest(normal_resuduals['residual'], stats.norm.cdf)

    ax.hlines(y=0, xmin=min(simulations_normal['simulation']),
              xmax=max(simulations_normal['simulation']), ls='--',
              color='gray')
    ax.scatter(simulations_normal['simulation'],
               normal_resuduals['residual'])
    ax.text(0.3, 0.85,
            f'Kolmogorov-Smirnov test results:\n'
            f'statistic: {ks_result[0]:.2f}\n'
            f'pvalue: {ks_result[1]:.2e} ', transform=ax.transAxes)

    ax.set_title("Residuals")
    ax.set_xlabel('simulated values')
    ax.set_ylabel('residuals')

    plt.tight_layout()
    return ax

