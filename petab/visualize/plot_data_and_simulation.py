"""Functions for plotting PEtab measurement files and simulation results in
the same format."""

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

from .plotter import MPLPlotter
from .plotting import VisSpecParser
from .. import problem
from ..C import *

# for typehints
IdsList = List[str]
NumList = List[int]

__all__ = [
    "plot_with_vis_spec",
    "plot_without_vis_spec",
    "plot_problem"
]


def plot_with_vis_spec(
        vis_spec_df: Union[str, pd.DataFrame],
        conditions_df: Union[str, pd.DataFrame],
        measurements_df: Optional[Union[str, pd.DataFrame]] = None,
        simulations_df: Optional[Union[str, pd.DataFrame]] = None,
        subplot_dir: Optional[str] = None,
        plotter_type: str = 'mpl',
        format_: str = 'png',
) -> Optional[Dict[str, plt.Subplot]]:
    """
    Plot measurements and/or simulations. Specification of the visualization
    routines is provided in visualization table.

    Parameters
    ----------
    vis_spec_df:
        A visualization table.
    conditions_df:
        A condition DataFrame in the PEtab format or path to the condition
        file.
    measurements_df:
        A measurement DataFrame in the PEtab format or path to the data file.
    simulations_df:
        A simulation DataFrame in the PEtab format or path to the simulation
        output data file.
    subplot_dir:
        A path to the folder where single subplots should be saved.
        PlotIDs will be taken as file names.
    plotter_type:
        Specifies which library should be used for plot generation. Currently,
        only matplotlib is supported.
    format_:
        File format for the generated figure.
        (See :py:func:`matplotlib.pyplot.savefig` for supported options).

    Returns
    -------
    ax: Axis object of the created plot.
    None: In case subplots are saved to a file.
    """

    if measurements_df is None and simulations_df is None:
        raise TypeError('Not enough arguments. Either measurements_data '
                        'or simulations_data should be provided.')

    vis_spec_parser = VisSpecParser(conditions_df, measurements_df,
                                    simulations_df)
    figure, dataprovider = vis_spec_parser.parse_from_vis_spec(vis_spec_df)

    if plotter_type == 'mpl':
        plotter = MPLPlotter(figure, dataprovider)
    else:
        raise NotImplementedError('Currently, only visualization with '
                                  'matplotlib is possible.')

    return plotter.generate_figure(subplot_dir, format_=format_)


def plot_without_vis_spec(
        conditions_df: Union[str, pd.DataFrame],
        grouping_list: Optional[List[IdsList]] = None,
        group_by: str = 'observable',
        measurements_df: Optional[Union[str, pd.DataFrame]] = None,
        simulations_df: Optional[Union[str, pd.DataFrame]] = None,
        plotted_noise: str = MEAN_AND_SD,
        subplot_dir: Optional[str] = None,
        plotter_type: str = 'mpl',
        format_: str = 'png',
) -> Optional[Dict[str, plt.Subplot]]:
    """
    Plot measurements and/or simulations. What exactly should be plotted is
    specified in a grouping_list.
    If grouping list is not provided, measurements (simulations) will be
    grouped by observable, i.e. all measurements for each observable will be
    visualized on one plot.

    Parameters
    ----------
    grouping_list:
        A list of lists. Each sublist corresponds to a plot, each subplot
        contains the Ids of datasets or observables or simulation conditions
        for this plot.
    group_by:
        Grouping type.
        Possible values: 'dataset', 'observable', 'simulation'.
    conditions_df:
        A condition DataFrame in the PEtab format or path to the condition
        file.
    measurements_df:
        A measurement DataFrame in the PEtab format or path to the data file.
    simulations_df:
        A simulation DataFrame in the PEtab format or path to the simulation
        output data file.
    plotted_noise:
        A string indicating how noise should be visualized:
        ['MeanAndSD' (default), 'MeanAndSEM', 'replicate', 'provided'].
    subplot_dir:
        A path to the folder where single subplots should be saved.
        PlotIDs will be taken as file names.
    plotter_type:
        Specifies which library should be used for plot generation. Currently,
        only matplotlib is supported.
    format_:
        File format for the generated figure.
        (See :py:func:`matplotlib.pyplot.savefig` for supported options).

    Returns
    -------
    ax: Axis object of the created plot.
    None: In case subplots are saved to a file.
    """

    if measurements_df is None and simulations_df is None:
        raise TypeError('Not enough arguments. Either measurements_data '
                        'or simulations_data should be provided.')

    vis_spec_parser = VisSpecParser(conditions_df, measurements_df,
                                    simulations_df)

    figure, dataprovider = vis_spec_parser.parse_from_id_list(
        grouping_list, group_by, plotted_noise)

    if plotter_type == 'mpl':
        plotter = MPLPlotter(figure, dataprovider)
    else:
        raise NotImplementedError('Currently, only visualization with '
                                  'matplotlib is possible.')

    return plotter.generate_figure(subplot_dir, format_=format_)


def plot_problem(
        petab_problem: problem.Problem,
        simulations_df: Optional[Union[str, pd.DataFrame]] = None,
        grouping_list: Optional[List[IdsList]] = None,
        group_by: str = 'observable',
        plotted_noise: str = MEAN_AND_SD,
        subplot_dir: Optional[str] = None,
        plotter_type: str = 'mpl'
) -> Optional[Dict[str, plt.Subplot]]:
    """
    Visualization using petab problem.
    If Visualization table is part of the petab_problem, it will be used for
    visualization. Otherwise, grouping_list will be used.
    If neither Visualization table nor grouping_list are available,
    measurements (simulations) will be grouped by observable, i.e. all
    measurements for each observable will be visualized on one plot.

    Parameters
    ----------
    petab_problem:
        A PEtab problem.
    simulations_df:
        A simulation DataFrame in the PEtab format or path to the simulation
        output data file.
    grouping_list:
        A list of lists. Each sublist corresponds to a plot, each subplot
        contains the Ids of datasets or observables or simulation conditions
        for this plot.
    group_by:
        Possible values: 'dataset', 'observable', 'simulation'.
    plotted_noise:
        A string indicating how noise should be visualized:
        ['MeanAndSD' (default), 'MeanAndSEM', 'replicate', 'provided'].
    subplot_dir:
        A string which is taken as path to the folder where single subplots
        should be saved. PlotIDs will be taken as file names.
    plotter_type:
        Specifies which library should be used for plot generation. Currently,
        only matplotlib is supported.

    Returns
    -------
    ax: Axis object of the created plot.
    None: In case subplots are saved to a file.
    """

    if petab_problem.visualization_df is not None:
        return plot_with_vis_spec(petab_problem.visualization_df,
                                  petab_problem.condition_df,
                                  petab_problem.measurement_df,
                                  simulations_df,
                                  subplot_dir,
                                  plotter_type)
    return plot_without_vis_spec(petab_problem.condition_df,
                                 grouping_list,
                                 group_by,
                                 petab_problem.measurement_df,
                                 simulations_df,
                                 plotted_noise,
                                 subplot_dir,
                                 plotter_type)
