"""PEtab visualization plotter classes"""
import os

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from .plotting import (Figure, DataProvider, Subplot, DataPlot)
from ..C import *

__all__ = ['Plotter', 'MPLPlotter', 'SeabornPlotter']


class Plotter(ABC):
    """
    Plotter abstract base class.

    Attributes
    ----------

    figure:
        Figure instance that serves as a markup for the figure that
        should be generated
    data_provider:
        Data provider
    """
    def __init__(self, figure: Figure, data_provider: DataProvider):
        self.figure = figure
        self.data_provider = data_provider

    @abstractmethod
    def generate_figure(
            self,
            subplot_dir: Optional[str] = None
    ) -> Optional[Dict[str, plt.Subplot]]:
        pass


class MPLPlotter(Plotter):
    """
    Matplotlib wrapper
    """
    def __init__(self, figure: Figure, data_provider: DataProvider):
        super().__init__(figure, data_provider)

    @staticmethod
    def _error_column_for_plot_type_data(plot_type_data: str) -> Optional[str]:
        """Translate PEtab plotTypeData value to column name of internal
        data representation

        Parameters
        ----------
            plot_type_data: PEtab plotTypeData value
        Returns
        -------
            Name of corresponding column
        """
        if plot_type_data == MEAN_AND_SD:
            return 'sd'
        if plot_type_data == MEAN_AND_SEM:
            return 'sem'
        if plot_type_data == PROVIDED:
            return 'noise_model'
        return None

    def generate_lineplot(
            self,
            ax: 'matplotlib.pyplot.Axes',
            dataplot: DataPlot,
            plotTypeData: str
    ) -> None:
        """
        Generate lineplot.

        It is possible to plot only data or only simulation or both.

        Parameters
        ----------
        ax:
            Axis object.
        dataplot:
            Visualization settings for the plot.
        plotTypeData:
            Specifies how replicates should be handled.
        """
        simu_color = None
        measurements_to_plot, simulations_to_plot = \
            self.data_provider.get_data_to_plot(dataplot,
                                                plotTypeData == PROVIDED)
        noise_col = self._error_column_for_plot_type_data(plotTypeData)

        label_base = dataplot.legendEntry

        if measurements_to_plot is not None \
                and not measurements_to_plot.data_to_plot.empty:
            # plotting all measurement data

            if plotTypeData == REPLICATE:
                replicates = np.stack(
                    measurements_to_plot.data_to_plot.repl.values)

                # plot first replicate
                p = ax.plot(
                    measurements_to_plot.conditions,
                    replicates[:, 0],
                    linestyle='-.',
                    marker='x', markersize=10, label=label_base
                )

                # plot other replicates with the same color
                ax.plot(
                    measurements_to_plot.conditions,
                    replicates[:, 1:],
                    linestyle='-.',
                    marker='x', markersize=10, color=p[0].get_color()
                )

            # construct errorbar-plots: noise specified above
            else:
                # sorts according to ascending order of conditions
                scond, smean, snoise = \
                    zip(*sorted(zip(
                        measurements_to_plot.conditions,
                        measurements_to_plot.data_to_plot['mean'],
                        measurements_to_plot.data_to_plot[noise_col])))
                p = ax.errorbar(
                    scond, smean, snoise,
                    linestyle='-.', marker='.', label=label_base
                )

            # simulations should have the same colors if both measurements
            # and simulations are plotted
            simu_color = p[0].get_color()

        # construct simulation plot
        if simulations_to_plot is not None:
            # markers will be displayed only for points that have measurement
            # counterpart
            if measurements_to_plot is not None:
                meas_conditions = measurements_to_plot.conditions.to_numpy() \
                    if isinstance(measurements_to_plot.conditions, pd.Series) \
                    else measurements_to_plot.conditions
                every = [condition in meas_conditions
                         for condition in simulations_to_plot.conditions]
            else:
                every = None
            # sorts according to ascending order of conditions
            xs, ys = zip(*sorted(zip(simulations_to_plot.conditions,
                                     simulations_to_plot.data_to_plot['mean'])
                                 ))
            ax.plot(
                xs, ys, linestyle='-', marker='o', markevery=every,
                label=label_base + " simulation", color=simu_color
            )

    def generate_barplot(
            self,
            ax: 'matplotlib.pyplot.Axes',
            dataplot: DataPlot,
            plotTypeData: str
    ) -> None:
        """
        Generate barplot.

        Parameters
        ----------
        ax:
            Axis object.
        dataplot:
            Visualization settings for the plot.
        plotTypeData:
            Specifies how replicates should be handled.
        """
        # TODO: plotTypeData == REPLICATE?
        noise_col = self._error_column_for_plot_type_data(plotTypeData)

        measurements_to_plot, simulations_to_plot = \
            self.data_provider.get_data_to_plot(dataplot,
                                                plotTypeData == PROVIDED)

        x_name = dataplot.legendEntry

        if simulations_to_plot:
            bar_kwargs = {
                'align': 'edge',
                'width': -1/3,
            }
        else:
            bar_kwargs = {
                'align': 'center',
                'width': 2/3,
            }

        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

        if measurements_to_plot is not None:
            ax.bar(x_name, measurements_to_plot.data_to_plot['mean'],
                   yerr=measurements_to_plot.data_to_plot[noise_col],
                   color=color, **bar_kwargs, label='measurement')

        if simulations_to_plot is not None:
            bar_kwargs['width'] = -bar_kwargs['width']
            ax.bar(x_name, simulations_to_plot.data_to_plot['mean'],
                   color='white', edgecolor=color, **bar_kwargs,
                   label='simulation')

    def generate_scatterplot(
            self,
            ax: 'matplotlib.pyplot.Axes',
            dataplot: DataPlot,
            plotTypeData: str
    ) -> None:
        """
        Generate scatterplot.

        Parameters
        ----------
        ax:
            Axis object.
        dataplot:
            Visualization settings for the plot.
        plotTypeData:
            Specifies how replicates should be handled.
        """
        measurements_to_plot, simulations_to_plot = \
            self.data_provider.get_data_to_plot(dataplot,
                                                plotTypeData == PROVIDED)

        if simulations_to_plot is None or measurements_to_plot is None:
            raise NotImplementedError('Scatter plots do not work without'
                                      ' simulation data')
        ax.scatter(measurements_to_plot.data_to_plot['mean'],
                   simulations_to_plot.data_to_plot['mean'],
                   label=getattr(dataplot, LEGEND_ENTRY))
        self._square_plot_equal_ranges(ax)

    def generate_subplot(
            self,
            ax: plt.Axes,
            subplot: Subplot
    ) -> None:
        """
        Generate subplot based on markup provided by subplot.

        Parameters
        ----------
        ax:
            Axis object.
        subplot:
            Subplot visualization settings.
        """
        # set yScale
        if subplot.yScale == LIN:
            ax.set_yscale("linear")
        elif subplot.yScale == LOG10:
            ax.set_yscale("log")
        elif subplot.yScale == LOG:
            ax.set_yscale("log", base=np.e)

        if subplot.plotTypeSimulation == BAR_PLOT:
            for data_plot in subplot.data_plots:
                self.generate_barplot(ax, data_plot, subplot.plotTypeData)

            # get rid of duplicate legends
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            x_names = [x.legendEntry for x in subplot.data_plots]
            ax.set_xticks(range(len(x_names)))
            ax.set_xticklabels(x_names)

            for label in ax.get_xmajorticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment("right")
        elif subplot.plotTypeSimulation == SCATTER_PLOT:
            for data_plot in subplot.data_plots:
                self.generate_scatterplot(ax, data_plot, subplot.plotTypeData)
        else:
            # set xScale
            if subplot.xScale == LIN:
                ax.set_xscale("linear")
            elif subplot.xScale == LOG10:
                ax.set_xscale("log")
            elif subplot.xScale == LOG:
                ax.set_xscale("log", base=np.e)
            # equidistant
            elif subplot.xScale == 'order':
                ax.set_xscale("linear")
                # check if conditions are monotone decreasing or increasing
                if np.all(np.diff(subplot.conditions) < 0):
                    # monot. decreasing -> reverse
                    xlabel = subplot.conditions[::-1]
                    conditions = range(len(subplot.conditions))[::-1]
                    ax.set_xticks(range(len(conditions)), xlabel)
                elif np.all(np.diff(subplot.conditions) > 0):
                    xlabel = subplot.conditions
                    conditions = range(len(subplot.conditions))
                    ax.set_xticks(range(len(conditions)), xlabel)
                else:
                    raise ValueError('Error: x-conditions do not coincide, '
                                     'some are mon. increasing, some '
                                     'monotonically decreasing')

            for data_plot in subplot.data_plots:
                self.generate_lineplot(ax, data_plot, subplot.plotTypeData)

        # show 'e' as basis not 2.7... in natural log scale cases
        def ticks(y, _):
            return r'$e^{{{:.0f}}}$'.format(np.log(y))

        if subplot.xScale == LOG:
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(ticks))
        if subplot.yScale == LOG:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))

        if subplot.plotTypeSimulation != BAR_PLOT:
            ax.legend()
        ax.set_title(subplot.plotName)
        if subplot.xlim:
            ax.set_xlim(subplot.xlim)
        if subplot.ylim:
            ax.set_ylim(subplot.ylim)
        ax.autoscale_view()

        # Beautify plots
        ax.set_xlabel(subplot.xLabel)
        ax.set_ylabel(subplot.yLabel)

    def generate_figure(
            self,
            subplot_dir: Optional[str] = None,
            format_: str = 'png',
    ) -> Optional[Dict[str, plt.Subplot]]:
        """
        Generate the full figure based on the markup in the figure attribute.

        Parameters
        ----------
        subplot_dir:
            A path to the folder where single subplots should be saved.
            PlotIDs will be taken as file names.
        format_:
            File format for the generated figure.
            (See :py:func:`matplotlib.pyplot.savefig` for supported options).

        Returns
        -------
        ax:
            Axis object of the created plot.
        None:
            In case subplots are saved to file.
        """
        if subplot_dir is None:
            # compute, how many rows and columns we need for the subplots
            num_row = int(np.round(np.sqrt(self.figure.num_subplots)))
            num_col = int(np.ceil(self.figure.num_subplots / num_row))

            fig, axes = plt.subplots(num_row, num_col, squeeze=False,
                                     figsize=self.figure.size)
            fig.set_layout_engine("tight")

            for ax in axes.flat[self.figure.num_subplots:]:
                ax.remove()

            axes = dict(zip([plot.plotId for plot in self.figure.subplots],
                            axes.flat))

        for subplot in self.figure.subplots:
            if subplot_dir is not None:
                fig, ax = plt.subplots(figsize=self.figure.size)
                fig.set_layout_engine("tight")
            else:
                ax = axes[subplot.plotId]

            try:
                self.generate_subplot(ax, subplot)
            except Exception as e:
                raise RuntimeError(
                    f"Error plotting {getattr(subplot, PLOT_ID)}.") from e

            if subplot_dir is not None:
                # TODO: why this doesn't work?
                plt.tight_layout()
                plt.savefig(os.path.join(subplot_dir,
                                         f'{subplot.plotId}.{format_}'))
                plt.close()

        if subplot_dir is None:
            # TODO: why this doesn't work?
            plt.tight_layout()
            return axes

    @staticmethod
    def _square_plot_equal_ranges(
            ax: 'matplotlib.pyplot.Axes',
            lim: Optional[Union[List, Tuple]] = None
    ) -> 'matplotlib.pyplot.Axes':
        """
        Square plot with equal range for scatter plots.

        Returns
        -------
            Updated axis object.
        """

        ax.axis('square')

        if lim is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            lim = [np.min([xlim[0], ylim[0]]),
                   np.max([xlim[1], ylim[1]])]

        ax.set_xlim(lim)
        ax.set_ylim(lim)

        # Same tick mark on x and y
        ax.yaxis.set_major_locator(ax.xaxis.get_major_locator())

        return ax


class SeabornPlotter(Plotter):
    """
    Seaborn wrapper.
    """
    def __init__(self, figure: Figure, data_provider: DataProvider):
        super().__init__(figure, data_provider)

    def generate_figure(
            self,
            subplot_dir: Optional[str] = None
    ) -> Optional[Dict[str, plt.Subplot]]:
        pass
