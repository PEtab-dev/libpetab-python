import numpy as np
import pandas as pd

from typing import List
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from .plotting import (Figure, DataProvider, Subplot, DataPlot)
from ..problem import Problem
from .plotting_config import square_plot_equal_ranges
from ..C import *

# for typehints
IdsList = List[str]
NumList = List[int]


class Plotter:
    def __init__(self, figure: Figure, data_provider: DataProvider):
        """

        :param figure:
        """
        self.figure = figure
        self.data_provider = data_provider

    # def create_figure(self, num_subplots) -> Figure:
    #     pass

    def generate_figure(self):
        # TODO save plot
        pass


class MPLPlotter(Plotter):
    """
    matplotlib wrapper
    """
    def __init__(self, figure: Figure, data_provider: DataProvider):
        super().__init__(figure, data_provider)

    def generate_lineplot(self, ax, dataplot: DataPlot, plotTypeData):
        """
        It is possible to plot only data or only simulation or both

        Parameters
        ----------
        ax
        dataplot
        plotTypeData

        Returns
        -------

        """

        simu_colors = None
        data_to_plot = self.data_provider.get_data_to_plot(dataplot)

        # set type of noise
        if plotTypeData == MEAN_AND_SD:
            noise_col = 'sd'
        elif plotTypeData == MEAN_AND_SEM:
            noise_col = 'sem'
        elif plotTypeData == PROVIDED:
            noise_col = 'noise_model'

        # add xOffset
        data_to_plot.conditions += dataplot.xOffset
        label_base = dataplot.legendEntry

        if data_to_plot.measurements_to_plot is not None:
            # plotting all measurement data

            # TODO:
            if plotTypeData == REPLICATE:
                pass
                # p = ax.plot(
                #     data_to_plot.conditions[conditions.index.values],
                #     data_to_plot.measurements_to_plot.repl[
                #         data_to_plot.measurements_to_plot.repl.index.values], 'x',
                #     label=label_base
                # )

            # construct errorbar-plots: noise specified above
            else:
                # sort index for the case that indices of conditions and
                # measurements differ if indep_var='time', conditions is a
                # numpy array, for indep_var=observable its a Series
                if isinstance(data_to_plot.conditions, np.ndarray):
                    data_to_plot.conditions.sort()
                elif isinstance(data_to_plot.conditions,
                                pd.core.series.Series):
                    data_to_plot.conditions.sort_index(inplace=True)
                else:
                    raise ValueError('Strange: conditions object is '
                                     'neither numpy nor series...')
                data_to_plot.measurements_to_plot.sort_index(inplace=True)
                # sorts according to ascending order of conditions
                scond, smean, snoise = \
                    zip(*sorted(zip(
                        data_to_plot.conditions,
                        data_to_plot.measurements_to_plot['mean'],
                        data_to_plot.measurements_to_plot[noise_col])))
                p = ax.errorbar(
                    scond, smean, snoise,
                    linestyle='-.', marker='.', label=label_base
                )

                # simulations should have the same colors if both measurements
                # and simulations are plotted
                simu_colors = p[0].get_color()

        # construct simulation plot
        if data_to_plot.simulations_to_plot is not None:

            xs, ys = zip(*sorted(zip(data_to_plot.conditions,
                                     data_to_plot.simulations_to_plot)))
            ax.plot(
                xs, ys, linestyle='-', marker='o',
                label=label_base + " simulation", color=simu_colors
            )

    def generate_barplot(self, ax, dataplot: DataPlot, plotTypeData: str):

        # set type of noise
        if plotTypeData == MEAN_AND_SD:
            noise_col = 'sd'
        elif plotTypeData == MEAN_AND_SEM:
            noise_col = 'sem'
        elif plotTypeData == PROVIDED:
            noise_col = 'noise_model'

        simu_colors = None
        data_to_plot = self.data_provider.get_data_to_plot(dataplot)

        x_name = dataplot.legendEntry

        if data_to_plot.simulations_to_plot:
            bar_kwargs = {
                'align': 'edge',
                'width': -1/3,
            }
        else:
            bar_kwargs = {
                'align': 'center',
                'width': 2/3,
            }

        # TODO shouldn't be any seaborn dependency here
        color = sns.color_palette()[0]

        if data_to_plot.measurements_to_plot is not None:
            p = ax.bar(x_name, data_to_plot.measurements_to_plot['mean'],
                       yerr=data_to_plot.measurements_to_plot[noise_col],
                       color=color, **bar_kwargs)
            simu_colors = p[0].get_facecolor()

        if data_to_plot.simulations_to_plot is not None:
            bar_kwargs['width'] = -bar_kwargs['width']
            ax.bar(x_name, data_to_plot.simulations_to_plot, color='white',
                   edgecolor=simu_colors, **bar_kwargs)

    def generate_scatterplot(self, ax, dataplot: DataPlot):

        data_to_plot = self.data_provider.get_data_to_plot(dataplot)

        if data_to_plot.simulations_to_plot is None:
            raise NotImplementedError('Scatter plots do not work without'
                                      ' simulation data')
        ax.scatter(data_to_plot.measurements_to_plot['mean'],
                   data_to_plot.simulations_to_plot,
                   label=getattr(dataplot, LEGEND_ENTRY))
        ax = square_plot_equal_ranges(ax)

    def generate_dataplot(self, ax, dataplot: DataPlot, plotTypeData):
        pass

    def generate_subplot(self,
                         ax,
                         subplot: Subplot):

        # plot_lowlevel

        # set yScale
        if subplot.yScale == LIN:
            ax.set_yscale("linear")
        elif subplot.yScale == LOG10:
            ax.set_yscale("log")
        elif subplot.yScale == LOG:
            ax.set_yscale("log", basey=np.e)

        # ms thing should be inside a single plot

        # TODO:
        # if subplot.measurements_to_plot:
        #     # add yOffset
        #     subplot.measurements_to_plot.loc[:, 'mean'] = \
        #         subplot.measurements_to_plot['mean'] + subplot.yOffset
        #     subplot.measurements_to_plot.loc[:, 'repl'] = \
        #         subplot.measurements_to_plot['repl'] + subplot.yOffset
        #
        # if subplot.simulations_to_plot:
        #     ms.loc[:, 'sim'] = ms['sim'] + subplot.vis_spec.yOffset

        if subplot.plotTypeSimulation == BAR_PLOT:
            for data_plot in subplot.data_plots:
                self.generate_barplot(ax, data_plot, subplot.plotTypeData)
        elif subplot.plotTypeSimulation == SCATTER_PLOT:
            for data_plot in subplot.data_plots:
                self.generate_scatterplot(ax, data_plot)
        else:

            # set xScale
            if subplot.xScale == LIN:
                ax.set_xscale("linear")
            elif subplot.xScale == LOG10:
                ax.set_xscale("log")
            elif subplot.xScale == LOG:
                ax.set_xscale("log", basex=np.e)
            # equidistant
            elif subplot.xScale == 'order':
                ax.set_xscale("linear")
                # check if conditions are monotone decreasing or increasing
                # todo: conditions
                if np.all(
                        np.diff(subplot.conditions) < 0):  # monot. decreasing
                    xlabel = subplot.conditions[::-1]  # reversing
                    conditions = range(len(subplot.conditions))[
                                 ::-1]  # reversing
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

        if not subplot.plotTypeSimulation == BAR_PLOT:
            ax.legend()
        ax.set_title(subplot.plotName)
        ax.relim()
        ax.autoscale_view()

        # Beautify plots
        ax.set_xlabel(
            subplot.xLabel)
        ax.set_ylabel(
            subplot.yLabel)

        return ax

    def generate_figure(self):

        # Set Options for plots
        # possible options: see: plt.rcParams.keys()
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['figure.figsize'] = self.figure.size
        plt.rcParams['errorbar.capsize'] = 2

        # Set Colormap
        # sns.set(style="ticks", palette="colorblind") ?

        # compute, how many rows and columns we need for the subplots
        num_row = int(np.round(np.sqrt(self.figure.num_subplots)))
        num_col = int(np.ceil(self.figure.num_subplots / num_row))

        fig, axes = plt.subplots(num_row, num_col, squeeze=False)

        for ax in axes.flat[self.figure.num_subplots:]:
            ax.remove()

        axes = dict(zip([plot.plotId for plot in self.figure.subplots],
                        axes.flat))

        for idx, subplot in enumerate(self.figure.subplots):
            self.generate_subplot(axes[subplot.plotId], subplot)

        fig.tight_layout()

        return axes


class SeabornPlotter(Plotter):
    """
    seaborn wrapper
    """
    def __init__(self, figure: Figure, data_provider: DataProvider):
        super().__init__(figure, data_provider)

    def generate_plot(self):
        pass

def plot_measurements():
    pass

def plot_simulations():
    pass

def plot_problem():
    pass
