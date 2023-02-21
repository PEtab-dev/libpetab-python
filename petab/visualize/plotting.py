"""PEtab visualization data selection and visualization settings classes"""
import warnings
from numbers import Number, Real
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import pandas as pd

from .helper_functions import (create_dataset_id_list_new,
                               generate_dataset_id_col)
from .. import conditions, core, measurements
from ..C import *
from ..problem import Problem

__all__ = ['DataSeries', 'DataPlot', 'Subplot', 'Figure', 'DataProvider',
           'VisSpecParser']

# for typehints
IdsList = List[str]
NumList = List[int]


# also for type hints
# TODO: split into dataplot and subplot level dicts?
# TODO: add when only python>=3.8 is supported
# class VisDict(TypedDict):
#     PLOT_NAME: str
#     PLOT_TYPE_SIMULATION: str
#     PLOT_TYPE_DATA: str
#     X_VALUES: str
#     X_OFFSET: List[Number]
#     X_LABEL: str
#     X_SCALE: str
#     Y_VALUES: List[str]
#     Y_OFFSET: List[Number]
#     Y_LABEL: str
#     Y_SCALE: str
#     LEGEND_ENTRY: List[Number]
#     DATASET_ID: List[str]


class DataSeries:
    """
    Data for one individual line
    """
    def __init__(self, conditions_: Optional[Union[np.ndarray, pd.Series]],
                 data_to_plot: Optional[pd.DataFrame] = None):

        self.data_to_plot = data_to_plot
        self.data_to_plot.sort_index(inplace=True)

        self.conditions = conditions_
        self.inf_point = np.inf in self.conditions if \
            self.conditions is not None else False
        # sort index for the case that indices of conditions and
        # measurements differ. if indep_var='time', conditions is a
        # numpy array, if indep_var=observable it's a Series
        if isinstance(self.conditions, np.ndarray):
            self.conditions.sort()
        elif isinstance(self.conditions, pd.Series):
            self.conditions.sort_index(inplace=True)

    def add_x_offset(self, offset) -> None:
        """
        Offset for the independent variable.

        Parameters
        ----------
        offset:
            Offset value.

        """
        if self.conditions is not None:
            self.conditions += offset

    def add_y_offset(self, offset):
        self.data_to_plot['mean'] += offset
        self.data_to_plot['repl'] += offset

    def add_offsets(self, x_offset=0, y_offset=0) -> None:
        """
        Data offsets.

        Parameters
        ----------
        x_offset:
            Offset for the independent variable.
        y_offset:
            Offsets for the observable.
        """
        self.add_x_offset(x_offset)
        self.add_y_offset(y_offset)


class DataPlot:
    """
    Visualization specification of a plot of one data series, e.g. for
    an individual line on a subplot.
    """
    def __init__(self,
                 plot_settings: dict):
        """
        Constructor.

        Parameters
        ----------
        plot_settings: A plot spec for one dataplot
            (only VISUALIZATION_DF_SINGLE_PLOT_LEVEL_COLS)
        """

        for key, val in plot_settings.items():
            setattr(self, key, val)

        if DATASET_ID not in vars(self):
            raise ValueError(f'{DATASET_ID} must be specified')
        if X_VALUES not in vars(self):  # TODO: singular?
            setattr(self, X_VALUES, TIME)
        if X_OFFSET not in vars(self):
            setattr(self, X_OFFSET, 0)
        if Y_VALUES not in vars(self):
            setattr(self, Y_VALUES, '')
        if Y_OFFSET not in vars(self):
            setattr(self, Y_OFFSET, 0.)
        if LEGEND_ENTRY not in vars(self):
            setattr(self, LEGEND_ENTRY, getattr(self, DATASET_ID))

    @classmethod
    def from_df(cls, plot_spec: pd.DataFrame):

        vis_spec_dict = plot_spec.to_dict()

        return cls(vis_spec_dict)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


class Subplot:
    """
    Visualization specification of a subplot.
    """
    def __init__(self,
                 plot_id: str,
                 plot_settings: dict,
                 dataplots: Optional[List[DataPlot]] = None):
        """
        Constructor.

        Parameters
        ----------
        plot_id:
            Plot ID.
        plot_settings:
            Plot spec for a subplot (only VISUALIZATION_DF_SUBPLOT_LEVEL_COLS).
        dataplots:
            A list of data plots that should be plotted on one subplot.
        """
        # parameters of a specific subplot

        setattr(self, PLOT_ID, plot_id)
        for key, val in plot_settings.items():
            setattr(self, key, val)

        if PLOT_NAME not in vars(self):
            setattr(self, PLOT_NAME, '')
        if PLOT_TYPE_SIMULATION not in vars(self):
            setattr(self, PLOT_TYPE_SIMULATION, LINE_PLOT)
        if PLOT_TYPE_DATA not in vars(self):
            setattr(self, PLOT_TYPE_DATA, MEAN_AND_SD)
        if X_LABEL not in vars(self):
            setattr(self, X_LABEL, TIME)  # TODO: getattr(self, X_VALUES)
        if X_SCALE not in vars(self):
            setattr(self, X_SCALE, LIN)
        if Y_LABEL not in vars(self):
            setattr(self, Y_LABEL, 'values')
        if Y_SCALE not in vars(self):
            setattr(self, Y_SCALE, LIN)

        self.data_plots = dataplots if dataplots is not None else []
        self.xlim = None
        self.ylim = None

    @classmethod
    def from_df(cls, plot_id: str, vis_spec: pd.DataFrame,
                dataplots: Optional[List[DataPlot]] = None):

        vis_spec_dict = {}
        for col in vis_spec:
            if col in VISUALIZATION_DF_SUBPLOT_LEVEL_COLS:
                entry = vis_spec.loc[:, col]
                entry = np.unique(entry)
                if entry.size > 1:
                    warnings.warn(f'For {PLOT_ID} {plot_id} in column '
                                  f'{col} contradictory settings ({entry})'
                                  f'. Proceeding with first entry '
                                  f'({entry[0]}).')
                entry = entry[0]

                # check if values are allowed
                if col in [Y_SCALE, X_SCALE] and entry not in \
                        OBSERVABLE_TRANSFORMATIONS:
                    raise ValueError(f'{X_SCALE} and {Y_SCALE} have to be '
                                     f'one of the following: '
                                     + ', '.join(OBSERVABLE_TRANSFORMATIONS))
                elif col == PLOT_TYPE_DATA and entry not in \
                        PLOT_TYPES_DATA:
                    raise ValueError(f'{PLOT_TYPE_DATA} has to be one of the '
                                     f'following: '
                                     + ', '.join(PLOT_TYPES_DATA))
                elif col == PLOT_TYPE_SIMULATION and entry not in \
                        PLOT_TYPES_SIMULATION:
                    raise ValueError(f'{PLOT_TYPE_SIMULATION} has to be one of'
                                     f' the following: '
                                     + ', '.join(PLOT_TYPES_SIMULATION))

                # append new entry to dict
                vis_spec_dict[col] = entry
            else:
                warnings.warn(f'Column {col} cannot be used to specify subplot'
                              f', only settings from the following columns can'
                              f' be used:'
                              + ', '.join(VISUALIZATION_DF_SUBPLOT_LEVEL_COLS))
        return cls(plot_id, vis_spec_dict, dataplots)

    def add_dataplot(self, dataplot: DataPlot) -> None:
        """
        Add data plot.

        Parameters
        ----------
        dataplot:
            Data plot visualization settings.

        """
        self.data_plots.append(dataplot)

    def set_axes_limits(self,
                        xlim: Optional[Tuple[Optional[Real],
                                             Optional[Real]]] = None,
                        ylim: Optional[Tuple[Optional[Real],
                                             Optional[Real]]] = None):
        """
        Set axes limits for all subplots. If xlim or ylim or any of the tuple
        items is None, corresponding limit is left unchanged.

        Parameters
        ----------
        xlim:
            X axis limits.
        ylim:
            Y axis limits.
        """
        self.xlim = xlim
        self.ylim = ylim


class Figure:
    """
    Visualization specification of a figure.

    Contains information regarding how data should be visualized.
    """
    def __init__(self, subplots: Optional[List[Subplot]] = None,
                 size: Tuple = (20, 15),
                 title: Optional[Tuple] = None):
        """
        Constructor.

        Parameters
        ----------
        subplots: A list of visualization specifications for each subplot
        size: Figure size
        title: Figure title
        """

        # TODO: Isensee measurements table in doc/examples doesn't correspond
        #       to documentation: observableTransformation and
        #       noiseDistribution columns replicateId problem
        # TODO: Should we put in the documentation which combination of fields
        #  must be unique in the measurement table and add such check?
        #  obs_id + sim_cond_id + preeq_cod_id (if exists) + time +
        #  replicate_id (if exists)?
        self.size = size
        self.title = title
        self.subplots = subplots if subplots is not None else []

    @property
    def num_subplots(self) -> int:
        return len(self.subplots)

    def add_subplot(self, subplot: Subplot) -> None:
        """
        Add subplot.

        Parameters
        ----------
        subplot:
            Subplot visualization settings.

        """
        self.subplots.append(subplot)

    def set_axes_limits(self,
                        xlim: Optional[Tuple[Optional[Real],
                                             Optional[Real]]] = None,
                        ylim: Optional[Tuple[Optional[Real],
                                             Optional[Real]]] = None) -> None:
        """
        Set axes limits for all subplots. If xlim or ylim or any of the tuple
        items is None, corresponding limit is left unchanged.

        Parameters
        ----------
        xlim:
            X axis limits.
        ylim:
            Y axis limits.
        """

        for subplot in self.subplots:
            subplot.set_axes_limits(xlim, ylim)

    def save_to_tsv(self, output_file_path: str = 'visuSpec.tsv') -> None:
        """
        Save full Visualization specification table.

        Note that datasetId column in the resulting table might have been
        generated even though datasetId column in Measurement table is missing
        or is different. Please, correct it manually.

        Parameters
        ----------
        output_file_path:
            File path to which the generated visualization specification is
            saved.
        """
        # TODO: what if datasetIds were generated?

        warnings.warn(f'Note: please check that {DATASET_ID} column '
                      f'corresponds to {DATASET_ID} column in Measurement '
                      f'(Simulation) table.')

        visu_dict = {}
        for subplot in self.subplots:
            subplot_level = {key: subplot.__dict__[key] for key in
                             subplot.__dict__ if key in
                             VISUALIZATION_DF_SUBPLOT_LEVEL_COLS}

            for dataplot in subplot.data_plots:
                dataset_level = {key: dataplot.__dict__[key] for key in
                                 dataplot.__dict__ if key in
                                 VISUALIZATION_DF_SINGLE_PLOT_LEVEL_COLS}
                row = {**subplot_level, **dataset_level}
                for key, value in row.items():
                    if key in visu_dict:
                        visu_dict[key].append(value)
                    else:
                        visu_dict[key] = [row[key]]
        visu_df = pd.DataFrame.from_dict(visu_dict)
        visu_df.to_csv(output_file_path, sep='\t', index=False)


class DataProvider:
    """
    Handles data selection.
    """
    def __init__(self,
                 exp_conditions: pd.DataFrame,
                 measurements_data: Optional[pd.DataFrame] = None,
                 simulations_data: Optional[pd.DataFrame] = None):
        self.conditions_data = exp_conditions

        if measurements_data is None and simulations_data is None:
            raise TypeError('Not enough arguments. Either measurements_data '
                            'or simulations_data should be provided.')
        self.measurements_data = measurements_data
        self.simulations_data = simulations_data

    @staticmethod
    def _matches_plot_spec(df: pd.DataFrame,
                           plot_spec: 'DataPlot',
                           dataset_id) -> pd.Series:
        """
        Construct an index for subsetting of the dataframe according to what
        is specified in plot_spec.

        Parameters
        ----------
            df:
                A pandas data frame to subset, can be from measurement file or
                simulation file.
            plot_spec:
                A visualization spec from the visualization file.

        Returns
        -------
        Boolean series that can be used for subsetting of the passed
        dataframe
        """
        subset = (
            (df[DATASET_ID] == dataset_id)
        )
        if getattr(plot_spec, Y_VALUES) == '':
            if len(df.loc[subset, OBSERVABLE_ID].unique()) > 1:
                raise ValueError(
                    f'{Y_VALUES} must be specified in visualization table if '
                    f'multiple different observables are available.'
                )
        else:
            subset &= (df[OBSERVABLE_ID] == getattr(plot_spec, Y_VALUES))
        return subset

    def _get_independent_var_values(self, data_df: pd.DataFrame,
                                    dataplot: DataPlot
                                    ) -> Tuple[np.ndarray, str, pd.Series]:
        """
        Get independent variable values.

        Parameters
        ----------
        data_df:
            A pandas data frame to subset, can be from measurement file or
            simulation file.
        dataplot:
            Data plot visualization settings.

        Returns
        -------
        col_name_unique:
            A name of the column from Measurement (Simulation) table, which
            specifies independent variable values (depends on the xValues entry
            of visualization specification).
            Possible values:

            * TIME (independent variable values will be taken from the TIME
              column of Measurement (Simulation) table)

            * SIMULATION_CONDITION_ID (independent variable values will be
              taken from one of the columns of Condition table)

        uni_condition_id:
            Time points
            or
            contains all unique condition IDs which should be
            plotted together as one dataplot. Independent variable values will
            be collected for these conditions
        conditions_:
            An independent variable values or None for the BarPlot case
            possible values: time points, None, vales of independent variable
            (Parameter or Species, specified in the xValues entry of
            visualization specification) for each condition_id in
            uni_condition_id

        """

        indep_var = getattr(dataplot, X_VALUES)

        dataset_id = getattr(dataplot, DATASET_ID)

        single_m_data = data_df[self._matches_plot_spec(
            data_df, dataplot, dataset_id)]

        # gather simulationConditionIds belonging to datasetId
        uni_condition_id, uind = np.unique(
            single_m_data[SIMULATION_CONDITION_ID],
            return_index=True)
        # keep the ordering which was given by user from top to bottom
        # (avoid ordering by names '1','10','11','2',...)'
        uni_condition_id = uni_condition_id[np.argsort(uind)]
        col_name_unique = SIMULATION_CONDITION_ID

        if indep_var == TIME:
            # obtain unique observation times
            uni_condition_id = single_m_data[TIME].unique()
            col_name_unique = TIME
            conditions_ = uni_condition_id
        elif indep_var == 'condition':
            conditions_ = None
        else:
            # indep_var = parameterOrStateId case ?
            # extract conditions (plot input) from condition file
            ind_cond = self.conditions_data.index.isin(uni_condition_id)
            conditions_ = self.conditions_data[ind_cond][indep_var]

        return uni_condition_id, col_name_unique, conditions_

    def get_data_series(
            self,
            data_df: pd.DataFrame,
            data_col: Literal['measurement', 'simulation'],
            dataplot: DataPlot,
            provided_noise: bool
    ) -> DataSeries:
        """
        Get data to plot from measurement or simulation DataFrame.

        Parameters
        ----------
        data_df: measurement or simulation DataFrame
        data_col: data column, i.e. 'measurement' or 'simulation'
        dataplot: visualization specification
        provided_noise:
            True if numeric values for the noise level are provided in the
            data table

        Returns
        -------
        Data to plot
        """
        uni_condition_id, col_name_unique, conditions_ = \
            self._get_independent_var_values(data_df, dataplot)

        dataset_id = getattr(dataplot, DATASET_ID)

        # get data subset selected based on provided dataset_id
        # and observable_ids
        single_m_data = data_df[self._matches_plot_spec(
            data_df, dataplot, dataset_id)]

        # create empty dataframe for means and SDs
        measurements_to_plot = pd.DataFrame(
            columns=['mean', 'noise_model', 'sd', 'sem', 'repl'],
            index=uni_condition_id
        )

        for var_cond_id in uni_condition_id:

            subset = (single_m_data[col_name_unique] == var_cond_id)

            # what has to be plotted is selected
            data_measurements = single_m_data.loc[
                subset,
                data_col
            ]

            # TODO: all this rather inside DataSeries?
            # process the data
            measurements_to_plot.at[var_cond_id, 'mean'] = np.mean(
                data_measurements)
            measurements_to_plot.at[var_cond_id, 'sd'] = np.std(
                data_measurements)

            if provided_noise and np.any(subset):
                if len(single_m_data.loc[
                           subset, NOISE_PARAMETERS].unique()) > 1:
                    raise NotImplementedError(
                        f"Datapoints with inconsistent {NOISE_PARAMETERS} "
                        f"is currently not implemented. Stopping.")
                tmp_noise = \
                    single_m_data.loc[subset, NOISE_PARAMETERS].values[0]
                if isinstance(tmp_noise, str):
                    raise NotImplementedError(
                        "No numerical noise values provided in the "
                        "measurement table. Stopping.")
                if isinstance(tmp_noise, Number) or \
                        tmp_noise.dtype == 'float64':
                    measurements_to_plot.at[
                        var_cond_id, 'noise_model'] = tmp_noise

            # standard error of mean
            measurements_to_plot.at[var_cond_id, 'sem'] = \
                np.std(data_measurements) / np.sqrt(
                    len(data_measurements))

            # single replicates
            measurements_to_plot.at[var_cond_id, 'repl'] = \
                data_measurements.values

        data_series = DataSeries(conditions_, measurements_to_plot)
        data_series.add_offsets(dataplot.xOffset, dataplot.yOffset)
        return data_series

    def get_data_to_plot(self, dataplot: DataPlot, provided_noise: bool
                         ) -> Tuple[DataSeries, DataSeries]:
        """
        Get data to plot.

        Parameters
        ----------
        dataplot: visualization specification
        provided_noise:
            True if numeric values for the noise level are provided in the
            measurement table

        Returns
        -----------
        measurements_to_plot,
        simulations_to_plot
        """
        measurements_to_plot = None
        simulations_to_plot = None

        if self.measurements_data is not None:
            measurements_to_plot = self.get_data_series(self.measurements_data,
                                                        MEASUREMENT,
                                                        dataplot,
                                                        provided_noise)

        if self.simulations_data is not None:
            simulations_to_plot = self.get_data_series(self.simulations_data,
                                                       SIMULATION,
                                                       dataplot,
                                                       provided_noise)
        return measurements_to_plot, simulations_to_plot


class VisSpecParser:
    """
    Parser of visualization specification provided by user either in the form
    of Visualization table or as a list of lists with datasets ids or
    observable ids or condition ids. Figure instance is created containing
    information regarding how data should be visualized. In addition to the
    Figure instance, a DataProvider instance is created that will be
    responsible for the data selection and manipulation.
    """
    def __init__(
            self,
            conditions_data: Union[str, Path, pd.DataFrame],
            exp_data: Optional[Union[str, Path, pd.DataFrame]] = None,
            sim_data: Optional[Union[str, Path, pd.DataFrame]] = None,
    ):
        if isinstance(conditions_data, (str, Path)):
            conditions_data = conditions.get_condition_df(conditions_data)

        # import from file in case experimental data is provided in file
        if isinstance(exp_data, (str, Path)):
            exp_data = measurements.get_measurement_df(exp_data)

        if isinstance(sim_data, (str, Path)):
            sim_data = core.get_simulation_df(sim_data)

        if exp_data is None and sim_data is None:
            raise TypeError('Not enough arguments. Either measurements_data '
                            'or simulations_data should be provided.')

        self.conditions_data = conditions_data
        self.measurements_data = exp_data
        self.simulations_data = sim_data

    @classmethod
    def from_problem(cls, petab_problem: Problem, sim_data):
        return cls(petab_problem.condition_df,
                   petab_problem.measurement_df,
                   sim_data)

    @property
    def _data_df(self):
        return self.measurements_data if self.measurements_data is not \
            None else self.simulations_data

    @staticmethod
    def create_subplot(
            plot_id: str,
            subplot_vis_spec: pd.DataFrame
    ) -> Subplot:
        """
        Create subplot.

        Parameters
        ----------
        plot_id:
            Plot id.
        subplot_vis_spec:
            A visualization specification DataFrame that contains specification
            for the subplot and corresponding dataplots.

        Returns
        -------

                Subplot
        """
        subplot_columns = [col for col in subplot_vis_spec.columns if col in
                           VISUALIZATION_DF_SUBPLOT_LEVEL_COLS]
        subplot = Subplot.from_df(plot_id,
                                  subplot_vis_spec.loc[:, subplot_columns])

        dataplot_cols = [col for col in subplot_vis_spec.columns if col in
                         VISUALIZATION_DF_SINGLE_PLOT_LEVEL_COLS]
        dataplot_spec = subplot_vis_spec.loc[:, dataplot_cols]

        for _, row in dataplot_spec.iterrows():
            data_plot = DataPlot.from_df(row)
            subplot.add_dataplot(data_plot)

        return subplot

    def parse_from_vis_spec(
            self,
            vis_spec: Optional[Union[str, Path, pd.DataFrame]],
    ) -> Tuple[Figure, DataProvider]:
        """
        Get visualization settings from a visualization specification.

        Parameters
        ----------
        vis_spec:
            Visualization specification DataFrame in the PEtab format
            or a path to a visualization file.

        Returns
        -------

        A figure template with visualization settings and a data provider
        """
        # import visualization specification, if file was specified
        if isinstance(vis_spec, (str, Path)):
            vis_spec = core.get_visualization_df(vis_spec)

        if DATASET_ID not in vis_spec.columns:
            self._add_dataset_id_col()
            vis_spec = self._expand_vis_spec_settings(vis_spec)
        else:
            if self.measurements_data is not None \
                    and DATASET_ID not in self.measurements_data:
                raise ValueError(f"grouping by datasetId was requested, but "
                                 f"{DATASET_ID} column is missing from "
                                 f"measurement table")
            if self.simulations_data is not None \
                    and DATASET_ID not in self.simulations_data:
                raise ValueError(f"grouping by datasetId was requested, but "
                                 f"{DATASET_ID} column is missing from "
                                 f"simulation table")

        figure = Figure()

        # get unique plotIDs preserving the order from the original vis spec
        _, idx = np.unique(vis_spec[PLOT_ID], return_index=True)
        plot_ids = vis_spec[PLOT_ID].iloc[np.sort(idx)]

        # loop over unique plotIds
        for plot_id in plot_ids:
            # get indices for specific plotId
            ind_plot = (vis_spec[PLOT_ID] == plot_id)

            subplot = self.create_subplot(plot_id, vis_spec[ind_plot])
            figure.add_subplot(subplot)

        return figure, DataProvider(self.conditions_data,
                                    self.measurements_data,
                                    self.simulations_data)

    def parse_from_id_list(self,
                           ids_per_plot: Optional[List[IdsList]] = None,
                           group_by: str = 'observable',
                           plotted_noise: Optional[str] = MEAN_AND_SD
                           ) -> Tuple[Figure, DataProvider]:
        """
        Get visualization settings from a list of ids and a grouping parameter.

        Parameters
        ----------
        ids_per_plot:
            A list of lists. Each sublist corresponds to a plot, each subplot
            contains the Ids of datasets or observables or simulation
            conditions for this plot.
            e.g.

            ::

                dataset_ids_per_plot = [['dataset_1', 'dataset_2'],
                                        ['dataset_1', 'dataset_4',
                                         'dataset_5']]

            or

            ::

                cond_id_list = [['model1_data1'],
                                ['model1_data2', 'model1_data3'],
                                ['model1_data4', 'model1_data5'],
                                ['model1_data6']].

        group_by:
            Grouping type. Possible values: 'dataset', 'observable',
            'simulation'.
        plotted_noise:
            String indicating how noise should be visualized:
            ['MeanAndSD' (default), 'MeanAndSEM', 'replicate', 'provided'].

        Returns
        -------
        A figure template with visualization settings and a data provider

        """

        if ids_per_plot is None:
            # this is the default case. If no grouping is specified,
            # all observables are plotted. One observable per plot.
            unique_obs_list = self._data_df[OBSERVABLE_ID].unique()
            ids_per_plot = [[obs_id] for obs_id in unique_obs_list]

        if group_by == 'dataset' and DATASET_ID not in self._data_df:
            raise ValueError(f"grouping by datasetId was requested, but "
                             f"{DATASET_ID} column is missing from data table")

        if group_by != 'dataset':
            # datasetId_list will be created (possibly overwriting previous
            # list - only in the local variable, not in the tsv-file)
            self._add_dataset_id_col()

        columns_dict = self._get_vis_spec_dependent_columns_dict(
            group_by, ids_per_plot)

        columns_dict[PLOT_TYPE_DATA] = [plotted_noise]*len(
            columns_dict[DATASET_ID])

        vis_spec_df = pd.DataFrame(columns_dict)

        return self.parse_from_vis_spec(vis_spec_df)

    def _add_dataset_id_col(self) -> None:
        """
        Add dataset_id column to the measurement table and simulations table
        (possibly overwrite).
        """

        if self.measurements_data is not None:
            if DATASET_ID in self.measurements_data.columns:
                self.measurements_data = self.measurements_data.drop(
                    DATASET_ID, axis=1)
            self.measurements_data.insert(
                loc=self.measurements_data.columns.size,
                column=DATASET_ID,
                value=generate_dataset_id_col(self.measurements_data))

        if self.simulations_data is not None:
            if DATASET_ID in self.simulations_data.columns:
                self.simulations_data = self.simulations_data.drop(DATASET_ID,
                                                                   axis=1)
            self.simulations_data.insert(
                loc=self.simulations_data.columns.size,
                column=DATASET_ID,
                value=generate_dataset_id_col(self.simulations_data))

    def _get_vis_spec_dependent_columns_dict(
        self,
        group_by: str,
        id_list: Optional[List[IdsList]] = None
    ) -> Dict:
        """
        Helper method for creating values for columns PLOT_ID, DATASET_ID,
        LEGEND_ENTRY, Y_VALUES for visualization specification file.

        Parameters
        ----------
        group_by:
            Grouping type.
            Possible values: 'dataset', 'observable', 'simulation'.
        id_list:
            Grouping list. Each sublist corresponds to a subplot and
            contains the Ids of datasets or observables or simulation
            conditions for this subplot.

        Returns
        -------
        A dictionary with values for columns PLOT_ID, DATASET_ID, \
        LEGEND_ENTRY, Y_VALUES for visualization specification.
        """

        if group_by != 'dataset':
            dataset_id_list = create_dataset_id_list_new(self._data_df,
                                                         group_by, id_list)
        else:
            dataset_id_list = id_list

        dataset_id_column = [i_dataset for sublist in dataset_id_list
                             for i_dataset in sublist]

        dataset_label_column = [self._create_legend(i_dataset) for sublist in
                                dataset_id_list for i_dataset in sublist]

        # such dataset ids were generated that each dataset_id always
        # corresponds to one observable
        yvalues_column = [self._data_df.loc[self._data_df[DATASET_ID] ==
                                            dataset_id, OBSERVABLE_ID].iloc[0]
                          for sublist in dataset_id_list for dataset_id in
                          sublist]

        # get number of plots and create plotId-lists
        plot_id_column = ['plot%s' % str(ind + 1) for ind, inner_list in
                          enumerate(dataset_id_list) for _ in inner_list]

        return {
            PLOT_ID: plot_id_column,
            DATASET_ID: dataset_id_column,
            LEGEND_ENTRY: dataset_label_column,
            Y_VALUES: yvalues_column
        }

    def _create_legend(self, dataset_id: str) -> str:
        """
        Create a legend for the dataset ids.

        Parameters
        ----------
        dataset_id:
            Dataset id.

        Returns
        -------
        A legend.
        """
        # relies on the fact that dataset ids were created based on cond_ids
        # and obs_ids. Therefore, in the following query all pairs will be
        # the same
        cond_id, obs_id = self._data_df[self._data_df[DATASET_ID] ==
                                        dataset_id][[SIMULATION_CONDITION_ID,
                                                     OBSERVABLE_ID]].iloc[0, :]
        tmp = self.conditions_data.loc[cond_id]
        if CONDITION_NAME not in tmp.index or \
                pd.isna(tmp[CONDITION_NAME]):
            cond_name = cond_id
        else:
            cond_name = tmp[CONDITION_NAME]
        return f'{cond_name} - {obs_id}'

    def _expand_vis_spec_settings(self, vis_spec: pd.DataFrame):
        """
        Expand visualization specification for the case when DATASET_ID is not
        in vis_spec.columns.

        Parameters
        -------
        vis_spec:
            Visualization specification DataFrame in the PEtab format
            or a path to a visualization file.

        Returns
        -------
        A visualization specification DataFrame.
        """
        if DATASET_ID in vis_spec.columns:
            raise ValueError(f"visualization specification expansion is "
                             f"unnecessary if column {DATASET_ID} is present")

        if vis_spec.empty:
            # in case of empty spec all measurements corresponding to each
            # observable will be plotted on a separate subplot
            observable_ids = self._data_df[OBSERVABLE_ID].unique()

            vis_spec_exp_rows = [
                self._vis_spec_rows_for_obs(obs_id, {PLOT_ID: f'plot{idx}'})
                for idx, obs_id in enumerate(observable_ids)
            ]
            return pd.concat(vis_spec_exp_rows, ignore_index=True)

        vis_spec_exp_rows = []
        for _, row in vis_spec.iterrows():
            if Y_VALUES in row:
                vis_spec_exp_rows.append(
                    self._vis_spec_rows_for_obs(row[Y_VALUES], row.to_dict())
                )
            else:
                observable_ids = self._data_df[OBSERVABLE_ID].unique()

                for obs_id in observable_ids:
                    vis_spec_exp_rows.append(
                        self._vis_spec_rows_for_obs(obs_id, row.to_dict())
                    )
        return pd.concat(vis_spec_exp_rows, ignore_index=True)

    def _vis_spec_rows_for_obs(self, obs_id: str, settings: dict
                               ) -> pd.DataFrame:
        """
        Create vis_spec for one observable.

        For each dataset_id corresponding to the observable with the specified
        id create a vis_spec entry with provided settings.

        Parameters
        ----------
        obs_id:
            Observable ID.
        settings:
            Additional visualization settings. For each key that is a
            valid visualization specification column name, the setting
            will be added to the resulting visualization specification.

        Returns
        -------
        A visualization specification DataFrame.
        """
        columns_to_expand = [PLOT_ID, PLOT_NAME, PLOT_TYPE_SIMULATION,
                             PLOT_TYPE_DATA, X_VALUES, X_OFFSET, X_LABEL,
                             X_SCALE, Y_OFFSET, Y_LABEL, Y_SCALE,
                             LEGEND_ENTRY]

        dataset_ids = self._data_df[
            self._data_df[OBSERVABLE_ID] ==
            obs_id][DATASET_ID].unique()
        n_rows = len(dataset_ids)
        columns_dict = {DATASET_ID: dataset_ids,
                        Y_VALUES: [obs_id] * n_rows}

        for column in settings:
            if column in columns_to_expand:
                columns_dict[column] = [settings[column]] * n_rows

        if LEGEND_ENTRY not in columns_dict:
            columns_dict[LEGEND_ENTRY] = \
                [self._create_legend(dataset_id) for dataset_id
                 in columns_dict[DATASET_ID]]
        return pd.DataFrame(columns_dict)
