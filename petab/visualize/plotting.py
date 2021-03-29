import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple, Union, TypedDict

from .helper_functions import (generate_dataset_id_col,
                               create_dataset_id_list_new,
                               expand_vis_spec_settings)
from .. import measurements, core, conditions
from ..problem import Problem
from ..C import *
from numbers import Number
import warnings

# for typehints
IdsList = List[str]
NumList = List[int]


# also for type hints
# TODO: split into dataplot and subplot level dicts?
class VisDict(TypedDict):
    PLOT_NAME: str
    PLOT_TYPE_SIMULATION: str
    PLOT_TYPE_DATA: str
    X_VALUES: str
    X_OFFSET: List[Number]
    X_LABEL: str
    X_SCALE: str
    Y_VALUES: List[str]
    Y_OFFSET: List[Number]
    Y_LABEL: str
    Y_SCALE: str
    LEGEND_ENTRY: List[Number]
    DATASET_ID: List[str]


class VisualizationSpec:
    def __init__(self,
                 plot_id: str,
                 plot_settings: VisDict,
                 fig_id: str = 'fig0'
                 ):
        """
        visualization specification for one plot

        Parameters
        ----------
        plot_id:
        plot_settings:
        fig_id:
        """
        # vis spec file + additioal styles/settings ?
        self.figureId = fig_id
        setattr(self, PLOT_ID, plot_id)
        for key, val in plot_settings.items():
            setattr(self, key, val)
        if PLOT_NAME not in vars(self):
            setattr(self, PLOT_NAME, getattr(self, PLOT_ID))
        if PLOT_TYPE_SIMULATION not in vars(self):
            setattr(self, PLOT_TYPE_SIMULATION, LINE_PLOT)
        if PLOT_TYPE_DATA not in vars(self):
            setattr(self, PLOT_TYPE_DATA, MEAN_AND_SD)
        # TODO datasetId optional so default should be created
        if X_VALUES not in vars(self):
            setattr(self, X_VALUES, TIME)
        if X_OFFSET not in vars(self):
            setattr(self, X_OFFSET, 0)
        if X_LABEL not in vars(self):
            setattr(self, X_LABEL, getattr(self, X_VALUES))

        if X_SCALE not in vars(self):
            setattr(self, X_SCALE, LIN)
        # TODO yValues optional but should be created one level above
        # TODO in docs: yValues list of observables, how default label?
        if Y_LABEL not in vars(self):
            setattr(self, Y_LABEL, 'values')
        if Y_OFFSET not in vars(self):
            setattr(self, Y_OFFSET, 0.)
        if LEGEND_ENTRY not in vars(self):
            setattr(self, LEGEND_ENTRY, getattr(self, DATASET_ID))

    @staticmethod
    def from_df(vis_spec_df: Union[pd.DataFrame, str]) -> \
            List['VisualizationSpec']:
        # check if file path or pd.DataFrame is passed
        if isinstance(vis_spec_df, str):
            vis_spec_df = pd.read_csv(vis_spec_df, sep='\t', index_col=PLOT_ID)
        elif vis_spec_df.index.name != PLOT_ID:
            vis_spec_df.set_index(PLOT_ID, inplace=True)
        uni_plot_ids = vis_spec_df.index.unique().to_list()
        vis_spec_list = []
        # create a VisualizationSpec object for each PlotId
        for plot_id in uni_plot_ids:
            vis_spec_dict = {}
            for col in vis_spec_df:
                print(plot_id, col)
                entry = vis_spec_df.loc[plot_id, col]
                if col in VISUALIZATION_DF_SUBPLOT_LEVEL_COLS:
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
                                     + ', '.join(PLOT_TYPES_DATA))
                # append new entry to dict
                vis_spec_dict[col] = entry
            vis_spec_list.append(VisualizationSpec(plot_id, vis_spec_dict))
        return vis_spec_list


class DataSeries:
    """
    data for one individual line
    """
    def __init__(self, conditions_: Optional[Union[np.ndarray, pd.Series]],
                 measurements_to_plot: Optional[pd.DataFrame] = None,
                 simulations_to_plot: Optional[pd.DataFrame] = None):

        self.conditions = conditions_
        if measurements_to_plot is None and simulations_to_plot is None:
            raise TypeError('Not enough arguments. Either measurements_to_plot'
                            ' or simulations_to_plot should be provided.')
        self.measurements_to_plot = measurements_to_plot
        self.simulations_to_plot = simulations_to_plot

    def add_x_offset(self, offset):
        if self.conditions is not None:
            self.conditions += offset

    def add_y_offset(self, offset):
        if self.measurements_to_plot is not None:
            self.measurements_to_plot['mean'] = \
                self.measurements_to_plot['mean'] + offset
            self.measurements_to_plot['repl'] = \
                self.measurements_to_plot['repl'] + offset

        if self.simulations_to_plot is not None:
            self.simulations_to_plot = [x + offset for x in
                                        self.simulations_to_plot]

    def add_offsets(self, x_offset=0, y_offset=0):
        self.add_x_offset(x_offset)
        self.add_y_offset(y_offset)


class DataPlot:
    def __init__(self,
                 plot_settings: VisDict):
        """
        Visualization specification of a plot of one data series, e.g. for
        an individual line on a subplot

        Parameters
        ----------
        plot_settings: plot spec for one dataplot
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


class Subplot:
    def __init__(self,
                 plot_id: str,
                 plot_settings: VisDict,
                 dataplots: Optional[List[DataPlot]] = None):
        """
        Visualization specification of a subplot

        Parameters
        ----------
        plot_settings: plot spec for a subplot
                       (only VISUALIZATION_DF_SUBPLOT_LEVEL_COLS)
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
                                     + ', '.join(PLOT_TYPES_DATA))

                # append new entry to dict
                vis_spec_dict[col] = entry
            else:
                warnings.warn(f'Column {col} cannot be used to specify subplot'
                              f', only settings from the following columns can'
                              f' be used:'
                              + ', '.join(VISUALIZATION_DF_SUBPLOT_LEVEL_COLS))
        return cls(plot_id, vis_spec_dict, dataplots)

    def add_dataplot(self, dataplot: DataPlot):
        self.data_plots.append(dataplot)


class Figure:
    def __init__(self, subplots: Optional[List[Subplot]] = None,
                 size: Tuple = (20, 15),
                 title: Optional[Tuple] = None):
        """
        Visualization specification of a figure. Contains information
        regarding how data should be visualized.

        Parameters
        ----------
        subplots
        size
        title
        """

        # TODO: Isensee meas table doesn't correspond to documentation
        self.size = size
        self.title = title
        self.subplots = subplots if subplots is not None else []

        # TODO: Should we put in the documentation which combination of fields
        #  must be unique? is there such check

    @property
    def num_subplots(self) -> int:
        return len(self.subplots)

    def add_subplot(self, subplot: Subplot):
        self.subplots.append(subplot)


class DataProvider:
    """
    Handles data selection
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
        constructs an index for subsetting of the dataframe according to what
        is specified in plot_spec.

        Parameters:
            df:
                pandas data frame to subset, can be from measurement file or
                simulation file
            plot_spec:
                visualization spec from the visualization file

        Returns:
            index:
                Boolean series that can be used for subsetting of the passed
                dataframe
        """
        subset = (
            (df[DATASET_ID] == dataset_id)
        )
        if getattr(plot_spec, Y_VALUES) == '':
            if len(df.loc[subset, OBSERVABLE_ID].unique()) > 1:
                ValueError(
                    f'{Y_VALUES} must be specified in visualization table if '
                    f'multiple different observables are available.'
                )
        else:
            subset &= (df[OBSERVABLE_ID] == getattr(plot_spec, Y_VALUES))
        return subset

    def _get_uni_condition_id(self, data_df: pd.DataFrame, dataplot: DataPlot
                              ) -> Tuple[np.ndarray, str, pd.Series]:
        """
        get independant variable values

        Parameters
        ----------
        data_df:
            pandas data frame to subset, can be from measurement file or
            simulation file
        dataplot:

        Returns
        -------
        col_name_unique:
            the name of the column from Measurement table, whose entries
            should be unique (depends on entry in column xValues of
            visualization specification)
            possible values: TIME or SIMULATION_CONDITION_ID
        uni_condition_id:
            time points
            or
            contains all unique condition IDs which should be
            plotted in one figure (can be found in measurementData file,
            column simulationConditionId)
        conditions_:
            possible values: time points, None, vales of independent variable
            for each condition_id in uni_condition_id

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
            # TODO: not described in docs?
            conditions_ = None
        else:
            # indep_var = parameterOrStateId case ?
            # extract conditions (plot input) from condition file
            ind_cond = self.conditions_data.index.isin(uni_condition_id)
            conditions_ = self.conditions_data[ind_cond][indep_var]

        return uni_condition_id, col_name_unique, conditions_

    def get_data_to_plot(self, dataplot: DataPlot, provided_noise: bool
                         ) -> DataSeries:
        """
        get data to plot

        Parameters
        ----------
        dataplot:
        provided_noise:
            True if if numeric values for the noise level are provided in the
            measurement table
        """

        measurements_to_plot = None
        simulations_to_plot = None

        # handle one "line" of plot

        dataset_id = getattr(dataplot, DATASET_ID)

        if self.measurements_data is not None:
            uni_condition_id, col_name_unique, conditions_ = \
                self._get_uni_condition_id(self.measurements_data, dataplot)
        else:
            uni_condition_id, col_name_unique, conditions_ = \
                self._get_uni_condition_id(self.simulations_data, dataplot)

        if self.measurements_data is not None:
            # define index to reduce exp_data to data linked to datasetId

            # get measurements_df subset selected based on provided dataset_id
            # and observable_ids
            single_m_data = self.measurements_data[self._matches_plot_spec(
                self.measurements_data, dataplot, dataset_id)]

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
                    MEASUREMENT
                ]

                # TODO: all this rather inside DataSeries?
                # process the data
                measurements_to_plot.at[var_cond_id, 'mean'] = np.mean(
                    data_measurements)
                measurements_to_plot.at[var_cond_id, 'sd'] = np.std(
                    data_measurements)

                if provided_noise & sum(subset):
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

        if self.simulations_data is not None:
            simulations_to_plot = []

            single_s_data = self.simulations_data[self._matches_plot_spec(
                self.simulations_data, dataplot, dataset_id)]

            for var_cond_id in uni_condition_id:
                simulation_measurements = single_s_data.loc[
                    single_s_data[col_name_unique] == var_cond_id,
                    SIMULATION
                ]

                simulations_to_plot.append(np.mean(
                    simulation_measurements
                ))

        data_series = DataSeries(conditions_, measurements_to_plot,
                                 simulations_to_plot)
        data_series.add_offsets(dataplot.xOffset, dataplot.yOffset)

        return data_series


class VisSpecParser:
    """
    Parser of visualization specification provided by user either in the form
    of Visualization table or as a list of lists with datasets ids or
    observable ids or condition ids. Figure instance is created containing
    information regarding how data should be visualized. In addition to the
    Figure instance, a DataProvider instance is created that will be
    responsible for the data selection and manipulation.

    """
    def __init__(self,
                 conditions_data: Union[str, pd.DataFrame],
                 exp_data: Optional[Union[str, pd.DataFrame]] = None,
                 sim_data: Optional[Union[str, pd.DataFrame]] = None,
                 ):
        if isinstance(conditions_data, str):
            conditions_data = conditions.get_condition_df(conditions_data)

        # import from file in case experimental data is provided in file
        if isinstance(exp_data, str):
            exp_data = measurements.get_measurement_df(exp_data)

        if isinstance(sim_data, str):
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
    def create_subplot(plot_id,
                       subplot_vis_spec: pd.DataFrame) -> Subplot:
        """
        create subplot

        Parameters
        ----------
        plot_id:
        subplot_vis_spec:
            visualization specification DataFrame that contains specification
            for the subplot and corresponding dataplots

        Returns
        -------

        subplot
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

    def parse_from_vis_spec(self,
                            vis_spec: Optional[Union[str, pd.DataFrame]],
                            ) -> Tuple[Figure, DataProvider]:

        # import visualization specification, if file was specified
        if isinstance(vis_spec, str):
            vis_spec = core.get_visualization_df(vis_spec)

        if DATASET_ID not in vis_spec.columns:
            self.add_dataset_id_col()
            if Y_VALUES in vis_spec.columns:
                plot_id_list = np.unique(vis_spec[PLOT_ID])

                observable_id_list = [vis_spec[vis_spec[PLOT_ID] ==
                                               plot_id].loc[:, Y_VALUES].values
                                      for plot_id in plot_id_list]

                columns_dict = self.get_vis_spec_dependent_columns_dict(
                    'observable', observable_id_list)

            else:
                # PLOT_ID is there, but NOT DATASET_ID and not Y_VALUES,
                # but potentially some settings.
                # TODO: multiple plotids with diff settings

                unique_obs_list = self._data_df[OBSERVABLE_ID].unique()
                observable_id_list = [[obs_id] for obs_id in unique_obs_list]

                columns_dict = self.get_vis_spec_dependent_columns_dict(
                    'observable', observable_id_list)

            vis_spec = expand_vis_spec_settings(vis_spec, columns_dict)
        else:
            if DATASET_ID not in self._data_df:
                raise ValueError(f"grouping by datasetId was requested, but "
                                 f"{DATASET_ID} column is missing from data "
                                 f"table")

        figure = Figure()

        # get unique plotIDs
        plot_ids = np.unique(vis_spec[PLOT_ID])

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
                           plotted_noise: Optional[str] = MEAN_AND_SD):
        """

        Parameters
        ----------
        ids_per_plot:
            e.g. dataset_ids_per_plot = [['dataset_1', 'dataset_2'],
                                         ['dataset_1', 'dataset_4',
                                          'dataset_5']]

            or cond_id_list = [['model1_data1'],
                               ['model1_data2', 'model1_data3'],
                               ['model1_data4', 'model1_data5'],
                               ['model1_data6']]
        group_by
            ['dataset', 'observable', 'simulation']
            # TODO: why simulation btw?
        plotted_noise
            String indicating how noise should be visualized:
            ['MeanAndSD' (default), 'MeanAndSEM', 'replicate', 'provided']
        Returns
        -------

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
            self.add_dataset_id_col()

        columns_dict = self.get_vis_spec_dependent_columns_dict(
            group_by, ids_per_plot)

        columns_dict[PLOT_TYPE_DATA] = [plotted_noise]*len(
            columns_dict[DATASET_ID])

        vis_spec_df = pd.DataFrame(columns_dict)

        return self.parse_from_vis_spec(vis_spec_df)

    def add_dataset_id_col(self):
        """
        add dataset_id column to the measurement table and simulations table
        (possibly overwrite)
        Returns
        -------

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

    def get_vis_spec_dependent_columns_dict(
        self,
        group_by: str,
        id_list: Optional[List[IdsList]] = None
    ) -> Dict:
        """
        Helper function for creating values for columns PLOT_ID, DATASET_ID,
        LEGEND_ENTRY, Y_VALUES for visualization specification file.

        Returns:
            A tuple of experimental DataFrame and a dictionary with values for
            columns PLOT_ID, DATASET_ID, LEGEND_ENTRY, Y_VALUES for
            visualization specification file.
        """

        legend_dict = self.create_legend_dict(self._data_df)
        # yvalues_dict = {}
        # TODO: generate legends and yvalues

        if group_by != 'dataset':
            dataset_id_list = create_dataset_id_list_new(self._data_df,
                                                         group_by, id_list)
        else:
            dataset_id_list = id_list

        dataset_id_column = [i_dataset for sublist in dataset_id_list
                             for i_dataset in sublist]

        if group_by != 'dataset':
            dataset_label_column = [legend_dict[i_dataset] for sublist in
                                    dataset_id_list for i_dataset in sublist]
            # yvalues_column = [yvalues_dict[i_dataset] for sublist in
            #                   dataset_id_list for i_dataset in sublist]
        else:
            dataset_label_column = dataset_id_column
            # yvalues_column = ['']*len(dataset_id_column)

        # such datasetids were generated that each dataset_id always
        # corresponds to one observable
        yvalues_column = [self._data_df.loc[self._data_df[DATASET_ID] ==
                                            dataset_id, OBSERVABLE_ID].iloc[0]
                          for sublist in dataset_id_list for dataset_id in
                          sublist]

        # TODO: is it really needed?
        # get number of plots and create plotId-lists
        # if group_by == 'observable':
        #     obs_uni = list(np.unique(exp_data[OBSERVABLE_ID]))
        #     # copy of dataset ids, for later replacing with plot ids
        #     plot_id_column = dataset_id_column.copy()
        #     for i_obs in range(0, len(obs_uni)):
        #         # get dataset_ids which include observable name
        #         matching = [s for s in dataset_id_column if
        #                     obs_uni[i_obs] in s]
        #         # replace the dataset ids with plot id with grouping of
        #         # observables
        #         for m_i in matching:
        #             plot_id_column = [sub.replace(m_i, 'plot%s' %
        #                                           str(i_obs + 1))
        #                               for sub in plot_id_column]
        # else:
        # get number of plots and create plotId-lists
        plot_id_column = ['plot%s' % str(ind + 1) for ind, inner_list in
                          enumerate(dataset_id_list) for _ in inner_list]

        columns_dict = {PLOT_ID: plot_id_column,
                        DATASET_ID: dataset_id_column,
                        LEGEND_ENTRY: dataset_label_column,
                        Y_VALUES: yvalues_column}
        return columns_dict

    def create_legend_dict(self, data_df: pd.DataFrame):
        legend_dict = {}

        for _, row in data_df.iterrows():
            cond_id = row[SIMULATION_CONDITION_ID]
            obs_id = row[OBSERVABLE_ID]
            dataset_id = row[DATASET_ID]

            # create nicer legend entries from condition names instead of IDs
            if dataset_id not in legend_dict.keys():
                tmp = self.conditions_data.loc[cond_id]
                if CONDITION_NAME not in tmp.index or \
                        pd.isna(tmp[CONDITION_NAME]):
                    cond_name = cond_id
                else:
                    cond_name = tmp[CONDITION_NAME]
                legend_dict[dataset_id] = cond_name + ' - ' + \
                    obs_id

        return legend_dict
