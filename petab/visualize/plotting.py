import numpy as np
import pandas as pd

from typing import List, Optional, Tuple, Union, TypedDict

from .helper_functions import (generate_dataset_id_col,
                               create_dataset_id_list_new,
                               matches_plot_spec_new)
from .. import problem, measurements, core, conditions
from ..problem import Problem
from ..C import *
from collections.abc import Sequence
from numbers import Number
import warnings

# for typehints
IdsList = List[str]
NumList = List[int]


# also for type hints
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
                    entry=entry[0]

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

        Parameters
        ----------
        plot_settings: plot spec for one dataplot
                       (only VISUALIZATION_DF_SINGLE_PLOT_LEVEL_COLS)
        """

        for key, val in plot_settings.items():
            if not isinstance(val, list):
                setattr(self, key, val)

        # TODO datasetId mandatory here

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

    @staticmethod
    def from_df(plot_spec: pd.DataFrame):

        vis_spec_dict = plot_spec.to_dict()

        return DataPlot(vis_spec_dict)


class Subplot:
    def __init__(self,
                 plot_id: str,
                 plot_settings: VisDict,
                 dataplots: List[DataPlot]):
        """

        Parameters
        ----------
        plot_settings: plot spec for a subplot
                       (only VISUALIZATION_DF_SUBPLOT_LEVEL_COLS)
        """
        # parameters of a specific subplot

        setattr(self, PLOT_ID, plot_id)
        for key, val in plot_settings.items():
            if not isinstance(val, list):
                setattr(self, key, val)

        if PLOT_NAME not in vars(self):
            setattr(self, PLOT_NAME, getattr(self, PLOT_ID))
        if PLOT_TYPE_SIMULATION not in vars(self):
            setattr(self, PLOT_TYPE_SIMULATION, LINE_PLOT)
        if PLOT_TYPE_DATA not in vars(self):
            setattr(self, PLOT_TYPE_DATA, MEAN_AND_SD)
        if X_LABEL not in vars(self):
            setattr(self, X_LABEL, TIME)  #getattr(self, X_VALUES)
        if X_SCALE not in vars(self):
            setattr(self, X_SCALE, LIN)
        if Y_LABEL not in vars(self):
            setattr(self, Y_LABEL, 'values')
        if Y_SCALE not in vars(self):
            setattr(self, Y_SCALE, LIN)

        self.data_plots = dataplots

    @staticmethod
    def from_df(plot_id: str, vis_spec: pd.DataFrame,
                dataplots: List[DataPlot]):

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
                raise
        return Subplot(plot_id, vis_spec_dict, dataplots)

    def add_dataplot(self, dataplot: DataPlot):
        self.data_plots.append(dataplot)


class Figure:
    def __init__(self, subplots: List[Subplot],
                 size: Tuple = (20, 10),
                 title: Optional[Tuple] = None):
        """

        Parameters
        ----------
        subplots
        size
        title
        """

        # TODO: Isensee meas table doesn't correspond to documentation
        self.size = size
        self.title = title
        self.subplots = subplots

        # TODO: Should we put in the documentation which combination of fields
        #  must be unique? is there such  check
        # TODO: vis_spec_df without datasetId was provided

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
                 exp_conditions: Union[str, pd.DataFrame],
                 measurements_data: Union[str, pd.DataFrame],
                 simulations_data: Optional[Union[str, pd.DataFrame]] = None):
        self.conditions_data = exp_conditions

        if measurements_data is None and simulations_data is None:
            raise TypeError('Not enough arguments. Either measurements_data '
                            'or simulations_data should be provided.')
        self.measurements_data = measurements_data
        self.simulations_data = simulations_data
        # validation of dfs?
        # extending
        pass

    # def check_datarequest_consistency(self):
    # TODO: not needed anymore?
    #     # check if data request is meaningful
    #     # check_vis_spec_consistency functionality
    #     pass

    def get_uni_condition_id(self, df: pd.DataFrame, dataplot: DataPlot):
        """

        Parameters
        ----------
        df:
            pandas data frame to subset, can be from measurement file or
            simulation file
        dataplot:

        Returns
        -------

        """

        indep_var = getattr(dataplot, X_VALUES)

        dataset_id = getattr(dataplot, DATASET_ID)

        # TODO: move matches_plot_spec to this class?
        single_m_data = df[matches_plot_spec_new(
            df, dataplot, dataset_id)]

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
            conditions = uni_condition_id
        elif indep_var == 'condition':
            # TODO: not described in docs?
            conditions = None
        else:
            # parameterOrStateId case ?
            # extract conditions (plot input) from condition file
            ind_cond = self.conditions_data.index.isin(uni_condition_id)
            conditions = self.conditions_data[ind_cond][indep_var]

        return uni_condition_id, col_name_unique, conditions

    def get_data_to_plot(self, dataplot: DataPlot, plotTypeData: str
                         ) -> DataSeries:
        """

        Parameters
        ----------
        dataplot:
        plotTypeData:
        """

        measurements_to_plot = None
        simulations_to_plot = None

        # handle one "line" of plot

        dataset_id = getattr(dataplot, DATASET_ID)

        if self.measurements_data is not None:
            uni_condition_id, col_name_unique, conditions = \
                self.get_uni_condition_id(self.measurements_data, dataplot)
        else:
            uni_condition_id, col_name_unique, conditions = \
                self.get_uni_condition_id(self.simulations_data, dataplot)

        if self.measurements_data is not None:
            # define index to reduce exp_data to data linked to datasetId

            single_m_data = self.measurements_data[matches_plot_spec_new(
                self.measurements_data, dataplot, dataset_id)]

            # create empty dataframe for means and SDs
            measurements_to_plot = pd.DataFrame(
                columns=['mean', 'noise_model', 'sd', 'sem', 'repl'],
                index=uni_condition_id
            )

            for var_cond_id in uni_condition_id:
                # TODO: should be self.measurements ?

                subset = (single_m_data[col_name_unique] == var_cond_id)
                # if vis_spec[Y_VALUES] == '':
                #     if len(single_m_data.loc[subset, OBSERVABLE_ID].unique()) > 1:
                #         ValueError(
                #             f'{Y_VALUES} must be specified in visualization table if '
                #             f'multiple different observables are available.'
                #         )
                # else:
                #     subset &= (single_m_data[OBSERVABLE_ID] == vis_spec[Y_VALUES])

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

                # TODO: one level above?
                if (plotTypeData == PROVIDED) & sum(subset):
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
                    if isinstance(tmp_noise,
                                  Number) or tmp_noise.dtype == 'float64':
                        measurements_to_plot.at[
                            var_cond_id, 'noise_model'] = tmp_noise

                # standard error of mean
                measurements_to_plot.at[var_cond_id, 'sem'] = \
                    np.std(data_measurements) / np.sqrt(
                        len(data_measurements))

                # single replicates
                measurements_to_plot.at[var_cond_id, 'repl'] = \
                    data_measurements

        if self.simulations_data is not None:
            simulations_to_plot = []
            for var_cond_id in uni_condition_id:
                # TODO: put == var_cond_id back in matches_plot_spec_new?
                simulation_measurements = self.simulations_data.loc[
                    matches_plot_spec_new(self.simulations_data,
                                          dataplot, dataset_id) &
                    (self.simulations_data[col_name_unique] == var_cond_id),
                    SIMULATION
                ]

                simulations_to_plot.append(np.mean(
                    simulation_measurements
                ))

        data_series = DataSeries(conditions, measurements_to_plot,
                                 simulations_to_plot)
        data_series.add_offsets(dataplot.xOffset, dataplot.yOffset)

        return data_series


class VisSpecParser:
    """


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

        self.conditions_data = conditions_data
        self.measurements_data = exp_data
        self.simulations_data = sim_data

    def create_subplot(self,
                       plot_id,
                       subplot_vis_spec: pd.DataFrame) -> Subplot:
        """

        Parameters
        ----------
        plot_id:
        subplot_vis_spec:

        Returns
        -------
        """

        subplot_columns = [col for col in subplot_vis_spec.columns if col in
                           VISUALIZATION_DF_SUBPLOT_LEVEL_COLS]
        subplot = Subplot.from_df(plot_id,
                                  subplot_vis_spec.loc[:, subplot_columns], [])

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

        # TODO: vis_spec_df without datasetId was provided

        # import visualization specification, if file was specified
        if isinstance(vis_spec, str):
            vis_spec = core.get_visualization_df(vis_spec)

        figure = Figure([])

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

    def parse_from_dataset_ids(self,
                               dataset_ids_per_plot: Union[List[IdsList],
                                                           List[NumList]],
                               plotted_noise: Optional[str] = MEAN_AND_SD
                               ) -> Tuple[Figure, DataProvider]:
        """

        Parameters
        ----------
        dataset_ids_per_plot:
            e.g. dataset_ids_per_plot = [['dataset_1', 'dataset_2'],
                                         ['dataset_1', 'dataset_4',
                                          'dataset_5']]
        plotted_noise:
            String indicating how noise should be visualized:
            ['MeanAndSD' (default), 'MeanAndSEM', 'replicate', 'provided']

        Returns
        -------
        """

        group_by = 'dataset'
        dataset_id_column = [i_dataset for sublist in dataset_ids_per_plot
                             for i_dataset in sublist]
        dataset_label_column = dataset_id_column
        yvalues_column = [''] * len(dataset_id_column)

        # get number of plots and create plotId-lists
        plot_id_column = ['plot%s' % str(ind + 1) for ind, inner_list in
                          enumerate(dataset_ids_per_plot) for _ in
                          inner_list]

        columns_dict = {PLOT_ID: plot_id_column,
                        DATASET_ID: dataset_id_column,
                        LEGEND_ENTRY: dataset_label_column,
                        Y_VALUES: yvalues_column,
                        PLOT_TYPE_DATA: [plotted_noise]*len(dataset_id_column)}
        vis_spec_df = pd.DataFrame(columns_dict)

        return self.parse_from_vis_spec(vis_spec_df)

    def parse_from_conditions_list(self,
                                   conditions_per_plot: Union[List[IdsList],
                                                              List[NumList]],
                                   plotted_noise: Optional[str] = MEAN_AND_SD
                                   ) -> Tuple[Figure, DataProvider]:
        """

        Parameters
        ----------
        conditions_per_plot
            e.g. cond_id_list = [['model1_data1'],
                                 ['model1_data2', 'model1_data3'],
                                 ['model1_data4', 'model1_data5'],
                                 ['model1_data6']]
        plotted_noise

        Returns
        -------

        """
        data_df = self.measurements_data if self.measurements_data is not \
            None else self.simulations_data

        if all(isinstance(x, int) for sublist in conditions_per_plot
               for x in sublist):
            # TODO: should unique_simcond_list be taken from conditons_df or
            #       measurements_df?
            unique_simcond_list = data_df[
                SIMULATION_CONDITION_ID].unique()
            conditions_id_list = [[unique_simcond_list[i_cond] for i_cond in
                                   i_cond_list] for i_cond_list in
                                  conditions_per_plot]
        elif all(isinstance(x, str) for sublist in conditions_per_plot
                 for x in sublist):
            conditions_id_list = conditions_per_plot
        else:
            raise TypeError("conditions_per_plot should be a list of lists. "
                            "Each sublist corresponds to a plot. "
                            "Elements of sublists should be either "
                            "condition ids (str) or condition numbers (int) ")

        group_by = 'simulation'  # TODO: why simulation btw?
        # datasetId_list will be created (possibly overwriting previous list
        #  - only in the local variable, not in the tsv-file)

        self.add_dataset_id_col()
        dataset_id_list = create_dataset_id_list_new(data_df,
                                                     group_by,
                                                     conditions_id_list)

        dataset_id_column = [i_dataset for sublist in dataset_id_list
                             for i_dataset in sublist]

        # TODO:
        # dataset_label_column = [legend_dict[i_dataset] for sublist in
        #                         dataset_id_list for i_dataset in sublist]
        # yvalues_column = [yvalues_dict[i_dataset] for sublist in
        #                   dataset_id_list for i_dataset in sublist]
        dataset_label_column = dataset_id_column
        yvalues_column = ['']*len(dataset_id_column)

        # get number of plots and create plotId-lists
        plot_id_column = ['plot%s' % str(ind + 1) for ind, inner_list in
                          enumerate(dataset_id_list) for _ in inner_list]

        columns_dict = {PLOT_ID: plot_id_column,
                        DATASET_ID: dataset_id_column,
                        LEGEND_ENTRY: dataset_label_column,
                        Y_VALUES: yvalues_column,
                        PLOT_TYPE_DATA: [plotted_noise]*len(dataset_id_column)}
        vis_spec_df = pd.DataFrame(columns_dict)

        return self.parse_from_vis_spec(vis_spec_df)

    def parse_from_observables_list(self,
                                    observables_per_plot: Union[List[IdsList],
                                                                List[NumList]],
                                    plotted_noise: Optional[str] = MEAN_AND_SD
                                    ) -> Tuple[Figure, DataProvider]:
        data_df = self.measurements_data if self.measurements_data is not \
            None else self.simulations_data

        if all(isinstance(x, int) for sublist in observables_per_plot
               for x in sublist):
            unique_obs_list = data_df[OBSERVABLE_ID].unique()
            observable_id_list = [[unique_obs_list[i_obs] for i_obs in
                                   i_obs_list] for i_obs_list in
                                  observables_per_plot]
        elif all(isinstance(x, str) for sublist in observables_per_plot
                 for x in sublist):
            observable_id_list = observables_per_plot
        else:
            raise TypeError("observables_per_plot should be a list of lists. "
                            "Each sublist corresponds to a plot. "
                            "Elements of sublists should be either observable "
                            "ids (str) or observable numbers (int) ")

        group_by = 'observable'
        # datasetId_list will be created (possibly overwriting previous list
        #  - only in the local variable, not in the tsv-file)

        self.add_dataset_id_col()
        dataset_id_list = create_dataset_id_list_new(data_df,
                                                     group_by,
                                                     observable_id_list)

        dataset_id_column = [i_dataset for sublist in dataset_id_list
                             for i_dataset in sublist]

        # TODO:
        # dataset_label_column = [legend_dict[i_dataset] for sublist in
        #                         dataset_id_list for i_dataset in sublist]
        # yvalues_column = [yvalues_dict[i_dataset] for sublist in
        #                   dataset_id_list for i_dataset in sublist]
        dataset_label_column = dataset_id_column
        yvalues_column = ['']*len(dataset_id_column)

        # TODO: is it really needed?
        # # get number of plots and create plotId-lists
        # obs_uni = list(exp_data[OBSERVABLE_ID].unique())
        # # copy of dataset ids, for later replacing with plot ids
        # plot_id_column = dataset_id_column.copy()
        # for i_obs in range(0, len(obs_uni)):
        #     # get dataset_ids which include observable name
        #     matching = [s for s in dataset_id_column if obs_uni[i_obs] in s]
        #     # replace the dataset ids with plot id with grouping of
        #     # observables
        #     for m_i in matching:
        #         plot_id_column = [sub.replace(m_i, 'plot%s' % str(i_obs + 1))
        #                           for sub in plot_id_column]

        # get number of plots and create plotId-lists
        plot_id_column = ['plot%s' % str(ind + 1) for ind, inner_list in
                          enumerate(dataset_id_list) for _ in inner_list]

        columns_dict = {PLOT_ID: plot_id_column,
                        DATASET_ID: dataset_id_column,
                        LEGEND_ENTRY: dataset_label_column,
                        Y_VALUES: yvalues_column,
                        PLOT_TYPE_DATA: [plotted_noise]*len(dataset_id_column)}

        vis_spec_df = pd.DataFrame(columns_dict)

        return self.parse_from_vis_spec(vis_spec_df)

    def add_dataset_id_col(self):
        # add dataset_id column to the measurement table and simulations table
        # (possibly overwrite)

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

    # def create_legend(self, dataset_id_column):
    #     legend_dict = {}
    #     yvalues_dict = {}
    #
    #     tmp_simcond = list(self.measurements_data[SIMULATION_CONDITION_ID])
    #     tmp_obs = list(self.measurements_data[OBSERVABLE_ID])
    #
    #     for dataset_id in dataset_id_column:
    #
    #         # create nicer legend entries from condition names instead of IDs
    #         if dataset_id not in legend_dict.keys():
    #             cond_id = self.measurements_data.loc[]
    #             tmp = self.conditions_data.loc[self.conditions_data.index
    #                                            == cond_id]
    #             if CONDITION_NAME not in tmp.columns or tmp[
    #                 CONDITION_NAME].isna().any():
    #                 tmp.loc[:, CONDITION_NAME] = tmp.index.tolist()
    #             legend_dict[dataset_id] = tmp[CONDITION_NAME][0] + ' - ' + \
    #                                       tmp_obs[ind]
    #             yvalues_dict[dataset_id] = tmp_obs[ind]
    #
    #     return legend_dict
