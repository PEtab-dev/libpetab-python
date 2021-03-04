import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple, Union

from ..problem import Problem
from ..C import *
from collections.abc import Sequence

# for typehints
IdsList = List[str]
NumList = List[int]


class VisualisationSpec_old:
    def __init__(self,
                 vis_spec: Union[str, pd.DataFrame],
                 exp_data: pd.DataFrame,
                 exp_conditions: pd.DataFrame
                 ):
        # vis spec file + additioal styles/settings ?
        self.create_or_update(exp_conditions,
                              exp_data,
                              vis_spec)

    def create_or_update(self,
                         exp_data: pd.DataFrame,
                         exp_conditions: pd.DataFrame,
                         vis_spec: Optional[pd.DataFrame] = None,
                         dataset_id_list: Optional[List[IdsList]] = None,
                         sim_cond_id_list: Optional[List[IdsList]] = None,
                         sim_cond_num_list: Optional[List[NumList]] = None,
                         observable_id_list: Optional[List[IdsList]] = None,
                         observable_num_list: Optional[List[NumList]] = None,
                         plotted_noise: Optional[str] = MEAN_AND_SD):
        pass


class VisualisationSpec:
    def __init__(self,
                 plot_id: str,
                 plot_settings: Dict,
                 fig_id: str = 'fig0'
                 ):
        # vis spec file + additioal styles/settings ?
        self.figureId = fig_id
        setattr(self, PLOT_ID, plot_id)
        for key,val in plot_settings.items():
            setattr(self,key,val)
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
        # TODO yValues list of observables, so how can it be label
        if Y_LABEL not in vars(self):
            setattr(self, Y_LABEL, 'values')
        if Y_OFFSET not in vars(self):
            setattr(self, Y_OFFSET, 0)
        if LEGEND_ENTRY not in vars(self):
            setattr(self, LEGEND_ENTRY, getattr(self, DATASET_ID))


class Figure:
    def __init__(self, vis_spec):
        """

        :param vis_spec: the whole vis spec
        """
        # TODO: rename this class?
        self.num_subplots = 1
        self.title = ''
        self.subplots = []  # list of SinglePlots
        # TODO: what type is this vis_spec - ?
        self.vis_spec = vis_spec

        # get unique plotIDs
        plot_ids = np.unique(self.vis_spec[PLOT_ID])

        # loop over unique plotIds
        for plot_id in plot_ids:
            # get subplot data to plot

            # get indices for specific plotId
            # get entrances of vis spec corresponding to this plot?
            ind_plot = (vis_spec[PLOT_ID] == plot_id)

            subplot_type = vis_spec.loc[ind_plot, PLOT_TYPE_SIMULATION] # take first one?

            # add plots
            self._add_subplot(vis_spec[ind_plot], subplot_type)

    def _add_subplot(self,
                     subplot_vis_spec,
                     subplot_type: str = 'LinePlot'):

        if subplot_type == 'BarPlot':
            subplot = BarPlot(subplot_vis_spec, self.measurements_df, self.simulation_df)
        elif subplot_type == 'ScatterPlot':
            subplot = ScatterPlot(subplot_vis_spec, self.measurements_df, self.simulation_df)
        else:
            subplot = LinePlot(subplot_vis_spec, self.measurements_df, self.simulation_df)

        self.subplots.append(subplot)


class DataToPlot:
    """
    data for one individial line
    """
    def __init__(self):
        # so far created based on the dataframe returned by get_data_to_plot
        self.xValues = []  # what is in the condition_ids parameter of get_data_to_plot
        self.mean = []  # means of replicates
        self.noise_model = None
        self.sd = []
        self.sem = []  # standard error of mean
        self.repl = []  # single replicates


class DataProvider:
    """
    Handles data selection
    """
    def __init__(self,
                 exp_conditions: Union[str, pd.DataFrame],
                 measurements: Union[str, pd.DataFrame],
                 simulations: Optional[Union[str, pd.DataFrame]] = None):
        self.exp_conditions = exp_conditions
        self.measurements = measurements
        self.simulations = simulations
        # validation of dfs?
        # extending
        pass

    def check_datarequest_consistency(self):
        # check if data request is meaningful
        # check_vis_spec_consistency functionality
        pass

    def group_by_measurement(self):
        pass

    def select_by_dataset_ids(self, dataset_ids_per_plot: IdsList
                              ) -> List[Tuple[DataToPlot, DataToPlot]]:
        """

        :param dataset_ids_per_plot:
        :return:

        datasets = [['dataset_1', 'dataset_2'],
                   ['dataset_1', 'dataset_4', 'dataset_5']]
        """

        data_per_plot = []

        for plot_id, dataset_ids in enumerate(dataset_ids_per_plot):
            self.check_datarequest_consistency(dataset_ids)
            # TODO: probably vis spec shouldn't be create here
            plot_vis_spec = VisualisationSpec.from_dataset_ids(
                f'plot_{plot_id}', dataset_ids, plotted_noise)
            # vis spec that is created for the first plot
            # plotId | datasetId | legendEntry | yValues | plotTypeData
            # plot1  | dataset_1 | dataset_1   |         | plotted_noise
            # plot1  | dataset_2 | dataset_2   |         | plotted_noise

            # plus non-mandatory columns are fulled with defaults
            data_per_plot.append(self.select_by_vis_spec(plot_vis_spec))

        return data_per_plot

    def select_by_condition_ids(self, condition_ids: IdsList):
        pass

    def select_by_condition_numbers(self, condition_nums: NumList):
        pass

    def select_by_observable_ids(self, observable_ids: IdsList):
        pass

    def select_by_observable_numbers(self, observable_nums: NumList):
        pass

    def select_by_vis_spec(self, vis_spec: VisualisationSpec
                           ) -> Tuple[DataToPlot, DataToPlot]:
        measurements_to_plot = None
        simulations_to_plot = None
        return measurements_to_plot, simulations_to_plot


class SinglePlot:
    def __init__(self,
                 plot_spec,
                 measurements_df: Optional[pd.DataFrame],
                 simulations_df: Optional[pd.DataFrame]):
        self.id = None
        self.plot_spec = plot_spec  # dataframe, vis spec of a single plot
        self.measurements_df = measurements_df
        self.simulations_df = simulations_df

        # if both meas and simu dfs are None error
        self.measurements_to_plot = self.get_measurements_to_plot()  # dataframe?
        self.simulations_to_plot = self.get_simulations_to_plot()

        self.xValues = plot_spec[X_VALUES]

        # parameters of a specific plot
        self.title = ''
        self.xLabel = X_LABEL
        self.yLabel = Y_LABEL


    def matches_plot_spec(df: pd.DataFrame,
                          col_id: str,
                          x_value: Union[float, str],
                          plot_spec: pd.Series) -> pd.Series:
        """
        constructs an index for subsetting of the dataframe according to what is
        specified in plot_spec.

        Parameters:
            df:
                pandas data frame to subset, can be from measurement file or
                simulation file
            col_id:
                name of the column that will be used for indexing in x variable
            x_value:
                subsetted x value
            plot_spec:
                visualization spec from the visualization file

        Returns:
            index:
                Boolean series that can be used for subsetting of the passed
                dataframe
        """

        subset = (
                (df[col_id] == x_value) &
                (df[DATASET_ID] == plot_spec[DATASET_ID])
        )
        if plot_spec[Y_VALUES] == '':
            if len(df.loc[subset, OBSERVABLE_ID].unique()) > 1:
                ValueError(
                    f'{Y_VALUES} must be specified in visualization table if '
                    f'multiple different observables are available.'
                )
        else:
            subset &= (df[OBSERVABLE_ID] == plot_spec[Y_VALUES])
        return subset

    def get_measurements_to_plot(self) -> Optional[pd.DataFrame]:

        # get datasetID and independent variable of first entry of plot1
        dataset_id = self.plot_spec[DATASET_ID]
        indep_var = self.xValues

        # define index to reduce exp_data to data linked to datasetId
        ind_dataset = self.measurements_df[DATASET_ID] == dataset_id

        # gather simulationConditionIds belonging to datasetId
        uni_condition_id, uind = np.unique(
            self.measurements_df[ind_dataset][SIMULATION_CONDITION_ID],
            return_index=True)
        # keep the ordering which was given by user from top to bottom
        # (avoid ordering by names '1','10','11','2',...)'
        uni_condition_id = uni_condition_id[np.argsort(uind)]
        col_name_unique = SIMULATION_CONDITION_ID

        # Case separation of independent parameter: condition, time or custom
        if indep_var == TIME:
            # obtain unique observation times
            uni_condition_id = np.unique(measurements_df[ind_dataset][TIME])
            col_name_unique = TIME

        # create empty dataframe for means and SDs
        meas_to_plot = pd.DataFrame(
            columns=['mean', 'noise_model', 'sd', 'sem', 'repl'],
            index=uni_condition_id
        )
        for var_cond_id in uni_condition_id:

            # TODO (#117): Here not the case: So, if entries in measurement file:
            #  preequCondId, time, observableParams, noiseParams,
            #  are not the same, then  -> differ these data into
            #  different groups!
            # now: go in simulationConditionId, search group of unique
            # simulationConditionId e.g. rows 0,6,12,18 share the same
            # simulationCondId, then check if other column entries are the same
            # (now: they are), then take intersection of rows 0,6,12,18 and checked
            # other same columns (-> now: 0,6,12,18) and then go on with code.
            # if there is at some point a difference in other columns, say e.g.
            # row 12,18 have different noiseParams than rows 0,6, the actual code
            # would take rows 0,6 and forget about rows 12,18

            # compute mean and standard deviation across replicates
            subset = matches_plot_spec(self.measurements_df,
                                       col_id, var_cond_id,
                                       self.plot_spec)
            data_measurements = self.measurements_df.loc[
                subset,
                MEASUREMENT
            ]

            meas_to_plot.at[var_cond_id, 'mean'] = np.mean(data_measurements)
            meas_to_plot.at[var_cond_id, 'sd'] = np.std(data_measurements)

            if (plot_spec.plotTypeData == PROVIDED) & sum(subset):
                if len(self.measurements_df.loc[subset, NOISE_PARAMETERS].unique()) > 1:
                    raise NotImplementedError(
                        f"Datapoints with inconsistent {NOISE_PARAMETERS} is "
                        f"currently not implemented. Stopping.")
                tmp_noise = self.measurements_df.loc[subset, NOISE_PARAMETERS].values[0]
                if isinstance(tmp_noise, str):
                    raise NotImplementedError(
                        "No numerical noise values provided in the measurement "
                        "table. Stopping.")
                if isinstance(tmp_noise, Number) or tmp_noise.dtype == 'float64':
                    meas_to_plot.at[var_cond_id, 'noise_model'] = tmp_noise

            # standard error of mean
            meas_to_plot.at[var_cond_id, 'sem'] = \
                np.std(data_measurements) / np.sqrt(len(data_measurements))

            # single replicates
            meas_to_plot.at[var_cond_id, 'repl'] = \
                data_measurements

        return meas_to_plot

    def get_simulations_to_plot(self) -> Optional[pd.DataFrame]:
        if self.simulations_df is None:
            return None

         # get datasetID and independent variable of first entry of plot1
        dataset_id = self.plot_spec[DATASET_ID]
        indep_var = self.xValues

        # define index to reduce exp_data to data linked to datasetId
        ind_dataset = self.simulations_df[DATASET_ID] == dataset_id

        # gather simulationConditionIds belonging to datasetId
        uni_condition_id, uind = np.unique(
            self.simulations_df[ind_dataset][SIMULATION_CONDITION_ID],
            return_index=True)
        # keep the ordering which was given by user from top to bottom
        # (avoid ordering by names '1','10','11','2',...)'
        uni_condition_id = uni_condition_id[np.argsort(uind)]
        col_name_unique = SIMULATION_CONDITION_ID

        # Case separation of independent parameter: condition, time or custom
        if indep_var == TIME:
            # obtain unique observation times
            uni_condition_id = np.unique(self.simulations_df_df[ind_dataset][TIME])
            col_name_unique = TIME

        # create empty dataframe for means and SDs
        sim_to_plot = pd.DataFrame(
            columns=['sim'],
            index=uni_condition_id
        )
        for var_cond_id in uni_condition_id:

            simulation_measurements = self.simulations_df.loc[
                matches_plot_spec(self.simulations_df, col_id, var_cond_id,
                                  self.plot_spec),
                SIMULATION
            ]
            sim_to_plot.at[var_cond_id, 'sim'] = np.mean(
                simulation_measurements
            )
        return sim_to_plot


class LinePlot(SinglePlot):
    def __init__(self,
                 vis_spec,
                 measurements_df: Optional[pd.DataFrame],
                 simulations_df: Optional[pd.DataFrame]):
        super().__init__(vis_spec, measurements_df, simulations_df)

    # def add_legends(self):
    #     # legends with rotation
    #     pass


class BarPlot(SinglePlot):
    def __init__(self,
                 vis_spec,
                 measurements_df: Optional[pd.DataFrame],
                 simulations_df: Optional[pd.DataFrame]):
        super().__init__(vis_spec, measurements_df, simulations_df)

    # def add_legends(self):
    #     # legends with rotation
    #     pass


class ScatterPlot(SinglePlot):
    def __init__(self,
                 vis_spec,
                 measurements_df: Optional[pd.DataFrame],
                 simulations_df: Optional[pd.DataFrame]):
        super().__init__(vis_spec, measurements_df, simulations_df)






# class Validator:
#     def __init__(self):
#         pass
#
#     def check_vis_spec(self,
#                        vis_spec):
#         # TODO: create_or_update_vis_spec functionality
#         pass
#
#     def check_measurements_df(self):
#         # TODO: create_or_update_vis_spec functionality
#         pass
