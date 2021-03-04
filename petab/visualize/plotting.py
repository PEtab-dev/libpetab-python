import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple, Union

from ..problem import Problem
from ..C import *

# for typehints
IdsList = List[str]
NumList = List[int]


class VisualisationSpec:
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


class Figure:
    def __init__(self):
        self.num_subplots = 1
        self.subplots = []  # list of SinglePlots


class DataToPlot:
    def __init__(self):
        sim_data = None
        meas_data = None


class SinglePlot:
    def __init__(self,
                 plot_spec,
                 measurements_df: Optional[pd.DataFrame],
                 simulations_df: Optional[pd.DataFrame]):
        self.id = None
        self.plot_spec = None  # dataframe, vis spec of a single plot

        # if both meas and simu dfs are None error
        self.measurements_to_plot = self.get_measurements_to_plot()  # dataframe?
        self.simulations_to_plot = self.get_simulations_to_plot()

        self.xValues = plot_spec[X_VALUES]

        # parameters of a specific plot
        self.title = ''
        self.xLabel = X_LABEL
        self.yLabel = Y_LABEL

    def get_measurements_to_plot(self) -> Optional[pd.DataFrame]:

        # create empty dataframe for means and SDs
        meas_to_plot = pd.DataFrame(
            columns=['mean', 'noise_model', 'sd', 'sem', 'repl'],
            index=condition_ids
        )
        for var_cond_id in condition_ids:

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
            subset = matches_plot_spec(m_data, col_id, var_cond_id, plot_spec)
            data_measurements = m_data.loc[
                subset,
                MEASUREMENT
            ]

            meas_to_plot.at[var_cond_id, 'mean'] = np.mean(data_measurements)
            meas_to_plot.at[var_cond_id, 'sd'] = np.std(data_measurements)

            if (plot_spec.plotTypeData == PROVIDED) & sum(subset):
                if len(m_data.loc[subset, NOISE_PARAMETERS].unique()) > 1:
                    raise NotImplementedError(
                        f"Datapoints with inconsistent {NOISE_PARAMETERS} is "
                        f"currently not implemented. Stopping.")
                tmp_noise = m_data.loc[subset, NOISE_PARAMETERS].values[0]
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
        return None


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
