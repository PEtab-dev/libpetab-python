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
                 vis_spec,
                 measurements_df: Optional[pd.DataFrame],
                 simulations_df: Optional[pd.DataFrame]):
        self.id = None
        self.vis_spec = None  # dataframe, vis spec of a single plot

        # if both meas and simu dfs are None error
        self.measurements_to_plot = self.get_measurements_to_plot()  # dataframe?
        self.simulations_to_plot = self.get_simulations_to_plot()

        self.xValues = vis_spec[X_VALUES]

        # parameters of a specific plot
        self.title = ''
        self.xLabel = X_LABEL
        self.yLabel = Y_LABEL

    def get_measurements_to_plot(self) -> Optional[pd.DataFrame]:
        return None

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
