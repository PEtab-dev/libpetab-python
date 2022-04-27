"""
This file should contain the functions, which PEtab internally needs for
plotting, but which are not meant to be used by non-developers and should
hence not be directly visible/usable when using `import petab.visualize`.
"""

from typing import List

import pandas as pd

from ..C import *

# for typehints
IdsList = List[str]
NumList = List[int]
__all__ = [
    'create_dataset_id_list_new',
    'generate_dataset_id_col',
]


def generate_dataset_id_col(exp_data: pd.DataFrame) -> List[str]:
    """
    Generate DATASET_ID column from condition_ids and observable_ids.

    Parameters
    ----------
    exp_data:
        A measurement (simulation) DataFrame in the PEtab format.

    Returns
    -------
        A list with generated datasetIds for each entry in the measurement
        (simulation) DataFrame
    """

    # create a column of dummy datasetIDs and legend entries: preallocate
    dataset_id_column = []

    # loop over experimental data table, create datasetId for each entry
    tmp_simcond = list(exp_data[SIMULATION_CONDITION_ID])
    tmp_obs = list(exp_data[OBSERVABLE_ID])

    for ind, cond_id in enumerate(tmp_simcond):
        # create and add dummy datasetID
        dataset_id = cond_id + '_' + tmp_obs[ind]
        dataset_id_column.append(dataset_id)

    return dataset_id_column


def create_dataset_id_list_new(df: pd.DataFrame,
                               group_by: str,
                               id_list: List[IdsList]
                               ) -> List[IdsList]:
    """
    Create dataset ID list from a list of simulation condition IDs or
    observable IDs.

    Parameters:
        df: Measurements or simulations DataFrame.
        group_by: Defines  grouping of data to plot.
        id_list:
            Grouping list. Each sublist corresponds to a subplot in a figure,
            and contains the IDs of observables or simulation conditions for
            the subplot.

    Returns:
        A list of datasetIds

    """
    if DATASET_ID not in df.columns:
        raise ValueError(f'{DATASET_ID} column must be in exp_data DataFrame')

    dataset_id_list = []

    if group_by == 'simulation':
        groupping_col = SIMULATION_CONDITION_ID
    elif group_by == 'observable':
        groupping_col = OBSERVABLE_ID
        if id_list is None:
            # this is the default case. If no grouping is specified,
            # all observables are plotted. One observable per plot.
            unique_obs_list = df[OBSERVABLE_ID].unique()
            id_list = [[obs_id] for obs_id in unique_obs_list]
    else:
        raise ValueError

    for sublist in id_list:
        plot_id_list = []
        for cond_id in sublist:
            plot_id_list.extend(list(
                df[df[groupping_col] == cond_id][
                    DATASET_ID].unique()))
        dataset_id_list.append(plot_id_list)
    return dataset_id_list
