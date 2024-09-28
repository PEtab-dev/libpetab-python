"""Functions operating on the PEtab measurement table"""
# noqa: F405

import itertools
import math
import numbers
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from . import core, lint, observables
from .C import *  # noqa: F403
from ..v1.measurements import *


__all__ = [
    "get_measured_experiments",
    "get_experiment_measurements",
]


def get_measured_experiments(measurement_df: pd.DataFrame) -> list[str]:
    """Get the list of experiments for which there are measurements.

    Arguments:
        measurement_df: PEtab measurement table

    Returns:
        The list of experiment IDs, sorted alphabetically.
    """
    return sorted(measurement_df[EXPERIMENT_ID].unique())


def get_experiment_measurements(
    measurement_df: pd.DataFrame, experiment_id: str
):
    """Get the measurements associated with a specific experiment.

    Arguments:
        measurement_df:
            PEtab measurement DataFrame.
        experiment_id:
            The experiment ID.

    Returns:
        The measurements for the experiment.
    """
    experiment_measurement_df = measurement_df.loc[
        measurement_df[EXPERIMENT_ID] == experiment_id
    ]
    return experiment_measurement_df


# TODO create_measurement_df, measurements_have_replicates


def measurement_is_at_steady_state(time: float) -> bool:
    """Deprecated. See `petab.core.time_is_at_steady_state`."""
    warnings.warn(
        "Use `petab.core.time_is_at_steady_state` instead.",
        DeprecationWarning,
    )
    return core.time_is_at_steady_state(time=time)
