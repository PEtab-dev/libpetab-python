"""Functions for normalizing PEtab tables.

Setting data types, adding missing columns and replacing NA by default values.
"""

from typing import TypeVar

import pandas as pd

from .C import *

__all__ = [
    "normalize_parameter_df",
    "normalize_measurement_df",
    "normalize_condition_df",
    "normalize_observable_df",
]

DataFrameOrNone = TypeVar("DataFrameOrNone", pd.DataFrame, None)


def normalize_parameter_df(
    df: DataFrameOrNone, inplace: bool = False
) -> DataFrameOrNone:
    """Normalize parameter table.

    Arguments:
        df: Parameter table
        inplace: Modify DataFrame in place
    Returns:
        The updated DataFrame
    """
    if df is None:
        return

    if not inplace:
        df = df.copy()

    col_to_type = {
        PARAMETER_ID: str,
        PARAMETER_NAME: str,
        PARAMETER_SCALE: str,
        NOMINAL_VALUE: float,
        LOWER_BOUND: float,
        UPPER_BOUND: float,
        ESTIMATE: bool,
        INITIALIZATION_PRIOR_TYPE: str,
        INITIALIZATION_PRIOR_PARAMETERS: str,  # TODO -> tuple?
        OBJECTIVE_PRIOR_TYPE: str,
        OBJECTIVE_PRIOR_PARAMETERS: str,  # TODO -> tuple?
    }
    for col, dtype in col_to_type.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
        else:
            df[col] = pd.NA

    # TODO only if estimate is True?
    df[INITIALIZATION_PRIOR_TYPE] = df[INITIALIZATION_PRIOR_TYPE].fillna(
        PARAMETER_SCALE_UNIFORM
    )
    # TODO: lb;ub
    #  df[INITIALIZATION_PRIOR_PARAMETERS]

    return df


def normalize_measurement_df(
    df: DataFrameOrNone, inplace: bool = False
) -> DataFrameOrNone:
    """Normalize measurement table.

    Arguments:
        df: Measurement table
        inplace: Modify DataFrame in place
    Returns:
        The updated DataFrame
    """
    if df is None:
        return

    if not inplace:
        df = df.copy()

    col_to_type = {
        OBSERVABLE_ID: str,
        PREEQUILIBRATION_CONDITION_ID: str,
        SIMULATION_CONDITION_ID: str,
        MEASUREMENT: float,
        TIME: float,
        OBSERVABLE_PARAMETERS: str,  # TODO -> tuple?
        NOISE_PARAMETERS: str,  # TODO -> tuple?
    }
    for col, dtype in col_to_type.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
        else:
            df[col] = pd.NA

    return df


def normalize_condition_df(
    df: DataFrameOrNone, inplace: bool = False
) -> DataFrameOrNone:
    """Normalize condition table.

    Arguments:
        df: Condition table
        inplace: Modify DataFrame in place
    Returns:
        The updated DataFrame
    """
    if df is None:
        return

    if not inplace:
        df = df.copy()

    # TODO: always as string even if everything is numeric?
    # TODO: everything but ID and Name to sympy?
    df = df.astype(str)

    return df


def normalize_observable_df(
    df: DataFrameOrNone, inplace: bool = False
) -> DataFrameOrNone:
    """Normalize observable table.

    Arguments:
        df: Observable table
        inplace: Modify DataFrame in place
    Returns:
        The updated DataFrame
    """
    if df is None:
        return

    if not inplace:
        df = df.copy()

    col_to_type = {
        OBSERVABLE_ID: str,
        OBSERVABLE_NAME: str,
        OBSERVABLE_FORMULA: str,  # TODO -> sympy?
        NOISE_FORMULA: str,  # TODO -> sympy?
        NOISE_DISTRIBUTION: str,
        OBSERVABLE_TRANSFORMATION: str,
    }
    for col, dtype in col_to_type.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
        else:
            df[col] = pd.NA

    df[OBSERVABLE_TRANSFORMATION] = df[OBSERVABLE_TRANSFORMATION].fillna(LIN)
    df[NOISE_DISTRIBUTION] = df[NOISE_DISTRIBUTION].fillna(NORMAL)
    df[NOISE_FORMULA] = df[NOISE_FORMULA].fillna(1.0)
