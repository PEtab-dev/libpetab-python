"""Validation of PEtab visualization files"""
import logging

import pandas as pd

from .. import C, Problem
from ..C import VISUALIZATION_DF_REQUIRED_COLS


logger = logging.getLogger(__name__)


def validate_visualization_df(
        problem: Problem
) -> bool:
    """Validate visualization table

    Arguments:
        problem: The PEtab problem containing a visualization table

    Returns:
        ``True`` if errors occurred, ``False`` otherwise
    """
    vis_df = problem.visualization_df
    if vis_df is None or vis_df.empty:
        return False

    errors = False

    if missing_req_cols := (set(VISUALIZATION_DF_REQUIRED_COLS)
                            - set(vis_df.columns)):
        logger.error(f"Missing required columns {missing_req_cols} "
                     "in visualization table.")
        errors = True

    # Set all unspecified optional values to their defaults to simplify
    # validation
    vis_df = vis_df.copy()
    _apply_defaults(vis_df)

    if unknown_types := (set(vis_df[C.PLOT_TYPE_SIMULATION].unique())
                         - set(C.PLOT_TYPES_SIMULATION)):
        logger.error(f"Unknown {C.PLOT_TYPE_SIMULATION}: {unknown_types}. "
                     f"Must be one of {C.PLOT_TYPES_SIMULATION}")
        errors = True

    if unknown_types := (set(vis_df[C.PLOT_TYPE_DATA].unique())
                         - set(C.PLOT_TYPES_DATA)):
        logger.error(f"Unknown {C.PLOT_TYPE_DATA}: {unknown_types}. "
                     f"Must be one of {C.PLOT_TYPES_DATA}")
        errors = True

    if unknown_scale := (set(vis_df[C.X_SCALE].unique())
                         - set(C.X_SCALES)):
        # TODO: handle 'order'
        logger.error(f"Unknown {C.X_SCALE}: {unknown_scale}. "
                     f"Must be one of {C.X_SCALES}")
        errors = True

    if unknown_scale := (set(vis_df[C.Y_SCALE].unique())
                         - set(C.Y_SCALES)):
        logger.error(f"Unknown {C.Y_SCALE}: {unknown_scale}. "
                     f"Must be one of {C.Y_SCALES}")
        errors = True

    return errors


def _apply_defaults(vis_df: pd.DataFrame):
    """
    Set default values.

    Adds default values to the given visualization table where no value was
    specified.
    """
    def set_default(column: str, value):
        if column not in vis_df:
            vis_df[column] = value
        elif value is not None:
            vis_df[column].fillna(value)

    set_default(C.PLOT_NAME, "")
    set_default(C.PLOT_TYPE_SIMULATION, C.LINE_PLOT)
    set_default(C.PLOT_TYPE_DATA, C.MEAN_AND_SD)
    set_default(C.DATASET_ID, None)
    set_default(C.X_VALUES, C.TIME)
    set_default(C.X_OFFSET, 0)
    set_default(C.X_LABEL, vis_df[C.X_VALUES])
    set_default(C.X_SCALE, C.LIN)
    set_default(C.Y_OFFSET, 0)
    set_default(C.Y_LABEL, vis_df[C.Y_VALUES])
    set_default(C.Y_SCALE, C.LIN)
    set_default(C.LEGEND_ENTRY, vis_df[C.DATASET_ID])
