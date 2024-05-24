"""Functions operating on the PEtab timecouse table."""

from pathlib import Path
from typing import TypeVar

import pandas as pd

from . import lint
from .C import TIMECOURSE_ID, TIMECOURSE_NAME, TIMECOURSE


__all__ = [
    "get_timecourse_df",
    "write_timecourse_df",
]


def get_timecourse_df(timecourse_file: str | Path | None) -> pd.DataFrame:
    """Read the provided timecourse file into a ``pandas.Dataframe``.

    Arguments:
        timecourse_file:
            Location of PEtab timecourse file, or a ``pandas.Dataframe``.

    Returns:
        The timecourses dataframe.
    """
    if timecourse_file is None:
        return timecourse_file

    if isinstance(timecourse_file, (str, Path)):
        timecourse_file = pd.read_csv(
            timecourse_file,
            sep='\t',
            float_precision='round_trip',
        )

    lint.assert_no_leading_trailing_whitespace(
        timecourse_file.columns.values, TIMECOURSE
    )

    if not isinstance(timecourse_file.index, pd.RangeIndex):
        timecourse_file.reset_index(inplace=True)

    try:
        timecourse_file.set_index([TIMECOURSE_ID], inplace=True)
    except KeyError:
        raise KeyError(
            f'Timecourse table missing mandatory field {TIMECOURSE_ID}.')

    return timecourse_file


def write_timecourse_df(df: pd.DataFrame, filename: str | Path) -> None:
    """Write the provided PEtab timecourse table to disk.

    Arguments:
        df:
            The PEtab timecourse table.
        filename:
            The table will be written to this location.
    """
    df.to_csv(filename, sep='\t', index=True)


