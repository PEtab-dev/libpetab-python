"""Functions operating on the PEtab condition table"""

from __future__ import annotations

from itertools import chain
from pathlib import Path

import pandas as pd
import sympy as sp

from .. import v2
from ..v1.lint import assert_no_leading_trailing_whitespace
from .C import *

__all__ = [
    "get_condition_df",
    "write_condition_df",
]


def get_condition_df(
    condition_file: str | pd.DataFrame | Path | None,
) -> pd.DataFrame | None:
    """Read the provided condition file into a ``pandas.Dataframe``.

    Arguments:
        condition_file: File name of PEtab condition file or pandas.Dataframe
    """
    if condition_file is None:
        return condition_file

    if isinstance(condition_file, str | Path):
        condition_file = pd.read_csv(
            condition_file, sep="\t", float_precision="round_trip"
        )

    assert_no_leading_trailing_whitespace(
        condition_file.columns.values, "condition"
    )

    return condition_file


def write_condition_df(df: pd.DataFrame, filename: str | Path) -> None:
    """Write PEtab condition table

    Arguments:
        df: PEtab condition table
        filename: Destination file name
    """
    df = get_condition_df(df)
    df.to_csv(filename, sep="\t", index=False)


def get_condition_table_free_symbols(problem: v2.Problem) -> set[sp.Basic]:
    """Free symbols from condition table assignments.

    Collects all free symbols from the condition table `targetValue` column.

    :returns: Set of free symbols.
    """
    return set(
        chain.from_iterable(
            change.target_value.free_symbols
            for condition in problem.conditions_table.conditions
            for change in condition.changes
            if change.target_value is not None
        )
    )
