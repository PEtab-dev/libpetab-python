"""Functionality related to the PEtab entity mapping table"""
from pathlib import Path
from typing import Union, Optional
from .models import Model
import pandas as pd

from . import lint
from .C import *  # noqa: F403

__all__ = [
    'get_mapping_df',
    'write_mapping_df',
    'check_mapping_df',
]


def get_mapping_df(
        mapping_file: Union[None, str, Path, pd.DataFrame]
) -> pd.DataFrame:
    """
    Read the provided mapping file into a ``pandas.Dataframe``.

    Arguments:
        mapping_file: Name of file to read from or pandas.Dataframe

    Returns:
        Mapping DataFrame
    """
    if mapping_file is None:
        return mapping_file

    if isinstance(mapping_file, (str, Path)):
        mapping_file = pd.read_csv(mapping_file, sep='\t',
                                   float_precision='round_trip')

    for col in MAPPING_DF_REQUIRED_COLS:
        if col not in mapping_file.reset_index().columns:
            raise KeyError(
                f"Mapping table missing mandatory field {PETAB_ENTITY_ID}.")

        lint.assert_no_leading_trailing_whitespace(
            mapping_file.reset_index()[col].values, col)

    if not isinstance(mapping_file.index, pd.RangeIndex):
        mapping_file.reset_index(inplace=True)

    mapping_file.set_index([PETAB_ENTITY_ID], inplace=True)

    return mapping_file


def write_mapping_df(df: pd.DataFrame, filename: Union[str, Path]) -> None:
    """Write PEtab mapping table

    Arguments:
        df: PEtab mapping table
        filename: Destination file name
    """
    df = get_mapping_df(df)
    df.to_csv(filename, sep='\t', index=True)


def check_mapping_df(
        df: pd.DataFrame,
        model: Optional[Model] = None,
) -> None:
    """Run sanity checks on PEtab mapping table

    Arguments:
        df: PEtab mapping DataFrame
        model: Model for additional checking of parameter IDs

    Raises:
        AssertionError: in case of problems
    """
    lint._check_df(df, MAPPING_DF_REQUIRED_COLS[1:], "mapping")

    if df.index.name != PETAB_ENTITY_ID:
        raise AssertionError(
            f"Mapping table has wrong index {df.index.name}."
            f"expected {PETAB_ENTITY_ID}.")

    lint.check_ids(df.index.values, kind=PETAB_ENTITY_ID)

    if model:
        for model_entity_id in df[MODEL_ENTITY_ID]:
            if not model.has_entity_with_id(model_entity_id):
                raise AssertionError(
                    "Mapping table maps to unknown "
                    f"model entity ID {model_entity_id}."
                )