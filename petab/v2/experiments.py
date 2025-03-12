"""Functions operating on the PEtab experiments table."""

from pathlib import Path

import pandas as pd

__all__ = ["get_experiment_df", "write_experiment_df"]


def get_experiment_df(
    experiments_file: str | pd.DataFrame | Path | None,
) -> pd.DataFrame | None:
    """
    Read the provided observable file into a ``pandas.Dataframe``.

    Arguments:
        experiments_file: Name of the file to read from or pandas.Dataframe.

    Returns:
        Observable DataFrame
    """
    if experiments_file is None:
        return experiments_file

    if isinstance(experiments_file, str | Path):
        experiments_file = pd.read_csv(
            experiments_file, sep="\t", float_precision="round_trip"
        )

    return experiments_file


def write_experiment_df(df: pd.DataFrame, filename: str | Path) -> None:
    """Write PEtab experiments table

    Arguments:
        df: PEtab experiments table
        filename: Destination file name. The parent directory will be created
            if necessary.
    """
    df = get_experiment_df(df)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, sep="\t", index=False)
