"""Tests related to ``petab.v2.experiments``."""
from tempfile import TemporaryDirectory

import pandas as pd

from petab.v2.C import CONDITION_ID, EXPERIMENT_ID, TIME
from petab.v2.experiments import get_experiment_df, write_experiment_df


def test_experiment_df_io():
    # Test None
    assert get_experiment_df(None) is None

    # Test DataFrame
    df = pd.DataFrame(
        {
            EXPERIMENT_ID: ["e1", "e2"],
            CONDITION_ID: ["c1", "c2"],
            TIME: [0, 1],
        }
    )
    df = get_experiment_df(df)
    assert df.shape == (2, 3)

    # Test writing to file and round trip
    with TemporaryDirectory() as tmpdir:
        tmpfile = f"{tmpdir}/experiment.csv"
        write_experiment_df(df, tmpfile)
        df2 = get_experiment_df(tmpfile)
        assert df.equals(df2)
