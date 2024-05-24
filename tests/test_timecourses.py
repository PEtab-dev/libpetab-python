"""Tests related to petab.conditions"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import petab
from petab import timecourses
from petab.C import *


def test_get_timecourse_df():
    """Test timecourses.get_timecourse_df."""
    # condition df missing ids
    timecourse_df = pd.DataFrame(
        data={
            TIMECOURSE: ["0:condition1;5:condition2", "-inf:condition1;0:condition2"],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        file_name = fh.name
        timecourse_df.to_csv(fh, sep="\t", index=False)

    with pytest.raises(KeyError):
        petab.get_timecourse_df(file_name)

    os.remove(file_name)

    # with ids
    timecourse_df = pd.DataFrame(
        data={
            TIMECOURSE_ID: ["timecourse1", "timecourse2"],
            TIMECOURSE: ["0:condition1;5:condition2", "-inf:condition1;0:condition2"],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        file_name = fh.name
        timecourse_df.to_csv(fh, sep="\t", index=False)

    df = petab.get_timecourse_df(file_name)  # .replace(np.nan, "")
    assert (df == timecourse_df.set_index(TIMECOURSE_ID)).all().all()

    os.remove(file_name)

    # test other arguments
    assert (petab.get_timecourse_df(timecourse_df) == timecourse_df).all().all()
    assert petab.get_timecourse_df(None) is None


def test_write_timecourse_df():
    """Test timecourses.write_timecourse_df."""
    timecourse_df = pd.DataFrame(
        data={
            TIMECOURSE_ID: ["timecourse1", "timecourse2"],
            TIMECOURSE: ["0:condition1;5:condition2", "-inf:condition1;0:condition2"],
        }
    ).set_index(TIMECOURSE_ID)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = Path(temp_dir) / "timecourses.tsv"
        petab.write_timecourse_df(timecourse_df, file_name)
        re_df = petab.get_timecourse_df(file_name)
        assert (timecourse_df == re_df).all().all()
