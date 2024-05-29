"""Tests related to petab.conditions"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import petab
from petab.C import *


def test_get_timecourse_df():
    """Test `timecourses.get_timecourse_df`."""
    # condition df missing ids
    timecourse_df = pd.DataFrame(
        data={
            TIMECOURSE: [
                "0:condition1;5:condition2",
                "-inf:condition1;0:condition2",
            ],
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
            TIMECOURSE: [
                "0:condition1;5:condition2",
                "-inf:condition1;0:condition2",
            ],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        file_name = fh.name
        timecourse_df.to_csv(fh, sep="\t", index=False)

    df = petab.get_timecourse_df(file_name)  # .replace(np.nan, "")
    assert (df == timecourse_df.set_index(TIMECOURSE_ID)).all().all()

    os.remove(file_name)

    # test other arguments
    assert (
        (petab.get_timecourse_df(timecourse_df) == timecourse_df).all().all()
    )
    assert petab.get_timecourse_df(None) is None


def test_write_timecourse_df():
    """Test `timecourses.write_timecourse_df`."""
    timecourse_df = pd.DataFrame(
        data={
            TIMECOURSE_ID: ["timecourse1", "timecourse2"],
            TIMECOURSE: [
                "0:condition1;5:condition2",
                "-inf:condition1;0:condition2",
            ],
        }
    ).set_index(TIMECOURSE_ID)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = Path(temp_dir) / "timecourses.tsv"
        petab.write_timecourse_df(timecourse_df, file_name)
        re_df = petab.get_timecourse_df(file_name)
        assert (timecourse_df == re_df).all().all()


def test_nested_timecourses():
    """Test the denesting in `Timecourse.from_df`.

    Implicitly tests general construction and use of the `Timecourse` and
    `Period` classes, up to loading them from a PEtab timecourse table.
    """
    # Define timecourse table with both `restartEvery` and "nested timecourse"
    # nesting
    timecourse_data = [
        {
            "timecourseId": "weekly_radiation_schedule",
            "timecourse": "0.0: radiation_on; 5.0: radiation_off",
        },
        {
            "timecourseId": "full_radiation_schedule",
            "timecourse": (
                "0.0: radiation_off; 10.0:7.0: weekly_radiation_schedule; "
                "30.0: radiation_off"
            ),
        },
    ]
    timecourse_df = pd.DataFrame(data=timecourse_data)
    timecourse_df.set_index("timecourseId", inplace=True)
    timecourses = petab.timecourses.Timecourse.from_df(timecourse_df)

    # Reconstruct df
    reconstructed_df = pd.DataFrame(
        data=[timecourse.to_series() for timecourse in timecourses.values()]
    )
    reconstructed_df.index.name = TIMECOURSE_ID
    reconstructed_df = petab.get_timecourse_df(reconstructed_df)

    expected_reconstructed_df = pd.DataFrame(
        data=[
            pd.Series(
                {TIMECOURSE: "0.0:radiation_on;5.0:radiation_off"},
                name="weekly_radiation_schedule",
            ),
            pd.Series(
                {
                    TIMECOURSE: (
                        "0.0:radiation_off;10.0:radiation_on;15.0:radiation_off;"
                        "17.0:radiation_on;22.0:radiation_off;24.0:radiation_on;"
                        "29.0:radiation_off;30.0:radiation_off"
                    )
                },
                name="full_radiation_schedule",
            ),
        ]
    )
    expected_reconstructed_df.index.name = TIMECOURSE_ID

    # The reconstructed df has the expected denesting of the original
    # timecourse df.
    pd.testing.assert_frame_equal(reconstructed_df, expected_reconstructed_df)
