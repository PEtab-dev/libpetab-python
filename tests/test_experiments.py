"""Tests related to petab.conditions"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import petab
from petab.C import *


def test_get_experiment_df():
    """Test `experiments.get_experiment_df`."""
    # condition df missing ids
    experiment_df = pd.DataFrame(
        data={
            EXPERIMENT: [
                "0:condition1;5:condition2",
                "-inf:condition1;0:condition2",
            ],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        file_name = fh.name
        experiment_df.to_csv(fh, sep="\t", index=False)

    with pytest.raises(KeyError):
        petab.get_experiment_df(file_name)

    os.remove(file_name)

    # with ids
    experiment_df = pd.DataFrame(
        data={
            EXPERIMENT_ID: ["experiment1", "experiment2"],
            EXPERIMENT: [
                "0:condition1;5:condition2",
                "-inf:condition1;0:condition2",
            ],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        file_name = fh.name
        experiment_df.to_csv(fh, sep="\t", index=False)

    df = petab.get_experiment_df(file_name)  # .replace(np.nan, "")
    assert (df == experiment_df.set_index(EXPERIMENT_ID)).all().all()

    os.remove(file_name)

    # test other arguments
    assert (
        (petab.get_experiment_df(experiment_df) == experiment_df).all().all()
    )
    assert petab.get_experiment_df(None) is None


def test_write_experiment_df():
    """Test `experiments.write_experiment_df`."""
    experiment_df = pd.DataFrame(
        data={
            EXPERIMENT_ID: ["experiment1", "experiment2"],
            EXPERIMENT: [
                "0:condition1;5:condition2",
                "-inf:condition1;0:condition2",
            ],
        }
    ).set_index(EXPERIMENT_ID)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = Path(temp_dir) / "experiments.tsv"
        petab.write_experiment_df(experiment_df, file_name)
        re_df = petab.get_experiment_df(file_name)
        assert (experiment_df == re_df).all().all()


def test_nested_experiments():
    """Test the denesting in `Experiment.from_df`.

    Implicitly tests general construction and use of the `Experiment` and
    `Period` classes, up to loading them from a PEtab experiment table.
    """
    # Define experiment table with both `restartEvery` and "nested experiment"
    # nesting
    experiment_data = [
        {
            "experimentId": "weekly_radiation_schedule",
            "experiment": "0.0: radiation_on; 5.0: radiation_off",
        },
        {
            "experimentId": "full_radiation_schedule",
            "experiment": (
                "0.0: radiation_off; 10.0:7.0: weekly_radiation_schedule; "
                "30.0: radiation_off"
            ),
        },
    ]
    experiment_df = pd.DataFrame(data=experiment_data)
    experiment_df.set_index("experimentId", inplace=True)
    experiments = petab.experiments.Experiment.from_df(experiment_df)

    # Reconstruct df
    reconstructed_df = pd.DataFrame(
        data=[experiment.to_series() for experiment in experiments.values()]
    )
    reconstructed_df.index.name = EXPERIMENT_ID
    reconstructed_df = petab.get_experiment_df(reconstructed_df)

    expected_reconstructed_df = pd.DataFrame(
        data=[
            pd.Series(
                {EXPERIMENT: "0.0:radiation_on;5.0:radiation_off"},
                name="weekly_radiation_schedule",
            ),
            pd.Series(
                {
                    EXPERIMENT: (
                        "0.0:radiation_off;10.0:radiation_on;15.0:radiation_off;"
                        "17.0:radiation_on;22.0:radiation_off;24.0:radiation_on;"
                        "29.0:radiation_off;30.0:radiation_off"
                    )
                },
                name="full_radiation_schedule",
            ),
        ]
    )
    expected_reconstructed_df.index.name = EXPERIMENT_ID

    # The reconstructed df has the expected denesting of the original
    # experiment df.
    pd.testing.assert_frame_equal(reconstructed_df, expected_reconstructed_df)
