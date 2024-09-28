"""Tests related to petab.conditions"""
import os
import tempfile
from pathlib import Path
from io import StringIO

import pandas as pd
import pytest

#import petab
import petab.v1
import petab.v2
from petab.v2.experiments import Experiments, get_v1_tables_sequence
from petab.v2.C import *


@pytest.fixture
def nested_switch_experiments():
    table = """
experiment_id	input_id	input_value	time	repeat_every
switchOn	switch	1		
switchOff	switch	0		
switchSequence	switchOn		0	
switchSequence	switchOff		5	
experiment1	switchSequence		0	10
experiment1	switchOff		31	
"""
    experiments = Experiments.load(pd.read_csv(StringIO(table), sep="\t"))
    return experiments

@pytest.fixture
def nested_switch_measurements():
    table = """
observableId	experiment_id	time	measurement
obs1	experiment1	0	1
obs1	experiment1	1	2
obs1	experiment1	5	3
obs1	experiment1	6	4
obs1	experiment1	10	5
obs1	experiment1	11	6
obs1	experiment1	25	7
obs1	experiment1	26	8
obs1	experiment1	30	9
obs1	experiment1	31	10
obs1	experiment1	32	11
obs1	experiment1	100	12
"""
    return petab.v1.get_measurement_df(pd.read_csv(StringIO(table), sep="\t"))


def test_denesting(nested_switch_experiments):
    """The table should be denested to remove self-references."""
    expected_denested_table = """experiment_id	input_id	input_value	time	priority	repeat_every
experiment1	switch	1.0	0.0	0	
experiment1	switch	0.0	5.0	0	
experiment1	switch	1.0	10.0	0	
experiment1	switch	0.0	15.0	0	
experiment1	switch	1.0	20.0	0	
experiment1	switch	0.0	25.0	0	
experiment1	switch	1.0	30.0	0	
experiment1	switch	0.0	31.0	0	
"""
    expected_denested_table = Experiments.load(pd.read_csv(StringIO(expected_denested_table), sep="\t")).save()

    denested_table = pd.DataFrame(nested_switch_experiments.denest("experiment1").periods)

    # The table has no self-references, i.e., `input_id` is not an `experiment_id`, but rather e.g. something in the model.
    pd.testing.assert_frame_equal(denested_table, expected_denested_table)


def test_v1_tables_sequence(nested_switch_experiments, nested_switch_measurements):
    """The experiments should be converted to a sequence of PEtab problems."""
    # TODO test preequilibration
    #switch_sequence = {
    #    0.0: 1.0,
    #    5.0: 0.0,
    #    10.0: 1.0,
    #    15.0: 0.0,
    #    20.0: 1.0,
    #    25.0: 0.0,
    #    30.0: 1.0,
    #    31.0: 0.0,
    #}
    experiment_id = "experiment1"
    switch_sequence = {
        period.time: period.input_value
        for period in nested_switch_experiments.denest(experiment_id).periods
    }
    expected_condition_tables = [
        petab.v1.get_condition_df(pd.DataFrame(data={
            petab.v1.C.CONDITION_ID: [experiment_id],
            "switch": [switch_value],
        }))
        for switch_value in switch_sequence.values()
    ]

    switch_sequence[float('inf')] = 0.0

    start_time = None
    expected_measurement_tables = []
    for period_index, (end_time, switch_value) in enumerate(switch_sequence.items()):
        if start_time is None:
            start_time = end_time
            continue

        expected_measurement_table = nested_switch_measurements.loc[
            (nested_switch_measurements[TIME] >= start_time)
            &
            (nested_switch_measurements[TIME] < end_time)
        ]
        expected_measurement_table = expected_measurement_table.rename(columns={EXPERIMENT_ID: petab.v1.SIMULATION_CONDITION_ID})
        expected_measurement_tables.append(expected_measurement_table)

        start_time = end_time

    petab_v1_sequence = get_v1_tables_sequence(
        experiments=nested_switch_experiments,
        measurement_df=nested_switch_measurements,
    )[experiment_id]

    for (_, (condition_df, measurement_df)), expected_condition_df, expected_measurement_df in zip(petab_v1_sequence.items(), expected_condition_tables, expected_measurement_tables, strict=True):
        # Each period has produced the correct expected condition and
        # measurement table.
        pd.testing.assert_frame_equal(condition_df, expected_condition_df)
        pd.testing.assert_frame_equal(measurement_df, expected_measurement_df)


#def test_get_experiment_df():
#    """Test `experiments.get_experiment_df`."""
#    # condition df missing ids
#    experiment_df = pd.DataFrame(
#        data={
#            EXPERIMENT: [
#                "0:condition1;5:condition2",
#                "-inf:condition1;0:condition2",
#            ],
#        }
#    )
#
#    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
#        file_name = fh.name
#        experiment_df.to_csv(fh, sep="\t", index=False)
#
#    with pytest.raises(KeyError):
#        petab.get_experiment_df(file_name)
#
#    os.remove(file_name)
#
#    # with ids
#    experiment_df = pd.DataFrame(
#        data={
#            EXPERIMENT_ID: ["experiment1", "experiment2"],
#            EXPERIMENT: [
#                "0:condition1;5:condition2",
#                "-inf:condition1;0:condition2",
#            ],
#        }
#    )
#
#    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
#        file_name = fh.name
#        experiment_df.to_csv(fh, sep="\t", index=False)
#
#    df = petab.get_experiment_df(file_name)  # .replace(np.nan, "")
#    assert (df == experiment_df.set_index(EXPERIMENT_ID)).all().all()
#
#    os.remove(file_name)
#
#    # test other arguments
#    assert (
#        (petab.get_experiment_df(experiment_df) == experiment_df).all().all()
#    )
#    assert petab.get_experiment_df(None) is None
#
#
#def test_write_experiment_df():
#    """Test `experiments.write_experiment_df`."""
#    experiment_df = pd.DataFrame(
#        data={
#            EXPERIMENT_ID: ["experiment1", "experiment2"],
#            EXPERIMENT: [
#                "0:condition1;5:condition2",
#                "-inf:condition1;0:condition2",
#            ],
#        }
#    ).set_index(EXPERIMENT_ID)
#
#    with tempfile.TemporaryDirectory() as temp_dir:
#        file_name = Path(temp_dir) / "experiments.tsv"
#        petab.write_experiment_df(experiment_df, file_name)
#        re_df = petab.get_experiment_df(file_name)
#        assert (experiment_df == re_df).all().all()
#
#
#def test_nested_experiments():
#    """Test the denesting in `Experiment.from_df`.
#
#    Implicitly tests general construction and use of the `Experiment` and
#    `Period` classes, up to loading them from a PEtab experiment table.
#    """
#    # Define experiment table with both `restartEvery` and "nested experiment"
#    # nesting
#    experiment_data = [
#        {
#            "experimentId": "weekly_radiation_schedule",
#            "experiment": "0.0: radiation_on; 5.0: radiation_off",
#        },
#        {
#            "experimentId": "full_radiation_schedule",
#            "experiment": (
#                "0.0: radiation_off; 10.0:7.0: weekly_radiation_schedule; "
#                "30.0: radiation_off"
#            ),
#        },
#    ]
#    experiment_df = pd.DataFrame(data=experiment_data)
#    experiment_df.set_index("experimentId", inplace=True)
#    experiments = petab.experiments.Experiment.from_df(experiment_df)
#
#    # Reconstruct df
#    reconstructed_df = pd.DataFrame(
#        data=[experiment.to_series() for experiment in experiments.values()]
#    )
#    reconstructed_df.index.name = EXPERIMENT_ID
#    reconstructed_df = petab.get_experiment_df(reconstructed_df)
#
#    expected_reconstructed_df = pd.DataFrame(
#        data=[
#            pd.Series(
#                {EXPERIMENT: "0.0:radiation_on;5.0:radiation_off"},
#                name="weekly_radiation_schedule",
#            ),
#            pd.Series(
#                {
#                    EXPERIMENT: (
#                        "0.0:radiation_off;10.0:radiation_on;15.0:radiation_off;"
#                        "17.0:radiation_on;22.0:radiation_off;24.0:radiation_on;"
#                        "29.0:radiation_off;30.0:radiation_off"
#                    )
#                },
#                name="full_radiation_schedule",
#            ),
#        ]
#    )
#    expected_reconstructed_df.index.name = EXPERIMENT_ID
#
#    # The reconstructed df has the expected denesting of the original
#    # experiment df.
#    pd.testing.assert_frame_equal(reconstructed_df, expected_reconstructed_df)
