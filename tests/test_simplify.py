"""Tests for petab.simplify.*"""
from math import nan

import pandas as pd
import pytest
import simplesbml
from pandas.testing import *

from petab import Problem
from petab.C import *  # noqa: F403
from petab.models.sbml_model import SbmlModel
from petab.simplify import *


@pytest.fixture
def problem() -> Problem:
    ss_model = simplesbml.SbmlModel()
    ss_model.addParameter("some_parameter", val=1.0)
    ss_model.addParameter("same_value_for_all_conditions", val=1.0)

    observable_df = pd.DataFrame(
        {
            OBSERVABLE_ID: ["obs_used", "obs_unused", "obs_used_2"],
            OBSERVABLE_FORMULA: [1.0, 2.0, 3.0],
            NOISE_FORMULA: [1.0, 2.0, 3.0],
        }
    )
    observable_df.set_index(OBSERVABLE_ID, inplace=True)

    conditions_df = pd.DataFrame(
        {
            CONDITION_ID: ["condition_used_1",
                           "condition_unused",
                           "condition_used_2"],
            "some_parameter": [1.0, 2.0, 3.0],
            "same_value_for_all_conditions": [4.0] * 3,
        }
    )
    conditions_df.set_index(CONDITION_ID, inplace=True)

    measurement_df = pd.DataFrame(
            {
                OBSERVABLE_ID: ["obs_used", "obs_used_2", "obs_used"],
                MEASUREMENT: [1.0, 1.5, 2.0],
                SIMULATION_CONDITION_ID: ["condition_used_1",
                                          "condition_used_1",
                                          "condition_used_2"],
                TIME: [1.0] * 3,
            }
        )
    yield Problem(
        model=SbmlModel(sbml_model=ss_model.getModel()),
        condition_df=conditions_df,
        observable_df=observable_df,
        measurement_df=measurement_df,
    )


def test_remove_nan_measurements(problem):
    expected = pd.DataFrame(
            {
                OBSERVABLE_ID: ["obs_used"] * 2,
                MEASUREMENT: [1.0, 2.0],
                SIMULATION_CONDITION_ID:
                    ["condition_used_1", "condition_used_2"],
                TIME: [1.0] * 2,
            }
        )

    problem.measurement_df = pd.DataFrame(
            {
                OBSERVABLE_ID: ["obs_used", "obs_with_nan", "obs_used"],
                MEASUREMENT: [1.0, nan, 2.0],
                SIMULATION_CONDITION_ID: ["condition_used_1",
                                          "condition_used_1",
                                          "condition_used_2"],
                TIME: [1.0] * 3,
            }
        )
    assert not problem.measurement_df.equals(expected)

    remove_nan_measurements(problem)

    assert_frame_equal(problem.measurement_df, expected)


def test_remove_unused_observables(problem):
    expected = pd.DataFrame(
            {
                OBSERVABLE_ID: ["obs_used", "obs_used_2"],
                OBSERVABLE_FORMULA: [1.0, 3.0],
                NOISE_FORMULA: [1.0, 3.0],
            }
        )
    expected.set_index(OBSERVABLE_ID, inplace=True)
    assert not problem.observable_df.equals(expected)

    remove_unused_observables(problem)

    assert_frame_equal(problem.observable_df, expected)


def test_remove_unused_conditions(problem):
    expected = pd.DataFrame(
            {
                CONDITION_ID: ["condition_used_1",
                               "condition_used_2"],
                "some_parameter": [1.0, 3.0],
                "same_value_for_all_conditions": [4.0] * 2,
            }
        )
    expected.set_index(CONDITION_ID, inplace=True)
    assert not problem.condition_df.equals(expected)

    remove_unused_conditions(problem)

    assert_frame_equal(problem.condition_df, expected)


def test_condition_parameters_to_parameter_table(problem):
    expected_conditions = pd.DataFrame(
        {
            CONDITION_ID: ["condition_used_1",
                           "condition_unused",
                           "condition_used_2"],
            "some_parameter": [1.0, 2.0, 3.0],
        }
    )
    expected_conditions.set_index(CONDITION_ID, inplace=True)
    assert not problem.condition_df.equals(expected_conditions)

    expected_parameters = pd.DataFrame({
            PARAMETER_ID: ["same_value_for_all_conditions"],
            PARAMETER_SCALE: [LIN],
            LOWER_BOUND: [nan],
            UPPER_BOUND: [nan],
            NOMINAL_VALUE: [4.0],
            ESTIMATE: [0],
        })
    expected_parameters.set_index(PARAMETER_ID, inplace=True)
    assert problem.parameter_df is None

    condition_parameters_to_parameter_table(problem)

    assert_frame_equal(problem.condition_df, expected_conditions)
    assert_frame_equal(problem.parameter_df, expected_parameters)


def test_simplify_problem(problem):
    # simplify_problem checks whether the result is valid
    simplify_problem(problem)
