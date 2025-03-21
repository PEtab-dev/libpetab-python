import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import petab.v2 as petab
from petab.v2 import Problem
from petab.v2.C import (
    CONDITION_ID,
    ESTIMATE,
    LOWER_BOUND,
    MODEL_ENTITY_ID,
    NOISE_FORMULA,
    NOMINAL_VALUE,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    PARAMETER_ID,
    PETAB_ENTITY_ID,
    TARGET_ID,
    TARGET_VALUE,
    UPPER_BOUND,
)


def test_load_remote():
    """Test loading remote files"""
    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/update_v2/petabtests/cases/v2.0.0/sbml/0010/_0010.yaml"
    )
    petab_problem = Problem.from_yaml(yaml_url)

    assert (
        petab_problem.measurement_df is not None
        and not petab_problem.measurement_df.empty
    )

    assert petab_problem.validate() == []


def test_auto_upgrade():
    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/main/petabtests/cases/v1.0.0/sbml/0001/_0001.yaml"
    )
    problem = Problem.from_yaml(yaml_url)
    # TODO check something specifically different in a v2 problem
    assert isinstance(problem, Problem)


def test_problem_from_yaml_multiple_files():
    """Test loading PEtab version 2 yaml with multiple condition / measurement
    / observable files
    """
    yaml_config = """
    format_version: 2.0.0
    parameter_file:
    problems:
    - condition_files: [conditions1.tsv, conditions2.tsv]
      measurement_files: [measurements1.tsv, measurements2.tsv]
      observable_files: [observables1.tsv, observables2.tsv]
      model_files:
      experiment_files: [experiments1.tsv, experiments2.tsv]
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir, "problem.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_config)

        for i in (1, 2):
            problem = Problem()
            problem.add_condition(f"condition{i}", parameter1=i)
            petab.write_condition_df(
                problem.condition_df, Path(tmpdir, f"conditions{i}.tsv")
            )

            problem.add_experiment(f"experiment{i}", 0, f"condition{i}")
            petab.write_experiment_df(
                problem.experiment_df, Path(tmpdir, f"experiments{i}.tsv")
            )

            problem.add_measurement(f"observable{i}", f"experiment{i}", 1, 1)
            petab.write_measurement_df(
                problem.measurement_df, Path(tmpdir, f"measurements{i}.tsv")
            )

            problem.add_observable(f"observable{i}", 1, 1)
            petab.write_observable_df(
                problem.observable_df, Path(tmpdir, f"observables{i}.tsv")
            )

        petab_problem1 = petab.Problem.from_yaml(yaml_path)

        # test that we can load the problem from a dict with a custom base path
        yaml_config = petab.load_yaml(yaml_path)
        petab_problem2 = petab.Problem.from_yaml(yaml_config, base_path=tmpdir)

    for petab_problem in (petab_problem1, petab_problem2):
        assert petab_problem.measurement_df.shape[0] == 2
        assert petab_problem.observable_df.shape[0] == 2
        assert petab_problem.condition_df.shape[0] == 2
        assert petab_problem.experiment_df.shape[0] == 2


def test_modify_problem():
    """Test modifying a problem via the API."""
    problem = Problem()
    problem.add_condition("condition1", parameter1=1)
    problem.add_condition("condition2", parameter2=2)

    exp_condition_df = pd.DataFrame(
        data={
            CONDITION_ID: ["condition1", "condition2"],
            TARGET_ID: ["parameter1", "parameter2"],
            TARGET_VALUE: [1.0, 2.0],
        }
    )
    assert_frame_equal(
        problem.condition_df, exp_condition_df, check_dtype=False
    )

    problem.add_observable("observable1", "1")
    problem.add_observable("observable2", "2", noise_formula=2.2)

    exp_observable_df = pd.DataFrame(
        data={
            OBSERVABLE_ID: ["observable1", "observable2"],
            OBSERVABLE_FORMULA: [1, 2],
            NOISE_FORMULA: [np.nan, 2.2],
        }
    ).set_index([OBSERVABLE_ID])
    assert_frame_equal(
        problem.observable_df[[OBSERVABLE_FORMULA, NOISE_FORMULA]],
        exp_observable_df,
        check_dtype=False,
    )

    problem.add_parameter("parameter1", 1, 0, lb=1, ub=2)
    problem.add_parameter("parameter2", False, 2)

    exp_parameter_df = pd.DataFrame(
        data={
            PARAMETER_ID: ["parameter1", "parameter2"],
            ESTIMATE: [1, 0],
            NOMINAL_VALUE: [0.0, 2.0],
            LOWER_BOUND: [1.0, np.nan],
            UPPER_BOUND: [2.0, np.nan],
        }
    ).set_index([PARAMETER_ID])
    assert_frame_equal(
        problem.parameter_df[
            [ESTIMATE, NOMINAL_VALUE, LOWER_BOUND, UPPER_BOUND]
        ],
        exp_parameter_df,
        check_dtype=False,
    )

    problem.add_mapping("new_petab_id", "some_model_entity_id")

    exp_mapping_df = pd.DataFrame(
        data={
            PETAB_ENTITY_ID: ["new_petab_id"],
            MODEL_ENTITY_ID: ["some_model_entity_id"],
        }
    ).set_index([PETAB_ENTITY_ID])
    assert_frame_equal(problem.mapping_df, exp_mapping_df, check_dtype=False)
