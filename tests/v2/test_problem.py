import tempfile
from pathlib import Path

import pandas as pd

import petab.v2 as petab
from petab.v2 import Problem
from petab.v2.C import (
    CONDITION_ID,
    MEASUREMENT,
    NOISE_FORMULA,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    SIMULATION_CONDITION_ID,
    TIME,
)


def test_load_remote():
    """Test loading remote files"""
    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/main/petabtests/cases/v2.0.0/sbml/0001/_0001.yaml"
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
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir, "problem.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_config)

        for i in (1, 2):
            condition_df = pd.DataFrame(
                {
                    CONDITION_ID: [f"condition{i}"],
                }
            )
            condition_df.set_index([CONDITION_ID], inplace=True)
            petab.write_condition_df(
                condition_df, Path(tmpdir, f"conditions{i}.tsv")
            )

            measurement_df = pd.DataFrame(
                {
                    SIMULATION_CONDITION_ID: [f"condition{i}"],
                    OBSERVABLE_ID: [f"observable{i}"],
                    TIME: [i],
                    MEASUREMENT: [1],
                }
            )
            petab.write_measurement_df(
                measurement_df, Path(tmpdir, f"measurements{i}.tsv")
            )

            observables_df = pd.DataFrame(
                {
                    OBSERVABLE_ID: [f"observable{i}"],
                    OBSERVABLE_FORMULA: [1],
                    NOISE_FORMULA: [1],
                }
            )
            petab.write_observable_df(
                observables_df, Path(tmpdir, f"observables{i}.tsv")
            )

        petab_problem1 = petab.Problem.from_yaml(yaml_path)

        # test that we can load the problem from a dict with a custom base path
        yaml_config = petab.load_yaml(yaml_path)
        petab_problem2 = petab.Problem.from_yaml(yaml_config, base_path=tmpdir)

    for petab_problem in (petab_problem1, petab_problem2):
        assert petab_problem.measurement_df.shape[0] == 2
        assert petab_problem.observable_df.shape[0] == 2
        assert petab_problem.condition_df.shape[0] == 2
