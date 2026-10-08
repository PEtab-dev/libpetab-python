import logging
from pathlib import Path

import pandas as pd
import pytest

from petab import v1, v2
from petab.v2 import Problem
from petab.v2.petab1to2 import petab1to2, v1v2_observable_df


def test_v1v2_observable_df_noise_distribution():
    """Test that noiseDistribution is correctly merged with
    observableTransformation."""
    observable_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: ["obs1", "obs2"],
            v1.C.OBSERVABLE_FORMULA: ["a", "b"],
            v1.C.NOISE_FORMULA: ["1", "1"],
            v1.C.OBSERVABLE_TRANSFORMATION: [v1.C.LIN, v1.C.LOG],
            v1.C.NOISE_DISTRIBUTION: [v1.C.NORMAL, v1.C.NORMAL],
        }
    ).set_index(v1.C.OBSERVABLE_ID)

    new_df = v1v2_observable_df(observable_df)

    assert list(new_df[v2.C.NOISE_DISTRIBUTION]) == [
        v2.C.NORMAL,
        v2.C.LOG_NORMAL,
    ]


def test_petab1to2_remote():
    """Test that we can upgrade a remote PEtab 1.0.0 problem."""
    yaml_url = (
        "https://cdn.jsdelivr.net/gh/PEtab-dev/petab_test_suite"
        "@main/petabtests/cases/v1.0.0/sbml/0001/_0001.yaml"
    )

    problem = petab1to2(yaml_url)
    assert isinstance(problem, Problem)
    assert len(problem.measurements)


def test_petab1to2_validate(tmp_path):
    """Test that `validate=False` allows converting problems that fail
    linting for reasons unrelated to the conversion."""
    from petab.v1.models.sbml_model import SbmlModel

    v1_problem = v1.Problem(
        model=SbmlModel.from_antimony(
            "A -> B; k1 * A; A = 1; B = 0; k1 = 0.5"
        ),
        condition_df=v1.get_condition_df(
            pd.DataFrame({v1.C.CONDITION_ID: ["c0"], "A": [1.0]})
        ),
        observable_df=v1.get_observable_df(
            pd.DataFrame(
                {
                    v1.C.OBSERVABLE_ID: ["obs_B"],
                    v1.C.OBSERVABLE_FORMULA: ["B"],
                    v1.C.NOISE_FORMULA: [0.1],
                }
            )
        ),
        measurement_df=v1.get_measurement_df(
            pd.DataFrame(
                {
                    v1.C.OBSERVABLE_ID: ["obs_B", "obs_B"],
                    v1.C.SIMULATION_CONDITION_ID: ["c0", "c0"],
                    v1.C.TIME: [1.0, 2.0],
                    v1.C.MEASUREMENT: [0.3, 0.5],
                    v1.C.DATASET_ID: ["d1", "d1"],
                }
            )
        ),
        parameter_df=v1.get_parameter_df(
            pd.DataFrame(
                {
                    v1.C.PARAMETER_ID: ["k1"],
                    v1.C.PARAMETER_SCALE: [v1.C.LIN],
                    v1.C.LOWER_BOUND: [1e-3],
                    v1.C.UPPER_BOUND: [1e3],
                    v1.C.NOMINAL_VALUE: [0.5],
                    v1.C.ESTIMATE: [1],
                }
            )
        ),
        # refers to a dataset that is not in the measurement table
        visualization_df=pd.DataFrame(
            {
                v1.C.PLOT_ID: ["p1"],
                v1.C.DATASET_ID: ["no_such_dataset"],
                v1.C.X_VALUES: [v1.C.TIME],
            }
        ),
    )
    yaml_file = v1_problem.to_files_generic(tmp_path / "v1")

    with pytest.raises(ValueError, match="does not pass linting"):
        petab1to2(yaml_file)

    problem = petab1to2(yaml_file, validate=False)
    assert isinstance(problem, Problem)
    assert len(problem.measurements) == 2

    output_dir = tmp_path / "v2"
    output_dir.mkdir()
    petab1to2(yaml_file, output_dir, validate=False)
    problem = Problem.from_yaml(output_dir / Path(yaml_file).name)
    assert len(problem.measurements) == 2


try:
    import benchmark_models_petab

    parametrize_or_skip = pytest.mark.parametrize(
        "problem_id", benchmark_models_petab.MODELS
    )
except ImportError:
    parametrize_or_skip = pytest.mark.skip(
        reason="benchmark_models_petab not installed"
    )


@pytest.mark.filterwarnings(
    "ignore:.*Using `log-normal` instead.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*Initialisation priors in parameter table are not supported.*:"
    "UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
@parametrize_or_skip
def test_benchmark_collection(problem_id):
    """Test that we can upgrade all benchmark collection models."""
    logging.basicConfig(level=logging.DEBUG)

    if problem_id == "Froehlich_CellSystems2018":
        # this is mostly about 6M sympifications in the condition table
        pytest.skip("Too slow. Re-enable once we are faster.")

    yaml_path = benchmark_models_petab.get_problem_yaml_path(problem_id)
    try:
        problem = petab1to2(yaml_path)
    except NotImplementedError as e:
        pytest.skip(str(e))
    assert isinstance(problem, Problem)
    assert len(problem.measurements)
