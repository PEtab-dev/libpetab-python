import logging

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


def test_petab1to2_placeholder_order(tmp_path):
    """Test that placeholders are ordered by index, not as text, so that the
    positional overrides still match for 10+ placeholders."""
    obs_id = "obs_B"
    n_obs, n_noise = 12, 11
    observable_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: [obs_id],
            v1.C.OBSERVABLE_FORMULA: [
                " + ".join(
                    f"observableParameter{i}_{obs_id} * B^{i}"
                    for i in range(1, n_obs + 1)
                )
            ],
            v1.C.NOISE_FORMULA: [
                " + ".join(
                    f"noiseParameter{i}_{obs_id}"
                    for i in range(1, n_noise + 1)
                )
            ],
        }
    ).set_index(v1.C.OBSERVABLE_ID)
    # the i-th observable override is i, the i-th noise override is 100 + i
    measurement_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: [obs_id, obs_id],
            v1.C.SIMULATION_CONDITION_ID: ["c0", "c0"],
            v1.C.TIME: [1.0, 2.0],
            v1.C.MEASUREMENT: [0.3, 0.5],
            v1.C.OBSERVABLE_PARAMETERS: ";".join(
                str(i) for i in range(1, n_obs + 1)
            ),
            v1.C.NOISE_PARAMETERS: ";".join(
                str(100 + i) for i in range(1, n_noise + 1)
            ),
        }
    )
    condition_df = pd.DataFrame(
        data={v1.C.CONDITION_ID: ["c0"], "A": [1.0]}
    ).set_index(v1.C.CONDITION_ID)
    parameter_df = pd.DataFrame(
        data={
            v1.C.PARAMETER_ID: ["k1"],
            v1.C.PARAMETER_SCALE: [v1.C.LIN],
            v1.C.LOWER_BOUND: [1e-3],
            v1.C.UPPER_BOUND: [1e3],
            v1.C.NOMINAL_VALUE: [0.5],
            v1.C.ESTIMATE: [1],
        }
    ).set_index(v1.C.PARAMETER_ID)
    v1_problem = v1.Problem(
        model=v1.models.sbml_model.SbmlModel.from_antimony(
            "A -> B; k1 * A; A = 1; B = 0; k1 = 0.5"
        ),
        condition_df=condition_df,
        measurement_df=measurement_df,
        parameter_df=parameter_df,
        observable_df=observable_df,
    )

    problem = petab1to2(v1_problem.to_files_generic(tmp_path))

    observable = problem[obs_id]
    assert len(problem.measurements) == 2
    for measurement in problem.measurements:
        assert {
            str(placeholder): float(override)
            for placeholder, override in zip(
                observable.observable_placeholders,
                measurement.observable_parameters,
                strict=True,
            )
        } == {
            f"observableParameter{i}_{obs_id}": i for i in range(1, n_obs + 1)
        }
        assert {
            str(placeholder): float(override)
            for placeholder, override in zip(
                observable.noise_placeholders,
                measurement.noise_parameters,
                strict=True,
            )
        } == {
            f"noiseParameter{i}_{obs_id}": 100 + i
            for i in range(1, n_noise + 1)
        }


@pytest.mark.parametrize("type_", ["observable", "noise"])
def test_v1v2_observable_df_placeholder_gap(type_):
    """Test that non-consecutively numbered placeholders are rejected,
    because v1 overrides are positional."""
    formula = f"{type_}Parameter1_obs1 + {type_}Parameter3_obs1"
    observable_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: ["obs1"],
            v1.C.OBSERVABLE_FORMULA: [
                formula if type_ == "observable" else "a"
            ],
            v1.C.NOISE_FORMULA: [formula if type_ == "noise" else "1"],
        }
    ).set_index(v1.C.OBSERVABLE_ID)

    with pytest.raises(ValueError, match="not numbered consecutively"):
        v1v2_observable_df(observable_df)


def test_petab1to2_remote():
    """Test that we can upgrade a remote PEtab 1.0.0 problem."""
    yaml_url = (
        "https://cdn.jsdelivr.net/gh/PEtab-dev/petab_test_suite"
        "@main/petabtests/cases/v1.0.0/sbml/0001/_0001.yaml"
    )

    problem = petab1to2(yaml_url)
    assert isinstance(problem, Problem)
    assert len(problem.measurements)


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
