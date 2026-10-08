import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from petab import v1, v2
from petab.v1.calculate import calculate_llh as calculate_llh_v1
from petab.v1.models.sbml_model import SbmlModel
from petab.v1.priors import Prior
from petab.v2 import Problem
from petab.v2.calculate import calculate_llh as calculate_llh_v2
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


def test_v1v2_observable_df_log10():
    """Test that log10-transformed observables are converted to
    log-transformed ones with the noise formula scaled by ln(10)."""
    observable_df = pd.DataFrame(
        data={
            v1.C.OBSERVABLE_ID: ["obs1", "obs2", "obs3"],
            v1.C.OBSERVABLE_FORMULA: ["a", "b", "c"],
            v1.C.NOISE_FORMULA: [
                "noiseParameter1_obs1 + 0.1",
                "noiseParameter1_obs2 + 0.1",
                "noiseParameter1_obs3 + 0.1",
            ],
            v1.C.OBSERVABLE_TRANSFORMATION: [
                v1.C.LOG,
                v1.C.LOG10,
                v1.C.LOG10,
            ],
            v1.C.NOISE_DISTRIBUTION: [
                v1.C.NORMAL,
                v1.C.NORMAL,
                v1.C.LAPLACE,
            ],
        }
    ).set_index(v1.C.OBSERVABLE_ID)

    new_df = v1v2_observable_df(observable_df)

    assert list(new_df[v2.C.NOISE_DISTRIBUTION]) == [
        v2.C.LOG_NORMAL,
        v2.C.LOG_NORMAL,
        v2.C.LOG_LAPLACE,
    ]
    assert list(new_df[v2.C.NOISE_FORMULA]) == [
        "noiseParameter1_obs1 + 0.1",
        "log(10) * (noiseParameter1_obs2 + 0.1)",
        "log(10) * (noiseParameter1_obs3 + 0.1)",
    ]
    assert list(new_df[v2.C.NOISE_PLACEHOLDERS]) == [
        "noiseParameter1_obs1",
        "noiseParameter1_obs2",
        "noiseParameter1_obs3",
    ]


def _write_v1_problem(
    path: Path,
    *,
    observables: list[dict],
    measurements: list[dict],
    parameters: list[dict],
) -> str:
    """Write a PEtab v1 problem for the conversion A -> B with the given
    tables (as lists of rows) to ``path`` and return the YAML file name."""
    problem = v1.Problem(
        model=SbmlModel.from_antimony(
            "r1: A -> B; k1 * A; A = 1; B = 0; k1 = 0.5"
        ),
        condition_df=v1.get_condition_df(
            pd.DataFrame([{v1.C.CONDITION_ID: "c0", "A": 1.0}])
        ),
        observable_df=v1.get_observable_df(pd.DataFrame(observables)),
        measurement_df=v1.get_measurement_df(pd.DataFrame(measurements)),
        parameter_df=v1.get_parameter_df(pd.DataFrame(parameters)),
    )
    return problem.to_files_generic(path)


_K1 = {
    v1.C.PARAMETER_ID: "k1",
    v1.C.PARAMETER_SCALE: v1.C.LIN,
    v1.C.LOWER_BOUND: 1e-3,
    v1.C.UPPER_BOUND: 1e2,
    v1.C.NOMINAL_VALUE: 0.5,
    v1.C.ESTIMATE: 1,
}


@pytest.mark.parametrize("noise_distribution", [v1.C.NORMAL, v1.C.LAPLACE])
@pytest.mark.parametrize("noise_placeholder", [False, True])
def test_petab1to2_log10_noise_llh(
    tmp_path, noise_distribution, noise_placeholder
):
    """Test that the conversion of log10-transformed observables preserves
    the likelihood."""
    if noise_placeholder:
        noise_formula = "noiseParameter1_obs_B + offset"
        noise_parameters = "sigma_B"
        noise_parameter_rows = [
            {
                v1.C.PARAMETER_ID: "sigma_B",
                v1.C.PARAMETER_SCALE: v1.C.LIN,
                v1.C.LOWER_BOUND: 1e-3,
                v1.C.UPPER_BOUND: 1e2,
                v1.C.NOMINAL_VALUE: 0.1,
                v1.C.ESTIMATE: 1,
            },
            {
                v1.C.PARAMETER_ID: "offset",
                v1.C.PARAMETER_SCALE: v1.C.LIN,
                v1.C.LOWER_BOUND: 0,
                v1.C.UPPER_BOUND: 1,
                v1.C.NOMINAL_VALUE: 0.05,
                v1.C.ESTIMATE: 0,
            },
        ]
    else:
        noise_formula = "0.1"
        noise_parameters = ""
        noise_parameter_rows = []

    yaml_file = _write_v1_problem(
        tmp_path,
        observables=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "B",
                v1.C.OBSERVABLE_TRANSFORMATION: v1.C.LOG10,
                v1.C.NOISE_FORMULA: noise_formula,
                v1.C.NOISE_DISTRIBUTION: noise_distribution,
            }
        ],
        measurements=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.SIMULATION_CONDITION_ID: "c0",
                v1.C.TIME: t,
                v1.C.MEASUREMENT: y,
                v1.C.NOISE_PARAMETERS: noise_parameters,
            }
            for t, y in ((1.0, 0.3), (2.0, 0.5))
        ],
        parameters=[_K1, *noise_parameter_rows],
    )
    v1_problem = v1.Problem.from_yaml(yaml_file)
    v2_problem = petab1to2(yaml_file)

    def simulate(measurement_df):
        return measurement_df.rename(
            columns={v1.C.MEASUREMENT: v1.C.SIMULATION}
        ).assign(**{v1.C.SIMULATION: [0.35, 0.45]})

    llh_v1 = calculate_llh_v1(
        v1_problem.measurement_df,
        simulate(v1_problem.measurement_df),
        v1_problem.observable_df,
        v1_problem.parameter_df,
    )
    llh_v2 = calculate_llh_v2(
        v2_problem.measurement_df,
        simulate(v2_problem.measurement_df),
        v2_problem.observable_df,
        v2_problem.parameter_df,
    )
    assert llh_v2 == pytest.approx(llh_v1, rel=1e-12)


@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
@pytest.mark.parametrize(
    "parameter_scale, prior_type",
    [
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_NORMAL),
        (v1.C.LOG10, v1.C.PARAMETER_SCALE_LAPLACE),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_NORMAL),
        (v1.C.LOG, v1.C.PARAMETER_SCALE_LAPLACE),
        (v1.C.LIN, v1.C.PARAMETER_SCALE_NORMAL),
    ],
)
def test_petab1to2_parameter_scale_priors(
    tmp_path, parameter_scale, prior_type
):
    """Test that parameterScale{Normal,Laplace} priors are converted to
    priors with the same density."""
    k1 = _K1 | {
        v1.C.PARAMETER_SCALE: parameter_scale,
        v1.C.OBJECTIVE_PRIOR_TYPE: prior_type,
        v1.C.OBJECTIVE_PRIOR_PARAMETERS: "-0.5;0.7",
    }
    yaml_file = _write_v1_problem(
        tmp_path,
        observables=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.OBSERVABLE_FORMULA: "B",
                v1.C.NOISE_FORMULA: "0.1",
            }
        ],
        measurements=[
            {
                v1.C.OBSERVABLE_ID: "obs_B",
                v1.C.SIMULATION_CONDITION_ID: "c0",
                v1.C.TIME: 1.0,
                v1.C.MEASUREMENT: 0.3,
            }
        ],
        parameters=[k1],
    )
    v1_prior = Prior.from_par_dict(k1, type_="objective")
    v2_prior = petab1to2(yaml_file)["k1"].prior_dist

    x = np.logspace(-3, 2, 21)
    assert v2_prior.pdf(x) == pytest.approx(v1_prior.pdf(x), rel=1e-12)


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
