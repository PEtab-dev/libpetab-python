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

    with pytest.warns(UserWarning, match=r"observables obs2, obs3 were"):
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
    with pytest.warns(UserWarning, match=r"observables obs_B were"):
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

    with pytest.raises(ValueError, match="Non-consecutive numbering"):
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


def _petab1to2_with_conditions(
    tmp_path, condition_df: pd.DataFrame, conditions: list[tuple[str, str]]
) -> Problem:
    """Convert a minimal PEtab v1 problem with the given condition table.

    ``conditions`` are (preequilibrationConditionId, simulationConditionId)
    pairs, one measurement per pair.
    """
    problem = v1.Problem(
        model=SbmlModel.from_antimony(
            "A = 1; B = 0; k1 = 0.5; A -> B; k1 * A"
        ),
        condition_df=v1.get_condition_df(condition_df),
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
                    v1.C.OBSERVABLE_ID: "obs_B",
                    v1.C.PREEQUILIBRATION_CONDITION_ID: [
                        preeq for preeq, _ in conditions
                    ],
                    v1.C.SIMULATION_CONDITION_ID: [
                        sim for _, sim in conditions
                    ],
                    v1.C.TIME: 1.0,
                    v1.C.MEASUREMENT: 0.5,
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
    )
    return petab1to2(problem.to_files_generic(tmp_path))


def _measured_periods(
    problem: Problem,
) -> list[list[tuple[float, list[str]]] | None]:
    """The (time, conditionIds) of the experiment periods per measurement."""
    return [
        None
        if m.experiment_id is None
        else [
            (p.time, p.condition_ids) for p in problem[m.experiment_id].periods
        ]
        for m in problem.measurements
    ]


def test_petab1to2_empty_conditions_preequilibration(tmp_path):
    """Pre-equilibration is kept if the condition table sets nothing."""
    problem = _petab1to2_with_conditions(
        tmp_path,
        pd.DataFrame({v1.C.CONDITION_ID: ["c0", "steady"]}),
        [("steady", "c0")],
    )

    assert problem.conditions == []
    assert _measured_periods(problem) == [
        [(v2.C.TIME_PREEQUILIBRATION, []), (0, [])]
    ]
    assert problem.experiments[0].has_preequilibration


def test_petab1to2_mixed_empty_conditions(tmp_path):
    """Conditions that set nothing are handled condition-wise."""
    problem = _petab1to2_with_conditions(
        tmp_path,
        pd.DataFrame(
            {
                v1.C.CONDITION_ID: ["c0", "c1", "preeq_set", "preeq_empty"],
                "A": [1.0, None, 2.0, None],
            }
        ),
        [
            ("preeq_set", "c0"),
            ("preeq_set", "c1"),
            ("preeq_empty", "c0"),
            ("preeq_empty", "c1"),
            ("", "c0"),
            ("", "c1"),
        ],
    )

    assert {c.id for c in problem.conditions} == {"c0", "preeq_set"}
    preeq = v2.C.TIME_PREEQUILIBRATION
    assert _measured_periods(problem) == [
        [(preeq, ["preeq_set"]), (0, ["c0"])],
        [(preeq, ["preeq_set"]), (0, [])],
        [(preeq, []), (0, ["c0"])],
        [(preeq, []), (0, [])],
        [(0, ["c0"])],
        # no pre-equilibration, no changes -> no experiment
        None,
    ]


def test_petab1to2_empty_conditions_no_preequilibration(tmp_path):
    """Without pre-equilibration, conditions that set nothing don't need an
    experiment."""
    problem = _petab1to2_with_conditions(
        tmp_path,
        pd.DataFrame({v1.C.CONDITION_ID: ["c0"]}),
        [("", "c0")],
    )

    assert problem.experiments == []
    assert _measured_periods(problem) == [None]


def test_petab1to2_validate(tmp_path):
    """Test that `validate=False` allows converting problems that fail
    linting for reasons unrelated to the conversion."""
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
    "ignore:.*Initialisation priors in parameter table are not supported.*:"
    "UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*Parameter scales are not supported in PEtab v2.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:.*does not support log10-transformed observables.*:UserWarning"
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
