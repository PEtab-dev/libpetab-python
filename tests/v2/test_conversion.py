import logging

import pandas as pd
import pytest

from petab import v1, v2
from petab.v1.models.sbml_model import SbmlModel
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
