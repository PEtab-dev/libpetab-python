import logging

import pandas as pd
import pytest

from petab import v1, v2
from petab.v1 import Problem as v1Problem
from petab.v2 import Problem
from petab.v2.models.sbml_model import SbmlModel
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


def test_petab1to2_assignments_to_experiments(tmp_path):
    problem = v1Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        k1 = piecewise(1, time < 10, 2);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_condition("c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", "c1", time=0.0, measurement=10.0)
    problem.add_measurement("obs_a", "c1", time=20.0, measurement=5.0)
    problem.add_parameter("k2", estimate=True, scale="lin", lb=1e-3, ub=1e3)

    yaml_path = problem.to_files_generic(prefix_path=tmp_path)

    converted = petab1to2(yaml_path, assignments_to_experiments=True)

    # the value before the switch is kept in the model, so that only the
    #  switch itself needs a condition
    assert converted.model.sbml_model.getParameter("k1").getValue() == 1.0

    experiments_expected = pd.DataFrame(
        data={
            v2.C.EXPERIMENT_ID: ["experiment____c1"],
            v2.C.TIME: [10.0],
            v2.C.CONDITION_ID: ["cond_0"],
        }
    )

    conditions_expected = pd.DataFrame(
        data={
            v2.C.CONDITION_ID: ["cond_0"],
            v2.C.TARGET_ID: ["k1"],
            v2.C.TARGET_VALUE: [2.0],
        }
    )

    assert converted.experiment_df.equals(experiments_expected)
    assert converted.condition_df.equals(conditions_expected)
    assert converted.measurement_df[v2.C.EXPERIMENT_ID].tolist() == [
        "experiment____c1",
        "experiment____c1",
    ]


def test_petab1to2_assignments_to_experiments_nested(tmp_path):
    """Test conversion of piecewise assignments that are nested inside a
    larger expression and switch at a time given by some expression."""
    problem = v1Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        level = 3;
        k1 := level * piecewise(0, time - 10 < 0, 1);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_condition("c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", "c1", time=0.0, measurement=10.0)
    problem.add_parameter("k2", estimate=True, scale="lin", lb=1e-3, ub=1e3)

    yaml_path = problem.to_files_generic(prefix_path=tmp_path)

    converted = petab1to2(yaml_path, assignments_to_experiments=True)

    assert converted.model.sbml_model.getParameter("k1").getValue() == 0.0

    # `level` is not a parameter table entry, so it must not appear in the
    #  first period -- the pre-dose value is kept as a condition here
    experiments_expected = pd.DataFrame(
        data={
            v2.C.EXPERIMENT_ID: ["experiment____c1", "experiment____c1"],
            v2.C.TIME: [0.0, 10.0],
            v2.C.CONDITION_ID: ["cond_0", "cond_1"],
        }
    )

    # the factor outside the piecewise expression is preserved
    conditions_expected = pd.DataFrame(
        data={
            v2.C.CONDITION_ID: ["cond_0", "cond_1"],
            v2.C.TARGET_ID: ["k1", "k1"],
            v2.C.TARGET_VALUE: [0.0, "level"],
        }
    )

    assert converted.experiment_df.equals(experiments_expected)
    assert converted.condition_df.equals(conditions_expected)


def test_petab1to2_assignments_to_experiments_condition_specific(tmp_path):
    """Test conversion of piecewise assignments whose switching times and
    values are set by the condition table."""
    problem = v1Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        dose = 0;
        dose_time = 0;
        k1 := dose * piecewise(0, time - dose_time < 0, 1);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_condition("c1", dose=2.0, dose_time=10.0)
    problem.add_condition("c2", dose=3.0, dose_time=20.0)
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", "c1", time=30.0, measurement=10.0)
    problem.add_measurement("obs_a", "c2", time=30.0, measurement=5.0)
    problem.add_parameter("k2", estimate=True, scale="lin", lb=1e-3, ub=1e3)

    yaml_path = problem.to_files_generic(prefix_path=tmp_path)

    converted = petab1to2(yaml_path, assignments_to_experiments=True)

    # the pre-dose value is the same for both conditions and is kept in the
    #  model; each condition gets its own experiment and switching time
    assert converted.model.sbml_model.getParameter("k1").getValue() == 0.0

    experiments_expected = pd.DataFrame(
        data={
            v2.C.EXPERIMENT_ID: ["experiment____c1", "experiment____c2"],
            v2.C.TIME: [10.0, 20.0],
            v2.C.CONDITION_ID: ["cond_0", "cond_1"],
        }
    )

    # `dose` and `dose_time` are covered by the new conditions
    conditions_expected = pd.DataFrame(
        data={
            v2.C.CONDITION_ID: ["cond_0", "cond_1"],
            v2.C.TARGET_ID: ["k1", "k1"],
            v2.C.TARGET_VALUE: [2.0, 3.0],
        }
    )

    assert converted.experiment_df.equals(experiments_expected)
    assert converted.condition_df.equals(conditions_expected)
    assert converted.measurement_df[v2.C.EXPERIMENT_ID].tolist() == [
        "experiment____c1",
        "experiment____c2",
    ]


def test_petab1to2_assignments_to_experiments_preequilibration(tmp_path):
    """Test that the converted assignments are applied during
    preequilibration, using the preequilibration condition's values."""
    problem = v1Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        dose = 0;
        dose_time = 0;
        k1 := dose * piecewise(0, time - dose_time < 0, 1);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_condition("c1", dose=2.0, dose_time=10.0)
    # the dose is present from the start, i.e. also during preequilibration
    problem.add_condition("c2", dose=3.0, dose_time=0.0)
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", "c1", time=30.0, measurement=10.0, preeq_cond_id="c2"
    )
    problem.add_parameter("k2", estimate=True, scale="lin", lb=1e-3, ub=1e3)

    yaml_path = problem.to_files_generic(prefix_path=tmp_path)

    converted = petab1to2(yaml_path, assignments_to_experiments=True)

    # the dose is applied during preequilibration, so the reset to the
    #  pre-dose value at t=0 is not redundant here and must be kept
    experiments_expected = pd.DataFrame(
        data={
            v2.C.EXPERIMENT_ID: ["experiment__c2___c1"] * 3,
            v2.C.TIME: [v2.C.TIME_PREEQUILIBRATION, 0.0, 10.0],
            v2.C.CONDITION_ID: ["cond_2", "cond_0", "cond_1"],
        }
    )

    conditions_expected = pd.DataFrame(
        data={
            v2.C.CONDITION_ID: ["cond_0", "cond_1", "cond_2"],
            v2.C.TARGET_ID: ["k1", "k1", "k1"],
            v2.C.TARGET_VALUE: [0.0, 2.0, 3.0],
        }
    )

    assert converted.experiment_df.equals(experiments_expected)
    assert converted.condition_df.equals(conditions_expected)
