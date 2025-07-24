import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sympy as sp
from pandas.testing import assert_frame_equal
from pydantic import AnyUrl, ValidationError
from sympy.abc import x, y

import petab.v2 as petab
from petab.v2 import C, Problem
from petab.v2.C import (
    CONDITION_ID,
    ESTIMATE,
    LOWER_BOUND,
    MODEL_ENTITY_ID,
    NAME,
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
from petab.v2.core import *
from petab.v2.petab1to2 import petab1to2

example_dir_fujita = Path(__file__).parents[2] / "doc/example/example_Fujita"


def test_observable_table_round_trip():
    file = example_dir_fujita / "Fujita_observables.tsv"
    observables = ObservableTable.from_tsv(file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "observables.tsv"
        observables.to_tsv(tmp_file)
        observables2 = ObservableTable.from_tsv(tmp_file)
        assert observables == observables2


def test_condition_table_round_trip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        petab1to2(example_dir_fujita / "Fujita.yaml", tmp_dir)
        file = Path(tmp_dir, "Fujita_experimentalCondition.tsv")
        conditions = ConditionTable.from_tsv(file)
        tmp_file = Path(tmp_dir) / "conditions.tsv"
        conditions.to_tsv(tmp_file)
        conditions2 = ConditionTable.from_tsv(tmp_file)
        assert conditions == conditions2


def test_experiment_add_periods():
    """Test operators for Experiment"""
    exp = Experiment(id="exp1")
    assert exp.periods == []

    p1 = ExperimentPeriod(time=0, condition_ids=["p1"])
    p2 = ExperimentPeriod(time=1, condition_ids=["p2"])
    p3 = ExperimentPeriod(time=2, condition_ids=["p3"])
    exp += p1
    exp += p2

    assert exp.periods == [p1, p2]

    exp2 = exp + p3
    assert exp2.periods == [p1, p2, p3]
    assert exp.periods == [p1, p2]


def test_condition_table_add_changes():
    condition_table = ConditionTable()
    assert condition_table.conditions == []

    c1 = Condition(
        id="condition1",
        changes=[Change(target_id="k1", target_value=1)],
    )
    c2 = Condition(
        id="condition2",
        changes=[Change(target_id="k2", target_value=sp.sympify("2 * x"))],
    )

    condition_table += c1
    condition_table += c2

    assert condition_table.conditions == [c1, c2]


def test_measurments():
    Measurement(
        observable_id="obs1", time=1, experiment_id="exp1", measurement=1
    )
    Measurement(
        observable_id="obs1", time="1", experiment_id="exp1", measurement="1"
    )
    Measurement(
        observable_id="obs1", time="inf", experiment_id="exp1", measurement="1"
    )

    Measurement(
        observable_id="obs1",
        time=1,
        experiment_id="exp1",
        measurement=1,
        observable_parameters=["p1"],
        noise_parameters=["n1"],
    )

    Measurement(
        observable_id="obs1",
        time=1,
        experiment_id="exp1",
        measurement=1,
        observable_parameters=[1],
        noise_parameters=[2],
    )

    Measurement(
        observable_id="obs1",
        time=1,
        experiment_id="exp1",
        measurement=1,
        observable_parameters=[sp.sympify("x ** y")],
        noise_parameters=[sp.sympify("x ** y")],
    )

    assert (
        Measurement(
            observable_id="obs1",
            time=1,
            experiment_id="exp1",
            measurement=1,
            non_petab=1,
        ).non_petab
        == 1
    )

    with pytest.raises(ValidationError, match="got -inf"):
        Measurement(
            observable_id="obs1",
            time="-inf",
            experiment_id="exp1",
            measurement=1,
        )

    with pytest.raises(ValidationError, match="Invalid ID"):
        Measurement(
            observable_id="1_obs", time=1, experiment_id="exp1", measurement=1
        )

    with pytest.raises(ValidationError, match="Invalid ID"):
        Measurement(
            observable_id="obs", time=1, experiment_id=" exp1", measurement=1
        )


def test_observable():
    Observable(id="obs1", formula=x + y)
    Observable(id="obs1", formula="x + y", noise_formula="x + y")
    Observable(id="obs1", formula=1, noise_formula=2)
    Observable(
        id="obs1",
        formula="x + y",
        noise_formula="x + y",
        observable_parameters=["p1"],
        noise_parameters=["n1"],
    )
    Observable(
        id="obs1",
        formula=sp.sympify("x + y"),
        noise_formula=sp.sympify("x + y"),
        observable_parameters=[sp.Symbol("p1")],
        noise_parameters=[sp.Symbol("n1")],
    )
    assert Observable(id="obs1", formula="x + y", non_petab=1).non_petab == 1

    o = Observable(id="obs1", formula=x + y)
    assert o.observable_placeholders == []
    assert o.noise_placeholders == []

    o = Observable(
        id="obs1",
        formula="observableParameter1_obs1",
        noise_formula="noiseParameter1_obs1",
        observable_placeholders="observableParameter1_obs1",
        noise_placeholders="noiseParameter1_obs1",
    )
    assert o.observable_placeholders == [
        sp.Symbol("observableParameter1_obs1", real=True),
    ]
    assert o.noise_placeholders == [
        sp.Symbol("noiseParameter1_obs1", real=True)
    ]


def test_change():
    Change(target_id="k1", target_value=1)
    Change(target_id="k1", target_value="x * y")

    assert (
        Change(target_id="k1", target_value=x * y, non_petab="foo").non_petab
        == "foo"
    )
    with pytest.raises(ValidationError, match="Invalid ID"):
        Change(target_id="1_k", target_value=x)

    with pytest.raises(ValidationError, match="input_value=None"):
        Change(target_id="k1", target_value=None)


def test_period():
    ExperimentPeriod(time=0)
    ExperimentPeriod(time=1, condition_ids=["p1"])
    ExperimentPeriod(time="-inf", condition_ids=["p1"])

    assert (
        ExperimentPeriod(time="1", condition_id="p1", non_petab=1).non_petab
        == 1
    )

    with pytest.raises(ValidationError, match="got inf"):
        ExperimentPeriod(time="inf", condition_ids=["p1"])

    with pytest.raises(ValidationError, match="Invalid conditionId"):
        ExperimentPeriod(time=1, condition_ids=["1_condition"])

    with pytest.raises(ValidationError, match="type=missing"):
        ExperimentPeriod(condition_ids=["condition"])


def test_parameter():
    Parameter(id="k1", lb=1, ub=2)
    Parameter(id="k1", estimate=False, nominal_value=1)

    assert Parameter(id="k1", lb=1, ub=2, non_petab=1).non_petab == 1

    with pytest.raises(ValidationError, match="Invalid ID"):
        Parameter(id="1_k", lb=1, ub=2)

    with pytest.raises(ValidationError, match="upper"):
        Parameter(id="k1", lb=1)

    with pytest.raises(ValidationError, match="lower"):
        Parameter(id="k1", ub=1)

    with pytest.raises(ValidationError, match="less than"):
        Parameter(id="k1", lb=2, ub=1)


def test_experiment():
    Experiment(id="experiment1")

    # extra fields allowed
    assert Experiment(id="experiment1", non_petab=1).non_petab == 1

    # ID required
    with pytest.raises(ValidationError, match="Field required"):
        Experiment()

    # valid ID required
    with pytest.raises(ValidationError, match="Invalid ID"):
        Experiment(id="experiment 1")

    periods = [
        ExperimentPeriod(time=C.TIME_PREEQUILIBRATION, condition_ids=["c1"]),
        ExperimentPeriod(time=-1, condition_id="c1"),
        ExperimentPeriod(time=1, condition_id="c1"),
    ]
    e = Experiment(id="experiment1", periods=list(reversed(periods)))

    assert e.has_preequilibration is True

    assert e.sorted_periods == periods
    assert e.periods != periods

    e.sort_periods()
    assert e.periods == periods

    e.periods.pop(0)
    assert e.has_preequilibration is False


def test_condition_table():
    assert ConditionTable().free_symbols == set()

    assert (
        ConditionTable(
            conditions=[
                Condition(
                    id="condition1",
                    changes=[Change(target_id="k1", target_value="true")],
                )
            ]
        ).free_symbols
        == set()
    )

    assert ConditionTable(
        conditions=[
            Condition(
                id="condition1",
                changes=[Change(target_id="k1", target_value=x / y)],
            )
        ]
    ).free_symbols == {x, y}


def test_load_remote():
    """Test loading remote files"""
    from jsonschema.exceptions import ValidationError

    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/main/petabtests/cases/v2.0.0/sbml/0010/_0010.yaml"
    )

    try:
        petab_problem = Problem.from_yaml(yaml_url)

        assert (
            petab_problem.measurement_df is not None
            and not petab_problem.measurement_df.empty
        )

        assert petab_problem.validate() == []
    except ValidationError:
        # FIXME: Until v2 is finalized, the format of the tests will often be
        # out of sync with the schema.
        # Ignore validation errors for now.
        pass


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
    parameter_files: []
    condition_files: [conditions1.tsv, conditions2.tsv]
    measurement_files: [measurements1.tsv, measurements2.tsv]
    observable_files: [observables1.tsv, observables2.tsv]
    model_files: {}
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
        problem.observable_df[[OBSERVABLE_FORMULA, NOISE_FORMULA]].map(
            lambda x: float(x) if x != "" else None
        ),
        exp_observable_df,
        check_dtype=False,
    )

    problem.add_parameter("parameter1", True, 0, lb=1, ub=2)
    problem.add_parameter("parameter2", False, 2)

    exp_parameter_df = pd.DataFrame(
        data={
            PARAMETER_ID: ["parameter1", "parameter2"],
            ESTIMATE: ["true", "false"],
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
            NAME: [None],
        }
    ).set_index([PETAB_ENTITY_ID])
    assert_frame_equal(problem.mapping_df, exp_mapping_df, check_dtype=False)


def test_sample_startpoint_shape():
    """Test startpoint sampling."""
    problem = Problem()
    problem += Parameter(id="p1", estimate=True, lb=1, ub=2)
    problem += Parameter(
        id="p2",
        estimate=True,
        lb=2,
        ub=3,
        prior_distribution="normal",
        prior_parameters=[2.5, 0.5],
    )
    problem += Parameter(id="p3", estimate=False, nominal_value=1)

    n_starts = 10
    sp = problem.sample_parameter_startpoints(n_starts=n_starts)
    assert sp.shape == (n_starts, 2)


def test_problem_config_paths():
    """Test handling of URLS and local paths in ProblemConfig."""

    pc = petab.ProblemConfig(
        parameter_files=["https://example.com/params.tsv"],
        condition_files=["conditions.tsv"],
        measurement_files=["measurements.tsv"],
        observable_files=["observables.tsv"],
        experiment_files=["experiments.tsv"],
    )
    assert isinstance(pc.parameter_files[0], AnyUrl)
    assert isinstance(pc.condition_files[0], Path)
    assert isinstance(pc.measurement_files[0], Path)
    assert isinstance(pc.observable_files[0], Path)
    assert isinstance(pc.experiment_files[0], Path)

    # Auto-convert to Path on assignment
    pc.parameter_files = ["foo.tsv"]
    assert isinstance(pc.parameter_files[0], Path)

    # We can't easily intercept mutations to the list:
    #  pc.parameter_files[0] = "foo.tsv"
    #  assert isinstance(pc.parameter_files[0], Path)
    # see also https://github.com/pydantic/pydantic/issues/8575


def test_get_changes_for_period():
    """Test getting changes for a specific period."""
    problem = Problem()
    ch1 = Change(target_id="target1", target_value=1.0)
    ch2 = Change(target_id="target2", target_value=2.0)
    ch3 = Change(target_id="target3", target_value=3.0)
    cond1 = Condition(id="condition1_1", changes=[ch1])
    cond2 = Condition(id="condition1_2", changes=[ch2])
    cond3 = Condition(id="condition2", changes=[ch3])
    problem += cond1
    problem += cond2
    problem += cond3

    p1 = ExperimentPeriod(
        id="p1", time=0, condition_ids=["condition1_1", "condition1_2"]
    )
    p2 = ExperimentPeriod(id="p2", time=1, condition_ids=["condition2"])
    problem += Experiment(
        id="exp1",
        periods=[p1, p2],
    )
    assert problem.get_changes_for_period(p1) == [ch1, ch2]
    assert problem.get_changes_for_period(p2) == [ch3]


def test_get_measurements_for_experiment():
    """Test getting measurements for an experiment."""
    problem = Problem()
    problem += Condition(
        id="condition1",
        changes=[Change(target_id="target1", target_value=1.0)],
    )
    problem += Condition(
        id="condition2",
        changes=[Change(target_id="target2", target_value=2.0)],
    )

    e1 = Experiment(
        id="exp1",
        periods=[
            ExperimentPeriod(id="p1", time=0, condition_ids=["condition1"]),
        ],
    )
    e2 = Experiment(
        id="exp2",
        periods=[
            ExperimentPeriod(id="p2", time=1, condition_ids=["condition2"]),
        ],
    )
    problem += e1
    problem += e2

    m1 = Measurement(
        observable_id="observable1",
        experiment_id="exp1",
        time=0,
        measurement=10.0,
    )
    m2 = Measurement(
        observable_id="observable2",
        experiment_id="exp1",
        time=1,
        measurement=20.0,
    )
    m3 = Measurement(
        observable_id="observable3",
        experiment_id="exp2",
        time=1,
        measurement=30.0,
    )
    problem += m1
    problem += m2
    problem += m3

    assert problem.get_measurements_for_experiment(e1) == [m1, m2]
    assert problem.get_measurements_for_experiment(e2) == [m3]
