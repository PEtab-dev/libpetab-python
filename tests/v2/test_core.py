import tempfile
from pathlib import Path

import pytest
import sympy as sp
from pydantic import ValidationError
from sympy.abc import x, y

from petab.v2.core import *
from petab.v2.petab1to2 import petab1to2

example_dir_fujita = Path(__file__).parents[2] / "doc/example/example_Fujita"


def test_observables_table_round_trip():
    file = example_dir_fujita / "Fujita_observables.tsv"
    observables = ObservablesTable.from_tsv(file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "observables.tsv"
        observables.to_tsv(tmp_file)
        observables2 = ObservablesTable.from_tsv(tmp_file)
        assert observables == observables2


def test_conditions_table_round_trip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        petab1to2(example_dir_fujita / "Fujita.yaml", tmp_dir)
        file = Path(tmp_dir, "Fujita_experimentalCondition.tsv")
        conditions = ConditionsTable.from_tsv(file)
        tmp_file = Path(tmp_dir) / "conditions.tsv"
        conditions.to_tsv(tmp_file)
        conditions2 = ConditionsTable.from_tsv(tmp_file)
        assert conditions == conditions2


def test_experiment_add_periods():
    """Test operators for Experiment"""
    exp = Experiment(id="exp1")
    assert exp.periods == []

    p1 = ExperimentPeriod(time=0, condition_id="p1")
    p2 = ExperimentPeriod(time=1, condition_id="p2")
    p3 = ExperimentPeriod(time=2, condition_id="p3")
    exp += p1
    exp += p2

    assert exp.periods == [p1, p2]

    exp2 = exp + p3
    assert exp2.periods == [p1, p2, p3]
    assert exp.periods == [p1, p2]


def test_conditions_table_add_changes():
    conditions_table = ConditionsTable()
    assert conditions_table.conditions == []

    c1 = Condition(
        id="condition1",
        changes=[Change(target_id="k1", target_value=1)],
    )
    c2 = Condition(
        id="condition2",
        changes=[Change(target_id="k2", target_value=sp.sympify("2 * x"))],
    )

    conditions_table += c1
    conditions_table += c2

    assert conditions_table.conditions == [c1, c2]


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
    assert o.observable_placeholders == set()
    assert o.noise_placeholders == set()

    o = Observable(
        id="obs1",
        formula="observableParameter1_obs1",
        noise_formula="noiseParameter1_obs1",
    )
    assert o.observable_placeholders == {
        sp.Symbol("observableParameter1_obs1", real=True),
    }
    assert o.noise_placeholders == {
        sp.Symbol("noiseParameter1_obs1", real=True)
    }

    # TODO: this should raise an error
    #   (numbering is not consecutive / not starting from 1)
    # TODO: clarify if observableParameter0_obs1 would be allowed
    #  as regular parameter
    #
    # with pytest.raises(ValidationError):
    #  Observable(id="obs1", formula="observableParameter2_obs1")


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
    ExperimentPeriod(time=1, condition_id="p1")
    ExperimentPeriod(time="-inf", condition_id="p1")

    assert (
        ExperimentPeriod(time="1", condition_id="p1", non_petab=1).non_petab
        == 1
    )

    with pytest.raises(ValidationError, match="got inf"):
        ExperimentPeriod(time="inf", condition_id="p1")

    with pytest.raises(ValidationError, match="Invalid ID"):
        ExperimentPeriod(time=1, condition_id="1_condition")

    with pytest.raises(ValidationError, match="type=missing"):
        ExperimentPeriod(condition_id="condition")


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
    Experiment(
        id="experiment1", periods=[ExperimentPeriod(time=1, condition_id="c1")]
    )

    assert Experiment(id="experiment1", non_petab=1).non_petab == 1

    with pytest.raises(ValidationError, match="Field required"):
        Experiment()

    with pytest.raises(ValidationError, match="Invalid ID"):
        Experiment(id="experiment 1")


def test_conditions_table():
    assert ConditionsTable().free_symbols == set()

    assert (
        ConditionsTable(
            conditions=[
                Condition(
                    id="condition1",
                    changes=[Change(target_id="k1", target_value="true")],
                )
            ]
        ).free_symbols
        == set()
    )

    assert ConditionsTable(
        conditions=[
            Condition(
                id="condition1",
                changes=[Change(target_id="k1", target_value=x / y)],
            )
        ]
    ).free_symbols == {x, y}
