import tempfile
from pathlib import Path

import sympy as sp

from petab.v2.core import (
    Change,
    Condition,
    ConditionsTable,
    Experiment,
    ExperimentPeriod,
    ObservablesTable,
)
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
