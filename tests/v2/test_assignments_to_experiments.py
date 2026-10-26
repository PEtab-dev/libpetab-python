"""Tests for `petab.v2.converters.assignments_to_experiments`."""

import libsbml
import numpy as np
import pandas as pd
import pytest

from petab import v2
from petab.v1.models.sbml_model import SbmlModel
from petab.v2 import Problem
from petab.v2.converters import (
    DEFAULT_EXPERIMENT_ID,
    ExperimentsToSbmlConverter,
    assignments_to_experiments,
)
from petab.v2.math import sympify_petab


@pytest.mark.parametrize(
    "condition, before, after",
    [("time < 10", 1.0, 2.0), ("10 < time", 2.0, 1.0)],
)
def test_assignments_to_experiments(condition, before, after):
    """Test that switches in either direction are converted."""
    problem = Problem()
    ant_model = f"""
        model conversion
        species A, B;
        A = 10;
        B = 0;
        k1 := piecewise(1, {condition}, 2);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=0.0, measurement=10.0)
    problem.add_measurement("obs_a", time=20.0, measurement=5.0)
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    # the value before the switch is kept in the model, so that only the
    #  switch itself needs a condition; the experiment still starts at 0,
    #  where the first measurement is
    assert converted.model.sbml_model.getParameter("k1").getValue() == before

    experiments_expected = pd.DataFrame(
        data={
            v2.C.EXPERIMENT_ID: [DEFAULT_EXPERIMENT_ID] * 2,
            v2.C.TIME: [0.0, 10.0],
            v2.C.CONDITION_ID: ["", "cond_0"],
        }
    )

    conditions_expected = pd.DataFrame(
        data={
            v2.C.CONDITION_ID: ["cond_0"],
            v2.C.TARGET_ID: ["k1"],
            v2.C.TARGET_VALUE: [after],
        }
    )

    assert converted.experiment_df.equals(experiments_expected)
    assert converted.condition_df.equals(conditions_expected)
    assert converted.measurement_df[v2.C.EXPERIMENT_ID].tolist() == [
        DEFAULT_EXPERIMENT_ID,
        DEFAULT_EXPERIMENT_ID,
    ]


def test_assignments_to_experiments_unused_condition_id():
    """Test that new condition IDs don't clash with other IDs."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("""
        species A = 10;
        k1 := piecewise(1, time < 10, 2);
        R1: A -> ; k1 * A;
        """)
    problem.add_observable("obs_a", formula="A", noise_formula="cond_0")
    problem.add_measurement("obs_a", time=20.0, measurement=5.0)
    problem.add_parameter("cond_0", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    assert [c.id for c in converted.conditions] == ["cond_0_1"]
    # the experiment starts at 0, where nothing changes
    assert [
        (p.time, p.condition_ids) for p in converted.experiments[0].periods
    ] == [(0.0, []), (10.0, ["cond_0_1"])]


def test_assignments_to_experiments_nested():
    """Test conversion of piecewise assignments that are nested inside a
    larger expression and switch at a time given by some expression."""
    problem = Problem()
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

    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=0.0, measurement=10.0)
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    assert converted.model.sbml_model.getParameter("k1").getValue() == 0.0

    # `level` is not a parameter table entry, so it must not appear in the
    #  first period -- the switch may refer to it, since the experiment
    #  starts with an empty period at 0
    experiments_expected = pd.DataFrame(
        data={
            v2.C.EXPERIMENT_ID: [DEFAULT_EXPERIMENT_ID] * 2,
            v2.C.TIME: [0.0, 10.0],
            v2.C.CONDITION_ID: ["", "cond_0"],
        }
    )

    # the factor outside the piecewise expression is preserved
    conditions_expected = pd.DataFrame(
        data={
            v2.C.CONDITION_ID: ["cond_0"],
            v2.C.TARGET_ID: ["k1"],
            v2.C.TARGET_VALUE: ["level"],
        }
    )

    assert converted.experiment_df.equals(experiments_expected)
    assert converted.condition_df.equals(conditions_expected)


def test_assignments_to_experiments_parameter_table_default():
    """Test that a pre-switch value that refers to parameter table symbols is
    set at the start, since the model only has a number for it."""
    problem = Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        d1 = 1;
        d2 = 2;
        k1 := piecewise(d1, time < 10, d2);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=20.0, measurement=5.0)
    problem.add_parameter(
        "d1", estimate=True, lb=1e-3, ub=1e3, nominal_value=3
    )
    problem.add_parameter(
        "d2", estimate=True, lb=1e-3, ub=1e3, nominal_value=4
    )
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    experiments_expected = pd.DataFrame(
        data={
            v2.C.EXPERIMENT_ID: [DEFAULT_EXPERIMENT_ID] * 2,
            v2.C.TIME: [0.0, 10.0],
            v2.C.CONDITION_ID: ["cond_0", "cond_1"],
        }
    )

    conditions_expected = pd.DataFrame(
        data={
            v2.C.CONDITION_ID: ["cond_0", "cond_1"],
            v2.C.TARGET_ID: ["k1", "k1"],
            v2.C.TARGET_VALUE: ["d1", "d2"],
        }
    )

    assert converted.experiment_df.equals(experiments_expected)
    assert converted.condition_df.equals(conditions_expected)


def test_assignments_to_experiments_condition_specific():
    """Test conversion of piecewise assignments whose switching times and
    values are set by the condition table."""
    problem = Problem()
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
    problem.add_experiment("experiment____c1", 0, "c1")
    problem.add_experiment("experiment____c2", 0, "c2")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="experiment____c1", time=30.0, measurement=10.0
    )
    problem.add_measurement(
        "obs_a", experiment_id="experiment____c2", time=30.0, measurement=5.0
    )
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    # the pre-dose value is the same for both conditions and is kept in the
    #  model; each condition gets its own experiment and switching time, and
    #  each experiment still starts at 0, where its condition is now empty
    assert converted.model.sbml_model.getParameter("k1").getValue() == 0.0

    experiments_expected = pd.DataFrame(
        data={
            v2.C.EXPERIMENT_ID: ["experiment____c1"] * 2
            + ["experiment____c2"] * 2,
            v2.C.TIME: [0.0, 10.0, 0.0, 20.0],
            v2.C.CONDITION_ID: ["", "cond_0", "", "cond_1"],
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


def _simulate(problem, experiment_id, timepoints):
    """Simulate an experiment in BasiCO, after converting the experiments
    to SBML events, and return the trajectory of `A`."""
    import basico

    converted = ExperimentsToSbmlConverter(problem).convert()
    indicator = converted.model.sbml_model.getParameter(
        ExperimentsToSbmlConverter.get_experiment_indicator(experiment_id)
    )
    if indicator is not None:
        indicator.setValue(1)
    basico.load_model(converted.model.to_sbml_str())
    result = basico.run_time_course(
        values=timepoints, r_tol=1e-10, a_tol=1e-12
    )
    return result["A"].to_numpy()


@pytest.mark.parametrize("by_condition", [False, True])
def test_assignments_to_experiments_simulate_before_switch(by_condition):
    """Test that a converted experiment starts at 0, not at its first switch,
    if nothing changes at 0: without an experiment, or if the condition at
    0 only sets symbols of the converted assignment. Simulations before and
    after the switch match the original problem's."""
    problem = Problem()
    ant_model = f"""
        model conversion
        species A, B;
        A = 10;
        B = 0;
        t_switch = 0;
        k1 := piecewise(1, time < {"t_switch" if by_condition else 5}, 2);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)
    if by_condition:
        experiment_id = "experiment____c1"
        problem.add_condition("c1", t_switch=3.0)
        problem.add_experiment(experiment_id, 0, "c1")
    else:
        experiment_id = DEFAULT_EXPERIMENT_ID
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    timepoints = [0.0, 1.0, 2.0, 4.0, 6.0, 10.0]
    for t in timepoints:
        problem.add_measurement(
            "obs_a",
            experiment_id=experiment_id if by_condition else None,
            time=t,
            measurement=5.0,
        )
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    np.testing.assert_allclose(
        _simulate(converted, experiment_id, timepoints),
        _simulate(problem, experiment_id, timepoints),
        rtol=1e-6,
    )
    (experiment,) = converted.experiments
    assert experiment.id == experiment_id
    assert [(p.time, p.condition_ids) for p in experiment.periods] == [
        (0.0, []),
        (3.0 if by_condition else 5.0, ["cond_0"]),
    ]


def test_assignments_to_experiments_preequilibration():
    """Test that the converted assignments are applied during
    preequilibration, using the preequilibration condition's values."""
    problem = Problem()
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
    problem.add_experiment(
        "experiment__c2___c1", v2.C.TIME_PREEQUILIBRATION, "c2", 0, "c1"
    )
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a",
        experiment_id="experiment__c2___c1",
        time=30.0,
        measurement=10.0,
    )
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    with pytest.warns(UserWarning, match="during preequilibration"):
        converted = assignments_to_experiments(problem)

    # the dose is applied during preequilibration, so the reset to the
    #  pre-dose value at t=0 is not redundant here and must be kept
    experiments_expected = pd.DataFrame(
        data={
            v2.C.EXPERIMENT_ID: ["experiment__c2___c1"] * 3,
            v2.C.TIME: [v2.C.TIME_PREEQUILIBRATION, 0.0, 10.0],
            v2.C.CONDITION_ID: ["cond_0", "cond_1", "cond_2"],
        }
    )

    conditions_expected = pd.DataFrame(
        data={
            v2.C.CONDITION_ID: ["cond_0", "cond_1", "cond_2"],
            v2.C.TARGET_ID: ["k1", "k1", "k1"],
            v2.C.TARGET_VALUE: [3.0, 0.0, 2.0],
        }
    )

    assert converted.experiment_df.equals(experiments_expected)
    assert converted.condition_df.equals(conditions_expected)


@pytest.mark.parametrize(
    "assignment, match",
    [
        ("piecewise(0, time == 10, 1)", "single"),
        ("piecewise(0, time < 10 && time > 5, 1)", "single"),
        ("piecewise(0, time^2 < 10, 1)", "single"),
        ("piecewise(time^2, time < 10, 1)", "linearly"),
        ("piecewise(exp(time), time < 10, 1)", "linearly"),
        (
            "piecewise(0, time < t_dose, 1)",
            r"estimated parameters \['t_dose'\]",
        ),
        ("piecewise(0, time < A, 1)", "constant model parameters"),
        ("A * piecewise(0, time < 10, 1)", "constant model parameters"),
    ],
)
def test_assignments_to_experiments_unsupported(assignment, match):
    """Test that assignments outside the supported scope are rejected."""
    problem = Problem()
    ant_model = f"""
        model conversion
        species A, B;
        A = 10;
        B = 0;
        t_dose = 5;
        k1 := {assignment};
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=0.0, measurement=10.0)
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)
    problem.add_parameter("t_dose", estimate=True, lb=0, ub=10)

    with pytest.raises(NotImplementedError, match=match):
        assignments_to_experiments(problem)


@pytest.mark.parametrize(
    "condition_value, parameter, t_switch",
    [
        # a constant model parameter
        (None, None, 5.0),
        # a parameter fixed by the parameter table
        (None, {"estimate": False, "nominal_value": 7.0}, 7.0),
        # set by the condition table
        (8.0, None, 8.0),
    ],
)
def test_assignments_to_experiments_fixed_switching_time(
    condition_value, parameter, t_switch
):
    """Test that switching times may depend on parameters that are fixed or
    set by the condition table."""
    problem = Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        t_dose = 5;
        k1 := piecewise(0, time < t_dose, 1);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    experiment_id = None
    if condition_value is not None:
        experiment_id = "e1"
        problem.add_condition("c1", t_dose=condition_value)
        problem.add_experiment(experiment_id, 0, "c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id=experiment_id, time=20.0, measurement=5.0
    )
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)
    if parameter is not None:
        problem.add_parameter("t_dose", **parameter)

    converted = assignments_to_experiments(problem)

    periods = [
        (row[v2.C.TIME], row[v2.C.CONDITION_ID])
        for _, row in converted.experiment_df.iterrows()
    ]
    # the experiment starts at 0, where nothing changes
    assert periods[0] == (0.0, "")
    conditions = converted.condition_df.set_index(v2.C.CONDITION_ID)
    assert [
        (t, float(conditions.loc[condition_id, v2.C.TARGET_VALUE]))
        for t, condition_id in periods[1:]
    ] == [(t_switch, 1.0)]


def test_assignments_to_experiments_estimated_switching_time_condition():
    """Test that a switching time that the condition table sets to an
    estimated parameter is rejected."""
    problem = Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        t_dose = 5;
        t_est = 5;
        k1 := piecewise(0, time < t_dose, 1);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_condition("c1", t_dose="t_est")
    problem.add_experiment("e1", 0, "c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=5.0
    )
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)
    problem.add_parameter("t_est", estimate=True, lb=0, ub=10)

    with pytest.raises(NotImplementedError, match="evaluate to a real number"):
        assignments_to_experiments(problem)


@pytest.mark.parametrize(
    "assignment, default, switches",
    [
        # a sum of doses given at different times
        (
            "piecewise(0, time < 10, 1) + 2 * piecewise(0, time < 20, 1)",
            0.0,
            [(10.0, 1.0), (20.0, 3.0)],
        ),
        # phases one after another, as several pieces
        (
            "piecewise(1, time < 10, 2, time < 20, 3)",
            1.0,
            [(10.0, 2.0), (20.0, 3.0)],
        ),
        # and as nested piecewise functions
        (
            "piecewise(1, time < 10, piecewise(2, time < 20, 3))",
            1.0,
            [(10.0, 2.0), (20.0, 3.0)],
        ),
        # a switch that does not change the value is no period
        (
            "piecewise(1, time < 10, piecewise(1, time < 20, 3))",
            1.0,
            [(20.0, 3.0)],
        ),
    ],
)
def test_assignments_to_experiments_several_switches(
    assignment, default, switches
):
    """Test conversion of assignments that switch more than once."""
    problem = Problem()
    ant_model = f"""
        model conversion
        species A, B;
        A = 10;
        B = 0;
        k1 := {assignment};
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=30.0, measurement=5.0)
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    assert converted.model.sbml_model.getParameter("k1").getValue() == default
    periods = [
        (row[v2.C.TIME], row[v2.C.CONDITION_ID])
        for _, row in converted.experiment_df.iterrows()
    ]
    # the experiment starts at 0, where nothing changes
    assert periods[0] == (0.0, "")
    conditions = converted.condition_df.set_index(v2.C.CONDITION_ID)
    assert [
        (t, float(conditions.loc[condition_id, v2.C.TARGET_VALUE]))
        for t, condition_id in periods[1:]
    ] == switches


def test_assignments_to_experiments_condition_specific_doses():
    """Test conversion of two doses whose times and amounts are set by the
    condition table, as in Brannmark_JBC2010."""
    problem = Problem()
    doses = (
        "d1 * piecewise(0, time - t1 < 0, 1)"
        " + d2 * piecewise(0, time - t2 < 0, 1)"
    )
    ant_model = f"""
        model conversion
        species A, B;
        A = 10;
        B = 0;
        d1 = 0; t1 = 0; d2 = 0; t2 = 0;
        k1 := {doses};
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)

    problem.add_condition("c1", d1=2.0, t1=0.0, d2=0.0, t2=0.0)
    problem.add_condition("c2", d1=1.0, t1=0.0, d2=3.0, t2=5.0)
    problem.add_experiment("experiment____c1", 0, "c1")
    problem.add_experiment("experiment____c2", 0, "c2")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    for experiment_id in ("experiment____c1", "experiment____c2"):
        problem.add_measurement(
            "obs_a", experiment_id=experiment_id, time=30.0, measurement=5.0
        )
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    conditions = converted.condition_df.set_index(v2.C.CONDITION_ID)
    got = {
        (row[v2.C.EXPERIMENT_ID], row[v2.C.TIME]): float(
            conditions.loc[row[v2.C.CONDITION_ID], v2.C.TARGET_VALUE]
        )
        for _, row in converted.experiment_df.iterrows()
    }
    # c1: the dose is there from the start; c2: the first dose from the
    #  start, the second added at t = 5
    assert got == {
        ("experiment____c1", 0.0): 2.0,
        ("experiment____c2", 0.0): 1.0,
        ("experiment____c2", 5.0): 4.0,
    }


def _rate_rule(problem, target_id):
    rule = problem.model.sbml_model.getRateRuleByVariable(target_id)
    return None if rule is None else libsbml.formulaToL3String(rule.getMath())


def _changes_by_period(problem):
    """{(experiment ID, time): {target ID: target value}}"""
    conditions = problem.condition_df.set_index(v2.C.CONDITION_ID)
    out = {}
    for _, row in problem.experiment_df.iterrows():
        change = conditions.loc[[row[v2.C.CONDITION_ID]]]
        out.setdefault((row[v2.C.EXPERIMENT_ID], row[v2.C.TIME]), {}).update(
            {
                r[v2.C.TARGET_ID]: float(r[v2.C.TARGET_VALUE])
                for _, r in change.iterrows()
            }
        )
    return out


def test_assignments_to_experiments_ramp():
    """Test that a value changing linearly with time becomes a rate rule
    whose slope each period sets."""
    problem = Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        r = 0.5; a = 1;
        k1 := piecewise(r * time + a, time <= 10, 0);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=20.0, measurement=5.0)
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    sbml = converted.model.sbml_model
    # the model only makes the target differential; the slope is a
    #  placeholder until the tables set it
    assert _rate_rule(converted, "k1") == "k1_slope"
    assert sbml.getParameter("k1_slope").getValue() == 0.0
    # the ramp starts at time 0, and at the switch, the target and its slope
    #  drop to zero
    assert _changes_by_period(converted) == {
        (DEFAULT_EXPERIMENT_ID, 0.0): {"k1": 1.0, "k1_slope": 0.5},
        (DEFAULT_EXPERIMENT_ID, 10.0): {"k1": 0.0, "k1_slope": 0.0},
    }


def test_assignments_to_experiments_condition_specific_ramps():
    """Test pulses and ramps whose start, rate and end are set by the
    condition table, as in Fujita_SciSignal2010."""
    problem = Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        EGF_0 = 0; EGF_rate = 0; EGF_end = 0;
        k1 := piecewise(EGF_rate * time + EGF_0, time <= EGF_end, 0);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)
    # a pulse: a constant dose withdrawn at t = 60
    problem.add_condition("pulse", EGF_0=1.0, EGF_rate=0.0, EGF_end=60.0)
    # a ramp: growing from zero at 0.01 per unit time until t = 3600
    problem.add_condition("ramp", EGF_0=0.0, EGF_rate=0.01, EGF_end=3600.0)
    for c in ("pulse", "ramp"):
        problem.add_experiment(f"experiment____{c}", 0, c)
        problem.add_measurement(
            "obs_a",
            experiment_id=f"experiment____{c}",
            time=30.0,
            measurement=5.0,
        )
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    assert _rate_rule(converted, "k1") == "k1_slope"
    assert _changes_by_period(converted) == {
        ("experiment____pulse", 0.0): {"k1": 1.0, "k1_slope": 0.0},
        ("experiment____pulse", 60.0): {"k1": 0.0, "k1_slope": 0.0},
        ("experiment____ramp", 0.0): {"k1": 0.0, "k1_slope": 0.01},
        ("experiment____ramp", 3600.0): {"k1": 0.0, "k1_slope": 0.0},
    }


def test_assignments_to_experiments_ramp_preequilibration():
    """Test that a ramp is held at its value at time 0 during
    preequilibration and starts at time 0."""
    problem = Problem()
    ant_model = """
        model conversion
        species A, B;
        A = 10;
        B = 0;
        r = 0;
        k1 := piecewise(r * time + 1, time <= 10, 0);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)
    problem.add_condition("pre", r=0.0)
    problem.add_condition("main", r=2.0)
    problem.add_experiment(
        "experiment__pre___main", v2.C.TIME_PREEQUILIBRATION, "pre", 0, "main"
    )
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a",
        experiment_id="experiment__pre___main",
        time=20.0,
        measurement=5.0,
    )
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    with pytest.warns(UserWarning, match="during preequilibration"):
        converted = assignments_to_experiments(problem)

    assert _changes_by_period(converted) == {
        ("experiment__pre___main", v2.C.TIME_PREEQUILIBRATION): {
            "k1": 1.0,
            "k1_slope": 0.0,
        },
        ("experiment__pre___main", 0.0): {"k1": 1.0, "k1_slope": 2.0},
        ("experiment__pre___main", 10.0): {"k1": 0.0, "k1_slope": 0.0},
    }


@pytest.mark.parametrize("t_switch", [0, 10])
@pytest.mark.parametrize(
    "condition, holds_at_switch, holds_before",
    [
        ("time < {t}", False, True),
        ("time <= {t}", True, True),
        ("time > {t}", False, False),
        ("time >= {t}", True, False),
        ("{t} >= time", True, True),
    ],
)
def test_assignments_to_experiments_preequilibration_boundary(
    t_switch, condition, holds_at_switch, holds_before
):
    """Test that preequilibration uses the assignment's value at time 0,
    including whether the condition holds for a switch at time 0."""
    condition = condition.format(t=t_switch)
    problem = Problem()
    ant_model = f"""
        model conversion
        species A, B;
        A = 10;
        B = 0;
        k1 := piecewise(0, {condition}, 1);
        k2 = 0.5;
        R1: A -> B; k1 * A;
        R2: B -> A; k2 * B;
        end
    """
    problem.model = SbmlModel.from_antimony(ant_model)
    problem.add_experiment("e1", v2.C.TIME_PREEQUILIBRATION, [], 0, [])
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=5.0
    )
    problem.add_parameter("k2", estimate=True, lb=1e-3, ub=1e3)

    with pytest.warns(UserWarning, match="during preequilibration"):
        converted = assignments_to_experiments(problem)

    holds_at_0 = holds_at_switch if t_switch == 0 else holds_before
    (preeq_period,) = (
        p for p in converted.experiments[0].periods if p.is_preequilibration
    )
    assert [
        (c.target_id, float(c.target_value))
        for c in converted.get_changes_for_period(preeq_period)
    ] == [("k1", 0.0 if holds_at_0 else 1.0)]


def test_assignments_to_experiments_validate():
    """Test that the converted problem is linted.

    Here, the input problem is valid, but the measurement without an
    experiment ID is assigned to a new experiment `default_experiment`, an ID
    that an existing experiment already has.
    """
    problem = Problem()
    problem.model = _switch_model("piecewise(0, time < 10, 1)")
    problem.add_condition("c1", other=2.0)
    problem.add_experiment(DEFAULT_EXPERIMENT_ID, 0, "c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a",
        experiment_id=DEFAULT_EXPERIMENT_ID,
        time=20.0,
        measurement=5.0,
    )
    problem.add_measurement("obs_a", time=30.0, measurement=5.0)
    assert not problem.validate().has_errors()

    with pytest.raises(ValueError, match="did not pass linting"):
        assignments_to_experiments(problem)


def _changes_by_time(
    problem: Problem, experiment_id: str
) -> dict[float, dict[str, float]]:
    """The changes of each period of an experiment, by the period's time."""
    return {
        period.time: {
            c.target_id: float(c.target_value)
            for c in problem.get_changes_for_period(period)
        }
        for period in problem[experiment_id].sorted_periods
    }


def _switch_model(assignment: str, extra: str = "") -> SbmlModel:
    """A model with the assignment rule `k1 := <assignment>`."""
    return SbmlModel.from_antimony(f"""
        model conversion
        species A;
        A = 10;
        other = 1;
        {extra}
        k1 := {assignment};
        R1: A -> ; k1 * A;
        end
    """)


def test_assignments_to_experiments_switch_at_period_start():
    """Test that a switch at the start of a later period is kept, although
    that period does not change any symbol of the assignment."""
    problem = Problem()
    problem.model = _switch_model("piecewise(0, time < 10, 1)")
    problem.add_condition("c0", other=1.0)
    problem.add_condition("c1", other=2.0)
    problem.add_experiment("e1", 0, "c0", 10, "c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=1.0
    )

    converted = assignments_to_experiments(problem)

    assert _changes_by_time(converted, "e1") == {
        0.0: {"other": 1.0},
        10.0: {"other": 2.0, "k1": 1.0},
    }


def test_assignments_to_experiments_switch_at_preequilibration_end():
    """Test that a switch at time 0 is kept after preequilibration, where the
    condition still holds at time 0."""
    problem = Problem()
    problem.model = _switch_model("piecewise(0, time <= 0, 1)")
    problem.add_experiment("e1", v2.C.TIME_PREEQUILIBRATION, [], 0, [])
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=1.0
    )

    with pytest.warns(UserWarning, match="during preequilibration"):
        converted = assignments_to_experiments(problem)

    assert _changes_by_time(converted, "e1") == {
        v2.C.TIME_PREEQUILIBRATION: {"k1": 0.0},
        0.0: {"k1": 1.0},
    }


def test_assignments_to_experiments_shared_symbol():
    """Test that the changes of a symbol of a converted assignment stay if
    the model still refers to that symbol elsewhere."""
    problem = Problem()
    problem.model = _switch_model(
        "dose * piecewise(0, time < 10, 1)",
        extra="dose = 0; R2: -> A; dose;",
    )
    problem.add_condition("c1", dose=2.0)
    problem.add_experiment("e1", 0, "c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=1.0
    )

    converted = assignments_to_experiments(problem)

    assert _changes_by_time(converted, "e1") == {
        0.0: {"dose": 2.0},
        10.0: {"k1": 2.0},
    }


def test_assignments_to_experiments_initial_assignment():
    """Test that a time-dependent initial assignment stays in the model, since
    it is evaluated once, at the start of the simulation."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("""
        model conversion
        species A;
        A = 10;
        k1 = piecewise(0, time < 10, 1);
        R1: A -> ; k1 * A;
        end
    """)
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=20.0, measurement=1.0)

    converted = assignments_to_experiments(problem)

    assert converted.model.sbml_model.getInitialAssignment("k1") is not None
    assert not converted.experiments
    assert not converted.conditions


def test_assignments_to_experiments_preequilibration_model_value():
    """Test that a value at preequilibration, which starts the experiment,
    only refers to parameter table symbols: other model parameters take their
    values."""
    problem = Problem()
    problem.model = _switch_model(
        "piecewise(level, time < 10, 0)", extra="level = 3;"
    )
    problem.add_experiment("e1", v2.C.TIME_PREEQUILIBRATION, [], 0, [])
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=1.0
    )

    with pytest.warns(UserWarning, match="during preequilibration"):
        converted = assignments_to_experiments(problem)

    assert _changes_by_time(converted, "e1") == {
        v2.C.TIME_PREEQUILIBRATION: {"k1": 3.0},
        0.0: {},
        10.0: {"k1": 0.0},
    }


def test_assignments_to_experiments_species_target():
    """Test that an assignment to a species is rejected."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("""
        model conversion
        species A, S;
        A = 10;
        S := piecewise(0, time < 10, 1);
        R1: A -> ; S * A;
        end
    """)
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=20.0, measurement=1.0)

    with pytest.raises(NotImplementedError, match="not a parameter"):
        assignments_to_experiments(problem)


def test_assignments_to_experiments_nothing_to_convert():
    """Test that a problem without time-dependent assignments is unchanged,
    including an experiment that only preequilibrates and periods that are
    not in time order."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("""
        model conversion
        species A;
        A = 10;
        k2 = 0.5;
        R1: A -> ; k2 * A;
        end
    """)
    problem.add_condition("c1", k2=1.0)
    problem.add_condition("c2", k2=2.0)
    problem.add_experiment("e1", v2.C.TIME_PREEQUILIBRATION, "c1")
    # periods out of time order
    problem.add_experiment("e2", 10, "c2", 0, "c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    for experiment_id in ("e1", "e2"):
        problem.add_measurement(
            "obs_a", experiment_id=experiment_id, time=20.0, measurement=1.0
        )

    converted = assignments_to_experiments(problem)

    def periods(problem):
        return [
            [(p.time, p.condition_ids) for p in e.periods]
            for e in problem.experiments
        ]

    assert periods(converted) == periods(problem)
    assert converted.condition_df.equals(problem.condition_df)


@pytest.mark.parametrize("on", ["model", "species"])
def test_assignments_to_experiments_conversion_factor(on):
    """Test that the changes of a symbol of a converted assignment stay if
    the model refers to that symbol as a conversion factor."""
    problem = Problem()
    problem.model = _switch_model(
        "cf * piecewise(0, time < 10, 1)", extra="cf = 1;"
    )
    sbml_model = problem.model.sbml_model
    element = sbml_model if on == "model" else sbml_model.getSpecies("A")
    assert (
        element.setConversionFactor("cf") == libsbml.LIBSBML_OPERATION_SUCCESS
    )
    problem.add_condition("c1", cf=2.0)
    problem.add_experiment("e1", 0, "c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=1.0
    )

    converted = assignments_to_experiments(problem)

    assert _changes_by_time(converted, "e1") == {
        0.0: {"cf": 2.0},
        10.0: {"k1": 2.0},
    }


def test_assignments_to_experiments_slope_id_taken():
    """Test that a ramp's slope parameter gets an ID that the problem does not
    use yet, here because an observable parameter is called `k1_slope`."""
    problem = Problem()
    problem.model = _switch_model("piecewise(time, time < 10, 1)")
    problem.add_observable("obs_a", formula="k1_slope * A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=20.0, measurement=5.0)
    problem.add_parameter("k1_slope", estimate=True, lb=1e-3, ub=1e3)

    converted = assignments_to_experiments(problem)

    assert _rate_rule(converted, "k1") == "k1_slope_1"
    assert _changes_by_time(converted, DEFAULT_EXPERIMENT_ID) == {
        0.0: {"k1": 0.0, "k1_slope_1": 1.0},
        10.0: {"k1": 1.0, "k1_slope_1": 0.0},
    }


@pytest.mark.parametrize(
    "conditions",
    [
        # the changes of a condition are simultaneous: `d1` takes the old `b1`
        [(2.0, {"b1": 2.0, "d1": "b1 + 0.5"})],
        # a change is evaluated once: `d1` keeps the `b1` of its period
        [(2.0, {"d1": "b1 + 0.5"}), (3.0, {"b1": 2.0})],
    ],
)
def test_assignments_to_experiments_change_evaluated_at_period_start(
    conditions,
):
    """Test that a condition change's value is evaluated at the start of its
    period, with the values from before the period, also when a later switch
    uses it."""
    problem = Problem()
    problem.model = _switch_model(
        "piecewise(b0, time < 5, d1)", extra="b0 = 0.3; b1 = 1.2; d1 = 1;"
    )
    args = [0, []]
    for i, (t, changes) in enumerate(conditions):
        problem.add_condition(f"c{i}", **changes)
        args += [t, f"c{i}"]
    problem.add_experiment("e1", *args)
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=1.0
    )

    converted = assignments_to_experiments(problem)

    (switch,) = (p for p in converted["e1"].periods if p.time == 5.0)
    (change,) = converted.get_changes_for_period(switch)
    assert change.target_id == "k1"
    # 1.2 + 0.5, not 2.0 + 0.5
    assert float(change.target_value) == pytest.approx(1.7)


def test_assignments_to_experiments_change_refers_to_species():
    """Test that a change of a symbol of the assignment whose value refers to
    a species, which is not known at the start of the period, is
    rejected."""
    problem = Problem()
    problem.model = _switch_model(
        "piecewise(0, time < 5, d1)", extra="d1 = 1;"
    )
    problem.add_condition("c1", d1="A")
    problem.add_experiment("e1", 0, [], 2, "c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=1.0
    )

    with pytest.raises(NotImplementedError, match=r"refer to \['A'\]"):
        assignments_to_experiments(problem)


def test_assignments_to_experiments_petab_symbols():
    """Test that the values of new condition changes use PEtab's symbols, so
    that they equal the same expressions parsed by PEtab."""
    problem = Problem()
    problem.model = _switch_model(
        "piecewise(0, time < 5, d1)", extra="d1 = 1;"
    )
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement("obs_a", time=20.0, measurement=1.0)
    problem.add_parameter(
        "d1", estimate=True, nominal_value=2.0, lb=0.1, ub=10
    )

    converted = assignments_to_experiments(problem)

    assert [
        change.target_value
        for condition in converted.conditions
        for change in condition.changes
    ] == [sympify_petab("d1")]


@pytest.mark.parametrize("estimate", [False, True])
def test_assignments_to_experiments_switching_time_from_fixed_parameter(
    estimate,
):
    """Test that a condition may set a switching time to an expression in a
    parameter that the parameter table fixes, which takes its nominal value,
    but not in an estimated one."""
    problem = Problem()
    problem.model = _switch_model(
        "piecewise(0, time < t1, 1)", extra="t1 = 0; p = 1;"
    )
    problem.add_condition("c1", t1="p + 1")
    problem.add_experiment("e1", 0, "c1")
    problem.add_observable("obs_a", formula="A", noise_formula=1.0)
    problem.add_measurement(
        "obs_a", experiment_id="e1", time=20.0, measurement=1.0
    )
    if estimate:
        problem.add_parameter(
            "p", estimate=True, nominal_value=4.0, lb=0.1, ub=10
        )
        with pytest.raises(
            NotImplementedError, match="evaluate to a real number"
        ):
            assignments_to_experiments(problem)
        return
    problem.add_parameter("p", estimate=False, nominal_value=4.0)

    converted = assignments_to_experiments(problem)

    # the switch is at the nominal value 4 + 1, not the model value 1 + 1
    assert _changes_by_time(converted, "e1") == {0.0: {}, 5.0: {"k1": 1.0}}
