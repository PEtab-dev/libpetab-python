from math import inf

import pandas as pd
import pytest
import sympy as sp
from sbmlmath import sbml_math_to_sympy

from petab.v2 import Change, Condition, Experiment, ExperimentPeriod, Problem
from petab.v2.converters import ExperimentsToSbmlConverter
from petab.v2.models.sbml_model import SbmlModel


def test_experiments_to_events_converter():
    """Test the ExperimentsToSbmlConverter."""
    ant_model = """
    species X = 0
    X' = 1
    """
    problem = Problem()
    problem.model = SbmlModel.from_antimony(ant_model)
    problem.add_condition("c1", X=1)
    problem.add_condition("c2", X=2)
    problem.add_experiment("e1", -inf, "c1", 10, "c2")

    converter = ExperimentsToSbmlConverter(problem)
    converted = converter.convert()
    assert converted.validate().has_errors() is False

    assert isinstance(converted.model, SbmlModel)
    sbml_model = converted.model.sbml_model

    # one event -- the initial period is handled via initial assignments
    assert sbml_model.getNumEvents() == 1
    assert converted.conditions == [
        Condition(
            id="_petab_preequilibration_on",
            changes=[
                Change(
                    target_id="_petab_preequilibration_indicator",
                    target_value=1,
                )
            ],
        ),
        Condition(
            id="_petab_preequilibration_off",
            changes=[
                Change(
                    target_id="_petab_preequilibration_indicator",
                    target_value=0,
                )
            ],
        ),
        Condition(
            id="_petab_experiment_condition_e1",
            changes=[
                Change(
                    target_id="_petab_experiment_indicator_e1", target_value=1
                )
            ],
        ),
    ]
    assert converted.experiments == [
        Experiment(
            id="e1",
            periods=[
                ExperimentPeriod(
                    time=-inf,
                    condition_ids=[
                        "_petab_experiment_condition_e1",
                        "_petab_preequilibration_on",
                    ],
                ),
                ExperimentPeriod(
                    time=10.0,
                    condition_ids=[
                        "_petab_experiment_condition_e1",
                        "_petab_preequilibration_off",
                    ],
                ),
            ],
        ),
    ]


def test_initial_assignments_many_experiments():
    """The first-period changes of all experiments are combined into a single
    flat piecewise initial assignment per target.

    Nesting the piecewise once per experiment exceeded lxml's depth limit
    (https://github.com/PEtab-dev/libpetab-python/issues/540).
    """
    n_experiments = 300
    ant_model = """
    species A = 1
    species B
    k = 3
    B = 2 * k
    """
    problem = Problem()
    problem.model = SbmlModel.from_antimony(ant_model)
    for i in range(n_experiments):
        problem.add_condition(f"c{i}", A=i + 2, B=i + 2)
        problem.add_experiment(f"e{i}", 0, f"c{i}")
    # an experiment that does not change the initial values
    problem.add_condition("c_default", k=3)
    problem.add_experiment("e_default", 0, "c_default")

    converted = ExperimentsToSbmlConverter(problem).convert()
    sbml_model = converted.model.sbml_model

    indicators = {
        sp.Symbol(ExperimentsToSbmlConverter.get_experiment_indicator(exp.id))
        for exp in problem.experiments
    }
    for target_id, default in (("A", 1), ("B", 2 * sp.Symbol("k"))):
        expr = sbml_math_to_sympy(
            sbml_model.getInitialAssignmentBySymbol(target_id).getMath()
        )
        assert isinstance(expr, sp.Piecewise)
        assert len(expr.args) == n_experiments + 1
        assert not any(isinstance(e, sp.Piecewise) for e, _ in expr.args)

        for exp_id, expected in (
            ("e0", 2),
            (f"e{n_experiments - 1}", n_experiments + 1),
            ("e_default", default),
        ):
            active = sp.Symbol(
                ExperimentsToSbmlConverter.get_experiment_indicator(exp_id)
            )
            assert (
                expr.xreplace({ind: int(ind == active) for ind in indicators})
                == expected
            )


def test_simulate_experiment_to_events():
    """
    Convert PEtab experiment to SBML events and compare BasiCO simulation
    results.
    """
    import basico

    # the basic model for the PEtab problem
    ant_model1 = """
    compartment comp1 = 10
    compartment comp2 = 2
    # concentration-based species
    species s1c_comp1 in comp1 = 1
    species s1c_comp2 in comp2 = 2
    species s2c_comp1 in comp1 = 3
    species s2c_comp2 in comp2 = 4
    # amount-based species
    # (note that in antimony<3.1.0 the initial values are concentrations
    #  nonetheless)
    substanceOnly species s3a_comp1 in comp1 = 5 * comp1
    substanceOnly species s3a_comp2 in comp2 = 6 * comp2
    substanceOnly species s4a_comp1 in comp1 = 7 * comp1
    substanceOnly species s4a_comp2 in comp2 = 8 * comp2

    # something dynamic
    some_species in comp1 = 0
    some_species' = 1

    # set time-derivatives, otherwise BasiCO won't include them in the result
    s1c_comp1' = 0
    s1c_comp2' = 0
    s2c_comp1' = 0
    s2c_comp2' = 0
    s3a_comp1' = 0
    s3a_comp2' = 0
    s4a_comp1' = 0
    s4a_comp2' = 0
    """

    # append events, equivalent to the expected PEtab conversion result
    ant_model_expected = (
        ant_model1
        + """
    # resize compartment
    # The size of comp1 should be set to 20, the concentrations of the
    # contained concentration-based species and the amounts of the amount-based
    # species should remain unchanged. comp2 and everything therein is
    # unaffected.
    # I.e., post-event:
    #   s1c_comp1 = 1, s2c_comp1 = 3, s3a_comp1 = 5, s4a_comp1 = 7
    at time >= 1:
        comp1 = 20,
        s1c_comp1 = s1c_comp1 * 20 / comp1,
        s2c_comp1 = s2c_comp1 * 20 / comp1;

    # resize compartment *and* reassign concentration
    # The size of comp2 should be set to 4, the concentration/amount of
    # s1c_comp2/s3a_comp2 should be set to the given values,
    # the amounts for amount-based and concentrations for concentration-based
    # other species in comp2 should remain unchanged.
    # I.e., post-event:
    #   comp2 = 4
    #   s1c_comp2 = 5, s3a_comp2 = 16,
    #   s2c_comp2 = 4 (unchanged), s4a_comp2 = 8 (unchanged)
    # The post-event concentrations of concentration-based species are
    # (per SBML):
    #   new_conc = assigned_amount / new_volume
    #            = assigned_conc * old_volume / new_volume
    #   <=> assigned_conc = new_conc * new_volume / old_volume
    # The post-event amounts of amount-based species are:
    #   new_amount = assigned_amount (independent of volume change)
    at time >= 5:
        comp2 = 4,
        s3a_comp2 = 16,
        s1c_comp2 = 5 * 4 / comp2,
        s2c_comp2 = s2c_comp2 * 4 / comp2;
    """
    )

    # simulate expected model in BasiCO
    sbml_expected = SbmlModel.from_antimony(ant_model_expected).to_sbml_str()
    basico.load_model(sbml_expected)
    # output timepoints (initial, pre-/post-event, ...)
    timepoints = [0, 0.9, 1.1, 4.9, 5.1, 10]
    # Simulation will return all species as concentrations
    df_expected = basico.run_time_course(values=timepoints)
    # fmt: off
    assert (
            df_expected
            == pd.DataFrame(
            {'Values[some_species]': {0.0: 0.0, 0.9: 0.9,
                                      1.1: 1.0999999999999996, 4.9: 4.9,
                                      5.1: 5.100000000000001, 10.0: 10.0},
             's1c_comp1': {0.0: 1.0, 0.9: 1.0, 1.1: 1.0, 4.9: 1.0, 5.1: 1.0,
                           10.0: 1.0},
             's2c_comp1': {0.0: 3.0, 0.9: 3.0, 1.1: 3.0, 4.9: 3.0, 5.1: 3.0,
                           10.0: 3.0},
             's3a_comp1': {0.0: 5.0, 0.9: 5.0, 1.1: 2.5, 4.9: 2.5, 5.1: 2.5,
                           10.0: 2.5},
             's4a_comp1': {0.0: 7.0, 0.9: 7.0, 1.1: 3.5, 4.9: 3.5, 5.1: 3.5,
                           10.0: 3.5},
             's1c_comp2': {0.0: 2.0, 0.9: 2.0, 1.1: 2.0, 4.9: 2.0, 5.1: 5.0,
                           10.0: 5.0},
             's2c_comp2': {0.0: 4.0, 0.9: 4.0, 1.1: 4.0, 4.9: 4.0, 5.1: 4.0,
                           10.0: 4.0},
             's3a_comp2': {0.0: 6.0, 0.9: 6.0, 1.1: 6.0, 4.9: 6.0, 5.1: 4.0,
                           10.0: 4.0},
             's4a_comp2': {0.0: 8.0, 0.9: 8.0, 1.1: 8.0, 4.9: 8.0, 5.1: 4.0,
                           10.0: 4.0},
             'Compartments[comp1]': {0.0: 10.0, 0.9: 10.0, 1.1: 20.0,
                                     4.9: 20.0, 5.1: 20.0, 10.0: 20.0},
             'Compartments[comp2]': {0.0: 2.0, 0.9: 2.0, 1.1: 2.0, 4.9: 2.0,
                                     5.1: 4.0, 10.0: 4.0}}
        )
    ).all().all()
    # fmt: on

    # construct PEtab test problem
    problem = Problem()
    problem.model = SbmlModel.from_antimony(ant_model1)
    problem.add_condition("c0", comp1=10)
    problem.add_condition("c1", comp1=20)
    problem.add_condition("c2", comp2=4, s1c_comp2=5, s3a_comp2=16)
    problem.add_experiment("e1", 0, "c0", 1, "c1", 5, "c2")
    problem.assert_valid()

    # convert PEtab experiments to SBML events and simulate in BasiCO
    converter = ExperimentsToSbmlConverter(problem)
    converted = converter.convert()
    # set experiment indicator to simulate experiment "e1"
    converted.model.sbml_model.getParameter(
        "_petab_experiment_indicator_e1"
    ).setValue(1)
    sbml_actual = converted.model.to_sbml_str()
    basico.load_model(sbml_actual)
    df_actual = basico.run_time_course(values=timepoints)

    # compare results
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
    ):
        print("Expected:")
        print(df_expected)
        print("Actual:")
        print(df_actual)

    for col in df_expected.columns:
        assert (df_expected[col] == df_actual[col]).all()


def _event_priorities(sbml_model) -> dict[str, float | None]:
    """Get the numeric priorities of all events in the given model."""
    return {
        ev.getId(): float(sbml_math_to_sympy(ev.getPriority()))
        if ev.isSetPriority() and ev.getPriority().getMath() is not None
        else None
        for ev in sbml_model.getListOfEvents()
    }


def _problem_0030(model_events: str) -> Problem:
    """PEtab test suite case 0030 with the given model events."""
    ant_model = f"""
    compartment C = 4
    species S in C = 3
    p = 1
    S' = p
    {model_events}
    """
    problem = Problem()
    problem.model = SbmlModel.from_antimony(ant_model)
    problem.add_condition("condition1", S=2)
    problem.add_condition("condition2", S="S + C", C=8)
    problem.add_experiment("experiment1", 0, "condition1", 10, "condition2")
    return problem


E0 = "E0: at time >= 10, fromTrigger=false: C = C * 2"
E0_PRIO = "E0: at time >= 10, priority=-2, fromTrigger=false: C = C * 2"
E1 = "E1: at time >= 20, fromTrigger=false: C = C"
E1_PRIO = "E1: at time >= 20, priority=5, fromTrigger=false: C = C"
E1_PRIO_NEG_INF = "E1: at time >= 20, priority=-inf, fromTrigger=false: C = C"
E1_PRIO_INF = "E1: at time >= 20, priority=inf, fromTrigger=false: C = C"
E2_PRIO = "E2: at time >= 30, priority=5, fromTrigger=false: C = C"
PETAB_EVENT = "_petab_event_experiment1_1"


@pytest.mark.parametrize(
    "model_events,expected",
    [
        # single unprioritized event
        (E0, {"E0": 0, PETAB_EVENT: 1}),
        # one unprioritized event, others prioritized
        (f"{E0}\n{E1_PRIO}", {"E0": 4, "E1": 5, PETAB_EVENT: 6}),
        # no distinct lower priority available: no automatic assignment for
        #  the unprioritized event
        (
            f"{E0}\n{E1_PRIO_NEG_INF}",
            {"E0": None, "E1": -inf, PETAB_EVENT: 1},
        ),
        # all events prioritized: unchanged
        (f"{E0_PRIO}\n{E1_PRIO}", {"E0": -2, "E1": 5, PETAB_EVENT: 6}),
        # multiple unprioritized events: no automatic assignment
        (f"{E0}\n{E1}", {"E0": None, "E1": None, PETAB_EVENT: None}),
        # no model events
        ("", {PETAB_EVENT: None}),
    ],
)
def test_auto_event_priority(model_events, expected):
    """Test automatic priority assignment for pre-existing events."""
    problem = _problem_0030(model_events)
    orig_priorities = _event_priorities(problem.model.sbml_model)

    converted = ExperimentsToSbmlConverter(problem).convert()

    assert _event_priorities(converted.model.sbml_model) == expected
    # the original problem is not modified
    assert _event_priorities(problem.model.sbml_model) == orig_priorities


def test_auto_event_priority_empty_priority():
    """A priority without math is treated as unset."""
    problem = _problem_0030(E0)
    problem.model.sbml_model.getEvent("E0").createPriority()

    converted = ExperimentsToSbmlConverter(problem).convert()

    assert _event_priorities(converted.model.sbml_model) == {
        "E0": 0,
        PETAB_EVENT: 1,
    }
    # the original problem is not modified
    assert _event_priorities(problem.model.sbml_model) == {"E0": None}


def test_explicit_default_priority():
    """An explicit default priority takes precedence.

    It only applies to events without priority."""
    problem = _problem_0030(f"{E0}\n{E1}\n{E2_PRIO}")

    converted = ExperimentsToSbmlConverter(
        problem, default_priority=3
    ).convert()

    assert _event_priorities(converted.model.sbml_model) == {
        "E0": 3,
        "E1": 3,
        "E2": 5,
        PETAB_EVENT: 6,
    }


def test_infinite_priority():
    """Test handling of a pre-existing event with priority +inf.

    The condition-change events cannot get a higher priority, but a lower
    priority can still be assigned to an unprioritized event."""
    problem = _problem_0030(f"{E0}\n{E1_PRIO_INF}\n{E2_PRIO}")

    with pytest.warns(UserWarning, match="Event `E1` has priority inf"):
        converted = ExperimentsToSbmlConverter(problem).convert()

    assert _event_priorities(converted.model.sbml_model) == {
        "E0": 4,
        "E1": inf,
        "E2": 5,
        PETAB_EVENT: inf,
    }
