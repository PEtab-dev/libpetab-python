from math import inf

import pandas as pd

from petab.v2 import Change, Condition, Experiment, ExperimentPeriod, Problem
from petab.v2.converters import ExperimentsToEventsConverter
from petab.v2.models.sbml_model import SbmlModel


def test_experiments_to_events_converter():
    """Test the ExperimentsToEventsConverter."""
    ant_model = """
    species X = 0
    X' = 1
    """
    problem = Problem()
    problem.model = SbmlModel.from_antimony(ant_model)
    problem.add_condition("c1", X=1)
    problem.add_condition("c2", X=2)
    problem.add_experiment("e1", -inf, "c1", 10, "c2")

    converter = ExperimentsToEventsConverter(problem)
    converted = converter.convert()
    assert converted.validate().has_errors() is False

    assert isinstance(converted.model, SbmlModel)
    sbml_model = converted.model.sbml_model

    assert sbml_model.getNumEvents() == 2
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
    # (note: the initial values are concentrations nonetheless)
    substanceOnly species s3a_comp1 in comp1 = 5
    substanceOnly species s3a_comp2 in comp2 = 6
    substanceOnly species s4a_comp1 in comp1 = 7
    substanceOnly species s4a_comp2 in comp2 = 8

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
    converter = ExperimentsToEventsConverter(problem)
    converted = converter.convert()
    # set experiment indicator to simulate experiment "e1"
    converted.model.sbml_model.getParameter(
        "_petab_experiment_indicator_e1"
    ).setValue(1)
    sbml_actual = converted.model.to_sbml_str()
    print(converted.model.to_antimony())
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
