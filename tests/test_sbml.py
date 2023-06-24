import os
import sys

import pandas as pd
import pytest

sys.path.append(os.getcwd())
import petab  # noqa: E402


def create_test_data():
    # Create test model and data files
    import simplesbml

    ss_model = simplesbml.SbmlModel()
    ss_model.addCompartment(comp_id="compartment_1", vol=1)
    for i in range(1, 4):
        ss_model.addParameter(f"parameter_{i}", i)

    for i in range(1, 5):
        ss_model.addSpecies(f"[species_{i}]", 10 * i)

    ss_model.addAssignmentRule("species_2", "25")

    condition_df = pd.DataFrame({
        petab.CONDITION_ID: ["condition_1"],
        "parameter_3": ['parameter_2'],
        "species_1": [15],
        "species_2": [25],
        "species_3": ['parameter_1'],
        "species_4": ['not_a_model_parameter'],
        "compartment_1": [2],
    })
    condition_df.set_index([petab.CONDITION_ID], inplace=True)

    observable_df = pd.DataFrame({
        petab.OBSERVABLE_ID: ["observable_1"],
        petab.OBSERVABLE_FORMULA: ["2 * species_1"],
    })
    observable_df.set_index([petab.OBSERVABLE_ID], inplace=True)

    measurement_df = pd.DataFrame({
        petab.OBSERVABLE_ID: ["observable_1"],
        petab.SIMULATION_CONDITION_ID: ["condition_1"],
        petab.TIME: [0.0],
    })

    parameter_df = pd.DataFrame({
        petab.PARAMETER_ID:
            ["parameter_1", "parameter_2", "not_a_model_parameter"],
        petab.PARAMETER_SCALE: [petab.LOG10] * 3,
        petab.NOMINAL_VALUE: [1.25, 2.25, 3.25],
        petab.ESTIMATE: [0, 1, 0],
    })
    parameter_df.set_index([petab.PARAMETER_ID], inplace=True)

    return ss_model, condition_df, observable_df, measurement_df, parameter_df


def check_model(condition_model):
    assert condition_model.getSpecies(
        "species_1").getInitialConcentration() == 15
    assert condition_model.getSpecies(
        "species_2").getInitialConcentration() == 25
    assert condition_model.getSpecies(
        "species_3").getInitialConcentration() == 1.25
    assert condition_model.getSpecies(
        "species_4").getInitialConcentration() == 3.25
    assert len(condition_model.getListOfInitialAssignments()) == 0, \
        "InitialAssignment not removed"
    assert condition_model.getCompartment("compartment_1").getSize() == 2.0
    assert condition_model.getParameter("parameter_1").getValue() == 1.25
    assert condition_model.getParameter("parameter_2").getValue() == 2.25
    assert condition_model.getParameter("parameter_3").getValue() == 2.25


def test_get_condition_specific_models():
    """Test for petab.sbml.get_condition_specific_models"""
    # retrieve test data
    ss_model, condition_df, observable_df, measurement_df, parameter_df = \
        create_test_data()

    petab_problem = petab.Problem(
        model=petab.models.sbml_model.SbmlModel(ss_model.model),
        condition_df=condition_df,
        observable_df=observable_df,
        measurement_df=measurement_df,
        parameter_df=parameter_df
    )

    # create SBML model for condition with parameters updated from problem
    with pytest.warns(
            UserWarning,
            match="An SBML rule was removed to set the "
                  "component species_2 to a constant value."
    ):
        _, condition_model = petab.get_model_for_condition(
            petab_problem, "condition_1")

    check_model(condition_model)
