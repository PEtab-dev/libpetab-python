import libsbml
import sys
import os

import pandas as pd

sys.path.append(os.getcwd())
import petab  # noqa: E402


def test_get_condition_specific_models():
    """Test for petab.sbml.get_condition_specific_models"""
    # Create test model and data files
    sbml_document = libsbml.SBMLDocument(3, 1)
    sbml_model = sbml_document.createModel()

    c = sbml_model.createCompartment()
    c.setId("compartment_1")
    c.setSize(1)

    for i in range(1, 4):
        p = sbml_model.createParameter()
        p.setId(f"parameter_{i}")
        p.setValue(i)

    for i in range(1, 4):
        s = sbml_model.createSpecies()
        s.setId(f"species_{i}")
        s.setInitialConcentration(10 * i)

    ia = sbml_model.createInitialAssignment()
    ia.setSymbol("species_2")
    ia.setMath(libsbml.parseL3Formula("25"))

    condition_df = pd.DataFrame({
        petab.CONDITION_ID: ["condition_1"],
        "parameter_3": ['parameter_2'],
        "species_1": [15],
        "species_2": [25],
        "species_3": ['parameter_1'],
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
        petab.PARAMETER_ID: ["parameter_1", "parameter_2"],
        petab.PARAMETER_SCALE: [petab.LOG10] * 2,
        petab.NOMINAL_VALUE: [1.25, 2.25],
        petab.ESTIMATE: [0, 1],
    })
    parameter_df.set_index([petab.PARAMETER_ID], inplace=True)

    petab_problem = petab.Problem(
        sbml_model=sbml_model,
        condition_df=condition_df,
        observable_df=observable_df,
        measurement_df=measurement_df,
        parameter_df=parameter_df
    )

    # Actual test
    condition_doc, condition_model = petab.get_model_for_condition(
        petab_problem, "condition_1")

    assert condition_model.getSpecies(
        "species_1").getInitialConcentration() == 15
    assert condition_model.getSpecies(
        "species_2").getInitialConcentration() == 25
    assert condition_model.getSpecies(
        "species_3").getInitialConcentration() == 1.25
    assert len(condition_model.getListOfInitialAssignments()) == 0, \
        "InitialAssignment not removed"
    assert condition_model.getCompartment("compartment_1").getSize() == 2.0
    assert condition_model.getParameter("parameter_1").getValue() == 1.25
    assert condition_model.getParameter("parameter_2").getValue() == 2.25
    assert condition_model.getParameter("parameter_3").getValue() == 2.25
