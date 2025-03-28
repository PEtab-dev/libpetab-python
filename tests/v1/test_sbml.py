import os
import sys

import libsbml
import pandas as pd
import pytest

from petab.v1.models.sbml_model import SbmlModel

sys.path.append(os.getcwd())
import petab  # noqa: E402


def create_test_data():
    # Create test model and data files
    model = SbmlModel.from_antimony(
        "\n".join(
            [
                "compartment compartment_1 = 1",
                *(f"species species_{i} = 10 * {i}" for i in range(1, 5)),
                *(f"parameter_{i} = {i}" for i in range(1, 4)),
                "species_2 := 25",
            ]
        )
    )

    condition_df = pd.DataFrame(
        {
            petab.CONDITION_ID: ["condition_1"],
            "parameter_3": ["parameter_2"],
            "species_1": [15],
            "species_2": [25],
            "species_3": ["parameter_1"],
            "species_4": ["not_a_model_parameter"],
            "compartment_1": [2],
        }
    )
    condition_df.set_index([petab.CONDITION_ID], inplace=True)

    observable_df = pd.DataFrame(
        {
            petab.OBSERVABLE_ID: ["observable_1"],
            petab.OBSERVABLE_FORMULA: ["2 * species_1"],
        }
    )
    observable_df.set_index([petab.OBSERVABLE_ID], inplace=True)

    measurement_df = pd.DataFrame(
        {
            petab.OBSERVABLE_ID: ["observable_1"],
            petab.SIMULATION_CONDITION_ID: ["condition_1"],
            petab.TIME: [0.0],
        }
    )

    parameter_df = pd.DataFrame(
        {
            petab.PARAMETER_ID: [
                "parameter_1",
                "parameter_2",
                "not_a_model_parameter",
            ],
            petab.PARAMETER_SCALE: [petab.LOG10] * 3,
            petab.NOMINAL_VALUE: [1.25, 2.25, 3.25],
            petab.ESTIMATE: [0, 1, 0],
        }
    )
    parameter_df.set_index([petab.PARAMETER_ID], inplace=True)

    return model, condition_df, observable_df, measurement_df, parameter_df


def check_model(condition_model):
    assert (
        condition_model.getSpecies("species_1").getInitialConcentration() == 15
    )
    assert (
        condition_model.getSpecies("species_2").getInitialConcentration() == 25
    )
    assert (
        condition_model.getSpecies("species_3").getInitialConcentration()
        == 1.25
    )
    assert (
        condition_model.getSpecies("species_4").getInitialConcentration()
        == 3.25
    )
    assert len(condition_model.getListOfInitialAssignments()) == 0, (
        "InitialAssignment not removed"
    )
    assert condition_model.getCompartment("compartment_1").getSize() == 2.0
    assert condition_model.getParameter("parameter_1").getValue() == 1.25
    assert condition_model.getParameter("parameter_2").getValue() == 2.25
    assert condition_model.getParameter("parameter_3").getValue() == 2.25


def test_get_condition_specific_models():
    """Test for petab.sbml.get_condition_specific_models"""
    # retrieve test data
    (
        model,
        condition_df,
        observable_df,
        measurement_df,
        parameter_df,
    ) = create_test_data()

    petab_problem = petab.Problem(
        model=model,
        condition_df=condition_df,
        observable_df=observable_df,
        measurement_df=measurement_df,
        parameter_df=parameter_df,
    )

    # create SBML model for condition with parameters updated from problem
    with pytest.warns(
        UserWarning,
        match="An SBML rule was removed to set the "
        "component species_2 to a constant value.",
    ):
        _, condition_model = petab.get_model_for_condition(
            petab_problem, "condition_1"
        )

    check_model(condition_model)


def test_sbml_model_repr():
    sbml_document = libsbml.SBMLDocument()
    sbml_model = sbml_document.createModel()
    sbml_model.setId("test")
    petab_model = SbmlModel(sbml_model)
    assert repr(petab_model) == "<SbmlModel 'test'>"


def test_sbml_from_ant():
    ant_model = """
    model test
        R1: S1 -> S2; k1*S1
        k1 = 1
    end
    """
    petab_model = SbmlModel.from_antimony(ant_model)
    assert petab_model.model_id == "test"
    assert petab_model.get_parameter_value("k1") == 1.0
    assert set(petab_model.get_valid_parameters_for_parameter_table()) == {
        "k1"
    }
