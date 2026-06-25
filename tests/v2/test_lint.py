"""Test related to ``petab.v2.lint``."""

from copy import deepcopy

import pysb
import pytest

from petab.v2 import Problem
from petab.v2.lint import *
from petab.v2.models.pysb_model import PySBModel
from petab.v2.models.sbml_model import SbmlModel


@pytest.fixture
def uses_pysb():
    """Cleanup PySB auto-exported symbols before and after test"""
    pysb.SelfExporter.cleanup()
    yield ()
    pysb.SelfExporter.cleanup()


def test_check_experiments():
    """Test ``CheckExperimentTable``."""
    problem = Problem()

    check = CheckExperimentTable()
    assert check.run(problem) is None

    problem.add_experiment("e1", 0, "c1", 1, "c2")
    problem.add_experiment("e2", "-inf", "c1", 1, "c2")
    assert check.run(problem) is None

    tmp_problem = deepcopy(problem)
    tmp_problem["e1"].periods[0].time = tmp_problem["e1"].periods[1].time
    assert check.run(tmp_problem) is not None


def test_check_incompatible_targets():
    """Multiple conditions with overlapping targets cannot be applied
    at the same time."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("p1 = 1; p2 = 2")
    problem.add_experiment("e1", 0, "c1", 1, "c2")
    problem.add_condition("c1", p1="1")
    problem.add_condition("c2", p1="2", p2="2")
    check = CheckValidConditionTargets()
    assert check.run(problem) is None

    problem["e1"].periods[0].condition_ids.append("c2")
    assert (error := check.run(problem)) is not None
    assert "overlapping targets {'p1'}" in error.message


def test_invalid_model_id_in_measurements():
    """Test that measurements with an invalid model ID are caught."""
    problem = Problem()
    problem.models.append(SbmlModel.from_antimony("p1 = 1", model_id="model1"))
    problem.add_observable("obs1", "A", 1)
    problem.add_measurement("obs1", experiment_id="e1", time=0, measurement=1)

    check = CheckMeasurementModelId()

    # Single model -> model ID is optional
    assert (error := check.run(problem)) is None, error

    # Two models -> model ID must be set
    problem.models.append(SbmlModel.from_antimony("p2 = 2", model_id="model2"))
    assert (error := check.run(problem)) is not None
    assert "multiple models" in error.message

    # Set model ID to a non-existing model ID
    problem.measurements[0].model_id = "invalid_model_id"
    assert (error := check.run(problem)) is not None
    assert "does not match" in error.message

    # Use a valid model ID
    problem.measurements[0].model_id = "model1"
    assert (error := check.run(problem)) is None, error


def test_undefined_experiment_id_in_measurements():
    """Test that measurements with an undefined experiment ID are caught."""
    problem = Problem()
    problem.add_experiment("e1", 0, "c1")
    problem.add_observable("obs1", "A", 1)
    problem.add_measurement("obs1", experiment_id="e1", time=0, measurement=1)

    check = CheckUndefinedExperiments()

    # Valid experiment ID
    assert (error := check.run(problem)) is None, error

    # Invalid experiment ID
    problem.measurements[0].experiment_id = "invalid_experiment_id"
    assert (error := check.run(problem)) is not None
    assert "not defined" in error.message


def test_validate_initial_change_symbols():
    """Test validation of symbols in target value expressions for changes
    applied at the start of an experiment."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("p1 = 1; p2 = 2")
    problem.add_experiment("e1", 0, "c1", 1, "c2")
    problem.add_condition("c1", p1="p2 + time")
    problem.add_condition("c2", p1="p2", p2="p1")
    problem.add_parameter("p1", nominal_value=1, estimate=False)
    problem.add_parameter("p2", nominal_value=2, estimate=False)

    check = CheckInitialChangeSymbols()
    assert check.run(problem) is None

    # removing `p1` from the parameter table is okay, as `c2` is never
    #  used at the start of an experiment
    problem.parameter_tables[0].parameters.remove(problem["p1"])
    assert check.run(problem) is None

    # removing `p2` is not okay, as it is used at the start of an experiment
    problem.parameter_tables[0].parameters.remove(problem["p2"])
    assert (error := check.run(problem)) is not None
    assert "contains additional symbols: {'p2'}" in error.message


def test_check_mapping_table(uses_pysb):
    """Test checks related to the mapping table."""
    problem = Problem()

    # PySB model with a compartment and a monomer, and mapping of model entity
    # to a valid PEtab id
    pysb_model = pysb.Model("test_model")
    pysb.Monomer("A_")
    pysb.Initial(A_() ** pysb.Compartment("C"), pysb.Parameter("a0", 1))
    problem.model = PySBModel(model=pysb_model, model_id="test_model")
    problem.add_mapping("A", "A_() ** C")

    check = CheckMappingTable()
    assert check.run(problem) is None

    check = CheckAllParametersPresentInParameterTable()
    assert check.run(problem) is None

    # add a petab id without model id but with name for annotation
    problem.add_mapping(petab_id="p2", model_id=None, name="Parameter 2")
    problem.add_parameter("p2", estimate=True, nominal_value=1, lb=0, ub=10)

    check = CheckMappingTable()
    assert check.run(problem) is None

    # Invalid: petabEntityId is referenced in the model
    pysb.SelfExporter.cleanup()
    pysb_model_invalid = pysb.Model("test_model_invalid")
    pysb.Monomer("A_")
    pysb.Initial(A_() ** pysb.Compartment("C"), pysb.Parameter("A", 1))
    problem.model = PySBModel(
        model=pysb_model_invalid, model_id="test_model_invalid"
    )
    assert (error := check.run(problem)) is not None
    assert (
        "`A` is used in the mapping table and referenced directly"
        in error.message
    )
