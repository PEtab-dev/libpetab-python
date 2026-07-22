import numpy as np
import pytest
from petab_sciml import Input, Node
from pydantic import ConfigDict

from petab.v2.core import *
from petab.v2.core import ModelFile
from petab.v2.extensions.sciml import (
    Hybridization,
    NeuralNetConfig,
    SciMLConfig,
)
from petab.v2.extensions.sciml_lint import (
    CheckArrayDataFiles,
    CheckHybridizationTable,
    CheckNeuralNetworkModel,
    CheckSciMLConditionTable,
    CheckSciMLParameterTable,
)
from petab.v2.models.sbml_model import SbmlModel


def _get_test_problem():
    problem = Problem(
        config=ProblemConfig(
            format_version="2.0.0",
            model_files=ConfigDict(
                {"lv": ModelFile(location="lv.xml", language="sbml")}
            ),
            parameter_files=["parameters.tsv"],
            measurement_files=["measurements.tsv"],
            observable_files=["observables.tsv"],
            experiment_files=["experiments.tsv"],
            mapping_files=["mappings.tsv"],
            extensions={
                "sciml": SciMLConfig(
                    version="0.1.0",
                    array_files=["net1_ps.hdf5"],
                    hybridization_files=["hybridizations.tsv"],
                    neural_networks={
                        "net1": NeuralNetConfig(
                            location="net1.yaml",
                            pre_initialization=False,
                            format="YAML",
                        )
                    },
                )
            },
        )
    )
    problem.model = SbmlModel.from_antimony("""
    model lv
      species A, B;
      A = 0.442;
      B = 4.63;
      alpha = 1.3;
      gamma_ = 0.8;
      -> A; alpha * A;
      B -> ; 1.8 * B;
      A -> ; 0.9 * A * B;
      -> B; gamma_;
    end
    """)
    problem.add_experiment("e1", 0, "cond1")
    problem.add_mapping("net1_input1", "net1.inputs[0][0]")
    problem.add_mapping("net1_input2", "net1.inputs[1]")
    problem.add_mapping("net1_output1", "net1.outputs[0][0]")
    problem.add_mapping("net1_ps", "net1.parameters")
    problem.add_measurement("B_obs", time=1, measurement=1, experiment_id="e1")
    problem.add_observable("B_obs", "B", noise_formula="0.05")
    problem.add_parameter(
        "alpha", estimate=True, lb=0, ub=15, nominal_value=1.3
    )
    problem.add_parameter(
        "net1_ps", estimate=True, lb=-np.inf, ub=np.inf, nominal_value="array"
    )
    problem.extensions.sciml.add_hybridization("net1_input1", "A")
    problem.extensions.sciml.add_hybridization("net1_input2", "array")
    problem.extensions.sciml.add_hybridization("gamma_", "net1_output1")
    problem.extensions.sciml.add_neural_network_from_dict(
        "net1",
        nn_dict={
            "nn_model_id": "net1",
            "inputs": [{"input_id": "input0"}],
            "layers": [
                {
                    "layer_id": "layer1",
                    "layer_type": "Linear",
                    "args": {
                        "in_features": 2,
                        "out_features": 1,
                        "bias": True,
                    },
                }
            ],
            "forward": [
                {
                    "name": "net_input",
                    "op": "placeholder",
                    "target": "net_input",
                },
                {
                    "name": "layer1",
                    "op": "call_module",
                    "target": "layer1",
                    "args": ["net_input"],
                },
                {
                    "name": "tanh",
                    "op": "call_method",
                    "target": "tanh",
                    "args": ["layer1"],
                },
            ],
        },
    )

    # array data
    problem.extensions.sciml.add_array_data_from_dict(
        {
            "metadata": {"pytorch_format": True},
            "inputs": {
                "net1_input2": {
                    "cond1": np.random.randn(2),
                }
            },
            "parameters": {
                "net1": {
                    "layer1": {
                        "bias": np.random.randn(2),
                        "weight": np.random.randn(2),
                    }
                }
            },
        }
    )

    # set the filenames
    problem.config.filepath = "problem.yaml"
    problem.model.rel_path = "lv.xml"
    problem.experiment_tables[0].rel_path = "experiments.tsv"
    problem.mapping_tables[0].rel_path = "mappings.tsv"
    problem.measurement_tables[0].rel_path = "measurements.tsv"
    problem.observable_tables[0].rel_path = "observables.tsv"
    problem.parameter_tables[0].rel_path = "parameters.tsv"
    problem.extensions.sciml.hybridization_tables[
        0
    ].rel_path = "hybridizations.tsv"
    # problem.extensions.sciml.neural_networks[0].rel_path = "net1.yaml"
    # problem.extensions.sciml.array_data_files[0].rel_path = "net1_ps.hdf5"

    return problem


def test_lint():
    problem = _get_test_problem()
    assert problem.validate() == []


def test_lint_equinox_network_format():
    """Linter accepts non-YAML formats without reading the network file."""
    problem = _get_test_problem()
    # Replace the YAML network config with equinox format
    sciml_cfg = problem.config.extensions["sciml"]
    sciml_cfg.neural_networks["net1"] = NeuralNetConfig(
        location="net1.py",
        pre_initialization=False,
        format="equinox",
    )
    assert problem.validate() == []


# ---------------------------------------------------------------------------
# Neural network model checks
# ---------------------------------------------------------------------------


def test_nn_model_valid():
    problem = _get_test_problem()
    assert CheckNeuralNetworkModel().run(problem) is None


def test_nn_model_unsupported_layer():
    problem = _get_test_problem()
    problem.extensions.sciml.neural_networks[0].layers[
        0
    ].layer_type = "NotALayer"
    issue = CheckNeuralNetworkModel().run(problem)
    assert issue is not None
    assert "unsupported layer type" in issue.message


def test_nn_model_unsupported_activation():
    problem = _get_test_problem()
    problem.extensions.sciml.neural_networks[0].forward.append(
        Node(
            name="bogus",
            op="call_method",
            target="bogus_activation",
            args=["layer1"],
        )
    )
    issue = CheckNeuralNetworkModel().run(problem)
    assert issue is not None
    assert "activation function/operation" in issue.message


def test_nn_model_duplicate_input_ids():
    problem = _get_test_problem()
    problem.extensions.sciml.neural_networks[0].inputs = [
        Input(input_id="a"),
        Input(input_id="a"),
    ]
    issue = CheckNeuralNetworkModel().run(problem)
    assert issue is not None
    assert "duplicate input" in issue.message.lower()


def test_nn_model_invalid_input_id():
    problem = _get_test_problem()
    problem.extensions.sciml.neural_networks[0].inputs = [
        Input(input_id="1_invalid")
    ]
    issue = CheckNeuralNetworkModel().run(problem)
    assert issue is not None
    assert "not a valid PEtab identifier" in issue.message


# ---------------------------------------------------------------------------
# Hybridization table checks
# ---------------------------------------------------------------------------


def test_hybridization_valid():
    problem = _get_test_problem()
    assert CheckHybridizationTable().run(problem) is None


def test_hybridization_nn_output_as_target():
    problem = _get_test_problem()
    problem.extensions.sciml.add_hybridization("net1_output1", "1.0")
    issue = CheckHybridizationTable().run(problem)
    assert issue is not None
    assert "net1_output1" in issue.message
    assert "target ids" in issue.message


def test_hybridization_nn_parameters_as_target():
    problem = _get_test_problem()
    problem.extensions.sciml.add_hybridization("net1_ps", "1.0")
    issue = CheckHybridizationTable().run(problem)
    assert issue is not None
    assert "net1_ps" in issue.message
    assert "neural network parameters" in issue.message


def test_hybridization_array_input_not_assigned_array():
    problem = _get_test_problem()
    # net1_input2 is an array input, but assign it a value instead of `array`
    for hyb in problem.extensions.sciml.hybridization_tables[0].hybridizations:
        if hyb.target_id == "net1_input2":
            hyb.target_value = "5.0"
    issue = CheckHybridizationTable().run(problem)
    assert issue is not None
    assert "net1_input2" in issue.message
    assert "array data file" in issue.message


def test_hybridization_invalid_math_rejected():
    """Invalid PEtab math target values are rejected when parsed."""
    with pytest.raises(ValueError):
        Hybridization(target_id="alpha", target_value="1 +* 2")


def test_hybridization_pre_initialization_not_evaluable():
    problem = _get_test_problem()
    # net1_input1 is assigned the species `A`, which is not available before
    # simulation -- invalid for a pre-initialization network.
    problem.config.extensions["sciml"].neural_networks[
        "net1"
    ].pre_initialization = True
    issue = CheckHybridizationTable().run(problem)
    assert issue is not None
    assert "pre-initialization" in issue.message.lower()
    assert "net1_input1" in issue.message


# ---------------------------------------------------------------------------
# Condition table checks
# ---------------------------------------------------------------------------


def test_condition_valid():
    problem = _get_test_problem()
    assert CheckSciMLConditionTable().run(problem) is None


def test_condition_nn_output_as_target():
    problem = _get_test_problem()
    problem.add_condition("cond2", net1_output1="1.0")
    issue = CheckSciMLConditionTable().run(problem)
    assert issue is not None
    assert "net1_output1" in issue.message


def test_condition_nn_parameters_as_target():
    problem = _get_test_problem()
    problem.add_condition("cond2", net1_ps="1.0")
    issue = CheckSciMLConditionTable().run(problem)
    assert issue is not None
    assert "neural network parameters" in issue.message


def test_condition_nn_input_as_target_value():
    problem = _get_test_problem()
    problem.add_condition("cond2", gamma_="net1_input1")
    issue = CheckSciMLConditionTable().run(problem)
    assert issue is not None
    assert "net1_input1" in issue.message


# ---------------------------------------------------------------------------
# Array data file checks
# ---------------------------------------------------------------------------


def test_array_data_valid():
    problem = _get_test_problem()
    assert CheckArrayDataFiles().run(problem) is None


def test_array_data_missing_initial_condition():
    problem = _get_test_problem()
    # Add an experiment whose initial condition has no array data.
    problem.add_experiment("e2", 0, "cond2")
    issue = CheckArrayDataFiles().run(problem)
    assert issue is not None
    assert "net1_input2" in issue.message
    assert "cond2" in issue.message


# ---------------------------------------------------------------------------
# Parameter table checks
# ---------------------------------------------------------------------------


def test_parameter_valid():
    problem = _get_test_problem()
    assert CheckSciMLParameterTable().run(problem) is None


def test_parameter_nominal_array_without_array_data():
    problem = _get_test_problem()
    problem.extensions.sciml.array_data_files = []
    issue = CheckSciMLParameterTable().run(problem)
    assert issue is not None
    assert "net1_ps" in issue.message
    assert "no array data" in issue.message


def test_parameter_nn_parameters_not_declared():
    problem = _get_test_problem()
    # Remove the NN parameter entry, leaving net1's parameters undeclared.
    pt = problem.parameter_tables[0]
    pt.elements = [p for p in pt.elements if p.id != "net1_ps"]
    issue = CheckSciMLParameterTable().run(problem)
    assert issue is not None
    assert "net1" in issue.message
    assert "not declared" in issue.message


def test_parameter_posterior_requires_bounds_or_prior():
    problem = _get_test_problem()
    # Assign a prior to another parameter -> "posterior" mode. net1_ps is
    # estimated with infinite bounds and no prior -> improper.
    alpha = problem.parameter_tables[0]["alpha"]
    alpha.prior_distribution = "normal"
    alpha.prior_parameters = [0.0, 1.0]
    issue = CheckSciMLParameterTable().run(problem)
    assert issue is not None
    assert "net1_ps" in issue.message


# ---------------------------------------------------------------------------
# Full-problem integration
# ---------------------------------------------------------------------------


def test_validate_reports_sciml_errors():
    """A problem with an NN output as a hybridization target fails
    validation."""
    problem = _get_test_problem()
    problem.extensions.sciml.add_hybridization("net1_output1", "1.0")
    results = problem.validate()
    assert results.has_errors()
