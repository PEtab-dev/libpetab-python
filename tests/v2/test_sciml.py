import numpy as np
from pydantic import ConfigDict

from petab.v2.core import *
from petab.v2.core import ModelFile, NeuralNetConfig
from petab.v2.lint import sciml_validation_tasks
from petab.v2.models.sbml_model import SbmlModel


def _get_test_problem():
    problem = Problem()
    problem.validation_tasks = sciml_validation_tasks
    problem.config = ProblemConfig(
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
            "sciml": {
                "version": "0.1.0",
                "array_files": ["net1_ps.hdf5"],
                "hybridization_files": ["hybridizations.tsv"],
                "neural_networks": {
                    "net1": NeuralNetConfig(
                        location="net1.yaml",
                        pre_initialization=False,
                        format="YAML",
                    )
                },
            }
        },
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
    problem.add_experiment("e1", 0, "")
    problem.add_mapping("net1_input1", "net1.inputs[0][0]")
    problem.add_mapping("net1_input2", "net1.inputs[0][1]")
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
    problem.add_hybridization("net1_input1", "A")
    problem.add_hybridization("net1_input2", "B")
    problem.add_hybridization("gamma_", "net1_output1")
    problem.add_neural_network_from_dict(
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
    problem.add_array_data_from_dict(
        {
            "metadata": {"pytorch_format": True},
            "inputs": {},
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
    problem.hybridization_tables[0].rel_path = "hybridizations.tsv"
    # problem.neural_networks[0].rel_path = "net1.yaml"
    # problem.array_data_files[0].rel_path = "net1_ps.hdf5"

    return problem


def test_lint():
    problem = _get_test_problem()
    assert problem.validate() == []
