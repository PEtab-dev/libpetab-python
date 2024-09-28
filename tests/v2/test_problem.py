import pytest
from petab.v2 import Problem


def test_load_remote():
    """Test loading remote files"""
    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/main/petabtests/cases/v2.0.0/sbml/0001/_0001.yaml"
    )
    petab_problem = Problem.from_yaml(yaml_url)

    assert (
        petab_problem.measurement_df is not None
        and not petab_problem.measurement_df.empty
    )

    assert petab_problem.validate() == []


def test_auto_upgrade():
    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/main/petabtests/cases/v1.0.0/sbml/0001/_0001.yaml"
    )
    problem = Problem.from_yaml(yaml_url)
    # TODO check something specifically different in a v2 problem
    assert isinstance(problem, Problem)


@pytest.fixture
def petab_problem():
    """Test petab problem."""
    # create test model
    import simplesbml

    model = simplesbml.SbmlModel()
    model.addParameter("fixedParameter1", 0.0)
    model.addParameter("observable_1", 0.0)

    measurement_df = pd.DataFrame(
        data={
            OBSERVABLE_ID: ["obs1", "obs2"],
            MEASUREMENT: [0.1, 0.2],
            OBSERVABLE_PARAMETERS: ["", "p1;p2"],
            NOISE_PARAMETERS: ["p3;p4", "p5"],
        }
    )

    condition_df = pd.DataFrame(
        data={
            CONDITION_ID: ["condition1", "condition2"],
            CONDITION_NAME: ["", "Condition 2"],
            "fixedParameter1": [1.0, 2.0],
        }
    ).set_index(CONDITION_ID)

    experiment_df = pd.DataFrame(
        data={
            EXPERIMENT_ID: ["experiment1", "experiment2"],
            EXPERIMENT: [
                "0:condition1;5:condition2",
                "-inf:condition1;0:condition2",
            ],
        }
    ).set_index(EXPERIMENT_ID)

    parameter_df = pd.DataFrame(
        data={
            PARAMETER_ID: ["dynamicParameter1", "dynamicParameter2"],
            PARAMETER_NAME: ["", "..."],
            ESTIMATE: [1, 0],
        }
    ).set_index(PARAMETER_ID)

    observable_df = pd.DataFrame(
        data={
            OBSERVABLE_ID: ["obs1"],
            OBSERVABLE_NAME: ["julius"],
            OBSERVABLE_FORMULA: ["observable_1 * observableParameter1_obs1"],
            NOISE_FORMULA: ["0.1 * observable_1 * observableParameter1_obs1"],
        }
    ).set_index(OBSERVABLE_ID)

    with tempfile.TemporaryDirectory() as temp_dir:
        sbml_file_name = Path(temp_dir, "model.xml")
        libsbml.writeSBMLToFile(model.document, str(sbml_file_name))

        measurement_file_name = Path(temp_dir, "measurements.tsv")
        petab.write_measurement_df(measurement_df, measurement_file_name)

        condition_file_name = Path(temp_dir, "conditions.tsv")
        petab.write_condition_df(condition_df, condition_file_name)

        experiment_file_name = Path(temp_dir, "experiments.tsv")
        petab.write_experiment_df(experiment_df, experiment_file_name)

        parameter_file_name = Path(temp_dir, "parameters.tsv")
        petab.write_parameter_df(parameter_df, parameter_file_name)

        observable_file_name = Path(temp_dir, "observables.tsv")
        petab.write_observable_df(observable_df, observable_file_name)

        with pytest.deprecated_call():
            petab_problem = petab.Problem.from_files(
                sbml_file=sbml_file_name,
                measurement_file=measurement_file_name,
                condition_file=condition_file_name,
                experiment_file=experiment_file_name,
                parameter_file=parameter_file_name,
                observable_files=observable_file_name,
            )
            assert petab_problem.n_measurements == 2
            assert petab_problem.n_estimated == 1
            assert petab_problem.n_priors == 0

            yield petab_problem
