import copy
import pickle
import tempfile
import warnings
from io import StringIO
from math import nan
from pathlib import Path

import libsbml
import numpy as np
import pandas as pd
import petab
import pytest
from petab.C import *
from petab.models.sbml_model import SbmlModel
from yaml import safe_load


@pytest.fixture
def condition_df_2_conditions():
    condition_df = pd.DataFrame(data={
        'conditionId': ['condition1', 'condition2'],
        'conditionName': ['', 'Condition 2'],
        'fixedParameter1': [1.0, 2.0]
    })
    condition_df.set_index('conditionId', inplace=True)
    return condition_df


@pytest.fixture
def petab_problem():
    """Test petab problem."""
    # create test model
    import simplesbml
    model = simplesbml.SbmlModel()
    model.addParameter('fixedParameter1', 0.0)
    model.addParameter('observable_1', 0.0)

    measurement_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['obs1', 'obs2'],
        OBSERVABLE_PARAMETERS: ['', 'p1;p2'],
        NOISE_PARAMETERS: ['p3;p4', 'p5']
    })

    condition_df = pd.DataFrame(data={
        CONDITION_ID: ['condition1', 'condition2'],
        CONDITION_NAME: ['', 'Condition 2'],
        'fixedParameter1': [1.0, 2.0]
    }).set_index(CONDITION_ID)

    parameter_df = pd.DataFrame(data={
        PARAMETER_ID: ['dynamicParameter1', 'dynamicParameter2'],
        PARAMETER_NAME: ['', '...'],
    }).set_index(PARAMETER_ID)

    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['observable_1'],
        OBSERVABLE_NAME: ['julius'],
        OBSERVABLE_FORMULA: ['observable_1'],
        NOISE_FORMULA: [1],
    }).set_index(OBSERVABLE_ID)

    with tempfile.TemporaryDirectory() as temp_dir:
        sbml_file_name = Path(temp_dir, "model.xml")
        libsbml.writeSBMLToFile(model.document, str(sbml_file_name))

        measurement_file_name = Path(temp_dir, "measurements.tsv")
        petab.write_measurement_df(measurement_df, measurement_file_name)

        condition_file_name = Path(temp_dir, "conditions.tsv")
        petab.write_condition_df(condition_df, condition_file_name)

        parameter_file_name = Path(temp_dir, "parameters.tsv")
        petab.write_parameter_df(parameter_df, parameter_file_name)

        observable_file_name = Path(temp_dir, "observables.tsv")
        petab.write_observable_df(observable_df, observable_file_name)

        with pytest.deprecated_call():
            yield petab.Problem.from_files(
                sbml_file=sbml_file_name,
                measurement_file=measurement_file_name,
                condition_file=condition_file_name,
                parameter_file=parameter_file_name,
                observable_files=observable_file_name)


@pytest.fixture
def fujita_model_scaling():
    path = Path(__file__).parent.parent / 'doc' / 'example' / 'example_Fujita'

    sbml_file = path / 'Fujita_model.xml'
    condition_file = path / 'Fujita_experimentalCondition.tsv'
    measurement_file = path / 'Fujita_measurementData.tsv'
    parameter_file = path / 'Fujita_parameters_scaling.tsv'

    with pytest.deprecated_call():
        return petab.Problem.from_files(
            sbml_file=sbml_file,
            condition_file=condition_file,
            measurement_file=measurement_file,
            parameter_file=parameter_file,
        )


def test_split_parameter_replacement_list():
    assert petab.split_parameter_replacement_list('') == []
    assert petab.split_parameter_replacement_list('param1') == ['param1']
    assert petab.split_parameter_replacement_list('param1;param2') \
        == ['param1', 'param2']
    assert petab.split_parameter_replacement_list('1.0') == [1.0]
    assert petab.split_parameter_replacement_list('1.0;2.0') == [1.0, 2.0]
    assert petab.split_parameter_replacement_list('param1;2.2') \
        == ['param1', 2.2]
    assert petab.split_parameter_replacement_list(np.nan) == []
    assert petab.split_parameter_replacement_list(1.5) == [1.5]
    assert petab.split_parameter_replacement_list(None) == []

    with pytest.raises(ValueError):
        assert petab.split_parameter_replacement_list('1.0;')

    with pytest.raises(ValueError):
        assert petab.split_parameter_replacement_list(';1.0')


def test_get_measurement_parameter_ids():
    measurement_df = pd.DataFrame(
        data={
            OBSERVABLE_PARAMETERS: ['', 'p1;p2'],
            NOISE_PARAMETERS: ['p3;p4', 'p5']})
    expected = ['p1', 'p2', 'p3', 'p4', 'p5']
    actual = petab.get_measurement_parameter_ids(measurement_df)
    # ordering is arbitrary
    assert set(actual) == set(expected)


def test_serialization(petab_problem):
    # serialize and back
    problem_recreated = pickle.loads(pickle.dumps(petab_problem))

    assert problem_recreated.measurement_df.equals(
        petab_problem.measurement_df)

    assert problem_recreated.parameter_df.equals(
        petab_problem.parameter_df)

    assert problem_recreated.condition_df.equals(
        petab_problem.condition_df)

    # Can't test for equality directly, testing for number of parameters
    #  should do the job here
    assert len(problem_recreated.sbml_model.getListOfParameters()) \
        == len(petab_problem.sbml_model.getListOfParameters())


def test_get_priors_from_df():
    """Check petab.get_priors_from_df."""
    parameter_df = pd.DataFrame({
        PARAMETER_SCALE: [LOG10, LOG10, LOG10, LOG10, LOG10],
        LOWER_BOUND: [1e-8, 1e-9, 1e-10, 1e-11, 1e-5],
        UPPER_BOUND: [1e8, 1e9, 1e10, 1e11, 1e5],
        ESTIMATE: [1, 1, 1, 1, 0],
        INITIALIZATION_PRIOR_TYPE: ['', '',
                                    UNIFORM, NORMAL, ''],
        INITIALIZATION_PRIOR_PARAMETERS: ['', '-5;5', '1e-5;1e5', '0;1', '']
    })

    prior_list = petab.get_priors_from_df(parameter_df, mode=INITIALIZATION)

    # only give values for estimated parameters
    assert len(prior_list) == 4

    # correct types
    types = [entry[0] for entry in prior_list]
    assert types == [PARAMETER_SCALE_UNIFORM, PARAMETER_SCALE_UNIFORM,
                     UNIFORM, NORMAL]

    # correct scales
    scales = [entry[2] for entry in prior_list]
    assert scales == [LOG10] * 4

    # correct bounds
    bounds = [entry[3] for entry in prior_list]
    assert bounds == list(zip(parameter_df[LOWER_BOUND],
                              parameter_df[UPPER_BOUND]))[:4]

    # give correct value for empty
    prior_pars = [entry[1] for entry in prior_list]
    assert prior_pars[0] == (-8, 8)
    assert prior_pars[1] == (-5, 5)
    assert prior_pars[2] == (1e-5, 1e5)


def test_startpoint_sampling(fujita_model_scaling):
    n_starts = 10
    startpoints = fujita_model_scaling.sample_parameter_startpoints(n_starts)
    assert (np.isfinite(startpoints)).all
    assert startpoints.shape == (n_starts, 19)
    for sp in startpoints:
        assert np.log10(31.62) <= sp[0] <= np.log10(316.23)
        assert -3 <= sp[1] <= 3


def test_startpoint_sampling_dict(fujita_model_scaling):
    n_starts = 10
    startpoints = fujita_model_scaling.sample_parameter_startpoints_dict(
        n_starts)
    assert len(startpoints) == n_starts
    for startpoint in startpoints:
        assert set(startpoint.keys()) == set(fujita_model_scaling.x_free_ids)


def test_create_parameter_df(
        condition_df_2_conditions):  # pylint: disable=W0621
    """Test petab.create_parameter_df."""
    import simplesbml
    ss_model = simplesbml.SbmlModel()
    ss_model.addSpecies('[x1]', 1.0)
    ss_model.addParameter('fixedParameter1', 2.0)
    ss_model.addParameter('p0', 3.0)
    model = SbmlModel(sbml_model=ss_model.model)

    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['obs1', 'obs2'],
        OBSERVABLE_FORMULA: ['x1', '2*x1']
    }).set_index(OBSERVABLE_ID)

    # Add assignment rule target which should be ignored
    ss_model.addParameter('assignment_target', 0.0)
    ss_model.addAssignmentRule('assignment_target', "1.0")

    measurement_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['obs1', 'obs2'],
        OBSERVABLE_PARAMETERS: ['', 'p1;p2'],
        NOISE_PARAMETERS: ['p3;p4', 'p5']
    })

    # first model parameters, then row by row noise and sigma overrides
    expected = ['p3', 'p4', 'p1', 'p2', 'p5']

    # Test old API with passing libsbml.Model directly
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parameter_df = petab.create_parameter_df(
            ss_model.model,
            condition_df_2_conditions,
            observable_df,
            measurement_df)
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert parameter_df.index.values.tolist() == expected

    parameter_df = petab.create_parameter_df(
        model=model,
        condition_df=condition_df_2_conditions,
        observable_df=observable_df,
        measurement_df=measurement_df
    )
    assert parameter_df.index.values.tolist() == expected

    # test with condition parameter override:
    condition_df_2_conditions.loc['condition2', 'fixedParameter1'] \
        = 'overrider'
    expected = ['p3', 'p4', 'p1', 'p2', 'p5', 'overrider']

    parameter_df = petab.create_parameter_df(
        model=model,
        condition_df=condition_df_2_conditions,
        observable_df=observable_df,
        measurement_df=measurement_df,
    )
    actual = parameter_df.index.values.tolist()
    assert actual == expected

    # test with optional parameters
    expected = ['p0', 'p3', 'p4', 'p1', 'p2', 'p5', 'overrider']

    parameter_df = petab.create_parameter_df(
        model=model,
        condition_df=condition_df_2_conditions,
        observable_df=observable_df,
        measurement_df=measurement_df,
        include_optional=True)
    actual = parameter_df.index.values.tolist()
    assert actual == expected
    assert parameter_df.loc['p0', NOMINAL_VALUE] == 3.0


def test_flatten_timepoint_specific_output_overrides():
    """Test flatten_timepoint_specific_output_overrides"""
    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['obs1'],
        OBSERVABLE_FORMULA: [
            'observableParameter1_obs1 + observableParameter2_obs1'],
        NOISE_FORMULA: ['noiseParameter1_obs1']
    })
    observable_df.set_index(OBSERVABLE_ID, inplace=True)

    observable_df_expected = pd.DataFrame(data={
        OBSERVABLE_ID: [
             'obs1__obsParOverride1_1_0__noiseParOverride1__condition1',
             'obs1__obsParOverride2_1_0__noiseParOverride1__condition1',
             'obs1__obsParOverride2_1_0__noiseParOverride2__condition1',
        ],
        OBSERVABLE_FORMULA: [
            'observableParameter1_obs1__obsParOverride1_1_0__'
            'noiseParOverride1__condition1 + observableParameter2_obs1'
            '__obsParOverride1_1_0__noiseParOverride1__condition1',
            'observableParameter1_obs1__obsParOverride2_1_0__noiseParOverride1'
            '__condition1 + observableParameter2_obs1__obsParOverride2_1_0'
            '__noiseParOverride1__condition1',
            'observableParameter1_obs1__obsParOverride2_1_0'
            '__noiseParOverride2__condition1 + observableParameter2_obs1__'
            'obsParOverride2_1_0__noiseParOverride2__condition1'],
        NOISE_FORMULA: ['noiseParameter1_obs1__obsParOverride1_1_0__'
                        'noiseParOverride1__condition1',
                        'noiseParameter1_obs1__obsParOverride2_1_0__'
                        'noiseParOverride1__condition1',
                        'noiseParameter1_obs1__obsParOverride2_1_0__'
                        'noiseParOverride2__condition1']
    })
    observable_df_expected.set_index(OBSERVABLE_ID, inplace=True)

    # Measurement table with timepoint-specific overrides
    measurement_df = pd.DataFrame(data={
        OBSERVABLE_ID:
            ['obs1', 'obs1', 'obs1', 'obs1'],
        SIMULATION_CONDITION_ID:
            ['condition1', 'condition1', 'condition1', 'condition1'],
        PREEQUILIBRATION_CONDITION_ID:
            ['', '', '', ''],
        TIME:
            [1.0, 1.0, 2.0, 2.0],
        MEASUREMENT:
            [.1] * 4,
        OBSERVABLE_PARAMETERS:
            ['obsParOverride1;1.0', 'obsParOverride2;1.0',
             'obsParOverride2;1.0', 'obsParOverride2;1.0'],
        NOISE_PARAMETERS:
            ['noiseParOverride1', 'noiseParOverride1',
             'noiseParOverride2', 'noiseParOverride2']
    })

    measurement_df_expected = pd.DataFrame(data={
        OBSERVABLE_ID:
            ['obs1__obsParOverride1_1_0__noiseParOverride1__condition1',
             'obs1__obsParOverride2_1_0__noiseParOverride1__condition1',
             'obs1__obsParOverride2_1_0__noiseParOverride2__condition1',
             'obs1__obsParOverride2_1_0__noiseParOverride2__condition1'],
        SIMULATION_CONDITION_ID:
            ['condition1', 'condition1', 'condition1', 'condition1'],
        PREEQUILIBRATION_CONDITION_ID:
            ['', '', '', ''],
        TIME:
            [1.0, 1.0, 2.0, 2.0],
        MEASUREMENT:
            [.1] * 4,
        OBSERVABLE_PARAMETERS:
            ['obsParOverride1;1.0', 'obsParOverride2;1.0',
             'obsParOverride2;1.0', 'obsParOverride2;1.0'],
        NOISE_PARAMETERS:
            ['noiseParOverride1', 'noiseParOverride1',
             'noiseParOverride2', 'noiseParOverride2']
    })

    problem = petab.Problem(measurement_df=measurement_df,
                            observable_df=observable_df)

    assert petab.lint_problem(problem) is False

    # Ensure having timepoint-specific overrides
    assert petab.lint.measurement_table_has_timepoint_specific_mappings(
        measurement_df) is True

    petab.flatten_timepoint_specific_output_overrides(problem)

    # Timepoint-specific overrides should be gone now
    assert petab.lint.measurement_table_has_timepoint_specific_mappings(
        problem.measurement_df) is False

    assert problem.observable_df.equals(observable_df_expected) is True
    assert problem.measurement_df.equals(measurement_df_expected) is True

    assert petab.lint_problem(problem) is False

    simulation_df = copy.deepcopy(problem.measurement_df)
    simulation_df.rename(columns={MEASUREMENT: SIMULATION})
    unflattened_problem = petab.Problem(
        measurement_df=measurement_df,
        observable_df=observable_df,
    )
    unflattened_simulation_df = petab.core.unflatten_simulation_df(
        simulation_df=simulation_df,
        petab_problem=unflattened_problem,
    )
    # The unflattened simulation dataframe has the original observable IDs.
    assert (unflattened_simulation_df[OBSERVABLE_ID] == 'obs1').all()


def test_flatten_timepoint_specific_output_overrides_special_cases():
    """Test flatten_timepoint_specific_output_overrides
    for special cases:
    * no preequilibration
    * no observable parameters
    """
    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['obs1'],
        OBSERVABLE_FORMULA: ['species1'],
        NOISE_FORMULA: ['noiseParameter1_obs1']
    })
    observable_df.set_index(OBSERVABLE_ID, inplace=True)

    observable_df_expected = pd.DataFrame(data={
        OBSERVABLE_ID: ['obs1__noiseParOverride1__condition1',
                        'obs1__noiseParOverride2__condition1'],
        OBSERVABLE_FORMULA: [
            'species1',
            'species1'],
        NOISE_FORMULA: ['noiseParameter1_obs1__noiseParOverride1__condition1',
                        'noiseParameter1_obs1__noiseParOverride2__condition1']
    })
    observable_df_expected.set_index(OBSERVABLE_ID, inplace=True)

    # Measurement table with timepoint-specific overrides
    measurement_df = pd.DataFrame(data={
        OBSERVABLE_ID:
            ['obs1', 'obs1', 'obs1', 'obs1'],
        SIMULATION_CONDITION_ID:
            ['condition1', 'condition1', 'condition1', 'condition1'],
        TIME:
            [1.0, 1.0, 2.0, 2.0],
        MEASUREMENT:
            [.1] * 4,
        NOISE_PARAMETERS:
            ['noiseParOverride1', 'noiseParOverride1',
             'noiseParOverride2', 'noiseParOverride2'],
    })

    measurement_df_expected = pd.DataFrame(data={
        OBSERVABLE_ID:
            ['obs1__noiseParOverride1__condition1',
             'obs1__noiseParOverride1__condition1',
             'obs1__noiseParOverride2__condition1',
             'obs1__noiseParOverride2__condition1'],
        SIMULATION_CONDITION_ID:
            ['condition1', 'condition1', 'condition1', 'condition1'],
        TIME:
            [1.0, 1.0, 2.0, 2.0],
        MEASUREMENT:
            [.1] * 4,
        NOISE_PARAMETERS:
            ['noiseParOverride1', 'noiseParOverride1',
             'noiseParOverride2', 'noiseParOverride2'],
    })

    problem = petab.Problem(measurement_df=measurement_df,
                            observable_df=observable_df)

    assert petab.lint_problem(problem) is False

    # Ensure having timepoint-specific overrides
    assert petab.lint.measurement_table_has_timepoint_specific_mappings(
        measurement_df) is True

    petab.flatten_timepoint_specific_output_overrides(problem)

    # Timepoint-specific overrides should be gone now
    assert petab.lint.measurement_table_has_timepoint_specific_mappings(
        problem.measurement_df) is False

    assert problem.observable_df.equals(observable_df_expected) is True
    assert problem.measurement_df.equals(measurement_df_expected) is True

    assert petab.lint_problem(problem) is False


def test_concat_measurements():
    a = pd.DataFrame({MEASUREMENT: [1.0]})
    b = pd.DataFrame({TIME: [1.0]})

    with tempfile.TemporaryDirectory() as temp_dir:
        filename_a = Path(temp_dir) / "measurements.tsv"
        petab.write_measurement_df(a, filename_a)

        expected = pd.DataFrame({
            MEASUREMENT: [1.0, nan],
            TIME: [nan, 1.0]
        })

        assert expected.equals(
            petab.concat_tables([a, b],
                                petab.measurements.get_measurement_df))

        assert expected.equals(
            petab.concat_tables([filename_a, b],
                                petab.measurements.get_measurement_df))


def test_concat_condition_df():
    df1 = pd.DataFrame(data={
        CONDITION_ID: ['condition1', 'condition2'],
        'par1': [1.1, 1.2],
        'par2': [2.1, 2.2],
        'par3': [3.1, 3.2]
    }).set_index(CONDITION_ID)

    assert df1.equals(petab.concat_tables(df1, petab.get_condition_df))

    df2 = pd.DataFrame(data={
        CONDITION_ID: ['condition3'],
        'par1': [1.3],
        'par2': [2.3],
    }).set_index(CONDITION_ID)

    df_expected = pd.DataFrame(data={
        CONDITION_ID: ['condition1', 'condition2', 'condition3'],
        'par1': [1.1, 1.2, 1.3],
        'par2': [2.1, 2.2, 2.3],
        'par3': [3.1, 3.2, np.nan],
    }).set_index(CONDITION_ID)
    assert df_expected.equals(
        petab.concat_tables((df1, df2), petab.get_condition_df)
    )


def test_get_observable_ids(petab_problem):  # pylint: disable=W0621
    """Test if observable ids functions returns correct value."""
    assert set(petab_problem.get_observable_ids()) == {'observable_1'}


def test_parameter_properties(petab_problem):  # pylint: disable=W0621
    """
    Test the petab.Problem functions to get parameter values.
    """
    petab_problem = copy.deepcopy(petab_problem)
    petab_problem.parameter_df = pd.DataFrame(data={
        PARAMETER_ID: ['par1', 'par2', 'par3'],
        LOWER_BOUND: [0, 0.1, 0.1],
        UPPER_BOUND: [100, 100, 200],
        PARAMETER_SCALE: ['lin', 'log', 'log10'],
        NOMINAL_VALUE: [7, 8, 9],
        ESTIMATE: [1, 1, 0],
    }).set_index(PARAMETER_ID)
    assert petab_problem.x_ids == ['par1', 'par2', 'par3']
    assert petab_problem.x_free_ids == ['par1', 'par2']
    assert petab_problem.x_fixed_ids == ['par3']
    assert petab_problem.lb == [0, 0.1, 0.1]
    assert petab_problem.lb_scaled == [0, np.log(0.1), np.log10(0.1)]
    assert petab_problem.get_lb(fixed=False, scaled=True) == [0, np.log(0.1)]
    assert petab_problem.ub == [100, 100, 200]
    assert petab_problem.ub_scaled == [100, np.log(100), np.log10(200)]
    assert petab_problem.get_ub(fixed=False, scaled=True) == [100, np.log(100)]
    assert petab_problem.x_nominal == [7, 8, 9]
    assert petab_problem.x_nominal_scaled == [7, np.log(8), np.log10(9)]
    assert petab_problem.x_nominal_free == [7, 8]
    assert petab_problem.x_nominal_fixed == [9]
    assert petab_problem.x_nominal_free_scaled == [7, np.log(8)]
    assert petab_problem.x_nominal_fixed_scaled == [np.log10(9)]


def test_to_float_if_float():
    to_float_if_float = petab.core.to_float_if_float

    assert to_float_if_float(1) == 1.0
    assert to_float_if_float("1") == 1.0
    assert to_float_if_float("-1.0") == -1.0
    assert to_float_if_float("1e1") == 10.0
    assert to_float_if_float("abc") == "abc"
    assert to_float_if_float([]) == []


def test_to_files(petab_problem):  # pylint: disable=W0621
    """Test problem.to_files."""
    with tempfile.TemporaryDirectory() as outdir:
        # create target files
        sbml_file = Path(outdir, "model.xml")
        condition_file = Path(outdir, "conditions.tsv")
        measurement_file = Path(outdir, "measurements.tsv")
        parameter_file = Path(outdir, "parameters.tsv")
        observable_file = Path(outdir, "observables.tsv")

        # write contents to files
        petab_problem.to_files(
            model_file=sbml_file,
            condition_file=condition_file,
            measurement_file=measurement_file,
            parameter_file=parameter_file,
            visualization_file=None,
            observable_file=observable_file,
            yaml_file=None,
        )

        # exemplarily load some
        parameter_df = petab.get_parameter_df(parameter_file)
        same_nans = parameter_df.isna() == petab_problem.parameter_df.isna()
        assert ((parameter_df == petab_problem.parameter_df) | same_nans) \
            .all().all()


def test_load_remote():
    """Test loading remote files"""

    yaml_url = "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite" \
               "/main/petabtests/cases/v1.0.0/sbml/0001/_0001.yaml"
    petab_problem = petab.Problem.from_yaml(yaml_url)

    assert petab_problem.sbml_model is not None
    assert petab_problem.measurement_df is not None \
           and not petab_problem.measurement_df.empty


def test_problem_from_yaml_v1_empty():
    """Test loading PEtab version 1 yaml without any files"""
    yaml_config = """
    format_version: 1
    parameter_file:
    problems:
    - condition_files: []
      measurement_files: []
      observable_files: []
      sbml_files: []
    """
    yaml_config = safe_load(StringIO(yaml_config))
    petab.Problem.from_yaml(yaml_config)


def test_problem_from_yaml_v1_multiple_files():
    """Test loading PEtab version 1 yaml with multiple condition / measurement
    / observable files"""
    yaml_config = """
    format_version: 1
    parameter_file:
    problems:
    - condition_files: [conditions1.tsv, conditions2.tsv]
      measurement_files: [measurements1.tsv, measurements2.tsv]
      observable_files: [observables1.tsv, observables2.tsv]
      sbml_files: []
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir, "problem.yaml")
        with open(yaml_path, 'w') as f:
            f.write(yaml_config)

        for i in (1, 2):
            condition_df = pd.DataFrame({
                CONDITION_ID: [f"condition{i}"],
            })
            condition_df.set_index([CONDITION_ID], inplace=True)
            petab.write_condition_df(condition_df,
                                     Path(tmpdir, f"conditions{i}.tsv"))

            measurement_df = pd.DataFrame({
                SIMULATION_CONDITION_ID: [f"condition{i}"],
                OBSERVABLE_ID: [f"observable{i}"],
                TIME: [i],
                MEASUREMENT: [1]
            })
            petab.write_measurement_df(measurement_df,
                                       Path(tmpdir, f"measurements{i}.tsv"))

            observables_df = pd.DataFrame({
                OBSERVABLE_ID: [f"observable{i}"],
                OBSERVABLE_FORMULA: [1],
                NOISE_FORMULA: [1],
            })
            petab.write_observable_df(observables_df,
                                      Path(tmpdir, f"observables{i}.tsv"))

        petab_problem = petab.Problem.from_yaml(yaml_path)

    assert petab_problem.measurement_df.shape[0] == 2
    assert petab_problem.observable_df.shape[0] == 2
    assert petab_problem.condition_df.shape[0] == 2
