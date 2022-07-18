import os
import subprocess
from math import nan
from unittest.mock import patch

import pandas as pd
import pytest

import petab
from petab import lint
from petab.C import *

# import fixtures
pytest_plugins = [
    "tests.test_petab",
]


def test_assert_measured_observables_present():
    # create test model

    measurement_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['non-existing1'],
    })

    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['obs1'],
    })
    observable_df.set_index(OBSERVABLE_ID, inplace=True)

    with pytest.raises(AssertionError):
        lint.assert_measured_observables_defined(
            measurement_df, observable_df
        )


def test_condition_table_is_parameter_free():
    with patch('petab.get_parametric_overrides') \
            as mock_get_parametric_overrides:
        mock_get_parametric_overrides.return_value = []
        assert lint.condition_table_is_parameter_free(pd.DataFrame()) is True
        mock_get_parametric_overrides.assert_called_once()

        mock_get_parametric_overrides.reset_mock()
        mock_get_parametric_overrides.return_value = ['p1']
        assert lint.condition_table_is_parameter_free(pd.DataFrame()) is False
        mock_get_parametric_overrides.assert_called_once()


def test_measurement_table_has_timepoint_specific_mappings():
    # Ensure we fail if we have time-point specific assignments

    measurement_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['obs1', 'obs1'],
        SIMULATION_CONDITION_ID: ['condition1', 'condition1'],
        PREEQUILIBRATION_CONDITION_ID: [nan, nan],
        TIME: [1.0, 2.0],
        OBSERVABLE_PARAMETERS: ['obsParOverride', ''],
        NOISE_PARAMETERS: ['1.0', 1.0]
    })

    assert lint.measurement_table_has_timepoint_specific_mappings(
        measurement_df) is True

    # both measurements different anyways
    measurement_df.loc[1, OBSERVABLE_ID] = 'obs2'
    assert lint.measurement_table_has_timepoint_specific_mappings(
        measurement_df) is False

    # mixed numeric string
    measurement_df.loc[1, OBSERVABLE_ID] = 'obs1'
    measurement_df.loc[1, OBSERVABLE_PARAMETERS] = 'obsParOverride'
    assert lint.measurement_table_has_timepoint_specific_mappings(
        measurement_df) is False

    # different numeric values
    measurement_df.loc[1, NOISE_PARAMETERS] = 2.0
    assert lint.measurement_table_has_timepoint_specific_mappings(
        measurement_df) is True
    assert lint.measurement_table_has_timepoint_specific_mappings(
        measurement_df, allow_scalar_numeric_noise_parameters=True) is False


def test_observable_table_has_nontrivial_noise_formula():
    # Ensure we fail if we have nontrivial noise formulas

    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['0obsPar1noisePar', '2obsPar0noisePar',
                        '3obsPar0noisePar'],
        OBSERVABLE_FORMULA: ['1.0',
                             '1.0',
                             '1.0'],
        NOISE_FORMULA: ['noiseParameter1_0obsPar1noisePar + 3.0',
                        1e18,
                        '1e18']
    })

    assert lint.observable_table_has_nontrivial_noise_formula(observable_df)\
        is True

    observable_df.loc[0, NOISE_FORMULA] = 'sigma1'

    assert lint.observable_table_has_nontrivial_noise_formula(observable_df) \
        is False


def test_assert_overrides_match_parameter_count():
    # Ensure we recognize and fail if we have wrong number of overrides
    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['0obsPar1noisePar', '2obsPar0noisePar'],
        OBSERVABLE_FORMULA: ['1.0',
                             'observableParameter1_2obsPar0noisePar + '
                             'observableParameter2_2obsPar0noisePar'],
        NOISE_FORMULA: ['noiseParameter1_0obsPar1noisePar', '1.0']
    })
    observable_df.set_index(OBSERVABLE_ID, inplace=True)

    measurement_df_orig = pd.DataFrame(data={
        OBSERVABLE_ID: ['0obsPar1noisePar',
                        '2obsPar0noisePar'],
        SIMULATION_CONDITION_ID: ['condition1', 'condition1'],
        PREEQUILIBRATION_CONDITION_ID: ['', ''],
        TIME: [1.0, 2.0],
        OBSERVABLE_PARAMETERS: ['', 'override1;override2'],
        NOISE_PARAMETERS: ['noiseParOverride', '']
    })

    # valid
    petab.assert_overrides_match_parameter_count(
        measurement_df_orig, observable_df)

    # 0 noise parameters given, 1 expected
    measurement_df = measurement_df_orig.copy()
    measurement_df.loc[0, NOISE_PARAMETERS] = ''
    with pytest.raises(AssertionError):
        petab.assert_overrides_match_parameter_count(
            measurement_df, observable_df)

    # 2 noise parameters given, 1 expected
    measurement_df = measurement_df_orig.copy()
    measurement_df.loc[0, NOISE_PARAMETERS] = 'noiseParOverride;oneTooMuch'
    with pytest.raises(AssertionError):
        petab.assert_overrides_match_parameter_count(
            measurement_df, observable_df)

    # 1 noise parameter given, 0 allowed
    measurement_df = measurement_df_orig.copy()
    measurement_df.loc[1, NOISE_PARAMETERS] = 'oneTooMuch'
    with pytest.raises(AssertionError):
        petab.assert_overrides_match_parameter_count(
            measurement_df, observable_df)

    # 0 observable parameters given, 2 expected
    measurement_df = measurement_df_orig.copy()
    measurement_df.loc[1, OBSERVABLE_PARAMETERS] = ''
    with pytest.raises(AssertionError):
        petab.assert_overrides_match_parameter_count(
            measurement_df, observable_df)

    # 1 observable parameters given, 2 expected
    measurement_df = measurement_df_orig.copy()
    measurement_df.loc[1, OBSERVABLE_PARAMETERS] = 'oneMissing'
    with pytest.raises(AssertionError):
        petab.assert_overrides_match_parameter_count(
            measurement_df, observable_df)

    # 3 observable parameters given, 2 expected
    measurement_df = measurement_df_orig.copy()
    measurement_df.loc[1, OBSERVABLE_PARAMETERS] = \
        'override1;override2;oneTooMuch'
    with pytest.raises(AssertionError):
        petab.assert_overrides_match_parameter_count(
            measurement_df, observable_df)

    # 1 observable parameters given, 0 expected
    measurement_df = measurement_df_orig.copy()
    measurement_df.loc[0, OBSERVABLE_PARAMETERS] = \
        'oneTooMuch'
    with pytest.raises(AssertionError):
        petab.assert_overrides_match_parameter_count(
            measurement_df, observable_df)


def test_assert_no_leading_trailing_whitespace():

    test_df = pd.DataFrame(data={
        'testId': ['name1 ', 'name2'],
        'testText ': [' name1', 'name2'],
        'testNumeric': [1.0, 2.0],
        'testNone': None
    })

    with pytest.raises(AssertionError):
        lint.assert_no_leading_trailing_whitespace(
            test_df.columns.values, "test")

    with pytest.raises(AssertionError):
        lint.assert_no_leading_trailing_whitespace(
            test_df['testId'].values, "testId")

    with pytest.raises(AssertionError):
        lint.assert_no_leading_trailing_whitespace(
            test_df['testText '].values, "testText")

    lint.assert_no_leading_trailing_whitespace(
        test_df['testNumeric'].values, "testNumeric")

    lint.assert_no_leading_trailing_whitespace(
        test_df['testNone'].values, "testNone")


def test_assert_model_parameters_in_condition_or_parameter_table():
    import simplesbml
    from petab.models.sbml_model import SbmlModel
    ss_model = simplesbml.SbmlModel()
    ss_model.addParameter('parameter1', 0.0)
    ss_model.addParameter('noiseParameter1_', 0.0)
    ss_model.addParameter('observableParameter1_', 0.0)
    sbml_model = SbmlModel(sbml_model=ss_model.model)

    lint.assert_model_parameters_in_condition_or_parameter_table(
            sbml_model, pd.DataFrame(columns=['parameter1']), pd.DataFrame()
    )

    lint.assert_model_parameters_in_condition_or_parameter_table(
            sbml_model, pd.DataFrame(), pd.DataFrame(index=['parameter1']))

    with pytest.raises(AssertionError):
        lint.assert_model_parameters_in_condition_or_parameter_table(
            sbml_model,
            pd.DataFrame(columns=['parameter1']),
            pd.DataFrame(index=['parameter1']))

    lint.assert_model_parameters_in_condition_or_parameter_table(
            sbml_model, pd.DataFrame(), pd.DataFrame())

    ss_model.addAssignmentRule('parameter1', 'parameter2')
    lint.assert_model_parameters_in_condition_or_parameter_table(
        sbml_model, pd.DataFrame(), pd.DataFrame())


def test_assert_noise_distributions_valid():
    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['0obsPar1noisePar',
                        '2obsPar0noisePar'],
        NOISE_PARAMETERS: ['', ''],
        NOISE_DISTRIBUTION: ['', ''],
    })
    observable_df.set_index([OBSERVABLE_ID], inplace=True)

    lint.assert_noise_distributions_valid(observable_df)

    observable_df[OBSERVABLE_TRANSFORMATION] = [LIN, LOG]
    observable_df[NOISE_DISTRIBUTION] = [NORMAL, '']
    lint.assert_noise_distributions_valid(observable_df)

    observable_df[NOISE_DISTRIBUTION] = ['Normal', '']
    with pytest.raises(ValueError):
        lint.assert_noise_distributions_valid(observable_df)

    observable_df.drop(columns=NOISE_DISTRIBUTION, inplace=True)
    lint.assert_noise_distributions_valid(observable_df)


def test_check_measurement_df():
    """Check measurement (and observable) tables"""
    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['0obsPar1noisePar',
                        '2obsPar0noisePar'],
        OBSERVABLE_FORMULA: ['', ''],
        NOISE_FORMULA: ['', ''],
        NOISE_DISTRIBUTION: ['', ''],
    })
    observable_df.set_index([OBSERVABLE_ID], inplace=True)

    measurement_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['0obsPar1noisePar',
                        '2obsPar0noisePar'],
        SIMULATION_CONDITION_ID: ['condition1', 'condition1'],
        PREEQUILIBRATION_CONDITION_ID: ['', ''],
        TIME: [1.0, 2.0],
        MEASUREMENT: [1.0, 2.0],
        OBSERVABLE_PARAMETERS: ['', ''],
        NOISE_PARAMETERS: ['', ''],
    })

    lint.check_measurement_df(measurement_df, observable_df)

    # Incompatible measurement and transformation
    observable_df[OBSERVABLE_TRANSFORMATION] = [LOG, '']
    measurement_df[MEASUREMENT] = [-1.0, 0.0]
    with pytest.raises(ValueError):
        lint.check_measurement_df(measurement_df, observable_df)


def test_check_parameter_bounds():
    lint.check_parameter_bounds(pd.DataFrame(
        {LOWER_BOUND: [1], UPPER_BOUND: [2], ESTIMATE: [1]}))

    with pytest.raises(AssertionError):
        lint.check_parameter_bounds(pd.DataFrame(
            {LOWER_BOUND: [3], UPPER_BOUND: [2], ESTIMATE: [1]}))

    with pytest.raises(AssertionError):
        lint.check_parameter_bounds(pd.DataFrame(
            {LOWER_BOUND: [-1], UPPER_BOUND: [2],
             ESTIMATE: [1], PARAMETER_SCALE: [LOG10]}))

    with pytest.raises(AssertionError):
        lint.check_parameter_bounds(pd.DataFrame(
            {LOWER_BOUND: [-1], UPPER_BOUND: [2],
             ESTIMATE: [1], PARAMETER_SCALE: [LOG]}))


def test_assert_parameter_prior_type_is_valid():
    """Check lint.assert_parameter_prior_type_is_valid."""
    lint.assert_parameter_prior_type_is_valid(pd.DataFrame(
        {INITIALIZATION_PRIOR_TYPE: [UNIFORM, LAPLACE, ''],
         OBJECTIVE_PRIOR_TYPE: [NORMAL, LOG_NORMAL, '']}))
    lint.assert_parameter_prior_type_is_valid(pd.DataFrame())

    with pytest.raises(AssertionError):
        lint.assert_parameter_prior_type_is_valid(pd.DataFrame(
            {INITIALIZATION_PRIOR_TYPE: ['normel']}))


def test_assert_parameter_prior_parameters_are_valid():
    """Check lint.assert_parameter_prior_parameters_are_valid."""
    parameter_df = pd.DataFrame({
        INITIALIZATION_PRIOR_TYPE: [UNIFORM, '', ''],
        INITIALIZATION_PRIOR_PARAMETERS: ['0;1', '10;20', ''],
        OBJECTIVE_PRIOR_PARAMETERS: ['0;20', '10;20', '']
    })

    lint.assert_parameter_prior_parameters_are_valid(parameter_df)

    with pytest.raises(AssertionError):
        lint.assert_parameter_prior_parameters_are_valid(pd.DataFrame(
            {INITIALIZATION_PRIOR_TYPE: [NORMAL]}))

    with pytest.raises(AssertionError):
        lint.assert_parameter_prior_parameters_are_valid(pd.DataFrame(
            {OBJECTIVE_PRIOR_PARAMETERS: ['0;1;2']}))


def test_petablint_succeeds():
    """Run petablint and ensure we exit successfully for a file that should
    contain no errors"""
    dir_isensee = '../doc/example/example_Isensee/'
    dir_fujita = '../doc/example/example_Fujita/'

    # run with measurement file
    script_path = os.path.abspath(os.path.dirname(__file__))
    measurement_file = os.path.join(
        script_path, dir_isensee, 'Isensee_measurementData.tsv')
    result = subprocess.run(['petablint', '-m', measurement_file])
    assert result.returncode == 0

    # run with yaml
    yaml_file = os.path.join(script_path, dir_fujita, 'Fujita.yaml')
    result = subprocess.run(['petablint', '-v', '-y', yaml_file])
    assert result.returncode == 0

    parameter_file = os.path.join(
        script_path, dir_fujita, 'Fujita_parameters.tsv')
    result = subprocess.run(['petablint', '-v', '-p', parameter_file])
    assert result.returncode == 0


def test_assert_measurement_conditions_present_in_condition_table():
    condition_df = pd.DataFrame(data={
        CONDITION_ID: ['condition1', 'condition2'],
        CONDITION_NAME: ['', 'Condition 2'],
        'fixedParameter1': [1.0, 2.0]
    })
    condition_df.set_index(CONDITION_ID, inplace=True)

    measurement_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['', ''],
        SIMULATION_CONDITION_ID: ['condition1', 'condition1'],
        TIME: [1.0, 2.0],
        MEASUREMENT: [1.0, 2.0],
        OBSERVABLE_PARAMETERS: ['', ''],
        NOISE_PARAMETERS: ['', ''],
    })

    # check we can handle missing preeq condition
    lint.assert_measurement_conditions_present_in_condition_table(
        measurement_df=measurement_df, condition_df=condition_df)

    # check we can handle preeq condition
    measurement_df[PREEQUILIBRATION_CONDITION_ID] = ['condition1',
                                                     'condition2']

    lint.assert_measurement_conditions_present_in_condition_table(
        measurement_df=measurement_df, condition_df=condition_df)

    # check we detect missing condition
    measurement_df[PREEQUILIBRATION_CONDITION_ID] = ['missing_condition1',
                                                     'missing_condition2']
    with pytest.raises(AssertionError):
        lint.assert_measurement_conditions_present_in_condition_table(
            measurement_df=measurement_df, condition_df=condition_df)


def test_check_condition_df():
    """Check that we correctly detect errors in condition table"""
    import simplesbml
    from petab.models.sbml_model import SbmlModel
    ss_model = simplesbml.SbmlModel()
    model = SbmlModel(sbml_model=ss_model.model)
    condition_df = pd.DataFrame(data={
        CONDITION_ID: ['condition1'],
        'p1': [nan],
    })
    condition_df.set_index(CONDITION_ID, inplace=True)

    # parameter missing in model
    with pytest.raises(AssertionError):
        lint.check_condition_df(condition_df, model)

    # fix by adding output parameter
    observable_df = pd.DataFrame({
        OBSERVABLE_ID: ["obs1"],
        OBSERVABLE_FORMULA: ["p1"],
    })
    lint.check_condition_df(condition_df, model, observable_df)

    # fix by adding parameter
    ss_model.addParameter('p1', 1.0)
    lint.check_condition_df(condition_df, model)

    # species missing in model
    condition_df['s1'] = [3.0]
    with pytest.raises(AssertionError):
        lint.check_condition_df(condition_df, model)

    # fix:
    ss_model.addSpecies("[s1]", 1.0)
    lint.check_condition_df(condition_df, model)

    # compartment missing in model
    condition_df['c2'] = [4.0]
    with pytest.raises(AssertionError):
        lint.check_condition_df(condition_df, model)

    # fix:
    ss_model.addCompartment(comp_id='c2', vol=1.0)
    lint.check_condition_df(condition_df, model)


def test_check_ids():
    """Test check_ids"""

    lint.check_ids(['a1', '_1'])

    with pytest.raises(ValueError):
        lint.check_ids(['1'])


def test_check_parameter_df():
    """Check lint.check_parameter_df."""

    parameter_df = pd.DataFrame({
        PARAMETER_ID: ['par0', 'par1', 'par2'],
        PARAMETER_SCALE: [LOG10, LOG10, LIN],
        NOMINAL_VALUE: [1e-2, 1e-3, 1e-4],
        ESTIMATE: [1, 1, 0],
        LOWER_BOUND: [1e-5, 1e-6, 1e-7],
        UPPER_BOUND: [1e5, 1e6, 1e7]
    }).set_index(PARAMETER_ID)

    lint.check_parameter_df(df=parameter_df)

    # NOMINAL_VALUE empty, for non-estimated parameter
    parameter_df.loc['par2', NOMINAL_VALUE] = ""
    with pytest.raises(AssertionError):
        lint.check_parameter_df(df=parameter_df)

    # NOMINAL_VALUE column missing, but non-estimated parameter
    del parameter_df[NOMINAL_VALUE]
    with pytest.raises(AssertionError):
        lint.check_parameter_df(df=parameter_df)


def test_check_observable_df():
    """Check that we correctly detect errors in observable table"""

    observable_df = pd.DataFrame(data={
        OBSERVABLE_ID: ['obs1', 'obs2'],
        OBSERVABLE_FORMULA: ['x1', 'x2'],
        NOISE_FORMULA: ['sigma1', 'sigma2']
    }).set_index(OBSERVABLE_ID)

    lint.check_observable_df(observable_df)

    # Check that duplicated observables ids are detected
    bad_observable_df = observable_df.copy()
    bad_observable_df.index = ['obs1', 'obs1']
    with pytest.raises(AssertionError):
        lint.check_observable_df(bad_observable_df)

    # Check that missing noiseFormula is detected
    bad_observable_df = observable_df.copy()
    bad_observable_df.loc['obs1', NOISE_FORMULA] = nan
    with pytest.raises(AssertionError):
        lint.check_observable_df(bad_observable_df)


def test_condition_ids_are_unique():
    condition_df = pd.DataFrame(data={
        CONDITION_ID: ['condition1', 'condition1'],
        'parameter1': [1.0, 2.0]
    })
    condition_df.set_index(CONDITION_ID, inplace=True)

    with pytest.raises(AssertionError):
        lint.check_condition_df(condition_df)

    condition_df.index = ['condition0', 'condition1']
    condition_df.index.name = 'conditionId'
    lint.check_condition_df(condition_df)


def test_parameter_ids_are_unique():
    parameter_df = pd.DataFrame({
        PARAMETER_ID: ['par0', 'par0'],
        PARAMETER_SCALE: [LIN, LIN],
        ESTIMATE: [1, 1],
        LOWER_BOUND: [1e-5, 1e-6],
        UPPER_BOUND: [1e5, 1e6]

    }).set_index(PARAMETER_ID)

    with pytest.raises(AssertionError):
        lint.check_parameter_df(parameter_df)

    parameter_df.index = ['par0', 'par1']
    parameter_df.index.name = 'parameterId'
    lint.check_parameter_df(parameter_df)
