"""Check that deprecated functionality raises but still works."""
import pytest
import tempfile
from pathlib import Path

import petab

from .test_sbml import create_test_data, check_model
from .test_petab import petab_problem  # noqa: F401


def test_problem_with_sbml_model():
    """Test that a problem can be correctly created from sbml model."""
    # retrieve test data
    ss_model, condition_df, observable_df, measurement_df, parameter_df = \
        create_test_data()

    with pytest.deprecated_call():
        petab_problem = petab.Problem(  # noqa: F811
            sbml_model=ss_model.model,
            condition_df=condition_df,
            measurement_df=measurement_df,
            parameter_df=parameter_df,
        )

    with pytest.warns(UserWarning,
                      match="An SBML rule was removed to set the component "
                            "species_2 to a constant value."):
        _, condition_model = petab.get_model_for_condition(
            petab_problem, "condition_1")

    check_model(condition_model)


def test_to_files_with_sbml_model(petab_problem):  # noqa: F811
    """Test problem.to_files."""
    with tempfile.TemporaryDirectory() as outdir:
        # create target files
        sbml_file = Path(outdir, "model.xml")
        condition_file = Path(outdir, "conditions.tsv")
        measurement_file = Path(outdir, "measurements.tsv")
        parameter_file = Path(outdir, "parameters.tsv")
        observable_file = Path(outdir, "observables.tsv")

        # write contents to files
        with pytest.deprecated_call():
            petab_problem.to_files(
                sbml_file=sbml_file,
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
