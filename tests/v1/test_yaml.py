"""Test for petab.yaml"""

import tempfile
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from petab.yaml import create_problem_yaml, get_path_prefix, validate


def test_validate():
    data = {"format_version": "1"}

    # should fail because we miss some information
    with pytest.raises(ValidationError):
        validate(data)

    # should be well-formed
    file_ = (
        Path(__file__).parents[2]
        / "doc"
        / "example"
        / "example_Fujita"
        / "Fujita.yaml"
    )
    validate(file_)


def test_create_problem_yaml():
    with tempfile.TemporaryDirectory() as outdir:
        # test with single problem files
        # create target files
        sbml_file = Path(outdir, "model.xml")
        condition_file = Path(outdir, "conditions.tsv")
        measurement_file = Path(outdir, "measurements.tsv")
        parameter_file = Path(outdir, "parameters.tsv")
        observable_file = Path(outdir, "observables.tsv")
        yaml_file = Path(outdir, "problem.yaml")
        visualization_file = Path(outdir, "visualization.tsv")

        _create_dummy_sbml_model(sbml_file)

        for file in (
            condition_file,
            measurement_file,
            parameter_file,
            observable_file,
            visualization_file,
        ):
            file.touch()
        create_problem_yaml(
            sbml_file,
            condition_file,
            measurement_file,
            parameter_file,
            observable_file,
            yaml_file,
            visualization_file,
        )
        validate(yaml_file)

        # test for list of files
        # create additional target files
        sbml_file2 = Path(outdir, "model2.xml")
        condition_file2 = Path(outdir, "conditions2.tsv")
        measurement_file2 = Path(outdir, "measurements2.tsv")
        observable_file2 = Path(outdir, "observables2.tsv")
        yaml_file2 = Path(outdir, "problem2.yaml")
        for file in (
            condition_file2,
            measurement_file2,
            observable_file2,
        ):
            file.touch()

        _create_dummy_sbml_model(sbml_file2)

        sbml_files = [sbml_file, sbml_file2]
        condition_files = [condition_file, condition_file2]
        measurement_files = [measurement_file, measurement_file2]
        observable_files = [observable_file, observable_file2]
        create_problem_yaml(
            sbml_files,
            condition_files,
            measurement_files,
            parameter_file,
            observable_files,
            yaml_file2,
        )
        validate(yaml_file2)


def test_get_path_prefix():
    assert get_path_prefix("/some/dir/file.yaml") == str(Path("/some/dir"))
    assert get_path_prefix("some/dir/file.yaml") == str(Path("some/dir"))
    assert (
        get_path_prefix("https://petab.rocks/dir/file.yaml")
        == "https://petab.rocks/dir"
    )


def test_validate_remote():
    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/main/petabtests/cases/v1.0.0/sbml/0001/_0001.yaml"
    )

    validate(yaml_url)


def _create_dummy_sbml_model(sbml_file: Path | str):
    import libsbml

    sbml_doc = libsbml.SBMLDocument()
    sbml_doc.createModel()
    libsbml.writeSBMLToFile(sbml_doc, str(sbml_file))
