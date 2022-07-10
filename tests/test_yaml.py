"""Test for petab.yaml"""
import tempfile
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError
from petab.yaml import create_problem_yaml, validate


def test_validate():
    data = {
        'format_version': '1'
    }

    # should fail because we miss some information
    with pytest.raises(ValidationError):
        validate(data)

    # should be well-formed
    file_ = Path(__file__).parents[1] / "doc" / "example" / "example_Fujita"\
        / "Fujita.yaml"
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
        for file in (sbml_file, condition_file, measurement_file,
                     parameter_file, observable_file, visualization_file):
            file.touch()
        create_problem_yaml(sbml_file, condition_file, measurement_file,
                            parameter_file, observable_file, yaml_file,
                            visualization_file)
        validate(yaml_file)

        # test for list of files
        # create additional target files
        sbml_file2 = Path(outdir, "model2.xml")
        condition_file2 = Path(outdir, "conditions2.tsv")
        measurement_file2 = Path(outdir, "measurements2.tsv")
        observable_file2 = Path(outdir, "observables2.tsv")
        yaml_file2 = Path(outdir, "problem2.yaml")
        for file in (sbml_file2, condition_file2, measurement_file2,
                     observable_file2):
            file.touch()

        sbml_files = [sbml_file, sbml_file2]
        condition_files = [condition_file, condition_file2]
        measurement_files = [measurement_file, measurement_file2]
        observable_files = [observable_file, observable_file2]
        create_problem_yaml(sbml_files, condition_files, measurement_files,
                            parameter_file, observable_files, yaml_file2)
        validate(yaml_file2)
