"""Code regarding the PEtab YAML config files"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse, urlunparse

import jsonschema
import numpy as np
import pandas as pd
import yaml
from pandas.io.common import get_handle

from .C import *  # noqa: F403

# directory with PEtab yaml schema files
SCHEMA_DIR = Path(__file__).parent.parent / "schemas"
# map of version number to validation schema
SCHEMAS = {
    "1": SCHEMA_DIR / "petab_schema.v1.0.0.yaml",
    "1.0.0": SCHEMA_DIR / "petab_schema.v1.0.0.yaml",
    "2.0.0": SCHEMA_DIR / "petab_schema.v2.0.0.yaml",
}

__all__ = [
    "validate",
    "validate_yaml_syntax",
    "validate_yaml_semantics",
    "load_yaml",
    "is_composite_problem",
    "assert_single_condition_and_sbml_file",
    "write_yaml",
    "create_problem_yaml",
    "get_path_prefix",
]


def validate(
    yaml_config: dict | str | Path,
    path_prefix: None | str | Path = None,
):
    """Validate syntax and semantics of PEtab config YAML

    Arguments:
        yaml_config:
            PEtab YAML config as filename or dict.
        path_prefix:
            Base location for relative paths. Defaults to location of YAML
            file if a filename was provided for ``yaml_config`` or the current
            working directory.
    """
    validate_yaml_syntax(yaml_config)
    validate_yaml_semantics(yaml_config=yaml_config, path_prefix=path_prefix)


def validate_yaml_syntax(
    yaml_config: dict | str | Path, schema: None | dict | str = None
):
    """Validate PEtab YAML file syntax

    Arguments:
        yaml_config:
            PEtab YAML file to validate, as file name or dictionary
        schema:
            Custom schema for validation

    Raises:
        see :func:`jsonschema.validate`
    """
    yaml_config = load_yaml(yaml_config)

    if schema is None:
        # try get PEtab version from yaml file
        #  if this is not the available, the file is not valid anyways,
        #  but let's still use the latest PEtab schema for full validation
        version = (
            yaml_config.get(FORMAT_VERSION, None) or list(SCHEMAS.values())[-1]
        )
        try:
            schema = SCHEMAS[str(version)]
        except KeyError as e:
            raise ValueError(
                "Unknown PEtab version given in problem "
                f"specification: {version}"
            ) from e
    schema = load_yaml(schema)
    jsonschema.validate(instance=yaml_config, schema=schema)


def validate_yaml_semantics(
    yaml_config: dict | str | Path,
    path_prefix: None | str | Path = None,
):
    """Validate PEtab YAML file semantics

    Check for existence of files. Assumes valid syntax.

    Version number and contents of referenced files are not yet checked.

    Arguments:
        yaml_config:
            PEtab YAML config as filename or dict.
        path_prefix:
            Base location for relative paths. Defaults to location of YAML
            file if a filename was provided for ``yaml_config`` or the current
            working directory.

    Raises:
        AssertionError: in case of problems
    """
    if not path_prefix:
        if isinstance(yaml_config, str | Path):
            path_prefix = get_path_prefix(yaml_config)
        else:
            path_prefix = ""

    yaml_config = load_yaml(yaml_config)

    def _check_file(_filename: str, _field: str):
        # this could be a regular path or some local or remote URL
        # the simplest check is just trying to load the respective table or
        # sbml model
        if _field == SBML_FILES:
            from .models.sbml_model import SbmlModel

            try:
                SbmlModel.from_file(_filename)
            except Exception as e:
                raise AssertionError(
                    f"Failed to read '{_filename}' provided as '{_field}'."
                ) from e
            return

        try:
            pd.read_csv(_filename, sep="\t")
        except pd.errors.EmptyDataError:
            # at this stage, we don't care about the content
            pass
        except Exception as e:
            raise AssertionError(
                f"Failed to read '{_filename}' provided as '{_field}'."
            ) from e

    # Handles both a single parameter file, and a parameter file that has been
    # split into multiple subset files.
    for parameter_subset_file in list(
        np.array(yaml_config[PARAMETER_FILE]).flat
    ):
        _check_file(
            f"{path_prefix}/{parameter_subset_file}"
            if path_prefix
            else parameter_subset_file,
            parameter_subset_file,
        )

    for problem_config in yaml_config[PROBLEMS]:
        for field in [
            SBML_FILES,
            CONDITION_FILES,
            MEASUREMENT_FILES,
            VISUALIZATION_FILES,
            OBSERVABLE_FILES,
        ]:
            if field in problem_config:
                for filename in problem_config[field]:
                    _check_file(
                        f"{path_prefix}/{filename}"
                        if path_prefix
                        else filename,
                        field,
                    )


def load_yaml(yaml_config: dict | Path | str) -> dict:
    """Load YAML

    Convenience function to allow for providing YAML inputs as filename, URL
    or as dictionary.

    Arguments:
        yaml_config:
            PEtab YAML config as filename or dict or URL.

    Returns:
        The unmodified dictionary if ``yaml_config`` was dictionary.
        Otherwise the parsed the YAML file.
    """
    # already parsed? all PEtab problem yaml files are dictionaries
    if isinstance(yaml_config, dict):
        return yaml_config

    with get_handle(yaml_config, mode="r") as io_handle:
        data = yaml.safe_load(io_handle.handle)
    return data


def is_composite_problem(yaml_config: dict | str | Path) -> bool:
    """Does this YAML file comprise multiple models?

    Arguments:
        yaml_config: PEtab configuration as dictionary or YAML file name
    """
    yaml_config = load_yaml(yaml_config)
    return len(yaml_config[PROBLEMS]) > 1


def assert_single_condition_and_sbml_file(problem_config: dict) -> None:
    """Check that there is only a single condition file and a single SBML
    file specified.

    Arguments:
        problem_config:
            Dictionary as defined in the YAML schema inside the `problems`
            list.
    Raises:
        NotImplementedError:
            If multiple condition or SBML files specified.
    """
    if (
        len(problem_config[SBML_FILES]) > 1
        or len(problem_config[CONDITION_FILES]) > 1
    ):
        # TODO https://github.com/ICB-DCM/PEtab/issues/188
        # TODO https://github.com/ICB-DCM/PEtab/issues/189
        raise NotImplementedError(
            "Support for multiple models or condition files is not yet "
            "implemented."
        )


def write_yaml(yaml_config: dict[str, Any], filename: str | Path) -> None:
    """Write PEtab YAML file

    Arguments:
        yaml_config: Data to write
        filename: File to create
    """
    with open(filename, "w") as outfile:
        yaml.dump(
            yaml_config, outfile, default_flow_style=False, sort_keys=False
        )


def create_problem_yaml(
    sbml_files: str | Path | list[str | Path],
    condition_files: str | Path | list[str | Path],
    measurement_files: str | Path | list[str | Path],
    parameter_file: str | Path,
    observable_files: str | Path | list[str | Path],
    yaml_file: str | Path,
    visualization_files: str | Path | list[str | Path] | None = None,
    relative_paths: bool = True,
    mapping_files: str | Path | list[str | Path] = None,
) -> None:
    """Create and write default YAML file for a single PEtab problem

    Arguments:
        sbml_files: Path of SBML model file or list of such
        condition_files: Path of condition file or list of such
        measurement_files: Path of measurement file or list of such
        parameter_file: Path of parameter file
        observable_files: Path of observable file or list of such
        yaml_file: Path to which YAML file should be written
        visualization_files:
            Optional Path to visualization file or list of such
        relative_paths:
            whether all paths in the YAML file should be relative to the
            location of the YAML file. If ``False``, then paths are left
            unchanged.
        mapping_files: Path of mapping file
    """
    if isinstance(sbml_files, Path | str):
        sbml_files = [sbml_files]
    if isinstance(condition_files, Path | str):
        condition_files = [condition_files]
    if isinstance(measurement_files, Path | str):
        measurement_files = [measurement_files]
    if isinstance(observable_files, Path | str):
        observable_files = [observable_files]
    if isinstance(visualization_files, Path | str):
        visualization_files = [visualization_files]

    if relative_paths:
        yaml_file_dir = Path(yaml_file).parent

        def get_rel_to_yaml(paths: list[str] | None):
            if paths is None:
                return paths
            return [
                os.path.relpath(path, start=yaml_file_dir) for path in paths
            ]

        sbml_files = get_rel_to_yaml(sbml_files)
        condition_files = get_rel_to_yaml(condition_files)
        measurement_files = get_rel_to_yaml(measurement_files)
        observable_files = get_rel_to_yaml(observable_files)
        visualization_files = get_rel_to_yaml(visualization_files)
        parameter_file = get_rel_to_yaml([parameter_file])[0]
        mapping_files = get_rel_to_yaml(mapping_files)

    problem_dic = {
        CONDITION_FILES: condition_files,
        MEASUREMENT_FILES: measurement_files,
        SBML_FILES: sbml_files,
        OBSERVABLE_FILES: observable_files,
    }
    if mapping_files:
        problem_dic[MAPPING_FILES] = mapping_files

    if visualization_files is not None:
        problem_dic[VISUALIZATION_FILES] = visualization_files
    yaml_dic = {
        PARAMETER_FILE: parameter_file,
        FORMAT_VERSION: 1,
        PROBLEMS: [problem_dic],
    }
    write_yaml(yaml_dic, yaml_file)


def get_path_prefix(yaml_path: Path | str) -> str:
    """Get the path prefix from a PEtab problem yaml file.

    Get the path prefix to retrieve any files with relative paths referenced
    in the given PEtab problem yaml file.

    Arguments:
        yaml_path: PEtab problem YAML file path (local or URL).

    Returns:
        The path prefix for retrieving any referenced files with relative
        paths.
    """
    yaml_path = str(yaml_path)

    # yaml_config may be path or URL
    path_url = urlparse(yaml_path)
    if not path_url.scheme or (
        path_url.scheme != "file" and not path_url.netloc
    ):
        # a regular file path string
        return str(Path(yaml_path).parent)

    # a URL
    # extract parent path
    url_path = unquote(urlparse(yaml_path).path)
    parent_path = str(PurePosixPath(url_path).parent)
    path_prefix = urlunparse(
        (
            path_url.scheme,
            path_url.netloc,
            parent_path,
            path_url.params,
            path_url.query,
            path_url.fragment,
        )
    )
    return path_prefix
