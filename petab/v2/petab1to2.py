"""Convert PEtab version 1 problems to version 2."""
import shutil
from itertools import chain
from pathlib import Path

from pandas.io.common import get_handle, is_url

import petab.v1.C as C
from petab.models import MODEL_TYPE_SBML
from petab.v1 import Problem as ProblemV1
from petab.v2.lint import lint_problem as lint_v2_problem
from petab.yaml import get_path_prefix

from ..v1 import lint_problem as lint_v1_problem
from ..v1.yaml import load_yaml, validate, write_yaml
from ..versions import get_major_version

__all__ = ["petab1to2"]


def petab1to2(yaml_config: Path | str, output_dir: Path | str = None):
    """Convert from PEtab 1.0 to PEtab 2.0 format.

    Convert a PEtab problem from PEtab 1.0 to PEtab 2.0 format.

    Parameters
    ----------
    yaml_config: dict | Path | str
        The PEtab problem as dictionary or YAML file name.
    output_dir: Path | str
        The output directory to save the converted PEtab problem, or ``None``,
        to return a :class:`petab.v2.Problem` instance.

    Raises
    ------
    ValueError
        If the input is invalid or does not pass linting or if the generated
        files do not pass linting.
    """
    if output_dir is None:
        # TODO requires petab.v2.Problem
        raise NotImplementedError("Not implemented yet.")
    elif isinstance(yaml_config, dict):
        raise ValueError("If output_dir is given, yaml_config must be a file.")

    if isinstance(yaml_config, Path | str):
        yaml_file = str(yaml_config)
        path_prefix = get_path_prefix(yaml_file)
        yaml_config = load_yaml(yaml_config)
        get_src_path = lambda filename: f"{path_prefix}/{filename}"  # noqa: E731
    else:
        yaml_file = None
        path_prefix = None
        get_src_path = lambda filename: filename  # noqa: E731

    get_dest_path = lambda filename: f"{output_dir}/{filename}"  # noqa: E731

    # Validate original PEtab problem
    validate(yaml_config, path_prefix=path_prefix)
    if get_major_version(yaml_config) != 1:
        raise ValueError("PEtab problem is not version 1.")
    petab_problem = ProblemV1.from_yaml(yaml_file or yaml_config)
    if lint_v1_problem(petab_problem):
        raise ValueError("PEtab problem does not pass linting.")

    # Update YAML file
    new_yaml_config = _update_yaml(yaml_config)

    # Write new YAML file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    new_yaml_file = output_dir / Path(yaml_file).name
    write_yaml(new_yaml_config, new_yaml_file)

    # Update tables
    # condition tables, observable tables, SBML files, parameter table:
    #  no changes - just copy
    file = yaml_config[C.PARAMETER_FILE]
    _copy_file(get_src_path(file), get_dest_path(file))

    for problem_config in yaml_config[C.PROBLEMS]:
        for file in chain(
            problem_config.get(C.CONDITION_FILES, []),
            problem_config.get(C.OBSERVABLE_FILES, []),
            (
                model[C.MODEL_LOCATION]
                for model in problem_config.get(C.MODEL_FILES, {}).values()
            ),
            problem_config.get(C.MEASUREMENT_FILES, []),
            problem_config.get(C.VISUALIZATION_FILES, []),
        ):
            _copy_file(get_src_path(file), get_dest_path(file))

    # TODO: Measurements: preequilibration to experiments/timecourses once
    #  finalized
    ...

    # validate updated Problem
    validation_issues = lint_v2_problem(new_yaml_file)

    if validation_issues:
        raise ValueError(
            "Generated PEtab v2 problem did not pass linting: "
            f"{validation_issues}"
        )


def _update_yaml(yaml_config: dict) -> dict:
    """Update PEtab 1.0 YAML to PEtab 2.0 format."""
    yaml_config = yaml_config.copy()

    # Update format_version
    yaml_config[C.FORMAT_VERSION] = "2.0.0"

    # Add extensions
    yaml_config[C.EXTENSIONS] = []

    # Move models and set IDs (filename for now)
    for problem in yaml_config[C.PROBLEMS]:
        problem[C.MODEL_FILES] = {}
        models = problem[C.MODEL_FILES]
        for sbml_file in problem[C.SBML_FILES]:
            model_id = sbml_file.split("/")[-1].split(".")[0]
            models[model_id] = {
                C.MODEL_LANGUAGE: MODEL_TYPE_SBML,
                C.MODEL_LOCATION: sbml_file,
            }
            problem[C.MODEL_FILES] = problem.get(C.MODEL_FILES, {})
        del problem[C.SBML_FILES]

    return yaml_config


def _copy_file(src: Path | str, dest: Path | str):
    """Copy file."""
    src = str(src)
    dest = str(dest)

    if is_url(src):
        with get_handle(src, mode="r") as src_handle:
            with open(dest, "w") as dest_handle:
                dest_handle.write(src_handle.handle.read())
        return

    shutil.copy(str(src), str(dest))
