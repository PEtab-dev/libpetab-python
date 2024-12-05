"""Convert PEtab version 1 problems to version 2."""
import shutil
from contextlib import suppress
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from pandas.io.common import get_handle, is_url

import petab.v1.C
from petab.models import MODEL_TYPE_SBML
from petab.v1 import Problem as ProblemV1
from petab.yaml import get_path_prefix

from .. import v1, v2
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
    if v1.lint_problem(petab_problem):
        raise ValueError("Provided PEtab problem does not pass linting.")

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
    file = yaml_config[v2.C.PARAMETER_FILE]
    _copy_file(get_src_path(file), Path(get_dest_path(file)))

    for problem_config in yaml_config[v2.C.PROBLEMS]:
        for file in chain(
            problem_config.get(v2.C.OBSERVABLE_FILES, []),
            (
                model[v2.C.MODEL_LOCATION]
                for model in problem_config.get(v2.C.MODEL_FILES, {}).values()
            ),
            problem_config.get(v2.C.VISUALIZATION_FILES, []),
        ):
            _copy_file(get_src_path(file), Path(get_dest_path(file)))

        # Update condition table
        for condition_file in problem_config.get(v2.C.CONDITION_FILES, []):
            condition_df = v1.get_condition_df(get_src_path(condition_file))
            condition_df = _melt_condition_df(
                condition_df, petab_problem.model
            )
            v2.write_condition_df(condition_df, get_dest_path(condition_file))

        for measurement_file in problem_config.get(v2.C.MEASUREMENT_FILES, []):
            measurement_df = v1.get_measurement_df(
                get_src_path(measurement_file)
            )
            if (
                petab_problem.condition_df is not None
                and len(
                    set(petab_problem.condition_df.columns)
                    - {petab.v1.C.CONDITION_NAME}
                )
                == 0
            ):
                # can't have "empty" conditions with no overrides in v2
                # TODO: this needs to be done condition wise
                measurement_df[v2.C.SIMULATION_CONDITION_ID] = np.nan
                if (
                    v1.C.PREEQUILIBRATION_CONDITION_ID
                    in measurement_df.columns
                ):
                    measurement_df[v2.C.PREEQUILIBRATION_CONDITION_ID] = np.nan
            v2.write_measurement_df(
                measurement_df, get_dest_path(measurement_file)
            )
    # TODO: Measurements: preequilibration to experiments/timecourses once
    #  finalized
    ...

    # validate updated Problem
    validation_issues = v2.lint_problem(new_yaml_file)

    if validation_issues:
        raise ValueError(
            "Generated PEtab v2 problem did not pass linting: "
            f"{validation_issues}"
        )


def _update_yaml(yaml_config: dict) -> dict:
    """Update PEtab 1.0 YAML to PEtab 2.0 format."""
    yaml_config = yaml_config.copy()

    # Update format_version
    yaml_config[v2.C.FORMAT_VERSION] = "2.0.0"

    # Add extensions
    yaml_config[v2.C.EXTENSIONS] = []

    # Move models and set IDs (filename for now)
    for problem in yaml_config[v2.C.PROBLEMS]:
        problem[v2.C.MODEL_FILES] = {}
        models = problem[v2.C.MODEL_FILES]
        for sbml_file in problem[v1.C.SBML_FILES]:
            model_id = sbml_file.split("/")[-1].split(".")[0]
            models[model_id] = {
                v2.C.MODEL_LANGUAGE: MODEL_TYPE_SBML,
                v2.C.MODEL_LOCATION: sbml_file,
            }
            problem[v2.C.MODEL_FILES] = problem.get(v2.C.MODEL_FILES, {})
        del problem[v1.C.SBML_FILES]

    return yaml_config


def _copy_file(src: Path | str, dest: Path):
    """Copy file."""
    # src might be a URL - convert to Path if local
    src_url = urlparse(src)
    if not src_url.scheme:
        src = Path(src)
    elif src_url.scheme == "file" and not src_url.netloc:
        src = Path(src.removeprefix("file:/"))

    if is_url(src):
        with get_handle(src, mode="r") as src_handle:
            with open(dest, "w") as dest_handle:
                dest_handle.write(src_handle.handle.read())
        return

    try:
        if dest.samefile(src):
            return
    except FileNotFoundError:
        shutil.copy(str(src), str(dest))


def _melt_condition_df(
    condition_df: pd.DataFrame, model: v1.Model
) -> pd.DataFrame:
    """Melt condition table."""
    condition_df = condition_df.copy().reset_index()
    with suppress(KeyError):
        # TODO: are condition names still supported in v2?
        condition_df.drop(columns=[v2.C.CONDITION_NAME], inplace=True)

    condition_df = condition_df.melt(
        id_vars=[v1.C.CONDITION_ID],
        var_name=v2.C.TARGET_ID,
        value_name=v2.C.TARGET_VALUE,
    )

    if condition_df.empty:
        # This happens if there weren't any condition-specific changes
        return pd.DataFrame(
            columns=[
                v2.C.CONDITION_ID,
                v2.C.TARGET_ID,
                v2.C.VALUE_TYPE,
                v2.C.TARGET_VALUE,
            ]
        )

    targets = set(condition_df[v2.C.TARGET_ID].unique())
    valid_cond_pars = set(model.get_valid_parameters_for_parameter_table())
    # entities to which we assign constant values
    constant = targets & valid_cond_pars
    # entities to which we assign initial values
    initial = set()
    for target in targets - constant:
        if model.is_state_variable(target):
            initial.add(target)
        else:
            raise NotImplementedError(
                f"Unable to determine value type {target} in the condition "
                "table."
            )
    condition_df[v2.C.VALUE_TYPE] = condition_df[v2.C.TARGET_ID].apply(
        lambda x: v2.C.VT_INITIAL if x in initial else v2.C.VT_CONSTANT
    )
    return condition_df
