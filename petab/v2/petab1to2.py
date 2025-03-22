"""Convert PEtab version 1 problems to version 2."""

import shutil
from contextlib import suppress
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
from pandas.io.common import get_handle, is_url

from .. import v1, v2
from ..v1.yaml import get_path_prefix, load_yaml, validate
from ..versions import get_major_version
from .models import MODEL_TYPE_SBML
from .problem import ProblemConfig

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
    petab_problem = v1.Problem.from_yaml(yaml_file or yaml_config)
    # get rid of conditionName column if present (unsupported in v2)
    petab_problem.condition_df = petab_problem.condition_df.drop(
        columns=[v1.C.CONDITION_NAME], errors="ignore"
    )
    if v1.lint_problem(petab_problem):
        raise ValueError("Provided PEtab problem does not pass linting.")

    output_dir = Path(output_dir)

    # Update YAML file
    new_yaml_config = _update_yaml(yaml_config)
    new_yaml_config = ProblemConfig(**new_yaml_config)

    # Update tables
    # condition tables, observable tables, SBML files, parameter table:
    #  no changes - just copy
    file = yaml_config[v2.C.PARAMETER_FILE]
    _copy_file(get_src_path(file), Path(get_dest_path(file)))

    for problem_config in new_yaml_config.problems:
        for file in chain(
            problem_config.observable_files,
            (model.location for model in problem_config.model_files.values()),
            problem_config.visualization_files,
        ):
            _copy_file(get_src_path(file), Path(get_dest_path(file)))

        # Update condition table
        for condition_file in problem_config.condition_files:
            condition_df = v1.get_condition_df(get_src_path(condition_file))
            condition_df = v1v2_condition_df(condition_df, petab_problem.model)
            v2.write_condition_df(condition_df, get_dest_path(condition_file))

        # records for the experiment table to be created
        experiments = []

        def create_experiment_id(sim_cond_id: str, preeq_cond_id: str) -> str:
            if not sim_cond_id and not preeq_cond_id:
                return ""
            # check whether the conditions will exist in the v2 condition table
            sim_cond_exists = (
                petab_problem.condition_df.loc[sim_cond_id].notna().any()
            )
            preeq_cond_exists = (
                preeq_cond_id
                and petab_problem.condition_df.loc[preeq_cond_id].notna().any()
            )
            if not sim_cond_exists and not preeq_cond_exists:
                # if we have only all-NaN conditions, we don't create a new
                #  experiment
                return ""

            if preeq_cond_id:
                preeq_cond_id = f"{preeq_cond_id}_"
            exp_id = f"experiment__{preeq_cond_id}__{sim_cond_id}"
            if exp_id in experiments:  # noqa: B023
                i = 1
                while f"{exp_id}_{i}" in experiments:  # noqa: B023
                    i += 1
                exp_id = f"{exp_id}_{i}"
            return exp_id

        measured_experiments = (
            petab_problem.get_simulation_conditions_from_measurement_df()
        )
        for (
            _,
            row,
        ) in measured_experiments.iterrows():
            # generate a new experiment for each simulation / pre-eq condition
            #  combination
            sim_cond_id = row[v1.C.SIMULATION_CONDITION_ID]
            preeq_cond_id = row.get(v1.C.PREEQUILIBRATION_CONDITION_ID, "")
            exp_id = create_experiment_id(sim_cond_id, preeq_cond_id)
            if not exp_id:
                continue
            if preeq_cond_id:
                experiments.append(
                    {
                        v2.C.EXPERIMENT_ID: exp_id,
                        v2.C.CONDITION_ID: preeq_cond_id,
                        v2.C.TIME: v2.C.TIME_PREEQUILIBRATION,
                    }
                )
            experiments.append(
                {
                    v2.C.EXPERIMENT_ID: exp_id,
                    v2.C.CONDITION_ID: sim_cond_id,
                    v2.C.TIME: 0,
                }
            )
        if experiments:
            exp_table_path = output_dir / "experiments.tsv"
            if exp_table_path.exists():
                raise ValueError(
                    f"Experiment table file {exp_table_path} already exists."
                )
            problem_config.experiment_files.append("experiments.tsv")
            v2.write_experiment_df(
                v2.get_experiment_df(pd.DataFrame(experiments)), exp_table_path
            )

        for measurement_file in problem_config.measurement_files:
            measurement_df = v1.get_measurement_df(
                get_src_path(measurement_file)
            )
            # if there is already an experiment ID column, we rename it
            if v2.C.EXPERIMENT_ID in measurement_df.columns:
                measurement_df.rename(
                    columns={v2.C.EXPERIMENT_ID: f"experiment_id_{uuid4()}"},
                    inplace=True,
                )
            # add pre-eq condition id if not present or convert to string
            #  for simplicity
            if v1.C.PREEQUILIBRATION_CONDITION_ID in measurement_df.columns:
                measurement_df.fillna(
                    {v1.C.PREEQUILIBRATION_CONDITION_ID: ""}, inplace=True
                )
            else:
                measurement_df[v1.C.PREEQUILIBRATION_CONDITION_ID] = ""

            if (
                petab_problem.condition_df is not None
                and len(
                    set(petab_problem.condition_df.columns)
                    - {v1.C.CONDITION_NAME}
                )
                == 0
            ):
                # we can't have "empty" conditions with no overrides in v2,
                #  therefore, we drop the respective condition ID completely
                #   TODO: or can we?
                # TODO: this needs to be checked condition-wise, not globally
                measurement_df[v1.C.SIMULATION_CONDITION_ID] = ""
                if (
                    v1.C.PREEQUILIBRATION_CONDITION_ID
                    in measurement_df.columns
                ):
                    measurement_df[v1.C.PREEQUILIBRATION_CONDITION_ID] = ""
            # condition IDs to experiment IDs
            measurement_df.insert(
                0,
                v2.C.EXPERIMENT_ID,
                measurement_df.apply(
                    lambda row: create_experiment_id(
                        row[v1.C.SIMULATION_CONDITION_ID],
                        row.get(v1.C.PREEQUILIBRATION_CONDITION_ID, ""),
                    ),
                    axis=1,
                ),
            )
            del measurement_df[v1.C.SIMULATION_CONDITION_ID]
            del measurement_df[v1.C.PREEQUILIBRATION_CONDITION_ID]
            v2.write_measurement_df(
                measurement_df, get_dest_path(measurement_file)
            )

    # Write new YAML file
    new_yaml_file = output_dir / Path(yaml_file).name
    new_yaml_config.to_yaml(new_yaml_file)

    # validate updated Problem
    validation_issues = v2.lint_problem(new_yaml_file)

    if validation_issues:
        sev = v2.lint.ValidationIssueSeverity
        validation_issues.log(max_level=sev.WARNING)
        errors = "\n".join(
            map(
                str,
                (i for i in validation_issues if i.level > sev.WARNING),
            )
        )
        if errors:
            raise ValueError(
                "The generated PEtab v2 problem did not pass linting: "
                f"{errors}"
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


def v1v2_condition_df(
    condition_df: pd.DataFrame, model: v1.Model
) -> pd.DataFrame:
    """Convert condition table from petab v1 to v2."""
    condition_df = condition_df.copy().reset_index()
    with suppress(KeyError):
        # conditionName was dropped in PEtab v2
        condition_df.drop(columns=[v1.C.CONDITION_NAME], inplace=True)

    condition_df = condition_df.melt(
        id_vars=[v1.C.CONDITION_ID],
        var_name=v2.C.TARGET_ID,
        value_name=v2.C.TARGET_VALUE,
    ).dropna(subset=[v2.C.TARGET_VALUE])

    if condition_df.empty:
        # This happens if there weren't any condition-specific changes
        return pd.DataFrame(
            columns=[
                v2.C.CONDITION_ID,
                v2.C.TARGET_ID,
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
    return condition_df
