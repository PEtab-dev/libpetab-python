"""Convert PEtab version 1 problems to version 2."""

from __future__ import annotations

import re
import shutil
import warnings
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
from pandas.io.common import get_handle, is_url

from .. import v1, v2
from ..v1.math import sympify_petab
from ..v1.yaml import get_path_prefix, load_yaml, validate
from ..versions import get_major_version
from .models import MODEL_TYPE_SBML

__all__ = ["petab1to2"]


def petab1to2(
    yaml_config: Path | str, output_dir: Path | str = None
) -> v2.Problem | None:
    """Convert from PEtab 1.0 to PEtab 2.0 format.

    Convert a PEtab problem from PEtab 1.0 to PEtab 2.0 format.

    .. note::

       Some aspects of PEtab v1 were not well-defined. For example, model
       initialization order (e.g., applying initial assignments before or
       after condition table overrides) and the impact of compartment size
       changes were not specified. In such cases, we made assumptions that are
       consistent with the clarified PEtab v2 specifications,
       the PEtab test suite, or common practice.
       Therefore, it is recommended to carefully review the generated PEtab v2
       problem to ensure it aligns with the expected behavior.

    :param yaml_config:
        The PEtab problem as dictionary or YAML file name.
    :param output_dir:
        The output directory to save the converted PEtab problem, or ``None``,
        to return a :class:`petab.v2.Problem` instance.

    :raises ValueError:
        If the input is invalid or does not pass linting or if the generated
        files do not pass linting.
    """
    if output_dir is not None:
        return petab_files_1to2(yaml_config, output_dir)

    with TemporaryDirectory() as tmp_dir:
        petab_files_1to2(yaml_config, tmp_dir)
        return v2.Problem.from_yaml(Path(tmp_dir, Path(yaml_config).name))


def petab_files_1to2(yaml_config: Path | str | dict, output_dir: Path | str):
    """Convert PEtab files from PEtab 1.0 to PEtab 2.0.


    :param yaml_config:
        The PEtab problem as dictionary or YAML file name.
    :param output_dir:
        The output directory to save the converted PEtab problem.

    :raises ValueError:
        If the input is invalid or does not pass linting or if the generated
        files do not pass linting.
    """
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

    # Validate the original PEtab problem
    validate(yaml_config, path_prefix=path_prefix)
    if get_major_version(yaml_config) != 1:
        raise ValueError("PEtab problem is not version 1.")
    petab_problem = v1.Problem.from_yaml(yaml_file or yaml_config)
    # TODO: move to mapping table
    # get rid of conditionName column if present (unsupported in v2)
    petab_problem.condition_df = petab_problem.condition_df.drop(
        columns=[v1.C.CONDITION_NAME], errors="ignore"
    )
    if v1.lint_problem(petab_problem):
        raise ValueError("Provided PEtab problem does not pass linting.")

    output_dir = Path(output_dir)

    # Update YAML file
    new_yaml_config = _update_yaml(yaml_config)
    new_yaml_config = v2.ProblemConfig(**new_yaml_config)

    # Update tables

    # parameter table
    parameter_df = v1v2_parameter_df(petab_problem.parameter_df.copy())
    v2.write_parameter_df(
        parameter_df, get_dest_path(new_yaml_config.parameter_files[0])
    )

    # copy files that don't need conversion: models
    for file in (
        model.location for model in new_yaml_config.model_files.values()
    ):
        _copy_file(get_src_path(file), Path(get_dest_path(file)))

    # Update observable table
    for observable_file in new_yaml_config.observable_files:
        observable_df = v1.get_observable_df(get_src_path(observable_file))
        observable_df = v1v2_observable_df(
            observable_df,
        )
        v2.write_observable_df(observable_df, get_dest_path(observable_file))

    # Update condition table
    for condition_file in new_yaml_config.condition_files:
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
        new_yaml_config.experiment_files.append("experiments.tsv")
        v2.write_experiment_df(
            v2.get_experiment_df(pd.DataFrame(experiments)), exp_table_path
        )

    for measurement_file in new_yaml_config.measurement_files:
        measurement_df = v1.get_measurement_df(get_src_path(measurement_file))
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
                set(petab_problem.condition_df.columns) - {v1.C.CONDITION_NAME}
            )
            == 0
        ):
            # we can't have "empty" conditions with no overrides in v2,
            #  therefore, we drop the respective condition ID completely
            #   TODO: or can we?
            # TODO: this needs to be checked condition-wise, not globally
            measurement_df[v1.C.SIMULATION_CONDITION_ID] = ""
            if v1.C.PREEQUILIBRATION_CONDITION_ID in measurement_df.columns:
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

    # Write the new YAML file
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
    yaml_config[v2.C.EXTENSIONS] = {}

    # Move models and set IDs (filename for now)
    yaml_config[v2.C.MODEL_FILES] = {}
    for problem in yaml_config[v1.C.PROBLEMS]:
        models = {}
        for sbml_file in problem[v1.C.SBML_FILES]:
            model_id = sbml_file.split("/")[-1].split(".")[0]
            models[model_id] = {
                v2.C.MODEL_LANGUAGE: MODEL_TYPE_SBML,
                v2.C.MODEL_LOCATION: sbml_file,
            }
            yaml_config[v2.C.MODEL_FILES] |= models
        del problem[v1.C.SBML_FILES]

        for file_type in (
            v1.C.CONDITION_FILES,
            v1.C.MEASUREMENT_FILES,
            v1.C.OBSERVABLE_FILES,
        ):
            if file_type in problem:
                yaml_config[file_type] = problem[file_type]
                del problem[file_type]
    del yaml_config[v1.C.PROBLEMS]

    # parameter_file -> parameter_files
    if not isinstance(
        (par_files := yaml_config.pop(v1.C.PARAMETER_FILE, [])), list
    ):
        par_files = [par_files]
    yaml_config[v2.C.PARAMETER_FILES] = par_files

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

    return condition_df


def v1v2_observable_df(observable_df: pd.DataFrame) -> pd.DataFrame:
    """Convert observable table from petab v1 to v2.

    Perform all updates that can be done solely on the observable table:
    * drop observableTransformation, update noiseDistribution
    * update placeholder parameters
    """
    df = observable_df.copy().reset_index()

    # drop observableTransformation, update noiseDistribution
    #  if there is no observableTransformation, no need to update
    if v1.C.OBSERVABLE_TRANSFORMATION in df.columns:
        df[v1.C.OBSERVABLE_TRANSFORMATION] = df[
            v1.C.OBSERVABLE_TRANSFORMATION
        ].fillna(v1.C.LIN)

        if v1.C.NOISE_DISTRIBUTION in df:
            df[v1.C.NOISE_DISTRIBUTION] = df[v1.C.NOISE_DISTRIBUTION].fillna(
                v1.C.NORMAL
            )
        else:
            df[v1.C.NOISE_DISTRIBUTION] = v1.C.NORMAL

        # merge observableTransformation into noiseDistribution
        def update_noise_dist(row):
            dist = row.get(v1.C.NOISE_DISTRIBUTION)
            trans = row.get(v1.C.OBSERVABLE_TRANSFORMATION)

            if trans == v1.C.LIN:
                new_dist = dist
            else:
                new_dist = f"{trans}-{dist}"

            if new_dist == "log10-normal":
                warnings.warn(
                    f"Noise distribution `{new_dist}' for "
                    f"observable `{row[v1.C.OBSERVABLE_ID]}'"
                    f" is not supported in PEtab v2. "
                    "Using `log-normal` instead.",
                    # call to `petab1to2`
                    stacklevel=9,
                )
                new_dist = v2.C.LOG_NORMAL

            if new_dist not in v2.C.NOISE_DISTRIBUTIONS:
                raise NotImplementedError(
                    f"Noise distribution `{new_dist}' for "
                    f"observable `{row[v1.C.OBSERVABLE_ID]}'"
                    f" is not supported in PEtab v2."
                )

        df[v2.C.NOISE_DISTRIBUTION] = df.apply(update_noise_dist, axis=1)
        df.drop(columns=[v1.C.OBSERVABLE_TRANSFORMATION], inplace=True)

    def extract_placeholders(row: pd.Series, type_: str) -> str:
        """Extract placeholders from observable formula."""
        if type_ == "observable":
            formula = row[v1.C.OBSERVABLE_FORMULA]
        elif type_ == "noise":
            formula = row[v1.C.NOISE_FORMULA]
        else:
            raise ValueError(f"Unknown placeholder type: {type_}")

        if pd.isna(formula):
            return ""

        t = f"{re.escape(type_)}Parameter"
        o = re.escape(row[v1.C.OBSERVABLE_ID])

        pattern = re.compile(rf"(?:^|\W)({t}\d+_{o})(?=\W|$)")

        expr = sympify_petab(formula)
        # for 10+ placeholders, the current lexicographical sorting will result
        #  in incorrect ordering of the placeholder IDs, so that they don't
        #  align with the overrides in the measurement table, but who does
        #  that anyway?
        return v2.C.PARAMETER_SEPARATOR.join(
            sorted(
                str(sym)
                for sym in expr.free_symbols
                if sym.is_Symbol and pattern.match(str(sym))
            )
        )

    df[v2.C.OBSERVABLE_PLACEHOLDERS] = df.apply(
        extract_placeholders, args=("observable",), axis=1
    )
    df[v2.C.NOISE_PLACEHOLDERS] = df.apply(
        extract_placeholders, args=("noise",), axis=1
    )

    return df


def v1v2_parameter_df(
    parameter_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert parameter table from petab v1 to v2.

    Do all the necessary conversions to the parameter table that can
    be done with the parameter table alone.
    """
    df = parameter_df.copy().reset_index()

    # parameter.estimate: int -> bool
    df[v2.C.ESTIMATE] = df[v1.C.ESTIMATE].apply(
        lambda x: str(bool(int(x))).lower()
    )

    def update_prior(row):
        """Convert prior to v2 format."""
        prior_type = row.get(v1.C.OBJECTIVE_PRIOR_TYPE)
        if pd.isna(prior_type):
            prior_type = v1.C.UNIFORM

        pscale = row.get(v1.C.PARAMETER_SCALE)
        if pd.isna(pscale):
            pscale = v1.C.LIN

        if prior_type not in v1.C.PARAMETER_SCALE_PRIOR_TYPES:
            return prior_type

        new_prior_type = prior_type.removeprefix("parameterScale").lower()
        if pscale != v1.C.LIN:
            new_prior_type = f"{pscale}-{new_prior_type}"

        if new_prior_type == "log10-normal":
            warnings.warn(
                f"Prior distribution `{new_prior_type}' for parameter "
                f"`{row.name}' is not supported in PEtab v2. "
                "Using `log-normal` instead.",
                # call to `petab1to2`
                stacklevel=9,
            )
            new_prior_type = v2.C.LOG_NORMAL

        if new_prior_type not in v2.C.PRIOR_DISTRIBUTIONS:
            raise NotImplementedError(
                f"PEtab v2 does not support prior type `{new_prior_type}' "
                f"required for parameter `{row.name}'."
            )

        return new_prior_type

    # update parameterScale*-priors
    if v1.C.OBJECTIVE_PRIOR_TYPE in df.columns:
        df[v1.C.OBJECTIVE_PRIOR_TYPE] = df.apply(update_prior, axis=1)

    # rename objectivePrior* to prior*
    df.rename(
        columns={
            v1.C.OBJECTIVE_PRIOR_TYPE: v2.C.PRIOR_DISTRIBUTION,
            v1.C.OBJECTIVE_PRIOR_PARAMETERS: v2.C.PRIOR_PARAMETERS,
        },
        inplace=True,
        errors="ignore",
    )
    # some columns were dropped in PEtab v2
    df.drop(
        columns=[
            v1.C.INITIALIZATION_PRIOR_TYPE,
            v1.C.INITIALIZATION_PRIOR_PARAMETERS,
            v1.C.PARAMETER_SCALE,
        ],
        inplace=True,
        errors="ignore",
    )

    # if uniform, we need to explicitly set the parameters
    def update_prior_pars(row):
        prior_type = row.get(v2.C.PRIOR_DISTRIBUTION)
        prior_pars = row.get(v2.C.PRIOR_PARAMETERS)

        if prior_type in (v2.C.UNIFORM, v2.C.LOG_UNIFORM) and pd.isna(
            prior_pars
        ):
            return (
                f"{row[v2.C.LOWER_BOUND]}{v2.C.PARAMETER_SEPARATOR}"
                f"{row[v2.C.UPPER_BOUND]}"
            )

        return prior_pars

    df[v2.C.PRIOR_PARAMETERS] = df.apply(update_prior_pars, axis=1)

    return df
