"""Convert PEtab version 1 problems to version 2."""

from __future__ import annotations

import math
import shutil
import warnings
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, get_args
from urllib.parse import urlparse

import pandas as pd
import sympy as sp
from pandas.io.common import get_handle, is_url

from .. import v1, v2
from ..v1.parameters import scale, unscale
from ..v1.yaml import get_path_prefix, load_yaml
from ..v1.yaml import validate as validate_yaml
from ..versions import get_major_version
from .models import MODEL_TYPE_SBML
from .reparametrize import (
    _get_override_slots,
    _log,
    _log10,
    reparametrize_log,
)

__all__ = ["petab1to2"]

#: Options for reparametrizing log-scaled parameters
Reparametrize = Literal["none", "priors", "all"]


def petab1to2(
    yaml_config: Path | str,
    output_dir: Path | str | None = None,
    *,
    validate: bool = True,
    reparametrize: Reparametrize = "none",
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
    :param validate:
        Whether to lint the input PEtab v1 problem, and the
        output PEtab v2 problem.
        If ``True`` (default), both problems are validated.
        If ``False``, neither problem is validated.
        The ``yaml_config`` content is validated regardless.
    :param reparametrize:
        Which ``log``/``log10``-scaled parameters to estimate as
        ``log10_<id>``/``log_<id>`` (see :func:`petab.v2.reparametrize_log`).
        PEtab v2 has no ``parameterScale``, so without reparametrization,
        ``parameterScale*`` priors are converted to densities over the
        unscaled parameter, which changes the maximum a posteriori estimate.

        * ``"none"`` (default): don't reparametrize.
        * ``"priors"``: estimated parameters with ``parameterScale*`` priors.
        * ``"all"``: all estimated log-scaled parameters, except those with
          priors over the unscaled parameter (e.g., ``logNormal``).
          This also preserves the v1 default prior (uniform on parameter
          scale). Parameters without ``parameterScale*`` priors are not
          reparametrized (with a warning) if they override a placeholder
          that cannot be log-transformed.

        Parameters with a lower bound of ``0`` get a lower bound of ``-inf``.
        They are not reparametrized if their prior on log scale would be
        an improper uniform prior.
        Further log-scaled parameters are reparametrized if they share a
        placeholder with reparametrized ones in the measurement table.

    :raises ValueError:
        If the input is invalid or, if ``validate`` is ``True``, if the input
        does not pass linting or if the generated files do not pass linting.
    :raises NotImplementedError:
        If a parameter with a ``parameterScale*`` prior cannot be
        reparametrized, e.g., because it overrides a placeholder that
        cannot be log-transformed.
    """
    if output_dir is not None:
        return petab_files_1to2(
            yaml_config,
            output_dir,
            validate=validate,
            reparametrize=reparametrize,
        )

    with TemporaryDirectory() as tmp_dir:
        petab_files_1to2(
            yaml_config,
            tmp_dir,
            validate=validate,
            reparametrize=reparametrize,
        )
        return v2.Problem.from_yaml(Path(tmp_dir, Path(yaml_config).name))


def petab_files_1to2(
    yaml_config: Path | str | dict,
    output_dir: Path | str,
    *,
    validate: bool = True,
    reparametrize: Reparametrize = "none",
):
    """Convert PEtab files from PEtab 1.0 to PEtab 2.0.


    :param yaml_config:
        The PEtab problem as dictionary or YAML file name.
    :param output_dir:
        The output directory to save the converted PEtab problem.
    :param validate:
        Whether to lint the PEtab v1 problem before and the generated
        PEtab v2 problem after the conversion. See :func:`petab1to2`.
    :param reparametrize:
        Which log-scaled parameters to reparametrize.
        See :func:`petab1to2`.

    :raises ValueError:
        If the input is invalid or, if ``validate`` is ``True``, if the input
        does not pass linting or if the generated files do not pass linting.
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

    if reparametrize not in get_args(Reparametrize):
        raise ValueError(
            f"Invalid value for `reparametrize': {reparametrize}. "
            f"Must be one of {get_args(Reparametrize)}."
        )

    # Validate the original PEtab problem
    validate_yaml(yaml_config, path_prefix=path_prefix)
    if get_major_version(yaml_config) != 1:
        raise ValueError("PEtab problem is not version 1.")
    petab_problem = v1.Problem.from_yaml(yaml_file or yaml_config)
    # TODO: move to mapping table
    # get rid of conditionName column if present (unsupported in v2)
    petab_problem.condition_df = petab_problem.condition_df.drop(
        columns=[v1.C.CONDITION_NAME], errors="ignore"
    )
    if validate and v1.lint_problem(petab_problem):
        raise ValueError("Provided PEtab problem does not pass linting.")

    output_dir = Path(output_dir)

    # Update YAML file
    new_yaml_config = _update_yaml(yaml_config)
    new_yaml_config = v2.ProblemConfig(**new_yaml_config)

    # Update tables

    # parameter table
    parameter_df = v1v2_parameter_df(
        petab_problem.parameter_df.copy(), reparametrize=reparametrize
    )
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
    #  (tables without any condition-specific changes are omitted)
    condition_files = []
    for condition_file in new_yaml_config.condition_files:
        condition_df = v1.get_condition_df(get_src_path(condition_file))
        condition_df = v1v2_condition_df(condition_df, petab_problem.model)
        if condition_df.empty:
            continue
        v2.write_condition_df(condition_df, get_dest_path(condition_file))
        condition_files.append(condition_file)
    new_yaml_config.condition_files = condition_files

    # records for the experiment table to be created
    experiments = []

    def v2_condition_id(cond_id: str) -> str:
        """Get the condition ID to use in the v2 experiment table.

        Conditions without any changes (all-NaN rows) are not included in the
        v2 condition table. Periods applying such a condition get an empty
        condition ID ("use the model as is").
        """
        if petab_problem.condition_df.loc[cond_id].notna().any():
            return cond_id
        return ""

    def create_experiment_id(sim_cond_id: str, preeq_cond_id: str) -> str:
        if not sim_cond_id and not preeq_cond_id:
            return ""
        if not preeq_cond_id and not v2_condition_id(sim_cond_id):
            # without pre-equilibration and without any changes, there is
            #  nothing to be described by an experiment
            return ""

        if preeq_cond_id:
            preeq_cond_id = f"{preeq_cond_id}_"
        exp_id = f"experiment__{preeq_cond_id}__{sim_cond_id}"
        if exp_id in experiments:
            i = 1
            while f"{exp_id}_{i}" in experiments:
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
                    v2.C.TIME: v2.C.TIME_PREEQUILIBRATION,
                    v2.C.CONDITION_ID: v2_condition_id(preeq_cond_id),
                }
            )
        experiments.append(
            {
                v2.C.EXPERIMENT_ID: exp_id,
                v2.C.TIME: 0,
                v2.C.CONDITION_ID: v2_condition_id(sim_cond_id),
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

    measurement_dfs = {
        measurement_file: v1.get_measurement_df(get_src_path(measurement_file))
        for measurement_file in new_yaml_config.measurement_files
    }
    # if there is already an experiment ID column, we rename it,
    #  consistently across all measurement tables
    if any(
        v2.C.EXPERIMENT_ID in df.columns for df in measurement_dfs.values()
    ):
        all_columns = set().union(
            *(df.columns for df in measurement_dfs.values())
        )
        renamed_experiment_id = f"{v2.C.EXPERIMENT_ID}_v1"
        i = 1
        while renamed_experiment_id in all_columns:
            i += 1
            renamed_experiment_id = f"{v2.C.EXPERIMENT_ID}_v1_{i}"
        warnings.warn(
            f"The measurement table has a column `{v2.C.EXPERIMENT_ID}`, "
            "which is reserved in PEtab v2. "
            f"It was renamed to `{renamed_experiment_id}`.",
            # point to the caller of `petab1to2`
            stacklevel=3,
        )

    for measurement_file, measurement_df in measurement_dfs.items():
        if v2.C.EXPERIMENT_ID in measurement_df.columns:
            measurement_df = measurement_df.rename(
                columns={v2.C.EXPERIMENT_ID: renamed_experiment_id}
            )
        # add pre-eq condition id if not present or convert to string
        #  for simplicity
        if v1.C.PREEQUILIBRATION_CONDITION_ID in measurement_df.columns:
            measurement_df[v1.C.PREEQUILIBRATION_CONDITION_ID] = (
                measurement_df[v1.C.PREEQUILIBRATION_CONDITION_ID]
                .astype("object")
                .fillna("")
            )
        else:
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

    if reparametrize != "none" and (
        problem := _reparametrize_log_scaled(
            v2.Problem.from_yaml(new_yaml_file),
            petab_problem.parameter_df,
            reparametrize,
        )
    ):
        problem.to_files(output_dir)

    if not validate:
        return

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
        with (
            get_handle(src, mode="r") as src_handle,
            open(dest, "w") as dest_handle,
        ):
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
        condition_df = condition_df.drop(columns=[v1.C.CONDITION_NAME])

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
        def get_noise_dist(row):
            dist = row.get(v1.C.NOISE_DISTRIBUTION)
            trans = row.get(v1.C.OBSERVABLE_TRANSFORMATION)

            if trans == v1.C.LIN:
                new_dist = dist
            elif trans == v1.C.LOG10:
                # see `get_noise_formula`
                new_dist = f"{v1.C.LOG}-{dist}"
            else:
                new_dist = f"{trans}-{dist}"

            if new_dist not in v2.C.NOISE_DISTRIBUTIONS:
                raise NotImplementedError(
                    f"Noise distribution `{new_dist}' for "
                    f"observable `{row[v1.C.OBSERVABLE_ID]}'"
                    f" is not supported in PEtab v2."
                )

            return new_dist

        # PEtab v2 has no log10-based noise distributions. Since
        #  ln(y) = ln(10) * log10(y), a normal (Laplace) distribution of
        #  log10(y) with standard deviation (scale) sigma is a normal
        #  (Laplace) distribution of ln(y) with ln(10) * sigma.
        #  Therefore, log10-<dist> becomes log-<dist> with the noise
        #  formula multiplied by ln(10).
        def get_noise_formula(row):
            formula = row[v1.C.NOISE_FORMULA]
            if row[v1.C.OBSERVABLE_TRANSFORMATION] != v1.C.LOG10:
                return formula
            return f"log(10) * ({formula})"

        df[v2.C.NOISE_DISTRIBUTION] = df.apply(get_noise_dist, axis=1)
        df[v1.C.NOISE_FORMULA] = df.apply(get_noise_formula, axis=1)

        is_log10 = df[v1.C.OBSERVABLE_TRANSFORMATION] == v1.C.LOG10
        if is_log10.any():
            log10_ids = ", ".join(df.loc[is_log10, v1.C.OBSERVABLE_ID])
            warnings.warn(
                "PEtab v2 does not support log10-transformed observables. "
                f"The noise distributions of observables {log10_ids} were "
                "converted to log-normal or log-laplace, and their noise "
                "formulas were multiplied by `log(10)`. Parameters in these "
                "noise formulas (e.g., estimated noise parameters) keep their "
                "PEtab v1 meaning, i.e., they refer to the log10-transformed "
                "observable, not to its natural logarithm.",
                # call to `petab1to2`
                stacklevel=4,
            )

        df = df.drop(columns=[v1.C.OBSERVABLE_TRANSFORMATION])

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

        # v1 overrides are positional: the i-th override replaces
        #  `{type_}Parameter{i}_{observableId}`. This returns the placeholders
        #  ordered by index and fails if they are not numbered 1..n.
        try:
            placeholders = v1.get_formula_placeholders(
                formula, row[v1.C.OBSERVABLE_ID], type_
            )
        except AssertionError as e:
            raise ValueError(
                f"Cannot convert the {type_} placeholders of observable "
                f"`{row[v1.C.OBSERVABLE_ID]}': {e}"
            ) from e
        return v2.C.PARAMETER_SEPARATOR.join(placeholders)

    df[v2.C.OBSERVABLE_PLACEHOLDERS] = df.apply(
        extract_placeholders, args=("observable",), axis=1
    )
    df[v2.C.NOISE_PLACEHOLDERS] = df.apply(
        extract_placeholders, args=("noise",), axis=1
    )

    return df


def _has_objective_priors(parameter_df: pd.DataFrame) -> bool:
    """Whether any estimated parameter of a v1 parameter table has an
    objective prior."""
    if v1.C.OBJECTIVE_PRIOR_TYPE not in parameter_df:
        return False
    return bool(
        (
            parameter_df[v1.C.OBJECTIVE_PRIOR_TYPE].notna()
            & (parameter_df[v1.C.ESTIMATE].astype(int) == 1)
        ).any()
    )


def v1v2_parameter_df(
    parameter_df: pd.DataFrame,
    reparametrize: Reparametrize = "none",
) -> pd.DataFrame:
    """Convert parameter table from petab v1 to v2.

    Do all the necessary conversions to the parameter table that can
    be done with the parameter table alone.

    :param parameter_df: The PEtab v1 parameter table.
    :param reparametrize:
        The reparametrization option passed to :func:`petab1to2`.
        Only used for warnings, the reparametrization itself is done later.
    """
    df = parameter_df.copy().reset_index()

    # parameter.estimate: int -> bool
    df[v2.C.ESTIMATE] = df[v1.C.ESTIMATE].apply(
        lambda x: str(bool(int(x))).lower()
    )

    has_priors = _has_objective_priors(parameter_df)

    def update_prior(row) -> tuple[str | None, str | None]:
        """Convert prior type and prior parameters to v2 format."""
        prior_type = row.get(v1.C.OBJECTIVE_PRIOR_TYPE)
        if pd.isna(prior_type):
            if not has_priors or row[v2.C.ESTIMATE] != "true":
                # no prior
                return None, None
            # the v2 default if other parameters have priors
            prior_type = v1.C.UNIFORM
        prior_pars = row.get(v1.C.OBJECTIVE_PRIOR_PARAMETERS)

        pscale = row.get(v1.C.PARAMETER_SCALE)
        if pd.isna(pscale):
            pscale = v1.C.LIN

        if prior_type not in v1.C.PARAMETER_SCALE_PRIOR_TYPES:
            # v1 logNormal / logLaplace are based on the natural logarithm,
            #  just like v2 log-normal / log-laplace
            return {
                v1.C.LOG_NORMAL: v2.C.LOG_NORMAL,
                v1.C.LOG_LAPLACE: v2.C.LOG_LAPLACE,
            }.get(prior_type, prior_type), prior_pars

        if prior_type == v1.C.PARAMETER_SCALE_UNIFORM and pscale != v1.C.LIN:
            # A uniform distribution of ln(x) (log10(x)) on [a, b] is a
            #  log-uniform distribution of x on [e^a, e^b] ([10^a, 10^b]).
            #  v2 log-uniform takes the bounds of x as parameters.
            #  Missing parameters default to the parameter bounds (see
            #  `update_prior_pars` below), as in v1.
            if not pd.isna(prior_pars):
                # Clip [a, b] to the bounds before unscaling, as v1 does
                #  (see `petab.v1.priors.Prior`). This avoids overflow for
                #  intervals that exceed the floating-point range when
                #  unscaled.
                low, high = map(
                    float, str(prior_pars).split(v1.C.PARAMETER_SEPARATOR)
                )
                lb, ub = row[v1.C.LOWER_BOUND], row[v1.C.UPPER_BOUND]
                low = lb if low <= scale(lb, pscale) else unscale(low, pscale)
                high = (
                    ub if high >= scale(ub, pscale) else unscale(high, pscale)
                )
                prior_pars = (
                    f"{float(low)}{v2.C.PARAMETER_SEPARATOR}{float(high)}"
                )
            return v2.C.LOG_UNIFORM, prior_pars

        if pscale == v1.C.LOG10 and prior_type in (
            v1.C.PARAMETER_SCALE_NORMAL,
            v1.C.PARAMETER_SCALE_LAPLACE,
        ):
            # PEtab v2 has no log10-based priors. Since
            #  ln(x) = ln(10) * log10(x), a normal (Laplace) distribution of
            #  log10(x) with mean (location) mu and standard deviation
            #  (scale) sigma is a normal (Laplace) distribution of ln(x)
            #  with ln(10) * mu and ln(10) * sigma.
            pscale = v1.C.LOG
            prior_pars = v2.C.PARAMETER_SEPARATOR.join(
                str(float(p) * math.log(10))
                for p in str(prior_pars).split(v1.C.PARAMETER_SEPARATOR)
            )

        new_prior_type = prior_type.removeprefix("parameterScale").lower()
        if pscale != v1.C.LIN:
            new_prior_type = f"{pscale}-{new_prior_type}"

        if new_prior_type not in v2.C.PRIOR_DISTRIBUTIONS:
            raise NotImplementedError(
                f"PEtab v2 does not support prior type `{new_prior_type}' "
                f"required for parameter `{row[v1.C.PARAMETER_ID]}'."
            )

        return new_prior_type, prior_pars

    # update parameterScale*-priors
    if v1.C.OBJECTIVE_PRIOR_TYPE in df.columns:
        df[[v1.C.OBJECTIVE_PRIOR_TYPE, v1.C.OBJECTIVE_PRIOR_PARAMETERS]] = (
            df.apply(update_prior, axis=1, result_type="expand")
        )

    # rename objectivePrior* to prior*
    df = df.rename(
        columns={
            v1.C.OBJECTIVE_PRIOR_TYPE: v2.C.PRIOR_DISTRIBUTION,
            v1.C.OBJECTIVE_PRIOR_PARAMETERS: v2.C.PRIOR_PARAMETERS,
        },
        errors="ignore",
    )
    # some columns were dropped in PEtab v2
    if v1.C.INITIALIZATION_PRIOR_TYPE in df and (
        df[v1.C.INITIALIZATION_PRIOR_TYPE].notna().any()
    ):
        warnings.warn(
            "Initialisation priors in parameter table are not supported "
            "in PEtab v2.",
            stacklevel=9,
        )
    if not (df[v1.C.PARAMETER_SCALE] == v1.C.LIN).all():
        msg = "Parameter scales are not supported in PEtab v2."
        reparametrizable = (
            parameter_df[v1.C.PARAMETER_SCALE].isin([v1.C.LOG, v1.C.LOG10])
            & (parameter_df[v1.C.ESTIMATE].astype(int) == 1)
            & (parameter_df[v1.C.LOWER_BOUND] >= 0)
        )
        if reparametrize == "none" and reparametrizable.any():
            prior_types = parameter_df.get(v1.C.OBJECTIVE_PRIOR_TYPE)
            scale_priors = (
                []
                if prior_types is None
                else parameter_df.index[
                    reparametrizable
                    & prior_types.isin(v1.C.PARAMETER_SCALE_PRIOR_TYPES)
                ]
            )
            msg += " Log-scaled parameters are estimated on linear scale."
            if len(scale_priors):
                msg += (
                    f" Parameter-scale priors of {', '.join(scale_priors)} "
                    "were converted to the corresponding priors of the "
                    "unscaled parameters, which changes the maximum a "
                    "posteriori estimate."
                )
            msg += (
                " Use `reparametrize='priors'` or `'all'` to estimate them "
                "on log scale."
            )
        warnings.warn(msg, stacklevel=9)
    df = df.drop(
        columns=[
            v1.C.INITIALIZATION_PRIOR_TYPE,
            v1.C.INITIALIZATION_PRIOR_PARAMETERS,
            v1.C.PARAMETER_SCALE,
        ],
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


def _reparametrize_log_scaled(
    problem: v2.Problem,
    v1_parameter_df: pd.DataFrame,
    reparametrize: Reparametrize,
) -> v2.Problem | None:
    """Reparametrize log-scaled parameters of a converted problem.

    See :func:`petab1to2`.

    :param problem: The converted problem.
    :param v1_parameter_df: The original PEtab v1 parameter table.
    :param reparametrize: The parameters to reparametrize.
    :return:
        The reparametrized problem, or ``None`` if no parameter was
        reparametrized.
    """
    df = v1_parameter_df
    has_priors = _has_objective_priors(df)
    # for each base: candidates and their priors over the scaled parameter
    candidates = {10: {}, "e": {}}
    # parameters with parameter-scale priors
    required = {10: set(), "e": set()}
    selected = {10: set(), "e": set()}
    # selected parameters that cannot be reparametrized due to their bounds
    #  and priors
    zero_bound = []
    # non-estimated parameters whose priors cannot be converted
    no_prior = set()
    for par_id, row in df.iterrows():
        pscale = row[v1.C.PARAMETER_SCALE]
        if pscale not in (v1.C.LOG, v1.C.LOG10):
            continue
        base = 10 if pscale == v1.C.LOG10 else "e"
        log_function = _log10 if base == 10 else _log
        prior_type = row.get(v1.C.OBJECTIVE_PRIOR_TYPE)
        if pd.isna(prior_type):
            # v1 default
            prior_type = None
        elif prior_type not in v1.C.PARAMETER_SCALE_PRIOR_TYPES:
            # A prior over the unscaled parameter. Its conversion already
            #  preserves the objective function.
            continue
        estimate = int(row[v1.C.ESTIMATE])
        is_selected = estimate and (
            reparametrize == "all" or prior_type is not None
        )
        if estimate and (
            row[v1.C.LOWER_BOUND] < 0 or row[v1.C.UPPER_BOUND] <= 0
        ):
            # No valid bounds on log scale. A negative lower bound is invalid
            #  in v1; an upper bound of 0 implies `lb == ub == 0`.
            continue

        # The prior over the scaled parameter. Derived from the v1 table
        #  rather than from the converted v2 prior to avoid rounding errors.
        #  Without a prior type, but with other estimated parameters having
        #  priors, the v1 default is made explicit, as in
        #  `v1v2_parameter_df`.
        prior = None
        if prior_type is not None or (estimate and has_priors):
            lb, ub = row[v1.C.LOWER_BOUND], row[v1.C.UPPER_BOUND]
            if lb >= 0 and ub > 0:
                # v1 allows a lower bound of 0 for log-scaled parameters
                lb, ub = log_function(lb), log_function(ub)
            else:
                # missing or invalid bounds of a non-estimated parameter
                lb, ub = -math.inf, math.inf
            pars = row.get(v1.C.OBJECTIVE_PRIOR_PARAMETERS)
            if prior_type is None or pd.isna(pars):
                pars = [lb, ub]
            else:
                pars = [
                    float(x) for x in str(pars).split(v1.C.PARAMETER_SEPARATOR)
                ]
            if prior_type in (None, v1.C.PARAMETER_SCALE_UNIFORM):
                prior = (v2.C.UNIFORM, [max(pars[0], lb), min(pars[1], ub)])
            else:
                name = prior_type.removeprefix("parameterScale").lower()
                prior = (name, pars)
            if prior[0] == v2.C.UNIFORM and not all(
                map(math.isfinite, prior[1])
            ):
                if estimate:
                    # improper prior for a lower bound of 0
                    if is_selected:
                        zero_bound.append(par_id)
                    continue
                prior = None
                no_prior.add(par_id)

        candidates[base][par_id] = prior
        if estimate and prior_type is not None:
            required[base].add(par_id)
        if is_selected:
            selected[base].add(par_id)

    if zero_bound:
        warnings.warn(
            f"Log-scaled parameters {', '.join(zero_bound)} were not "
            "reparametrized, because their lower bound of 0 would result in "
            "an improper prior on log scale. "
            "They are estimated on linear scale, and any parameter-scale "
            "priors were converted to the corresponding priors of the "
            "unscaled parameters, which changes the maximum a posteriori "
            "estimate.",
            stacklevel=4,
        )

    # Placeholders can only be transformed if all their overrides are
    #  reparametrized or non-negative numbers. Add log-scaled parameters that
    #  share a placeholder with selected parameters. If that is not possible,
    #  drop selected parameters that don't have parameter-scale priors.
    #  Otherwise, `reparametrize_log` will raise.
    slots = _get_override_slots(problem).values()
    excluded = set()
    for base, base_selected in selected.items():
        changed = True
        while changed:
            changed = False
            for values in slots:
                ids = {str(v) for v in values if isinstance(v, sp.Symbol)}
                if not ids & base_selected:
                    continue
                other = ids - base_selected
                if other and other <= candidates[base].keys() - excluded:
                    base_selected |= other
                    changed = True
                elif other or any(v.is_number and v < 0 for v in values):
                    optional = (ids & base_selected) - required[base]
                    if optional:
                        base_selected -= optional
                        excluded |= optional
                        changed = True
    if excluded:
        warnings.warn(
            "Log-scaled parameters "
            f"{', '.join(sorted(excluded))} were not reparametrized, "
            "because they are used as overrides for placeholders that "
            "cannot be log-transformed.",
            stacklevel=4,
        )

    if not any(selected.values()):
        return None

    if no_prior := no_prior & set().union(*selected.values()):
        warnings.warn(
            "The priors of the non-estimated parameters "
            f"{', '.join(sorted(no_prior))} were dropped, because they "
            "cannot be converted to log scale without valid bounds.",
            stacklevel=4,
        )

    for base, base_selected in selected.items():
        if not base_selected:
            continue
        try:
            problem, _ = reparametrize_log(
                problem,
                # keep the parameter table order
                [p for p in candidates[base] if p in base_selected],
                base=base,
                priors={p: candidates[base][p] for p in base_selected},
            )
        except NotImplementedError as e:
            raise NotImplementedError(
                "Log-scaled parameters could not be reparametrized. "
                "Use `reparametrize='none'` to estimate them on linear "
                "scale.\n" + str(e)
            ) from e

    return problem
