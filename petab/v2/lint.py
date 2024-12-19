"""Validation of PEtab problems"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Set
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from .. import v2
from ..v1.lint import (
    _check_df,
    assert_measured_observables_defined,
    assert_measurements_not_null,
    assert_measurements_numeric,
    assert_model_parameters_in_condition_or_parameter_table,
    assert_no_leading_trailing_whitespace,
    assert_parameter_bounds_are_numeric,
    assert_parameter_estimate_is_boolean,
    assert_parameter_id_is_string,
    assert_parameter_prior_parameters_are_valid,
    assert_parameter_prior_type_is_valid,
    assert_parameter_scale_is_valid,
    assert_unique_observable_ids,
    assert_unique_parameter_ids,
    check_ids,
    check_observable_df,
    check_parameter_bounds,
)
from ..v1.measurements import (
    assert_overrides_match_parameter_count,
    split_parameter_replacement_list,
)
from ..v1.observables import get_output_parameters, get_placeholders
from ..v1.visualize.lint import validate_visualization_df
from ..v2.C import *
from .problem import Problem

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationIssueSeverity",
    "ValidationIssue",
    "ValidationResultList",
    "ValidationError",
    "ValidationTask",
    "CheckModel",
    "CheckTableExists",
    "CheckValidPetabIdColumn",
    "CheckMeasurementTable",
    "CheckConditionTable",
    "CheckObservableTable",
    "CheckParameterTable",
    "CheckExperimentTable",
    "CheckExperimentConditionsExist",
    "CheckAllParametersPresentInParameterTable",
    "CheckValidParameterInConditionOrParameterTable",
    "CheckVisualizationTable",
    "lint_problem",
    "default_validation_tasks",
]


class ValidationIssueSeverity(IntEnum):
    """The severity of a validation issue."""

    # INFO: Informational message, no action required
    INFO = 10
    # WARNING: Warning message, potential issues
    WARNING = 20
    # ERROR: Error message, action required
    ERROR = 30
    # CRITICAL: Critical error message, stops further validation
    CRITICAL = 40


@dataclass
class ValidationIssue:
    """The result of a validation task.

    Attributes:
        level: The level of the validation event.
        message: The message of the validation event.
    """

    level: ValidationIssueSeverity
    message: str

    def __post_init__(self):
        if not isinstance(self.level, ValidationIssueSeverity):
            raise TypeError(
                "`level` must be an instance of ValidationIssueSeverity."
            )

    def __str__(self):
        return f"{self.level.name}: {self.message}"


@dataclass
class ValidationError(ValidationIssue):
    """A validation result with level ERROR."""

    level: ValidationIssueSeverity = field(
        default=ValidationIssueSeverity.ERROR, init=False
    )
    task: str | None = None

    def __post_init__(self):
        if self.task is None:
            self.task = self._get_task_name()

    def _get_task_name(self):
        """Get the name of the ValidationTask that raised this error."""
        import inspect

        # walk up the stack until we find the ValidationTask.run method
        for frame_info in inspect.stack():
            frame = frame_info.frame
            if "self" in frame.f_locals:
                task = frame.f_locals["self"]
                if isinstance(task, ValidationTask):
                    return task.__class__.__name__


class ValidationResultList(list[ValidationIssue]):
    """A list of validation results.

    Contains all issues found during the validation of a PEtab problem.
    """

    def log(
        self,
        *,
        logger: logging.Logger = logger,
        min_level: ValidationIssueSeverity = ValidationIssueSeverity.INFO,
    ):
        """Log the validation results."""
        for result in self:
            if result.level < min_level:
                continue
            if result.level == ValidationIssueSeverity.INFO:
                logger.info(result.message)
            elif result.level == ValidationIssueSeverity.WARNING:
                logger.warning(result.message)
            elif result.level >= ValidationIssueSeverity.ERROR:
                logger.error(result.message)

        if not self:
            logger.info("PEtab format check completed successfully.")

    def has_errors(self) -> bool:
        """Check if there are any errors in the validation results."""
        return any(
            result.level >= ValidationIssueSeverity.ERROR for result in self
        )


def lint_problem(problem: Problem | str | Path) -> ValidationResultList:
    """Validate a PEtab problem.

    Arguments:
        problem:
            PEtab problem to check. Instance of :class:`Problem` or path
            to a PEtab problem yaml file.
    Returns:
        A list of validation results. Empty if no issues were found.
    """

    problem = Problem.get_problem(problem)

    return problem.validate()


class ValidationTask(ABC):
    """A task to validate a PEtab problem."""

    @abstractmethod
    def run(self, problem: Problem) -> ValidationIssue | None:
        """Run the validation task.

        Arguments:
            problem: PEtab problem to check.
        Returns:
            Validation results or ``None``
        """
        ...

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class CheckModel(ValidationTask):
    """A task to validate the model of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.model is None:
            return ValidationError("Model is missing.")

        if not problem.model.is_valid():
            # TODO get actual model validation messages
            return ValidationError("Model is invalid.")


class CheckTableExists(ValidationTask):
    """A task to check if a table exists in the PEtab problem."""

    def __init__(self, table_name: str):
        if table_name not in ["measurement", "observable", "parameter"]:
            # all others are optional
            raise ValueError(
                f"Table name {table_name} is not supported. "
                "Supported table names are 'measurement', 'observable', "
                "'parameter'."
            )
        self.table_name = table_name

    def run(self, problem: Problem) -> ValidationIssue | None:
        if getattr(problem, f"{self.table_name}_df") is None:
            return ValidationError(f"{self.table_name} table is missing.")


class CheckValidPetabIdColumn(ValidationTask):
    """A task to check that a given column contains only valid PEtab IDs."""

    def __init__(
        self,
        table_name: str,
        column_name: str,
        required_column: bool = True,
        ignore_nan: bool = False,
    ):
        self.table_name = table_name
        self.column_name = column_name
        self.required_column = required_column
        self.ignore_nan = ignore_nan

    def run(self, problem: Problem) -> ValidationIssue | None:
        df = getattr(problem, f"{self.table_name}_df")
        if df is None:
            return

        if self.column_name not in df.columns:
            if self.required_column:
                return ValidationError(
                    f"Column {self.column_name} is missing in "
                    f"{self.table_name} table."
                )
            return

        try:
            ids = df[self.column_name].values
            if self.ignore_nan:
                ids = ids[~pd.isna(ids)]
            check_ids(ids, kind=self.column_name)
        except ValueError as e:
            return ValidationError(str(e))


class CheckMeasurementTable(ValidationTask):
    """A task to validate the measurement table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.measurement_df is None:
            return

        df = problem.measurement_df
        try:
            _check_df(df, MEASUREMENT_DF_REQUIRED_COLS, "measurement")

            for column_name in MEASUREMENT_DF_REQUIRED_COLS:
                if not np.issubdtype(df[column_name].dtype, np.number):
                    assert_no_leading_trailing_whitespace(
                        df[column_name].values, column_name
                    )

            for column_name in MEASUREMENT_DF_OPTIONAL_COLS:
                if column_name in df and not np.issubdtype(
                    df[column_name].dtype, np.number
                ):
                    assert_no_leading_trailing_whitespace(
                        df[column_name].values, column_name
                    )

            if problem.observable_df is not None:
                assert_measured_observables_defined(df, problem.observable_df)
                assert_overrides_match_parameter_count(
                    df, problem.observable_df
                )

                if OBSERVABLE_TRANSFORMATION in problem.observable_df:
                    # Check for positivity of measurements in case of
                    #  log-transformation
                    assert_unique_observable_ids(problem.observable_df)
                    # If the above is not checked, in the following loop
                    # trafo may become a pandas Series
                    for measurement, obs_id in zip(
                        df[MEASUREMENT], df[OBSERVABLE_ID], strict=True
                    ):
                        trafo = problem.observable_df.loc[
                            obs_id, OBSERVABLE_TRANSFORMATION
                        ]
                        if measurement <= 0.0 and trafo in [LOG, LOG10]:
                            raise ValueError(
                                "Measurements with observable "
                                f"transformation {trafo} must be "
                                f"positive, but {measurement} <= 0."
                            )

            assert_measurements_not_null(df)
            assert_measurements_numeric(df)
        except AssertionError as e:
            return ValidationError(str(e))

        # TODO: introduce some option for validation of partial vs full
        #  problem. if this is supposed to be a complete problem, a missing
        #  condition table should be an error if the measurement table refers
        #  to conditions, otherwise it should maximally be a warning
        used_experiments = set(problem.measurement_df[EXPERIMENT_ID].values)
        # handle default-experiment
        used_experiments = set(
            filter(
                lambda x: not isinstance(x, float) or not np.isnan(x),
                used_experiments,
            )
        )
        # check that measured experiments exist
        available_experiments = (
            set(problem.experiment_df[EXPERIMENT_ID].unique())
            if problem.experiment_df is not None
            else set()
        )
        if missing_experiments := (used_experiments - available_experiments):
            return ValidationError(
                "Measurement table references experiments that "
                "are not specified in the experiments table: "
                + str(missing_experiments)
            )


class CheckConditionTable(ValidationTask):
    """A task to validate the condition table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.condition_df is None:
            return

        df = problem.condition_df

        try:
            _check_df(df, CONDITION_DF_REQUIRED_COLS, "condition")
            check_ids(df[CONDITION_ID], kind="condition")
            check_ids(df[TARGET_ID], kind="target")
        except AssertionError as e:
            return ValidationError(str(e))

        # TODO: check value types

        if problem.model is None:
            return

        # check targets are valid
        allowed_targets = set(
            problem.model.get_valid_ids_for_condition_table()
        )
        if problem.observable_df is not None:
            allowed_targets |= set(
                get_output_parameters(
                    model=problem.model,
                    observable_df=problem.observable_df,
                    mapping_df=problem.mapping_df,
                )
            )
        if problem.mapping_df is not None:
            allowed_targets |= set(problem.mapping_df.index.values)
        invalid = set(df[TARGET_ID].unique()) - allowed_targets
        if invalid:
            return ValidationError(
                f"Condition table contains invalid targets: {invalid}"
            )

        # TODO check that all value types are valid for the given targets


class CheckObservableTable(ValidationTask):
    """A task to validate the observable table of a PEtab problem."""

    def run(self, problem: Problem):
        if problem.observable_df is None:
            return

        try:
            check_observable_df(
                problem.observable_df,
            )
        except AssertionError as e:
            return ValidationIssue(
                level=ValidationIssueSeverity.ERROR, message=str(e)
            )


class CheckObservablesDoNotShadowModelEntities(ValidationTask):
    """A task to check that observable IDs do not shadow model entities."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.observable_df is None or problem.model is None:
            return

        shadowed_entities = [
            obs_id
            for obs_id in problem.observable_df.index
            if problem.model.has_entity_with_id(obs_id)
        ]
        if shadowed_entities:
            return ValidationError(
                f"Observable IDs {shadowed_entities} shadow model entities."
            )


class CheckParameterTable(ValidationTask):
    """A task to validate the parameter table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.parameter_df is None:
            return

        try:
            df = problem.parameter_df
            _check_df(df, PARAMETER_DF_REQUIRED_COLS[1:], "parameter")

            if df.index.name != PARAMETER_ID:
                return ValidationError(
                    f"Parameter table has wrong index {df.index.name}."
                    f" Expected {PARAMETER_ID}.",
                )

            check_ids(df.index.values, kind="parameter")

            for column_name in PARAMETER_DF_REQUIRED_COLS[
                1:
            ]:  # 0 is PARAMETER_ID
                if not np.issubdtype(df[column_name].dtype, np.number):
                    assert_no_leading_trailing_whitespace(
                        df[column_name].values, column_name
                    )

            # nominal value is required for non-estimated parameters
            non_estimated_par_ids = list(
                df.index[
                    (df[ESTIMATE] != 1)
                    | (
                        pd.api.types.is_string_dtype(df[ESTIMATE])
                        and df[ESTIMATE] != "1"
                    )
                ]
            )
            # TODO implement as validators
            #  `assert_has_fixed_parameter_nominal_values`
            #   and `assert_correct_table_dtypes`
            if non_estimated_par_ids:
                if NOMINAL_VALUE not in df:
                    return ValidationError(
                        "Parameter table contains parameters "
                        f"{non_estimated_par_ids} that are not "
                        "specified to be estimated, "
                        f"but column {NOMINAL_VALUE} is missing."
                    )
                try:
                    df.loc[non_estimated_par_ids, NOMINAL_VALUE].apply(float)
                except ValueError:
                    return ValidationError(
                        f"Expected numeric values for `{NOMINAL_VALUE}` "
                        "in parameter table "
                        "for all non-estimated parameters."
                    )

            assert_parameter_id_is_string(df)
            assert_parameter_scale_is_valid(df)
            assert_parameter_bounds_are_numeric(df)
            assert_parameter_estimate_is_boolean(df)
            assert_unique_parameter_ids(df)
            check_parameter_bounds(df)
            assert_parameter_prior_type_is_valid(df)
            assert_parameter_prior_parameters_are_valid(df)

        except AssertionError as e:
            return ValidationError(str(e))


class CheckExperimentTable(ValidationTask):
    """A task to validate the experiment table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.experiment_df is None:
            return

        df = problem.experiment_df

        try:
            _check_df(df, EXPERIMENT_DF_REQUIRED_COLS, "experiment")
        except AssertionError as e:
            return ValidationError(str(e))

        # valid timepoints
        invalid = []
        for time in df[TIME].values:
            try:
                time = float(time)
                if not np.isfinite(time) and time != -np.inf:
                    invalid.append(time)
            except ValueError:
                invalid.append(time)
        if invalid:
            return ValidationError(
                f"Invalid timepoints in experiment table: {invalid}"
            )


class CheckExperimentConditionsExist(ValidationTask):
    """A task to validate that all conditions in the experiment table exist
    in the condition table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.experiment_df is None:
            return

        if (
            problem.condition_df is None
            and problem.experiment_df is not None
            and not problem.experiment_df.empty
        ):
            return ValidationError(
                "Experiment table is non-empty, "
                "but condition table is missing."
            )

        required_conditions = problem.experiment_df[CONDITION_ID].unique()
        existing_conditions = problem.condition_df[CONDITION_ID].unique()

        missing_conditions = set(required_conditions) - set(
            existing_conditions
        )
        if missing_conditions:
            return ValidationError(
                f"Experiment table contains conditions that are not present "
                f"in the condition table: {missing_conditions}"
            )


class CheckAllParametersPresentInParameterTable(ValidationTask):
    """Ensure all required parameters are contained in the parameter table
    with no additional ones."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if (
            problem.model is None
            or problem.parameter_df is None
            or problem.observable_df is None
            or problem.measurement_df is None
        ):
            return

        required = get_required_parameters_for_parameter_table(problem)
        allowed = get_valid_parameters_for_parameter_table(problem)

        actual = set(problem.parameter_df.index)
        missing = required - actual
        extraneous = actual - allowed

        # missing parameters might be present under a different name based on
        # the mapping table
        if missing and problem.mapping_df is not None:
            model_to_petab_mapping = {}
            for map_from, map_to in zip(
                problem.mapping_df.index.values,
                problem.mapping_df[MODEL_ENTITY_ID],
                strict=True,
            ):
                if map_to in model_to_petab_mapping:
                    model_to_petab_mapping[map_to].append(map_from)
                else:
                    model_to_petab_mapping[map_to] = [map_from]
            missing = {
                missing_id
                for missing_id in missing
                if missing_id not in model_to_petab_mapping
                or all(
                    mapping_parameter not in actual
                    for mapping_parameter in model_to_petab_mapping[missing_id]
                )
            }

        if missing:
            return ValidationError(
                "Missing parameter(s) in the model or the "
                "parameters table: " + str(missing)
            )

        if extraneous:
            return ValidationError(
                "Extraneous parameter(s) in parameter table: "
                + str(extraneous)
            )


class CheckValidParameterInConditionOrParameterTable(ValidationTask):
    """A task to check that all required and only allowed model parameters are
    present in the condition or parameter table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if (
            problem.model is None
            or problem.condition_df is None
            or problem.parameter_df is None
        ):
            return

        try:
            assert_model_parameters_in_condition_or_parameter_table(
                problem.model,
                problem.condition_df,
                problem.parameter_df,
                problem.mapping_df,
            )
        except AssertionError as e:
            return ValidationIssue(
                level=ValidationIssueSeverity.ERROR, message=str(e)
            )


class CheckVisualizationTable(ValidationTask):
    """A task to validate the visualization table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.visualization_df is None:
            return

        if validate_visualization_df(problem):
            return ValidationIssue(
                level=ValidationIssueSeverity.ERROR,
                message="Visualization table is invalid.",
            )


def get_valid_parameters_for_parameter_table(
    problem: Problem,
) -> set[str]:
    """
    Get set of parameters which may be present inside the parameter table

    Arguments:
        model: PEtab model
        condition_df: PEtab condition table
        observable_df: PEtab observable table
        measurement_df: PEtab measurement table
        mapping_df: PEtab mapping table for additional checks

    Returns:
        Set of parameter IDs which PEtab allows to be present in the
        parameter table.
    """
    # - grab all allowed model parameters
    # - grab corresponding names from mapping table
    # - grab all output parameters defined in {observable,noise}Formula
    # - grab all parameters from measurement table
    # - grab all parametric overrides from condition table
    # - remove parameters for which condition table columns exist
    # - remove placeholder parameters
    #   (only partial overrides are not supported)
    model = problem.model
    condition_df = problem.condition_df
    observable_df = problem.observable_df
    measurement_df = problem.measurement_df
    mapping_df = problem.mapping_df

    # must not go into parameter table
    blackset = set()

    if observable_df is not None:
        placeholders = set(get_placeholders(observable_df))

        # collect assignment targets
        blackset |= placeholders

    if condition_df is not None:
        blackset |= set(condition_df.columns.values) - {CONDITION_NAME}

    # don't use sets here, to have deterministic ordering,
    #  e.g. for creating parameter tables
    parameter_ids = OrderedDict.fromkeys(
        p
        for p in model.get_valid_parameters_for_parameter_table()
        if p not in blackset
    )

    if mapping_df is not None:
        for from_id, to_id in mapping_df[MODEL_ENTITY_ID].items():
            if to_id in parameter_ids.keys():
                parameter_ids[from_id] = None

    if observable_df is not None:
        # add output parameters from observables table
        output_parameters = get_output_parameters(
            observable_df=observable_df, model=model
        )
        for p in output_parameters:
            if p not in blackset:
                parameter_ids[p] = None

    # Append parameters from measurement table, unless they occur as condition
    # table columns
    def append_overrides(overrides):
        for p in overrides:
            if isinstance(p, str) and p not in blackset:
                parameter_ids[p] = None

    if measurement_df is not None:
        for _, row in measurement_df.iterrows():
            # we trust that the number of overrides matches
            append_overrides(
                split_parameter_replacement_list(
                    row.get(OBSERVABLE_PARAMETERS, None)
                )
            )
            append_overrides(
                split_parameter_replacement_list(
                    row.get(NOISE_PARAMETERS, None)
                )
            )

    # Append parameter overrides from condition table
    if condition_df is not None:
        for p in v2.conditions.get_condition_table_free_symbols(problem):
            parameter_ids[str(p)] = None

    return set(parameter_ids.keys())


def get_required_parameters_for_parameter_table(
    problem: Problem,
) -> Set[str]:
    """
    Get set of parameters which need to go into the parameter table

    Arguments:
        problem: The PEtab problem
    Returns:
        Set of parameter IDs which PEtab requires to be present in the
        parameter table. That is all {observable,noise}Parameters from the
        measurement table as well as all parametric condition table overrides
        that are not defined in the model.
    """
    parameter_ids = set()

    # Add parameters from measurement table, unless they are fixed parameters
    def append_overrides(overrides):
        parameter_ids.update(
            p
            for p in overrides
            if isinstance(p, str)
            and (
                problem.condition_df is None
                or p not in problem.condition_df[TARGET_ID]
            )
        )

    for _, row in problem.measurement_df.iterrows():
        # we trust that the number of overrides matches
        append_overrides(
            split_parameter_replacement_list(
                row.get(OBSERVABLE_PARAMETERS, None)
            )
        )
        append_overrides(
            split_parameter_replacement_list(row.get(NOISE_PARAMETERS, None))
        )

    # remove `observable_ids` when
    # `get_output_parameters` is updated for PEtab v2/v1.1, where
    # observable IDs are allowed in observable formulae
    observable_ids = set(problem.observable_df.index)

    # Add output parameters except for placeholders
    for formula_type, placeholder_sources in (
        (
            # Observable formulae
            {"observables": True, "noise": False},
            # can only contain observable placeholders
            {"noise": False, "observables": True},
        ),
        (
            # Noise formulae
            {"observables": False, "noise": True},
            # can contain noise and observable placeholders
            {"noise": True, "observables": True},
        ),
    ):
        output_parameters = get_output_parameters(
            problem.observable_df,
            problem.model,
            mapping_df=problem.mapping_df,
            **formula_type,
        )
        placeholders = get_placeholders(
            problem.observable_df,
            **placeholder_sources,
        )
        parameter_ids.update(
            p
            for p in output_parameters
            if p not in placeholders and p not in observable_ids
        )

    # Add condition table parametric overrides unless already defined in the
    #  model
    parameter_ids.update(
        str(p)
        for p in v2.conditions.get_condition_table_free_symbols(problem)
        if not problem.model.has_entity_with_id(str(p))
    )

    # parameters that are overridden via the condition table are not allowed
    if problem.condition_df is not None:
        parameter_ids -= set(problem.condition_df[TARGET_ID].unique())

    return parameter_ids


#: Validation tasks that should be run on any PEtab problem
default_validation_tasks = [
    CheckTableExists("measurement"),
    CheckTableExists("observable"),
    CheckTableExists("parameter"),
    CheckModel(),
    CheckMeasurementTable(),
    CheckConditionTable(),
    CheckExperimentTable(),
    CheckValidPetabIdColumn("experiment", EXPERIMENT_ID, ignore_nan=True),
    CheckValidPetabIdColumn("experiment", CONDITION_ID),
    CheckExperimentConditionsExist(),
    CheckObservableTable(),
    CheckObservablesDoNotShadowModelEntities(),
    CheckParameterTable(),
    CheckAllParametersPresentInParameterTable(),
    # TODO: atomize checks, update to long condition table, re-enable
    # CheckVisualizationTable(),
    CheckValidParameterInConditionOrParameterTable(),
]
