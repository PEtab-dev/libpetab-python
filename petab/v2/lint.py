"""Validation of PEtab problems"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from petab.v1 import (
    assert_model_parameters_in_condition_or_parameter_table,
)
from petab.v1.C import (
    ESTIMATE,
    MODEL_ENTITY_ID,
    NOISE_PARAMETERS,
    NOMINAL_VALUE,
    OBSERVABLE_PARAMETERS,
    PARAMETER_DF_REQUIRED_COLS,
    PARAMETER_ID,
)
from petab.v1.conditions import get_parametric_overrides
from petab.v1.lint import (
    _check_df,
    assert_no_leading_trailing_whitespace,
    assert_parameter_bounds_are_numeric,
    assert_parameter_estimate_is_boolean,
    assert_parameter_id_is_string,
    assert_parameter_prior_parameters_are_valid,
    assert_parameter_prior_type_is_valid,
    assert_parameter_scale_is_valid,
    assert_unique_parameter_ids,
    check_ids,
    check_parameter_bounds,
)
from petab.v1.measurements import split_parameter_replacement_list
from petab.v1.observables import get_output_parameters, get_placeholders
from petab.v1.parameters import (
    get_valid_parameters_for_parameter_table,
)
from petab.v1.visualize.lint import validate_visualization_df

from ..v1 import (
    assert_measurement_conditions_present_in_condition_table,
    check_condition_df,
    check_measurement_df,
    check_observable_df,
)
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
    "CheckMeasurementTable",
    "CheckConditionTable",
    "CheckObservableTable",
    "CheckParameterTable",
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


class CheckMeasurementTable(ValidationTask):
    """A task to validate the measurement table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.measurement_df is None:
            return

        try:
            check_measurement_df(problem.measurement_df, problem.observable_df)

            if problem.condition_df is not None:
                # TODO: handle missing condition_df
                assert_measurement_conditions_present_in_condition_table(
                    problem.measurement_df, problem.condition_df
                )
        except AssertionError as e:
            return ValidationError(str(e))


class CheckConditionTable(ValidationTask):
    """A task to validate the condition table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.condition_df is None:
            return

        try:
            check_condition_df(
                problem.condition_df,
                model=problem.model,
                observable_df=problem.observable_df,
                mapping_df=problem.mapping_df,
            )
        except AssertionError as e:
            return ValidationError(str(e))


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

        allowed = get_valid_parameters_for_parameter_table(
            model=problem.model,
            condition_df=problem.condition_df,
            observable_df=problem.observable_df,
            measurement_df=problem.measurement_df,
            mapping_df=problem.mapping_df,
        )

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


def get_required_parameters_for_parameter_table(
    problem: Problem,
) -> set[str]:
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
            if isinstance(p, str) and p not in problem.condition_df.columns
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
        p
        for p in get_parametric_overrides(problem.condition_df)
        if not problem.model.has_entity_with_id(p)
    )

    # remove parameters that occur in the condition table and are overridden
    #  for ALL conditions
    for p in problem.condition_df.columns[
        ~problem.condition_df.isnull().any()
    ]:
        try:
            parameter_ids.remove(p)
        except KeyError:
            pass

    return parameter_ids


#: Validation tasks that should be run on any PEtab problem
default_validation_tasks = [
    CheckTableExists("measurement"),
    CheckTableExists("observable"),
    CheckTableExists("parameter"),
    CheckModel(),
    CheckMeasurementTable(),
    CheckConditionTable(),
    CheckObservableTable(),
    CheckObservablesDoNotShadowModelEntities(),
    CheckParameterTable(),
    CheckAllParametersPresentInParameterTable(),
    CheckVisualizationTable(),
    CheckValidParameterInConditionOrParameterTable(),
]
