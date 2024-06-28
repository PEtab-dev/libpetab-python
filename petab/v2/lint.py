"""Validation of PEtab problems"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from .problem import Problem

logger = logging.getLogger(__name__)


class ValidationEventLevel(IntEnum):
    """The level of a validation event."""

    # INFO: Informational message, no action required
    INFO = 10
    # WARNING: Warning message, potential issues
    WARNING = 20
    # ERROR: Error message, action required
    ERROR = 30
    # CRITICAL: Critical error message, stops further validation
    CRITICAL = 40


def lint_problem(problem: Problem | str | Path) -> ValidationResultList:
    """Validate a PEtab problem.

    Arguments:
        problem: PEtab problem to check. Instance of :class:`Problem` or path
        to a PEtab problem yaml file.
    Returns:
        A list of validation results. Empty if no issues were found.
    """

    if isinstance(problem, str | Path):
        problem = Problem.from_yaml(str(problem))

    return problem.validate()


class ValidationTask(ABC):
    """A task to validate a PEtab problem."""

    @abstractmethod
    def run(self, problem: Problem) -> ValidationResult | None:
        """Run the validation task.

        Arguments:
            problem: PEtab problem to check.
        Returns:
            Validation results or ``None``
        """
        ...

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class ModelValidationTask(ValidationTask):
    """A task to validate the model of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationResult | None:
        if problem.model is None:
            return ValidationResult(
                level=ValidationEventLevel.WARNING,
                message="Model is missing.",
            )

        if not problem.model.is_valid():
            # TODO get actual model validation messages
            return ValidationResult(
                level=ValidationEventLevel.ERROR,
                message="Model is invalid.",
            )


class MeasurementTableValidationTask(ValidationTask):
    """A task to validate the measurement table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationResult | None:
        if problem.measurement_df is None or problem.measurement_df.empty:
            return ValidationResult(
                level=ValidationEventLevel.ERROR,
                message="Measurement table is missing or empty.",
            )

        try:
            from ..v1 import (
                assert_measurement_conditions_present_in_condition_table,
                check_measurement_df,
            )

            check_measurement_df(problem.measurement_df, problem.observable_df)

            if problem.condition_df is not None:
                assert_measurement_conditions_present_in_condition_table(
                    problem.measurement_df, problem.condition_df
                )
        except AssertionError as e:
            return ValidationResult(
                level=ValidationEventLevel.ERROR, message=str(e)
            )


class ConditionTableValidationTask(ValidationTask):
    """A task to validate the condition table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationResult | None:
        if problem.condition_df is None or problem.condition_df.empty:
            return ValidationResult(
                level=ValidationEventLevel.ERROR,
                message="Condition table is missing or empty.",
            )

        try:
            from ..v1 import check_condition_df

            check_condition_df(
                problem.condition_df,
                model=problem.model,
                observable_df=problem.observable_df,
                mapping_df=problem.mapping_df,
            )
        except AssertionError as e:
            return ValidationResult(
                level=ValidationEventLevel.ERROR, message=str(e)
            )


class ObservableTableValidationTask(ValidationTask):
    """A task to validate the observable table of a PEtab problem."""

    def run(self, problem: Problem):
        if problem.observable_df is None or problem.observable_df.empty:
            return ValidationResult(
                level=ValidationEventLevel.ERROR,
                message="Observable table is missing or empty.",
            )

        try:
            from ..v1 import check_observable_df

            check_observable_df(
                problem.observable_df,
            )
        except AssertionError as e:
            return ValidationResult(
                level=ValidationEventLevel.ERROR, message=str(e)
            )
        if problem.model is not None:
            for obs_id in problem.observable_df.index:
                if problem.model.has_entity_with_id(obs_id):
                    return ValidationResult(
                        level=ValidationEventLevel.ERROR,
                        message=f"Observable ID {obs_id} shadows model "
                        "entity.",
                    )


class ParameterTableValidationTask(ValidationTask):
    """A task to validate the parameter table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationResult | None:
        if problem.parameter_df is None or problem.parameter_df.empty:
            return ValidationResult(
                level=ValidationEventLevel.ERROR,
                message="Parameter table is missing or empty.",
            )
        try:
            from petab.v1.C import (
                ESTIMATE,
                NOMINAL_VALUE,
                PARAMETER_DF_REQUIRED_COLS,
                PARAMETER_ID,
            )
            from petab.v1.lint import (
                _check_df,
                assert_all_parameters_present_in_parameter_df,
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

            df = problem.parameter_df
            _check_df(df, PARAMETER_DF_REQUIRED_COLS[1:], "parameter")

            if df.index.name != PARAMETER_ID:
                raise AssertionError(
                    f"Parameter table has wrong index {df.index.name}."
                    f"expected {PARAMETER_ID}."
                )

            check_ids(df.index.values, kind="parameter")

            for column_name in PARAMETER_DF_REQUIRED_COLS[
                1:
            ]:  # 0 is PARAMETER_ID
                if not np.issubdtype(df[column_name].dtype, np.number):
                    assert_no_leading_trailing_whitespace(
                        df[column_name].values, column_name
                    )

            # nominal value is generally optional, but required if any for any
            #  parameter estimate != 1
            non_estimated_par_ids = list(
                df.index[
                    (df[ESTIMATE] != 1)
                    | (
                        pd.api.types.is_string_dtype(df[ESTIMATE])
                        and df[ESTIMATE] != "1"
                    )
                ]
            )
            if non_estimated_par_ids:
                if NOMINAL_VALUE not in df:
                    raise AssertionError(
                        "Parameter table contains parameters "
                        f"{non_estimated_par_ids} that are not "
                        "specified to be estimated, "
                        f"but column {NOMINAL_VALUE} is missing."
                    )
                try:
                    df.loc[non_estimated_par_ids, NOMINAL_VALUE].apply(float)
                except ValueError as e:
                    raise AssertionError(
                        f"Expected numeric values for `{NOMINAL_VALUE}` "
                        "in parameter table "
                        "for all non-estimated parameters."
                    ) from e

            assert_parameter_id_is_string(df)
            assert_parameter_scale_is_valid(df)
            assert_parameter_bounds_are_numeric(df)
            assert_parameter_estimate_is_boolean(df)
            assert_unique_parameter_ids(df)
            check_parameter_bounds(df)
            assert_parameter_prior_type_is_valid(df)
            assert_parameter_prior_parameters_are_valid(df)

            if (
                problem.model
                and problem.measurement_df is not None
                and problem.condition_df is not None
            ):
                assert_all_parameters_present_in_parameter_df(
                    df,
                    problem.model,
                    problem.observable_df,
                    problem.measurement_df,
                    problem.condition_df,
                    problem.mapping_df,
                )

        except AssertionError as e:
            return ValidationResult(
                level=ValidationEventLevel.ERROR, message=str(e)
            )


class MiscValidationTask(ValidationTask):
    """A task to perform miscellaneous validation checks."""

    # TODO split further

    def run(self, problem: Problem) -> ValidationResult | None:
        from petab.v1 import (
            assert_model_parameters_in_condition_or_parameter_table,
        )

        if (
            problem.model is not None
            and problem.condition_df is not None
            and problem.parameter_df is not None
        ):
            try:
                assert_model_parameters_in_condition_or_parameter_table(
                    problem.model,
                    problem.condition_df,
                    problem.parameter_df,
                    problem.mapping_df,
                )
            except AssertionError as e:
                return ValidationResult(
                    level=ValidationEventLevel.ERROR, message=str(e)
                )

        if problem.visualization_df is not None:
            from petab.visualize.lint import validate_visualization_df

            # TODO: don't log directly
            if validate_visualization_df(problem):
                return ValidationResult(
                    level=ValidationEventLevel.ERROR,
                    message="Visualization table is invalid.",
                )
        else:
            return ValidationResult(
                level=ValidationEventLevel.WARNING,
                message="Visualization table is missing.",
            )

        if (
            problem.measurement_df is None
            or problem.condition_df is None
            or problem.model is None
            or problem.parameter_df is None
            or problem.observable_df is None
        ):
            return ValidationResult(
                level=ValidationEventLevel.WARNING,
                message="Not all files of the PEtab problem definition "
                "could be checked.",
            )


@dataclass
class ValidationResult:
    """The result of a validation task.

    Attributes:
        level: The level of the validation event.
        message: The message of the validation event.
    """

    level: ValidationEventLevel
    message: str

    def __str__(self):
        return f"{self.level.name}: {self.message}"


class ValidationResultList(list[ValidationResult]):
    """A list of validation results.

    Contains all issues found during the validation of a PEtab problem.
    """

    def log(self, min_level: ValidationEventLevel = ValidationEventLevel.INFO):
        """Log the validation results."""
        for result in self:
            if result.level < min_level:
                continue
            if result.level == ValidationEventLevel.INFO:
                logger.info(result.message)
            elif result.level == ValidationEventLevel.WARNING:
                logger.warning(result.message)
            elif result.level >= ValidationEventLevel.ERROR:
                logger.error(result.message)

        if not self:
            logger.info("PEtab format check completed successfully.")

    def has_errors(self) -> bool:
        """Check if there are any errors in the validation results."""
        return any(
            result.level >= ValidationEventLevel.ERROR for result in self
        )


#: Validation tasks that should be run on any PEtab problem
default_validation_tasks = [
    ModelValidationTask(),
    MeasurementTableValidationTask(),
    ConditionTableValidationTask(),
    ObservableTableValidationTask(),
    ParameterTableValidationTask(),
    MiscValidationTask(),
]
