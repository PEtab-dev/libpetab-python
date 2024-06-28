"""Validation of PEtab problems"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from .problem import Problem

logger = logging.getLogger(__name__)


def lint_problem(problem: Problem | str | Path) -> ValidationResultList:
    """Validate a PEtab problem.

    Arguments:
        problem: PEtab problem to check. Instance of :class:`Problem` or path
        to a PEtab problem yaml file.
    """

    if isinstance(problem, str | Path):
        problem = Problem.from_yaml(str(problem))

    return problem.validate()


class ValidationTask(ABC):
    """A task to validate a PEtab problem."""

    @abstractmethod
    def run(self, problem: Problem) -> list[ValidationResult]:
        """Run the validation task.

        Arguments:
            problem: PEtab problem to check.
        """
        ...


class ModelValidationTask(ValidationTask):
    """A task to validate the model of a PEtab problem."""

    def run(self, problem: Problem) -> list[ValidationResult]:
        if problem.model is None:
            return ValidationResultList(
                [
                    ValidationResult(
                        level=ValidationEventLevel.WARNING,
                        message="Model is missing.",
                    )
                ]
            )

        if problem.model.is_valid():
            return []

        return ValidationResultList(
            [
                ValidationResult(
                    level=ValidationEventLevel.ERROR,
                    message="Model is invalid.",
                )
            ]
        )


class MeasurementTableValidationTask(ValidationTask):
    """A task to validate the measurement table of a PEtab problem."""

    def run(self, problem: Problem) -> list[ValidationResult]:
        if problem.measurement_df is None or problem.measurement_df.empty:
            return ValidationResultList(
                [
                    ValidationResult(
                        level=ValidationEventLevel.ERROR,
                        message="Measurement table is missing or empty.",
                    )
                ]
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
            return ValidationResultList(
                [
                    ValidationResult(
                        level=ValidationEventLevel.ERROR, message=str(e)
                    )
                ]
            )
        return []


class ConditionTableValidationTask(ValidationTask):
    """A task to validate the condition table of a PEtab problem."""

    def run(self, problem: Problem) -> list[ValidationResult]:
        if problem.condition_df is None or problem.condition_df.empty:
            return ValidationResultList(
                [
                    ValidationResult(
                        level=ValidationEventLevel.ERROR,
                        message="Condition table is missing or empty.",
                    )
                ]
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
            return ValidationResultList(
                [
                    ValidationResult(
                        level=ValidationEventLevel.ERROR, message=str(e)
                    )
                ]
            )
        return []


class ObservableTableValidationTask(ValidationTask):
    """A task to validate the observable table of a PEtab problem."""

    def run(self, problem: Problem):
        if problem.observable_df is None or problem.observable_df.empty:
            return ValidationResultList(
                [
                    ValidationResult(
                        level=ValidationEventLevel.ERROR,
                        message="Observable table is missing or empty.",
                    )
                ]
            )

        try:
            from ..v1 import check_observable_df

            check_observable_df(
                problem.observable_df,
            )
        except AssertionError as e:
            return ValidationResultList(
                [
                    ValidationResult(
                        level=ValidationEventLevel.ERROR, message=str(e)
                    )
                ]
            )
        result = ValidationResultList()
        if problem.model is not None:
            for obs_id in problem.observable_df.index:
                if problem.model.has_entity_with_id(obs_id):
                    result.append(
                        ValidationResult(
                            level=ValidationEventLevel.ERROR,
                            message=f"Observable ID {obs_id} shadows model "
                            "entity.",
                        )
                    )
        return result


class ParameterTableValidationTask(ValidationTask):
    """A task to validate the parameter table of a PEtab problem."""

    def run(self, problem: Problem) -> list[ValidationResult]:
        if problem.parameter_df is None or problem.parameter_df.empty:
            return ValidationResultList(
                [
                    ValidationResult(
                        level=ValidationEventLevel.ERROR,
                        message="Parameter table is missing or empty.",
                    )
                ]
            )

        try:
            from ..v1 import check_parameter_df

            check_parameter_df(
                problem.parameter_df,
                problem.model,
                problem.observable_df,
                problem.measurement_df,
                problem.condition_df,
                problem.mapping_df,
            )
        except AssertionError as e:
            return ValidationResultList(
                [
                    ValidationResult(
                        level=ValidationEventLevel.ERROR, message=str(e)
                    )
                ]
            )
        return []


class MiscValidationTask(ValidationTask):
    """A task to perform miscellaneous validation checks."""

    def run(self, problem: Problem):
        result = ValidationResultList()
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
                result.append(
                    ValidationResult(
                        level=ValidationEventLevel.ERROR, message=str(e)
                    )
                )

        if problem.visualization_df is not None:
            from petab.visualize.lint import validate_visualization_df

            # TODO: don't log directly
            if validate_visualization_df(problem):
                result.append(
                    ValidationResult(
                        level=ValidationEventLevel.ERROR,
                        message="Visualization table is invalid.",
                    )
                )
        else:
            result.append(
                ValidationResult(
                    level=ValidationEventLevel.WARNING,
                    message="Visualization table is missing.",
                )
            )

        if (
            problem.measurement_df is None
            or problem.condition_df is None
            or problem.model is None
            or problem.parameter_df is None
            or problem.observable_df is None
        ):
            result.append(
                ValidationResult(
                    level=ValidationEventLevel.WARNING,
                    message="Not all files of the PEtab problem definition "
                    "could be checked.",
                )
            )
        return result


@dataclass
class ValidationResult:
    """The result of a validation task."""

    level: ValidationEventLevel
    message: str


class ValidationResultList(list[ValidationResult]):
    """A list of validation results."""

    def log(self, min_):
        """Log the validation results."""
        for result in self:
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


class ValidationEventLevel(IntEnum):
    # INFO: Informational message, no action required
    INFO = 10
    # WARNING: Warning message, potential issues
    WARNING = 20
    # ERROR: Error message, action required
    ERROR = 30
    # CRITICAL: Critical error message, stops further validation
    CRITICAL = 40


default_validation_taks = [
    ModelValidationTask(),
    MeasurementTableValidationTask(),
    ConditionTableValidationTask(),
    ObservableTableValidationTask(),
    ParameterTableValidationTask(),
    MiscValidationTask(),
]
