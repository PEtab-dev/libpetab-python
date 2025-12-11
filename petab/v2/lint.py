"""Validation of PEtab problems"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from collections.abc import Set
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import chain
from pathlib import Path

import pandas as pd
import sympy as sp

from ..v2.C import *
from .core import PriorDistribution, Problem

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationIssueSeverity",
    "ValidationIssue",
    "ValidationResultList",
    "ValidationError",
    "ValidationTask",
    "CheckModel",
    "CheckProblemConfig",
    "CheckMeasuredObservablesDefined",
    "CheckOverridesMatchPlaceholders",
    "CheckMeasuredExperimentsDefined",
    "CheckMeasurementModelId",
    "CheckPosLogMeasurements",
    "CheckValidConditionTargets",
    "CheckUniquePrimaryKeys",
    "CheckExperimentTable",
    "CheckExperimentConditionsExist",
    "CheckAllParametersPresentInParameterTable",
    "CheckValidParameterInConditionOrParameterTable",
    "CheckUnusedExperiments",
    "CheckObservablesDoNotShadowModelEntities",
    "CheckUnusedConditions",
    "CheckPriorDistribution",
    "CheckUndefinedExperiments",
    "CheckInitialChangeSymbols",
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
    task: str | None = None

    def __post_init__(self):
        if not isinstance(self.level, ValidationIssueSeverity):
            raise TypeError(
                "`level` must be an instance of ValidationIssueSeverity."
            )

    def __str__(self):
        return f"{self.level.name}: {self.message}"

    @staticmethod
    def _get_task_name() -> str | None:
        """Get the name of the ValidationTask that raised this error.

        Expected to be called from below a `ValidationTask.run`.
        """
        import inspect

        # walk up the stack until we find the ValidationTask.run method
        for frame_info in inspect.stack():
            frame = frame_info.frame
            if "self" in frame.f_locals:
                task = frame.f_locals["self"]
                if isinstance(task, ValidationTask):
                    return task.__class__.__name__
        return None


@dataclass
class ValidationError(ValidationIssue):
    """A validation result with level ERROR."""

    level: ValidationIssueSeverity = field(
        default=ValidationIssueSeverity.ERROR, init=False
    )

    def __post_init__(self):
        if self.task is None:
            self.task = self._get_task_name()


@dataclass
class ValidationWarning(ValidationIssue):
    """A validation result with level WARNING."""

    level: ValidationIssueSeverity = field(
        default=ValidationIssueSeverity.WARNING, init=False
    )

    def __post_init__(self):
        if self.task is None:
            self.task = self._get_task_name()


class ValidationResultList(list[ValidationIssue]):
    """A list of validation results.

    Contains all issues found during the validation of a PEtab problem.
    """

    def log(
        self,
        *,
        logger: logging.Logger = logger,
        min_level: ValidationIssueSeverity = ValidationIssueSeverity.INFO,
        max_level: ValidationIssueSeverity = ValidationIssueSeverity.CRITICAL,
    ):
        """Log the validation results.

        :param logger: The logger to use for logging.
            Defaults to the module logger.
        :param min_level: The minimum severity level to log.
        :param max_level: The maximum severity level to log.
        """
        for result in self:
            if result.level < min_level or result.level > max_level:
                continue
            msg = f"{result.level.name}: {result.message} [{result.task}]"
            if result.level == ValidationIssueSeverity.INFO:
                logger.info(msg)
            elif result.level == ValidationIssueSeverity.WARNING:
                logger.warning(msg)
            elif result.level >= ValidationIssueSeverity.ERROR:
                logger.error(msg)

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
            to a PEtab problem YAML file.
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


class CheckProblemConfig(ValidationTask):
    """A task to validate the configuration of a PEtab problem.

    This corresponds to checking the problem YAML file semantics.
    """

    def run(self, problem: Problem) -> ValidationIssue | None:
        if (config := problem.config) is None or config.base_path is None:
            # This is allowed, so we can validate in-memory problems
            #  that don't have the list of files populated
            return None
            # TODO: decide when this should be emitted
            # return ValidationWarning("Problem configuration is missing.")

        # TODO: we need some option for validating partial vs full problems
        # check for unset but required files
        missing_files = []
        if not config.parameter_files:
            missing_files.append("parameters")

        if not config.measurement_files:
            missing_files.append("measurements")

        if not config.observable_files:
            missing_files.append("observables")

        if missing_files:
            return ValidationError(
                f"Missing files: {', '.join(missing_files)}"
            )

        return None


class CheckModel(ValidationTask):
    """A task to validate the model of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.model is None:
            return ValidationError("Model is missing.")

        if not problem.model.is_valid():
            # TODO get actual model validation messages
            return ValidationError("Model is invalid.")

        return None


class CheckMeasuredObservablesDefined(ValidationTask):
    """A task to check that all observables referenced by the measurements
    are defined."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        used_observables = {m.observable_id for m in problem.measurements}
        defined_observables = {o.id for o in problem.observables}
        if undefined_observables := (used_observables - defined_observables):
            return ValidationError(
                f"Observable(s) {undefined_observables} are used in the "
                "measurement table but are not defined in the observable "
                "table."
            )

        return None


class CheckOverridesMatchPlaceholders(ValidationTask):
    """A task to check that the number of observable/noise parameters
    in the measurements matches the number of placeholders in the observables.
    """

    def run(self, problem: Problem) -> ValidationIssue | None:
        observable_parameters_count = {
            o.id: len(o.observable_placeholders) for o in problem.observables
        }
        noise_parameters_count = {
            o.id: len(o.noise_placeholders) for o in problem.observables
        }
        messages = []
        observables = {o.id: o for o in problem.observables}
        for m in problem.measurements:
            # check observable parameters
            try:
                expected = observable_parameters_count[m.observable_id]
            except KeyError:
                messages.append(
                    f"Observable {m.observable_id} is used in the measurement "
                    f"table but is not defined in the observable table."
                )
                continue

            actual = len(m.observable_parameters)

            if actual != expected:
                formula = observables[m.observable_id].formula
                messages.append(
                    f"Mismatch of observable parameter overrides for "
                    f"{m.observable_id} ({formula})"
                    f"in:\n{m}\n"
                    f"Expected {expected} but got {actual}"
                )

            # check noise parameters
            expected = noise_parameters_count[m.observable_id]
            actual = len(m.noise_parameters)
            if actual != expected:
                # no overrides defined, but a numerical sigma can be provided
                # anyway
                if len(m.noise_parameters) != 1 or (
                    len(m.noise_parameters) == 1
                    and m.noise_parameters[0].is_number
                ):
                    messages.append(
                        "No placeholders have been specified in the "
                        f"noise model for observable {m.observable_id}, "
                        "but a parameter ID "
                        "or multiple overrides were specified in the "
                        "noiseParameters column."
                    )
                else:
                    formula = observables[m.observable_id].noise_formula
                    messages.append(
                        f"Mismatch of noise parameter overrides for "
                        f"{m.observable_id} ({formula})"
                        f"in:\n{m}\n"
                        f"Expected {expected} but got {actual}"
                    )

        if messages:
            return ValidationError("\n".join(messages))

        return None


class CheckPosLogMeasurements(ValidationTask):
    """Check that measurements for observables with
    log-transformation are positive."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        from .core import NoiseDistribution as ND  # noqa: N813

        log_observables = {
            o.id
            for o in problem.observables
            if o.noise_distribution in [ND.LOG_NORMAL, ND.LOG_LAPLACE]
        }
        if log_observables:
            for m in problem.measurements:
                if m.measurement <= 0 and m.observable_id in log_observables:
                    return ValidationError(
                        "Measurements with observable "
                        f"log transformation must be "
                        f"positive, but {m.measurement} <= 0 for {m}"
                    )

        return None


class CheckMeasuredExperimentsDefined(ValidationTask):
    """A task to check that all experiments referenced by measurements
    are defined."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        # TODO: introduce some option for validation of partial vs full
        #  problem. if this is supposed to be a complete problem, a missing
        #  condition table should be an error if the measurement table refers
        #  to conditions, otherwise it should maximally be a warning
        used_experiments = {
            m.experiment_id
            for m in problem.measurements
            if m.experiment_id is not None
        }

        # check that measured experiments exist
        available_experiments = {e.id for e in problem.experiments}
        if missing_experiments := (used_experiments - available_experiments):
            return ValidationError(
                "Measurement table references experiments that "
                "are not specified in the experiments table: "
                + str(missing_experiments)
            )

        return None


class CheckValidConditionTargets(ValidationTask):
    """Check that all condition table targets are valid."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        allowed_targets = (
            set(problem.model.get_valid_ids_for_condition_table())
            if problem.model
            else set()
        )
        allowed_targets |= set(problem.get_output_parameters())
        allowed_targets |= {
            m.petab_id for m in problem.mappings if m.model_id is not None
        }

        used_targets = {
            change.target_id
            for cond in problem.conditions
            for change in cond.changes
        }

        if invalid := (used_targets - allowed_targets):
            return ValidationError(
                f"Condition table contains invalid targets: {invalid}"
            )

        # Check that changes of simultaneously applied conditions don't
        #  intersect
        for experiment in problem.experiments:
            for period in experiment.periods:
                if not period.condition_ids:
                    continue
                period_targets = set()
                for condition_id in period.condition_ids:
                    condition_targets = {
                        change.target_id
                        for cond in problem.conditions
                        if cond.id == condition_id
                        for change in cond.changes
                    }
                    if invalid := (period_targets & condition_targets):
                        return ValidationError(
                            "Simultaneously applied conditions for experiment "
                            f"{experiment.id} have overlapping targets "
                            f"{invalid} at time {period.time}."
                        )
                    period_targets |= condition_targets
        return None


class CheckUniquePrimaryKeys(ValidationTask):
    """Check that all primary keys are unique."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        # TODO: check that IDs are globally unique
        #  -- replaces CheckObservablesDoNotShadowModelEntities

        # check for uniqueness of all primary keys
        counter = Counter(c.id for c in problem.conditions)
        duplicates = {id_ for id_, count in counter.items() if count > 1}

        if duplicates:
            return ValidationError(
                f"Condition table contains duplicate IDs: {duplicates}"
            )

        counter = Counter(o.id for o in problem.observables)
        duplicates = {id_ for id_, count in counter.items() if count > 1}

        if duplicates:
            return ValidationError(
                f"Observable table contains duplicate IDs: {duplicates}"
            )

        counter = Counter(e.id for e in problem.experiments)
        duplicates = {id_ for id_, count in counter.items() if count > 1}

        if duplicates:
            return ValidationError(
                f"Experiment table contains duplicate IDs: {duplicates}"
            )

        counter = Counter(p.id for p in problem.parameters)
        duplicates = {id_ for id_, count in counter.items() if count > 1}

        if duplicates:
            return ValidationError(
                f"Parameter table contains duplicate IDs: {duplicates}"
            )

        return None


class CheckObservablesDoNotShadowModelEntities(ValidationTask):
    """A task to check that observable IDs do not shadow model entities."""

    # TODO: all PEtab entity IDs must be disjoint from the model entity IDs
    def run(self, problem: Problem) -> ValidationIssue | None:
        if not problem.observables or problem.model is None:
            return None

        shadowed_entities = [
            o.id
            for o in problem.observables
            if problem.model.has_entity_with_id(o.id)
        ]
        if shadowed_entities:
            return ValidationError(
                f"Observable IDs {shadowed_entities} shadow model entities."
            )

        return None


class CheckExperimentTable(ValidationTask):
    """A task to validate the experiment table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        messages = []
        for experiment in problem.experiments:
            # Check that there are no duplicate timepoints
            counter = Counter(period.time for period in experiment.periods)
            duplicates = {time for time, count in counter.items() if count > 1}
            if duplicates:
                messages.append(
                    f"Experiment {experiment.id} contains duplicate "
                    f"timepoints: {duplicates}"
                )

        if messages:
            return ValidationError("\n".join(messages))

        return None


class CheckExperimentConditionsExist(ValidationTask):
    """A task to validate that all conditions in the experiment table exist
    in the condition table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        messages = []
        available_conditions = {c.id for c in problem.conditions}
        for experiment in problem.experiments:
            missing_conditions = (
                set(
                    chain.from_iterable(
                        period.condition_ids for period in experiment.periods
                    )
                )
                - available_conditions
            )
            if missing_conditions:
                messages.append(
                    f"Experiment {experiment.id} requires conditions that are "
                    f"not present in the condition table: {missing_conditions}"
                )

        if messages:
            return ValidationError("\n".join(messages))

        return None


class CheckAllParametersPresentInParameterTable(ValidationTask):
    """Ensure all required parameters are contained in the parameter table
    with no additional ones."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.model is None:
            return None

        required = get_required_parameters_for_parameter_table(problem)
        allowed = get_valid_parameters_for_parameter_table(problem)

        actual = {p.id for p in problem.parameters}
        missing = required - actual
        extraneous = actual - allowed

        # missing parameters might be present under a different name based on
        # the mapping table
        if missing:
            model_to_petab_mapping = {}
            for m in problem.mappings:
                if m.model_id in model_to_petab_mapping:
                    model_to_petab_mapping[m.model_id].append(m.petab_id)
                else:
                    model_to_petab_mapping[m.model_id] = [m.petab_id]
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

        return None


class CheckValidParameterInConditionOrParameterTable(ValidationTask):
    """A task to check that all required and only allowed model parameters are
    present in the condition or parameter table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.model is None:
            return None

        allowed_in_condition_cols = set(
            problem.model.get_valid_ids_for_condition_table()
        )
        allowed_in_condition_cols |= {
            m.petab_id
            for m in problem.mappings
            if not pd.isna(m.model_id)
            and (
                # mapping table entities mapping to already allowed parameters
                m.model_id in allowed_in_condition_cols
                # mapping table entities mapping to species
                or problem.model.is_state_variable(m.model_id)
            )
        }

        allowed_in_parameter_table = get_valid_parameters_for_parameter_table(
            problem
        )

        entities_in_condition_table = {
            change.target_id
            for cond in problem.conditions
            for change in cond.changes
        }
        entities_in_parameter_table = {p.id for p in problem.parameters}

        disallowed_in_condition = {
            x
            for x in (entities_in_condition_table - allowed_in_condition_cols)
            # we only check model entities here, not output parameters
            if problem.model.has_entity_with_id(x)
        }
        if disallowed_in_condition:
            is_or_are = "is" if len(disallowed_in_condition) == 1 else "are"
            return ValidationError(
                f"{disallowed_in_condition} {is_or_are} not "
                "allowed to occur in condition table "
                "columns."
            )

        disallowed_in_parameters = {
            x
            for x in (entities_in_parameter_table - allowed_in_parameter_table)
            # we only check model entities here, not output parameters
            if problem.model.has_entity_with_id(x)
        }

        if disallowed_in_parameters:
            is_or_are = "is" if len(disallowed_in_parameters) == 1 else "are"
            return ValidationError(
                f"{disallowed_in_parameters} {is_or_are} not "
                "allowed to occur in the parameters table."
            )

        in_both = entities_in_condition_table & entities_in_parameter_table
        if in_both:
            is_or_are = "is" if len(in_both) == 1 else "are"
            return ValidationError(
                f"{in_both} {is_or_are} present in both "
                "the condition table and the parameter table."
            )

        return None


class CheckUnusedExperiments(ValidationTask):
    """A task to check for experiments that are not used in the measurement
    table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        used_experiments = {
            m.experiment_id
            for m in problem.measurements
            if m.experiment_id is not None
        }
        available_experiments = {e.id for e in problem.experiments}

        unused_experiments = available_experiments - used_experiments
        if unused_experiments:
            return ValidationWarning(
                f"Experiments {unused_experiments} are not used in the "
                "measurements table."
            )

        return None


class CheckUndefinedExperiments(ValidationTask):
    """A task to check for experiments that are used in the measurement
    table but not defined in the experiment table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        used_experiments = {
            m.experiment_id
            for m in problem.measurements
            if m.experiment_id is not None
        }
        available_experiments = {e.id for e in problem.experiments}

        if undefined_experiments := used_experiments - available_experiments:
            return ValidationWarning(
                f"Experiments {undefined_experiments} are used in the "
                "measurements table but are not defined in the experiments "
                "table."
            )

        return None


class CheckUnusedConditions(ValidationTask):
    """A task to check for conditions that are not used in the experiment
    table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        used_conditions = set(
            chain.from_iterable(
                p.condition_ids for e in problem.experiments for p in e.periods
            )
        )
        available_conditions = {c.id for c in problem.conditions}

        unused_conditions = available_conditions - used_conditions
        if unused_conditions:
            return ValidationWarning(
                f"Conditions {unused_conditions} are not used in the "
                "experiments table."
            )

        return None


class CheckInitialChangeSymbols(ValidationTask):
    """
    Check that changes of any first period of any experiment only refers to
    allowed symbols.

    The only allowed symbols are those that are present in the parameter table.
    """

    def run(self, problem: Problem) -> ValidationIssue | None:
        if not problem.experiments:
            return None

        if not problem.conditions:
            return None

        allowed_symbols = {p.id for p in problem.parameters}
        allowed_symbols.add(TIME_SYMBOL)
        # IDs of conditions that have already been checked
        valid_conditions = set()
        id_to_condition = {c.id: c for c in problem.conditions}

        messages = []
        for experiment in problem.experiments:
            if not experiment.periods:
                continue

            first_period = experiment.sorted_periods[0]
            for condition_id in first_period.condition_ids:
                if condition_id in valid_conditions:
                    continue

                try:
                    condition = id_to_condition[condition_id]
                except KeyError:
                    messages.append(
                        f"Unable to validate changes for condition "
                        f"{condition_id} applied at the start of "
                        f"experiment {experiment.id}, as the condition "
                        "does not exist."
                    )

                used_symbols = {
                    str(sym)
                    for change in condition.changes
                    for sym in change.target_value.free_symbols
                }
                invalid_symbols = used_symbols - allowed_symbols
                if invalid_symbols:
                    messages.append(
                        f"Condition {condition.id} is applied at the start of "
                        f"experiment {experiment.id}, and thus, its "
                        f"target value expressions must only contain "
                        f"symbols from the parameter table, or `time`. "
                        "However, it contains additional symbols: "
                        f"{invalid_symbols}. "
                    )

        if messages:
            return ValidationError("\n".join(messages))

        return None


class CheckPriorDistribution(ValidationTask):
    """A task to validate the prior distribution of a PEtab problem."""

    _num_pars = {
        PriorDistribution.CAUCHY: 2,
        PriorDistribution.CHI_SQUARED: 1,
        PriorDistribution.EXPONENTIAL: 1,
        PriorDistribution.GAMMA: 2,
        PriorDistribution.LAPLACE: 2,
        PriorDistribution.LOG_LAPLACE: 2,
        PriorDistribution.LOG_NORMAL: 2,
        PriorDistribution.LOG_UNIFORM: 2,
        PriorDistribution.NORMAL: 2,
        PriorDistribution.RAYLEIGH: 1,
        PriorDistribution.UNIFORM: 2,
    }

    def run(self, problem: Problem) -> ValidationIssue | None:
        messages = []
        for parameter in problem.parameters:
            if parameter.prior_distribution is None:
                continue

            if parameter.prior_distribution not in PRIOR_DISTRIBUTIONS:
                messages.append(
                    f"Prior distribution `{parameter.prior_distribution}' "
                    f"for parameter `{parameter.id}' is not valid."
                )
                continue

            if (
                exp_num_par := self._num_pars[parameter.prior_distribution]
            ) != len(parameter.prior_parameters):
                messages.append(
                    f"Prior distribution `{parameter.prior_distribution}' "
                    f"for parameter `{parameter.id}' requires "
                    f"{exp_num_par} parameters, but got "
                    f"{len(parameter.prior_parameters)} "
                    f"({parameter.prior_parameters})."
                )

            # TODO: check distribution parameter domains more specifically
            try:
                if parameter.estimate and parameter.prior_dist is not None:
                    # .prior_dist fails for non-estimated parameters
                    _ = parameter.prior_dist.sample(1)
            except Exception as e:
                messages.append(
                    f"Prior parameters `{parameter.prior_parameters}' "
                    f"for parameter `{parameter.id}' are invalid "
                    f"(hint: {e})."
                )

        if messages:
            return ValidationError("\n".join(messages))

        return None


class CheckMeasurementModelId(ValidationTask):
    """Validate model IDs of measurements."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        messages = []
        available_models = {m.model_id for m in problem.models}

        for measurement in problem.measurements:
            if not measurement.model_id:
                if len(available_models) < 2:
                    # If there is only one model, it is not required to specify
                    # the model ID in the measurement table.
                    continue

                messages.append(
                    f"Measurement `{measurement}' does not have a model ID, "
                    "but there are multiple models available. "
                    "Please specify the model ID in the measurement table."
                )
                continue

            if measurement.model_id not in available_models:
                messages.append(
                    f"Measurement `{measurement}' has model ID "
                    f"`{measurement.model_id}' which does not match "
                    "any of the available models: "
                    f"{available_models}."
                )

        if messages:
            return ValidationError("\n".join(messages))

        return None


def get_valid_parameters_for_parameter_table(
    problem: Problem,
) -> set[str]:
    """
    Get the set of parameters which may be present inside the parameter table

    :param problem: The PEtab problem

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

    # must not go into parameter table
    invalid = set(get_placeholders(problem))

    # condition table targets
    invalid |= {
        change.target_id
        for cond in problem.conditions
        for change in cond.changes
    }

    # don't use sets here, to have deterministic ordering,
    #  e.g., for creating parameter tables
    parameter_ids = OrderedDict.fromkeys(
        p
        for p in problem.model.get_valid_parameters_for_parameter_table()
        if p not in invalid
    )

    for mapping in problem.mappings:
        if mapping.model_id and mapping.model_id in parameter_ids.keys():
            parameter_ids[mapping.petab_id] = None

    # add output parameters from observable table
    output_parameters = problem.get_output_parameters()
    for p in output_parameters:
        if p not in invalid:
            parameter_ids[p] = None

    # Append parameters from measurement table, unless they occur as condition
    # table columns
    def append_overrides(overrides):
        for p in overrides:
            if isinstance(p, sp.Symbol) and (str_p := str(p)) not in invalid:
                parameter_ids[str_p] = None

    for measurement in problem.measurements:
        # we trust that the number of overrides matches
        append_overrides(measurement.observable_parameters)
        append_overrides(measurement.noise_parameters)

    # Append parameter overrides from condition table
    for ct in problem.condition_tables:
        for p in ct.free_symbols:
            parameter_ids[str(p)] = None

    return set(parameter_ids.keys())


def get_required_parameters_for_parameter_table(
    problem: Problem,
) -> Set[str]:
    """
    Get the set of parameters that need to go into the parameter table

    Arguments:
        problem: The PEtab problem
    Returns:
        Set of parameter IDs which PEtab requires to be present in the
        parameter table. That is all {observable,noise}Parameters from the
        measurement table as well as all parametric condition table overrides
        that are not defined in the model.
    """
    parameter_ids = set()
    condition_targets = {
        change.target_id
        for cond in problem.conditions
        for change in cond.changes
    }

    # Add parameters from measurement table, unless they are fixed parameters
    def append_overrides(overrides):
        parameter_ids.update(
            str_p
            for p in overrides
            if isinstance(p, sp.Symbol)
            and (str_p := str(p)) not in condition_targets
        )

    for m in problem.measurements:
        # we trust that the number of overrides matches
        append_overrides(m.observable_parameters)
        append_overrides(m.noise_parameters)

    # Add output parameters except for placeholders
    for formula_type, placeholder_sources in (
        (
            # Observable formulae
            {"observable": True, "noise": False},
            # can only contain observable placeholders
            {"noise": False, "observable": True},
        ),
        (
            # Noise formulae
            {"observable": False, "noise": True},
            # can contain noise and observable placeholders
            {"noise": True, "observable": True},
        ),
    ):
        output_parameters = problem.get_output_parameters(
            **formula_type,
        )
        placeholders = get_placeholders(
            problem,
            **placeholder_sources,
        )
        parameter_ids.update(
            p for p in output_parameters if p not in placeholders
        )

    # Add condition table parametric overrides unless already defined in the
    #  model
    parameter_ids.update(
        str(p)
        for ct in problem.condition_tables
        for p in ct.free_symbols
        if not problem.model.has_entity_with_id(str(p))
    )

    # parameters that are overridden via the condition table are not allowed
    parameter_ids -= condition_targets

    return parameter_ids


def get_placeholders(
    problem: Problem,
    observable: bool = True,
    noise: bool = True,
) -> list[str]:
    """Get all placeholder parameters from observable table observableFormulas
    and noiseFormulas.

    Arguments:
        problem: The PEtab problem
        observable: Include parameters from observableFormulas
        noise: Include parameters from noiseFormulas

    Returns:
        List of placeholder parameters from observable table observableFormulas
        and noiseFormulas.
    """
    # collect placeholder parameters overwritten by
    # {observable,noise}Parameters
    placeholders = []
    for o in problem.observables:
        if observable:
            placeholders.extend(map(str, o.observable_placeholders))
        if noise:
            placeholders.extend(map(str, o.noise_placeholders))

    from ..v1.core import unique_preserve_order

    return unique_preserve_order(placeholders)


#: Validation tasks that should be run on any PEtab problem
default_validation_tasks = [
    CheckProblemConfig(),
    CheckModel(),
    CheckUniquePrimaryKeys(),
    CheckMeasurementModelId(),
    CheckMeasuredObservablesDefined(),
    CheckPosLogMeasurements(),
    CheckOverridesMatchPlaceholders(),
    CheckValidConditionTargets(),
    CheckExperimentTable(),
    CheckExperimentConditionsExist(),
    CheckUndefinedExperiments(),
    CheckObservablesDoNotShadowModelEntities(),
    CheckAllParametersPresentInParameterTable(),
    CheckValidParameterInConditionOrParameterTable(),
    CheckUnusedExperiments(),
    CheckUnusedConditions(),
    CheckPriorDistribution(),
    CheckInitialChangeSymbols(),
    # TODO validate mapping table
]
