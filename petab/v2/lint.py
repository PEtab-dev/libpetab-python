"""Validation of PEtab problems"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from collections.abc import Set
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import pandas as pd
import sympy as sp

from .. import v2
from .problem import Problem

logger = logging.getLogger(__name__)

__all__ = [
    "ValidationIssueSeverity",
    "ValidationIssue",
    "ValidationResultList",
    "ValidationError",
    "ValidationTask",
    "CheckModel",
    "CheckProblemConfig",
    "CheckPosLogMeasurements",
    "CheckValidConditionTargets",
    "CheckUniquePrimaryKeys",
    "CheckExperimentTable",
    "CheckExperimentConditionsExist",
    "CheckAllParametersPresentInParameterTable",
    "CheckValidParameterInConditionOrParameterTable",
    "CheckVisualizationTable",
    "CheckUnusedExperiments",
    "CheckObservablesDoNotShadowModelEntities",
    "CheckUnusedConditions",
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
        if not config.parameter_file:
            missing_files.append("parameters")

        if not [p.measurement_files for p in config.problems]:
            missing_files.append("measurements")

        if not [p.observable_files for p in config.problems]:
            missing_files.append("observables")

        if missing_files:
            return ValidationError(
                f"Missing files: {', '.join(missing_files)}"
            )


class CheckModel(ValidationTask):
    """A task to validate the model of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.model is None:
            return ValidationError("Model is missing.")

        if not problem.model.is_valid():
            # TODO get actual model validation messages
            return ValidationError("Model is invalid.")


class CheckMeasuredObservablesDefined(ValidationTask):
    """A task to check that all observables referenced by the measurements
    are defined."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        used_observables = {
            m.observable_id for m in problem.measurement_table.measurements
        }
        defined_observables = {
            o.id for o in problem.observables_table.observables
        }
        if undefined_observables := (used_observables - defined_observables):
            return ValidationError(
                f"Observables {undefined_observables} used in "
                "measurement table but not defined in observables table."
            )


class CheckOverridesMatchPlaceholders(ValidationTask):
    """A task to check that the number of observable/noise parameters
    in the measurements match the number of placeholders in the observables."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        observable_parameters_count = {
            o.id: len(o.observable_placeholders)
            for o in problem.observables_table.observables
        }
        noise_parameters_count = {
            o.id: len(o.noise_placeholders)
            for o in problem.observables_table.observables
        }
        messages = []
        for m in problem.measurement_table.measurements:
            # check observable parameters
            try:
                expected = observable_parameters_count[m.observable_id]
            except KeyError:
                messages.append(
                    f"Observable {m.observable_id} used in measurement "
                    f"table is not defined."
                )
                continue

            actual = len(m.observable_parameters)

            if actual != expected:
                formula = problem.observables_table[m.observable_id].formula
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
                    formula = problem.observables_table[
                        m.observable_id
                    ].noise_formula
                    messages.append(
                        f"Mismatch of noise parameter overrides for "
                        f"{m.observable_id} ({formula})"
                        f"in:\n{m}\n"
                        f"Expected {expected} but got {actual}"
                    )

        if messages:
            return ValidationError("\n".join(messages))


class CheckPosLogMeasurements(ValidationTask):
    """Check that measurements for observables with
    log-transformation are positive."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        from .core import ObservableTransformation as ot

        log_observables = {
            o.id
            for o in problem.observables_table.observables
            if o.transformation in [ot.LOG, ot.LOG10]
        }
        if log_observables:
            for m in problem.measurement_table.measurements:
                if m.measurement <= 0 and m.observable_id in log_observables:
                    return ValidationError(
                        "Measurements with observable "
                        f"log transformation must be "
                        f"positive, but {m.measurement} <= 0 for {m}"
                    )


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
            for m in problem.measurement_table.measurements
            if m.experiment_id is not None
        }

        # check that measured experiments exist
        available_experiments = {
            e.id for e in problem.experiments_table.experiments
        }
        if missing_experiments := (used_experiments - available_experiments):
            return ValidationError(
                "Measurement table references experiments that "
                "are not specified in the experiments table: "
                + str(missing_experiments)
            )


class CheckValidConditionTargets(ValidationTask):
    """Check that all condition table targets are valid."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        allowed_targets = set(
            problem.model.get_valid_ids_for_condition_table()
        )
        allowed_targets |= set(get_output_parameters(problem))
        allowed_targets |= {
            m.petab_id
            for m in problem.mapping_table.mappings
            if m.model_id is not None
        }

        used_targets = {
            change.target_id
            for cond in problem.conditions_table.conditions
            for change in cond.changes
        }

        if invalid := (used_targets - allowed_targets):
            return ValidationError(
                f"Condition table contains invalid targets: {invalid}"
            )


class CheckUniquePrimaryKeys(ValidationTask):
    """Check that all primary keys are unique."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        # TODO: check that IDs are globally unique
        #  -- replaces CheckObservablesDoNotShadowModelEntities

        # check for uniqueness of all primary keys
        counter = Counter(c.id for c in problem.conditions_table.conditions)
        duplicates = {id for id, count in counter.items() if count > 1}

        if duplicates:
            return ValidationError(
                f"Condition table contains duplicate IDs: {duplicates}"
            )

        counter = Counter(o.id for o in problem.observables_table.observables)
        duplicates = {id for id, count in counter.items() if count > 1}

        if duplicates:
            return ValidationError(
                f"Observable table contains duplicate IDs: {duplicates}"
            )

        counter = Counter(e.id for e in problem.experiments_table.experiments)
        duplicates = {id for id, count in counter.items() if count > 1}

        if duplicates:
            return ValidationError(
                f"Experiment table contains duplicate IDs: {duplicates}"
            )

        counter = Counter(p.id for p in problem.parameter_table.parameters)
        duplicates = {id for id, count in counter.items() if count > 1}

        if duplicates:
            return ValidationError(
                f"Parameter table contains duplicate IDs: {duplicates}"
            )


class CheckObservablesDoNotShadowModelEntities(ValidationTask):
    """A task to check that observable IDs do not shadow model entities."""

    # TODO: all PEtab entity IDs must be disjoint from the model entity IDs
    def run(self, problem: Problem) -> ValidationIssue | None:
        if not problem.observables_table.observables or problem.model is None:
            return None

        shadowed_entities = [
            o.id
            for o in problem.observables_table.observables
            if problem.model.has_entity_with_id(o.id)
        ]
        if shadowed_entities:
            return ValidationError(
                f"Observable IDs {shadowed_entities} shadow model entities."
            )


class CheckExperimentTable(ValidationTask):
    """A task to validate the experiment table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        messages = []
        for experiment in problem.experiments_table.experiments:
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


class CheckExperimentConditionsExist(ValidationTask):
    """A task to validate that all conditions in the experiment table exist
    in the condition table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        messages = []
        available_conditions = {
            c.id for c in problem.conditions_table.conditions
        }
        for experiment in problem.experiments_table.experiments:
            missing_conditions = {
                period.condition_id
                for period in experiment.periods
                if period.condition_id is not None
            } - available_conditions
            if missing_conditions:
                messages.append(
                    f"Experiment {experiment.id} requires conditions that are "
                    f"not present in the condition table: {missing_conditions}"
                )

        if messages:
            return ValidationError("\n".join(messages))


class CheckAllParametersPresentInParameterTable(ValidationTask):
    """Ensure all required parameters are contained in the parameter table
    with no additional ones."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.model is None:
            return None

        required = get_required_parameters_for_parameter_table(problem)
        allowed = get_valid_parameters_for_parameter_table(problem)

        actual = {p.id for p in problem.parameter_table.parameters}
        missing = required - actual
        extraneous = actual - allowed

        # missing parameters might be present under a different name based on
        # the mapping table
        if missing:
            model_to_petab_mapping = {}
            for m in problem.mapping_table.mappings:
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
            for m in problem.mapping_table.mappings
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
            for cond in problem.conditions_table.conditions
            for change in cond.changes
        }
        entities_in_parameter_table = {
            p.id for p in problem.parameter_table.parameters
        }

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


class CheckUnusedExperiments(ValidationTask):
    """A task to check for experiments that are not used in the measurements
    table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        used_experiments = {
            m.experiment_id
            for m in problem.measurement_table.measurements
            if m.experiment_id is not None
        }
        available_experiments = {
            e.id for e in problem.experiments_table.experiments
        }

        unused_experiments = available_experiments - used_experiments
        if unused_experiments:
            return ValidationWarning(
                f"Experiments {unused_experiments} are not used in the "
                "measurements table."
            )


class CheckUnusedConditions(ValidationTask):
    """A task to check for conditions that are not used in the experiments
    table."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        used_conditions = {
            p.condition_id
            for e in problem.experiments_table.experiments
            for p in e.periods
            if p.condition_id is not None
        }
        available_conditions = {
            c.id for c in problem.conditions_table.conditions
        }

        unused_conditions = available_conditions - used_conditions
        if unused_conditions:
            return ValidationWarning(
                f"Conditions {unused_conditions} are not used in the "
                "experiments table."
            )


class CheckVisualizationTable(ValidationTask):
    """A task to validate the visualization table of a PEtab problem."""

    def run(self, problem: Problem) -> ValidationIssue | None:
        if problem.visualization_df is None:
            return None

        from ..v1.visualize.lint import validate_visualization_df

        if validate_visualization_df(problem):
            return ValidationIssue(
                level=ValidationIssueSeverity.ERROR,
                message="Visualization table is invalid.",
            )


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
        for cond in problem.conditions_table.conditions
        for change in cond.changes
    }

    # don't use sets here, to have deterministic ordering,
    #  e.g., for creating parameter tables
    parameter_ids = OrderedDict.fromkeys(
        p
        for p in problem.model.get_valid_parameters_for_parameter_table()
        if p not in invalid
    )

    for mapping in problem.mapping_table.mappings:
        if mapping.model_id and mapping.model_id in parameter_ids.keys():
            parameter_ids[mapping.petab_id] = None

    # add output parameters from observables table
    output_parameters = get_output_parameters(problem)
    for p in output_parameters:
        if p not in invalid:
            parameter_ids[p] = None

    # Append parameters from measurement table, unless they occur as condition
    # table columns
    def append_overrides(overrides):
        for p in overrides:
            if isinstance(p, sp.Symbol) and (str_p := str(p)) not in invalid:
                parameter_ids[str_p] = None

    for measurement in problem.measurement_table.measurements:
        # we trust that the number of overrides matches
        append_overrides(measurement.observable_parameters)
        append_overrides(measurement.noise_parameters)

    # Append parameter overrides from condition table
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
    condition_targets = {
        change.target_id
        for cond in problem.conditions_table.conditions
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

    for m in problem.measurement_table.measurements:
        # we trust that the number of overrides matches
        append_overrides(m.observable_parameters)
        append_overrides(m.noise_parameters)

    # TODO remove `observable_ids` when
    #  `get_output_parameters` is updated for PEtab v2/v1.1, where
    #  observable IDs are allowed in observable formulae
    observable_ids = {o.id for o in problem.observables_table.observables}

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
            problem,
            **formula_type,
        )
        placeholders = get_placeholders(
            problem,
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
    parameter_ids -= condition_targets

    return parameter_ids


def get_output_parameters(
    problem: Problem,
    observables: bool = True,
    noise: bool = True,
) -> list[str]:
    """Get output parameters

    Returns IDs of parameters used in observable and noise formulas that are
    not defined in the model.

    Arguments:
        problem: The PEtab problem
        observables: Include parameters from observableFormulas
        noise: Include parameters from noiseFormulas

    Returns:
        List of output parameter IDs
    """
    formulas = []
    if observables:
        formulas.extend(
            o.formula for o in problem.observables_table.observables
        )
    if noise:
        formulas.extend(
            o.noise_formula for o in problem.observables_table.observables
        )
    output_parameters = OrderedDict()

    for formula in formulas:
        free_syms = sorted(
            formula.free_symbols,
            key=lambda symbol: symbol.name,
        )
        for free_sym in free_syms:
            sym = str(free_sym)
            if problem.model.symbol_allowed_in_observable_formula(sym):
                continue

            # does it map to a model entity?

            if (
                (mapped := problem.mapping_table.get(sym)) is not None
                and mapped.model_id is not None
                and problem.model.symbol_allowed_in_observable_formula(
                    mapped.model_id
                )
            ):
                continue

            output_parameters[sym] = None

    return list(output_parameters.keys())


def get_placeholders(
    problem: Problem,
    observables: bool = True,
    noise: bool = True,
) -> list[str]:
    """Get all placeholder parameters from observable table observableFormulas
    and noiseFormulas.

    Arguments:
        problem: The PEtab problem
        observables: Include parameters from observableFormulas
        noise: Include parameters from noiseFormulas

    Returns:
        List of placeholder parameters from observable table observableFormulas
        and noiseFormulas.
    """
    # collect placeholder parameters overwritten by
    # {observable,noise}Parameters
    placeholders = []
    for o in problem.observables_table.observables:
        if observables:
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
    CheckMeasuredObservablesDefined(),
    CheckPosLogMeasurements(),
    CheckOverridesMatchPlaceholders(),
    CheckValidConditionTargets(),
    CheckExperimentTable(),
    CheckExperimentConditionsExist(),
    CheckObservablesDoNotShadowModelEntities(),
    CheckAllParametersPresentInParameterTable(),
    CheckValidParameterInConditionOrParameterTable(),
    CheckUnusedExperiments(),
    CheckUnusedConditions(),
    # TODO: atomize checks, update to long condition table, re-enable
    # CheckVisualizationTable(),
    # TODO validate mapping table
]
