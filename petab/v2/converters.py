"""Conversion of PEtab problems."""

from __future__ import annotations

import warnings
from copy import deepcopy

import libsbml
from sbmlmath import sbml_math_to_sympy, set_math

from .core import (
    Change,
    Condition,
    ConditionTable,
    Experiment,
    ExperimentPeriod,
    Problem,
)
from .models._sbml_utils import add_sbml_parameter, check
from .models.sbml_model import SbmlModel

__all__ = ["ExperimentsToEventsConverter"]


class ExperimentsToEventsConverter:
    """Convert PEtab experiments to SBML events.

    For an SBML-model-based PEtab problem, this class converts the PEtab
    experiments to events as far as possible.

    If the model already contains events, PEtab events are added with a higher
    priority than the existing events to guarantee that PEtab condition changes
    are applied before any pre-existing assignments.

    The PEtab problem must not contain any identifiers starting with
    ``_petab``.

    All periods and condition changes that are represented by events
    will be removed from the condition table.
    Each experiment will have at most one period with a start time of ``-inf``
    and one period with a finite start time. The associated changes with
    these periods are only the pre-equilibration indicator
    (if necessary), and the experiment indicator parameter.
    """

    #: ID of the parameter that indicates whether the model is in
    #  the pre-equilibration phase (1) or not (0).
    PREEQ_INDICATOR = "_petab_preequilibration_indicator"

    #: The condition ID of the condition that sets the
    #: pre-equilibration indicator to 1.
    CONDITION_ID_PREEQ_ON = "_petab_preequilibration_on"

    #: The condition ID of the condition that sets the
    #: pre-equilibration indicator to 0.
    CONDITION_ID_PREEQ_OFF = "_petab_preequilibration_off"

    def __init__(self, problem: Problem, default_priority: float = None):
        """Initialize the converter.

        :param problem: The PEtab problem to convert.
            This will not be modified.
        :param default_priority: The priority value to apply to any events that
            preexist in the model and do not have a priority set.

            In SBML, for event assignments that are to be applied at the same
            simulation time, the order of event execution is determined by the
            priority of the respective events.
            If no priority is set, the order is undefined.
            See SBML specs for details.
            To ensure that the PEtab condition-start-events are executed before
            any other events, all events should have a priority set.
        """
        if len(problem.models) > 1:
            #  https://github.com/PEtab-dev/libpetab-python/issues/392
            raise NotImplementedError(
                "Only single-model PEtab problems are supported."
            )
        if not isinstance(problem.model, SbmlModel):
            raise ValueError("Only SBML models are supported.")

        self._original_problem = problem
        self._new_problem = deepcopy(self._original_problem)

        self._model: libsbml.Model = self._new_problem.model.sbml_model
        self._preeq_indicator = self.PREEQ_INDICATOR

        # The maximum event priority that was found in the unprocessed model.
        self._max_event_priority = None
        # The priority that will be used for the PEtab events.
        self._petab_event_priority = None
        self._default_priority = default_priority
        self._preprocess()

    def _get_experiment_indicator_condition_id(
        self, experiment_id: str
    ) -> str:
        """Get the condition ID for the experiment indicator parameter."""
        return f"_petab_experiment_condition_{experiment_id}"

    def _preprocess(self) -> None:
        """Check whether we can handle the given problem and store some model
        information."""
        model = self._model
        if model.getLevel() < 3:
            # try to upgrade the SBML model
            if not model.getSBMLDocument().setLevelAndVersion(3, 2):
                raise ValueError(
                    "Cannot handle SBML models with SBML level < 3, "
                    "because they do not support initial values for event "
                    "triggers and automatic upconversion of the model failed."
                )

        # Apply default priority to all events that do not have a priority
        if self._default_priority is not None:
            for event in model.getListOfEvents():
                if (
                    not event.getPriority()
                    or event.getPriority().getMath() is None
                ):
                    priority = event.createPriority()
                    priority.setMath(
                        libsbml.parseL3Formula(str(self._default_priority))
                    )

        # Collect event priorities
        event_priorities = {
            ev.getId() or str(ev): sbml_math_to_sympy(ev.getPriority())
            for ev in model.getListOfEvents()
            if ev.getPriority() and ev.getPriority().getMath() is not None
        }

        # Check for non-constant event priorities and track the maximum
        #  priority used so far.
        for e, priority in event_priorities.items():
            if priority.free_symbols:
                # We'd need to find the maximum priority of all events,
                #  which is challenging/impossible to do in general.
                raise NotImplementedError(
                    f"Event `{e}` has a non-constant priority: {priority}. "
                    "This is currently not supported."
                )
            self._max_event_priority = max(
                self._max_event_priority or 0, float(priority)
            )

        self._petab_event_priority = (
            self._max_event_priority + 1
            if self._max_event_priority is not None
            else None
        )

        for event in model.getListOfEvents():
            # Check for undefined event priorities and warn
            if (prio := event.getPriority()) and prio.getMath() is None:
                warnings.warn(
                    f"Event `{event.getId()}` has no priority set. "
                    "Make sure that this event cannot trigger at the time of "
                    "a PEtab condition change, otherwise the behavior is "
                    "undefined. To avoid this warning, see the "
                    "`default_priority` parameter of "
                    f"{self.__class__.__name__}.",
                    stacklevel=1,
                )

            # Check for useValuesFromTrigger time
            if event.getUseValuesFromTriggerTime():
                # Non-PEtab-condition-change events must be executed *after*
                #  PEtab condition changes have been applied, based on the
                #  updated model state. This would be violated by
                #  useValuesFromTriggerTime=true.
                warnings.warn(
                    f"Event `{event.getId()}` has "
                    "`useValuesFromTriggerTime=true'. "
                    "Make sure that this event cannot trigger at the time of "
                    "a PEtab condition change, or consider changing "
                    "`useValuesFromTriggerTime' to `false'. Otherwise "
                    "simulation results may be incorrect.",
                    stacklevel=1,
                )

    def convert(self) -> Problem:
        """Convert the PEtab experiments to SBML events.

        :return: The converted PEtab problem.
        """

        self._add_preequilibration_indicator()

        for experiment in self._new_problem.experiments:
            self._convert_experiment(experiment)

        self._add_indicators_to_conditions()

        validation_results = self._new_problem.validate()
        validation_results.log()

        return self._new_problem

    def _convert_experiment(self, experiment: Experiment) -> None:
        """Convert a single experiment to SBML events."""
        model = self._model
        experiment.sort_periods()
        has_preequilibration = experiment.has_preequilibration

        # add experiment indicator
        exp_ind_id = self.get_experiment_indicator(experiment.id)
        if model.getElementBySId(exp_ind_id) is not None:
            raise ValueError(
                f"The model has entity with ID `{exp_ind_id}`. "
                "IDs starting with `petab_` are reserved for "
                f"{self.__class__.__name__} and should not be used in the "
                "model."
            )
        add_sbml_parameter(model, id_=exp_ind_id, constant=False, value=0)
        kept_periods = []
        for i_period, period in enumerate(experiment.periods):
            if period.is_preequilibration:
                # pre-equilibration cannot be represented in SBML,
                #  so we need to keep this period in the Problem.
                kept_periods.append(period)
            elif i_period == int(has_preequilibration):
                # we always keep the first non-pre-equilibration period
                #  to set the indicator parameters
                kept_periods.append(period)
            elif not period.condition_ids:
                # no condition, no changes, no need for an event,
                #  no need to keep the period unless it's the pre-equilibration
                #  or the only non-equilibration period (handled above)
                continue

            ev = self._create_period_start_event(
                experiment=experiment,
                i_period=i_period,
                period=period,
            )
            self._create_event_assignments_for_period(
                ev,
                [
                    self._new_problem[condition_id]
                    for condition_id in period.condition_ids
                ],
            )

        if len(kept_periods) > 2:
            raise AssertionError("Expected at most two periods to be kept.")

        # add conditions that set the indicator parameters
        for period in kept_periods:
            period.condition_ids = [
                self._get_experiment_indicator_condition_id(experiment.id),
                self.CONDITION_ID_PREEQ_ON
                if period.is_preequilibration
                else self.CONDITION_ID_PREEQ_OFF,
            ]

        experiment.periods = kept_periods

    def _create_period_start_event(
        self, experiment: Experiment, i_period: int, period: ExperimentPeriod
    ) -> libsbml.Event:
        """Create an event that triggers at the start of a period."""

        # TODO: for now, add separate events for each experiment x period,
        #  this could be optimized to reuse events

        ev = self._model.createEvent()
        check(ev.setId(f"_petab_event_{experiment.id}_{i_period}"))
        check(ev.setUseValuesFromTriggerTime(True))
        trigger = ev.createTrigger()
        check(trigger.setInitialValue(False))  # may trigger at t=0
        check(trigger.setPersistent(True))
        if self._petab_event_priority is not None:
            priority = ev.createPriority()
            set_math(priority, self._petab_event_priority)

        exp_ind_id = self.get_experiment_indicator(experiment.id)

        # Create trigger expressions
        # Since handling of == and !=, and distinguishing < and <=
        # (and > and >=), is a bit tricky in terms of root-finding,
        # we use these slightly more convoluted expressions.
        # (assuming that the indicator parameters are {0, 1})
        if period.is_preequilibration:
            trig_math = libsbml.parseL3Formula(
                f"({exp_ind_id} > 0.5) && ({self._preeq_indicator} > 0.5)"
            )
        else:
            trig_math = libsbml.parseL3Formula(
                f"({exp_ind_id} > 0.5) "
                f"&& ({self._preeq_indicator} < 0.5) "
                f"&& (time >= {period.time})"
            )
        check(trigger.setMath(trig_math))

        return ev

    def _add_preequilibration_indicator(
        self,
    ) -> None:
        """Add an indicator parameter for the pre-equilibration to the SBML
        model."""
        par_id = self._preeq_indicator
        if self._model.getElementBySId(par_id) is not None:
            raise ValueError(
                f"Entity with ID {par_id} already exists in the SBML model."
            )

        # add the pre-steady-state indicator parameter
        add_sbml_parameter(self._model, id_=par_id, value=0, constant=False)

    @staticmethod
    def get_experiment_indicator(experiment_id: str) -> str:
        """The ID of the experiment indicator parameter.

        The experiment indicator parameter is used to identify the
        experiment in the SBML model. It is a parameter that is set
        to 1 for the current experiment and 0 for all other
        experiments. The parameter is used in the event trigger
        to determine whether the event should be triggered.

        :param experiment_id: The ID of the experiment for which to create
            the experiment indicator parameter ID.
        """
        return f"_petab_experiment_indicator_{experiment_id}"

    @staticmethod
    def _create_event_assignments_for_period(
        event: libsbml.Event, conditions: list[Condition]
    ) -> None:
        """Create an event assignments for a given period."""
        for condition in conditions:
            for change in condition.changes:
                ExperimentsToEventsConverter._change_to_event_assignment(
                    change, event
                )

    @staticmethod
    def _change_to_event_assignment(
        change: Change, event: libsbml.Event
    ) -> None:
        """Convert a PEtab ``Change``  to an SBML event assignment."""
        sbml_model = event.getModel()

        ea = event.createEventAssignment()
        ea.setVariable(change.target_id)
        set_math(ea, change.target_value)

        # target needs const=False, and target may not exist yet
        #  (e.g., in case of output parameters added in the observable
        #  table)
        target = sbml_model.getElementBySId(change.target_id)
        if target is None:
            add_sbml_parameter(
                sbml_model, id_=change.target_id, constant=False, value=0
            )
        else:
            # We can safely change the `constant` attribute of the target.
            #  "Constant" does not imply "boundary condition" in SBML.
            target.setConstant(False)

        # the target value may depend on parameters that are only
        #  introduced in the PEtab parameter table - those need
        #  to be added to the model
        for sym in change.target_value.free_symbols:
            if sbml_model.getElementBySId(sym.name) is None:
                add_sbml_parameter(
                    sbml_model, id_=sym.name, constant=True, value=0
                )

    def _add_indicators_to_conditions(self) -> None:
        """After converting the experiments to events, add the indicator
        parameters for the pre-equilibration period and for the different
        experiments to the remaining conditions.
        Then remove all other conditions."""
        problem = self._new_problem

        # create conditions for indicator parameters
        problem += Condition(
            id=self.CONDITION_ID_PREEQ_ON,
            changes=[Change(target_id=self._preeq_indicator, target_value=1)],
        )

        problem += Condition(
            id=self.CONDITION_ID_PREEQ_OFF,
            changes=[Change(target_id=self._preeq_indicator, target_value=0)],
        )

        # add conditions for the experiment indicators
        for experiment in problem.experiments:
            cond_id = self._get_experiment_indicator_condition_id(
                experiment.id
            )
            changes = [
                Change(
                    target_id=self.get_experiment_indicator(experiment.id),
                    target_value=1,
                )
            ]
            problem += Condition(
                id=cond_id,
                changes=changes,
            )

        #  All changes have been encoded in event assignments and can be
        #  removed. Only keep the conditions setting our indicators.
        problem.condition_tables = [
            ConditionTable(
                [
                    condition
                    for condition in problem.conditions
                    if condition.id.startswith("_petab")
                ]
            )
        ]
