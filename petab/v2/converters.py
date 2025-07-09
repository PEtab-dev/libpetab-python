"""Conversion of PEtab problems."""

from __future__ import annotations

import warnings
from copy import deepcopy
from math import inf

import libsbml
from sbmlmath import sbml_math_to_sympy, set_math

from .core import Change, Condition, Experiment, ExperimentPeriod
from .models._sbml_utils import add_sbml_parameter, check
from .models.sbml_model import SbmlModel
from .problem import Problem

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

    def __init__(self, problem: Problem):
        """Initialize the converter.

        :param problem: The PEtab problem to convert.
            This will not be modified.
        """
        if not isinstance(problem.model, SbmlModel):
            raise ValueError("Only SBML models are supported.")

        self._original_problem = problem
        self._new_problem = deepcopy(self._original_problem)

        self._model = self._new_problem.model.sbml_model
        self._preeq_indicator = self.PREEQ_INDICATOR

        # The maximum event priority that was found in the unprocessed model.
        self._max_event_priority = None
        # The priority that will be used for the PEtab events.
        self._petab_event_priority = None

        self._preprocess()

    def _preprocess(self):
        """Check whether we can handle the given problem and store some model
        information."""
        model = self._model
        if model.getLevel() < 3:
            # try to upgrade the SBML model
            if not model.getSBMLDocument().setLevelAndVersion(3, 2):
                raise ValueError(
                    "Cannot handle SBML models with SBML level < 3, "
                    "because they do not support initial values for event "
                    "triggers and automatic upconversion failed."
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
                    "undefined.",
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

        problem = self._new_problem
        for experiment in problem.experiment_table.experiments:
            self._convert_experiment(problem, experiment)

        self._add_indicators_to_conditions(problem)

        validation_results = problem.validate()
        validation_results.log()

        return problem

    def _convert_experiment(self, problem: Problem, experiment: Experiment):
        """Convert a single experiment to SBML events."""
        model = self._model
        experiment.sort_periods()
        has_preequilibration = (
            len(experiment.periods) and experiment.periods[0].time == -inf
        )

        # add experiment indicator
        exp_ind_id = self.get_experiment_indicator(experiment.id)
        if model.getElementBySId(exp_ind_id) is not None:
            raise AssertionError(
                f"Entity with ID {exp_ind_id} exists already."
            )
        add_sbml_parameter(model, id_=exp_ind_id, constant=False, value=0)
        kept_periods = []
        for i_period, period in enumerate(experiment.periods):
            # check for non-zero initial times of the first period
            if (i_period == int(has_preequilibration)) and period.time != 0:
                # TODO: we could address that by offsetting all occurrences of
                #  the SBML time in the model (except for the newly added
                #  events triggers). Or we better just leave it to the
                #  simulator -- we anyways keep the first period in the
                #  returned Problem.
                raise NotImplementedError(
                    "Cannot represent non-zero initial time in SBML."
                )

            if period.time == -inf:
                # pre-equilibration cannot be represented in SBML,
                #  so we need to keep this period in the Problem.
                kept_periods.append(period)
            elif i_period == int(has_preequilibration):
                # we always keep the first non-pre-equilibration period
                #  to set the indicator parameters
                kept_periods.append(period)
            elif not period.changes:
                # no condition, no changes, no need for an event,
                #  no need to keep the period unless it's the pre-equilibration
                #  or the only non-equilibration period (handled above)
                continue

            ev = self._create_period_begin_event(
                experiment=experiment,
                i_period=i_period,
                period=period,
            )
            self._create_event_assignments_for_period(
                ev,
                [
                    problem.condition_table[condition_id]
                    for condition_id in period.condition_ids
                ],
            )

        if len(kept_periods) > 2:
            raise AssertionError("Expected at most two periods to be kept.")

        # add conditions that set the indicator parameters
        for period in kept_periods:
            period.condition_ids = [
                f"_petab_experiment_condition_{experiment.id}",
                "_petab_preequilibration"
                if period.time == -inf
                else "_petab_no_preequilibration",
            ]

        experiment.periods = kept_periods

    def _create_period_begin_event(
        self, experiment: Experiment, i_period: int, period: ExperimentPeriod
    ) -> libsbml.Event:
        """Create an event that triggers at the beginning of a period."""

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

        if period.time == -inf:
            trig_math = libsbml.parseL3Formula(
                f"({exp_ind_id} == 1) && ({self._preeq_indicator} == 1)"
            )
        else:
            trig_math = libsbml.parseL3Formula(
                f"({exp_ind_id} == 1) && ({self._preeq_indicator} != 1) "
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
    def _change_to_event_assignment(change: Change, event: libsbml.Event):
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
            # TODO: can that break models??
            target.setConstant(False)

        # the target value may depend on parameters that are only
        #  introduced in the PEtab parameter table - those need
        #  to be added to the model
        for sym in change.target_value.free_symbols:
            if sbml_model.getElementBySId(sym.name) is None:
                add_sbml_parameter(
                    sbml_model, id_=sym.name, constant=True, value=0
                )

    def _add_indicators_to_conditions(self, problem: Problem) -> None:
        """After converting the experiments to events, add the indicator
        parameters for the pre-equilibration period and for the different
        experiments to the remaining conditions.
        Then remove all other conditions."""

        # create conditions for indicator parameters
        problem.condition_table.conditions.append(
            Condition(
                id="_petab_preequilibration",
                changes=[
                    Change(target_id=self._preeq_indicator, target_value=1)
                ],
            )
        )
        problem.condition_table.conditions.append(
            Condition(
                id="_petab_no_preequilibration",
                changes=[
                    Change(target_id=self._preeq_indicator, target_value=0)
                ],
            )
        )
        # add conditions for the experiment indicators
        for experiment in problem.experiment_table.experiments:
            problem.condition_table.conditions.append(
                Condition(
                    id=f"_petab_experiment_condition_{experiment.id}",
                    changes=[
                        Change(
                            target_id=self.get_experiment_indicator(
                                experiment.id
                            ),
                            target_value=1,
                        )
                    ],
                )
            )

        #  All changes have been encoded in event assignments and can be
        #  removed. Only keep the conditions setting our indicators.
        problem.condition_table.conditions = [
            condition
            for condition in problem.condition_table.conditions
            if condition.id.startswith("_petab")
        ]
