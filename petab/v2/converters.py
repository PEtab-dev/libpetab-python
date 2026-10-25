"""Conversion of PEtab problems."""

from __future__ import annotations

import logging
import math
import warnings
from copy import deepcopy

import libsbml
import sympy as sp
from sbmlmath import TimeSymbol, sbml_math_to_sympy, set_math

from .core import (
    Change,
    Condition,
    ConditionTable,
    Experiment,
    ExperimentPeriod,
    Problem,
)
from .models._sbml_utils import (
    add_sbml_parameter,
    check,
    set_initial_assignment,
)
from .models.sbml_model import SbmlModel

__all__ = ["ExperimentsToSbmlConverter", "assignments_to_experiments"]

logger = logging.getLogger(__name__)


class ExperimentsToSbmlConverter:
    """Convert PEtab experiments to SBML.

    For an SBML-model-based PEtab problem, this class converts the PEtab
    experiments to initial assignments and events as far as possible.

    If the model already contains events, the events implementing the PEtab
    condition changes are added with a higher priority than the pre-existing
    events, so that these changes are applied before any pre-existing
    assignments.
    This requires that all event priorities in the original model are numeric
    constants.
    If exactly one pre-existing event has no priority, it is assigned a
    priority lower than all other priorities in the model.
    If multiple events have no priority, their order relative to the
    condition-change events is undefined, unless ``default_priority`` is
    provided.

    The PEtab problem must not contain any identifiers starting with
    ``_petab``.

    All periods and condition changes that are represented by initial
    assignments or events will be removed from the condition table.
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

    def __init__(
        self, problem: Problem, default_priority: float | None = None
    ):
        """Initialize the converter.

        :param problem: The PEtab problem to convert.
            This will not be modified.
        :param default_priority: The priority value to apply to any events that
            preexist in the model and do not have a priority set.

            In SBML, for event assignments that are to be applied at the same
            simulation time, the order of event execution is determined by the
            priority of the respective events.
            For events without priority, the order is undefined.
            See SBML specs for details.
            To ensure that the condition-change events are executed before
            any other events, all events should have a priority set.
            Note that, according to the SBML specs, events with equal
            priorities that are executed simultaneously must be executed in
            random order.
            If ``None``, a priority is only assigned automatically if exactly
            one pre-existing event has no priority.
        """
        if len(problem.models) > 1:
            #  https://github.com/PEtab-dev/libpetab-python/issues/392
            raise NotImplementedError(
                "Only single-model PEtab problems are supported."
            )
        if not isinstance(problem.model, SbmlModel):
            raise TypeError("Only SBML models are supported.")

        self._original_problem = problem
        self._new_problem = deepcopy(self._original_problem)

        self._model: libsbml.Model = self._new_problem.model.sbml_model
        self._preeq_indicator = self.PREEQ_INDICATOR

        sciml = self._new_problem.extensions.sciml
        self._array_data_condition_ids = (
            sciml._get_array_data_condition_ids() if sciml else set()
        )

        # The maximum event priority that was found in the unprocessed model.
        self._max_event_priority = None
        # The priority that will be used for the condition-change events.
        self._petab_event_priority = None
        self._default_priority = default_priority
        self._preprocess()

    @staticmethod
    def _get_experiment_indicator_condition_id(experiment_id: str) -> str:
        """Get the condition ID for the experiment indicator parameter."""
        return f"_petab_experiment_condition_{experiment_id}"

    def _preprocess(self) -> None:
        """Check whether we can handle the given problem and store some model
        information."""
        model = self._model
        # try to upgrade the SBML model if necessary
        if (
            model.getLevel() < 3
            and not model.getSBMLDocument().setLevelAndVersion(3, 2)
        ):
            raise ValueError(
                "Cannot handle SBML models with SBML level < 3, "
                "because they do not support initial values for event "
                "triggers and automatic upconversion of the model failed."
            )

        # Apply default priority to all events that do not have a priority
        if self._default_priority is not None:
            for event in model.getListOfEvents():
                if not self._has_priority(event):
                    priority = event.createPriority()
                    priority.setMath(
                        libsbml.parseL3Formula(str(self._default_priority))
                    )
        else:
            self._auto_assign_priority(self._get_event_priorities())

        # Track the maximum priority used so far
        event_priorities = self._get_event_priorities()
        for event_id, priority in event_priorities.items():
            if math.isnan(priority) or priority == math.inf:
                warnings.warn(
                    f"Event `{event_id}` has priority {priority}. "
                    "The condition-change events cannot be assigned a "
                    "higher priority. Make sure that this event cannot "
                    "trigger at the time of a PEtab condition change, "
                    "otherwise simulation results may be incorrect.",
                    stacklevel=1,
                )
        if non_nan := [
            p for p in event_priorities.values() if not math.isnan(p)
        ]:
            self._max_event_priority = max(0, *non_nan)

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

    @staticmethod
    def _has_priority(event: libsbml.Event) -> bool:
        """Check whether the given event has a priority expression."""
        return (
            event.isSetPriority() and event.getPriority().getMath() is not None
        )

    def _get_event_priorities(self) -> dict[str, float]:
        """Get the priorities of all events that have a priority.

        :return: Mapping of event IDs to priorities.
        :raises NotImplementedError: If any priority is not constant.
        """
        event_priorities = {}
        for ev in self._model.getListOfEvents():
            if not self._has_priority(ev):
                continue
            event_id = ev.getId() or str(ev)
            priority = sbml_math_to_sympy(ev.getPriority())
            if priority.free_symbols:
                # We'd need to find the maximum priority of all events,
                #  which is challenging/impossible to do in general.
                raise NotImplementedError(
                    f"Event `{event_id}` has a non-constant priority: "
                    f"{priority}. This is currently not supported."
                )
            event_priorities[event_id] = float(priority)
        return event_priorities

    def _auto_assign_priority(
        self, event_priorities: dict[str, float]
    ) -> None:
        """Assign a priority to a single unprioritized pre-existing event.

        If exactly one pre-existing event has no priority, and no other
        event has priority ``-inf`` or ``NaN``, the unprioritized event is
        assigned a priority lower than any other finite priority in the
        model.
        Without priority, the order of this event with respect to all other
        events is undefined, so assigning a priority only selects one of the
        admissible orders. A distinct priority cannot create ties, which
        would require random execution order (SBML L3V2 §4.12.3). This allows
        executing the condition-change events before this event.

        If more than one event has no priority, no priority is assigned,
        since this would impose an arbitrary order on them if they are
        executed simultaneously.

        :param event_priorities: The priorities of all prioritized events.
        """
        unprioritized = [
            ev
            for ev in self._model.getListOfEvents()
            if not self._has_priority(ev)
        ]
        if len(unprioritized) != 1:
            return

        priorities = event_priorities.values()
        if any(math.isnan(p) or p == -math.inf for p in priorities):
            # no distinct lower priority available
            return

        value = min((p for p in priorities if p != math.inf), default=1) - 1
        event = unprioritized[0]
        priority = event.createPriority()
        set_math(priority, value)
        logger.debug(
            "Assigned priority %s to event `%s` to ensure that the "
            "condition-change events are executed before this event.",
            value,
            event.getId(),
        )

    def convert(self) -> Problem:
        """Convert the PEtab experiments to SBML events.

        :return: The converted PEtab problem.
        """

        self._add_preequilibration_indicator()

        # Collect values for initial assignments for the different experiments.
        #  All expressions must be combined into a single initial assignment
        #  per target.
        # target_id -> [(experiment_indicator, target_value), ...]
        period0_assignments: dict[str, list[tuple[str, sp.Basic]]] = {}
        for experiment in self._new_problem.experiments:
            self._convert_experiment(experiment, period0_assignments)

        self._create_initial_assignments(period0_assignments)

        self._add_indicators_to_conditions()

        validation_results = self._new_problem.validate()
        validation_results.log()

        return self._new_problem

    def _convert_experiment(
        self,
        experiment: Experiment,
        period0_assignments: dict[str, list[tuple[str, sp.Basic]]],
    ) -> None:
        """
        Convert a single experiment to SBML events or initial assignments.

        :param experiment: The experiment to convert.
        :param period0_assignments: The changes of the first period are not
            applied directly, but are added to this dictionary
            (``target_id -> [(experiment_indicator, target_value), ...]``).
            They are combined across experiments into a single initial
            assignment per target by :meth:`_create_initial_assignments`.
        """
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
        kept_periods: list[ExperimentPeriod] = []

        for i_period, period in enumerate(experiment.sorted_periods):
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

            # Skip if condition ids are set by array data
            if self._array_data_condition_ids and set(
                period.condition_ids
            ).issubset(self._array_data_condition_ids):
                continue

            # Encode the period changes in the SBML model as events
            #  that trigger at the start of the period or,
            #  for the first period, as initial assignments.
            #  Initial assignments are required for the first period,
            #  because other initial assignments may depend on
            #  the changed values.
            #  Additionally, tools that don't support events can still handle
            #  single-period experiments.
            if i_period == 0:
                exp_ind_id = self.get_experiment_indicator(experiment.id)

                for change in self._new_problem.get_changes_for_period(period):
                    period0_assignments.setdefault(
                        change.target_id, []
                    ).append((exp_ind_id, change.target_value))
            else:
                ev = self._create_period_start_event(
                    experiment=experiment,
                    i_period=i_period,
                    period=period,
                )
                self._create_event_assignments_for_period(
                    ev,
                    self._new_problem.get_changes_for_period(period),
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

    def _create_initial_assignments(
        self, period0_assignments: dict[str, list[tuple[str, sp.Basic]]]
    ) -> None:
        """Create initial assignments for the first period of all experiments.

        Unlike events, we can't have different initial assignments for
        different experiments, so all changes of a target are combined into
        a single piecewise expression, conditioned on the experiment
        indicators.

        :param period0_assignments: The changes of the first period of all
            experiments
            (``target_id -> [(experiment_indicator, target_value), ...]``).
        """
        model = self._model
        free_symbols_in_assignments = set()
        for target_id, changes in period0_assignments.items():
            # The initial value might only be changed for a subset of
            #  experiments. We need to keep the original initial value
            #  for all other experiments.

            # Is there an initial assignment for this target already?
            # If not, fall back to the initial value of the target.
            if (
                ia := model.getInitialAssignmentBySymbol(target_id)
            ) is not None:
                default = sbml_math_to_sympy(ia.getMath())
            else:
                # use the initial value of the target as default
                target = model.getElementBySId(target_id)
                default = self._initial_value_from_element(target)

            # Only create the initial assignment if there is
            #  actually something to change.
            if expr_cond_pairs := [
                (target_value, sp.Symbol(exp_ind) > 0.5)
                for exp_ind, target_value in changes
                if target_value != default
            ]:
                expr = sp.Piecewise(
                    *expr_cond_pairs,
                    (default, True),
                )

                set_initial_assignment(model, target_id, expr)
                free_symbols_in_assignments |= expr.free_symbols

        # the target value may depend on parameters that are only
        #  introduced in the PEtab parameter table - those need
        #  to be added to the model
        for sym in free_symbols_in_assignments:
            if model.getElementBySId(sym.name) is None:
                add_sbml_parameter(model, id_=sym.name, constant=True, value=0)

    @staticmethod
    def _initial_value_from_element(target: libsbml.SBase) -> sp.Basic:
        """Get the initial value of an SBML element.

        The value of the size attribute of compartments,
        the initial concentration or amount of species (amount for
        `hasOnlySubstanceUnits=true`, concentration otherwise), and
        the value of parameters, not considering any initial assignment
        constructs.
        """
        if target is None:
            raise ValueError("`target` is None.")

        if target.getTypeCode() == libsbml.SBML_COMPARTMENT:
            return sp.Float(target.getSize())

        if target.getTypeCode() == libsbml.SBML_SPECIES:
            if target.getHasOnlySubstanceUnits():
                # amount-based -> return amount
                if target.isSetInitialAmount():
                    return sp.Float(target.getInitialAmount())
                return sp.Float(target.getInitialConcentration()) * sp.Symbol(
                    target.getCompartment()
                )
            # concentration-based -> return concentration
            if target.isSetInitialConcentration():
                return sp.Float(target.getInitialConcentration())

            return sp.Float(target.getInitialAmount()) / sp.Symbol(
                target.getCompartment()
            )

        if target.getTypeCode() == libsbml.SBML_PARAMETER:
            return sp.Float(target.getValue())

        raise NotImplementedError(
            "Cannot create initial assignment for unsupported SBML "
            f"entity type {target.getTypeCode()}."
        )

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
        event: libsbml.Event, changes: list[Change]
    ) -> None:
        """Create event assignments for a given period.

        Converts PEtab ``Change``s to equivalent SBML event assignments.

        Note that the SBML event assignment formula is not necessarily the same
        as the `targetValue` in PEtab.
        In SBML, concentrations are treated as derived quantities.
        Therefore, changing the size of a compartment will update the
        concentrations of all contained concentration-based species.
        In PEtab, such a change would not automatically update the species
        concentrations, but only the compartment size.

        Therefore, to correctly implement a PEtab change of a compartment size
        in SBML, we need to compensate for the automatic update of species
        concentrations by adding event assignments for all contained
        concentration-based species.

        :param event: The SBML event to which the assignments should be added.
        :param changes: The PEtab condition changes that are to be applied
            at the start of the period.
        """
        _add_assignment = ExperimentsToSbmlConverter._add_assignment
        sbml_model = event.getModel()
        # collect IDs of compartments that are changed in this period
        changed_compartments = {
            change.target_id
            for change in changes
            if sbml_model.getElementBySId(change.target_id) is not None
            and sbml_model.getElementBySId(change.target_id).getTypeCode()
            == libsbml.SBML_COMPARTMENT
        }

        for change in changes:
            sbml_target = sbml_model.getElementBySId(change.target_id)

            if sbml_target is None:
                raise ValueError(
                    f"Cannot create event assignment for change of "
                    f"`{change.target_id}`: No such entity in the SBML model."
                )

            target_type = sbml_target.getTypeCode()
            if target_type == libsbml.SBML_COMPARTMENT:
                # handle the actual compartment size change
                _add_assignment(event, change.target_id, change.target_value)

                # Changing a compartment size affects all contained
                #  concentration-based species - we need to add event
                #  assignments for those to compensate for the automatic
                #  update of their concentrations.
                # The event assignment will set the concentration to
                #   new_conc = assigned_amount / new_volume
                #            = assigned_conc * old_volume / new_volume
                #   <=> assigned_conc = new_conc * new_volume / old_volume
                # Therefore, the event assignment is not just `new_conc`,
                #  but `new_conc * new_volume / old_volume`.

                # concentration-based species in the changed compartment
                conc_species = [
                    species.getId()
                    for species in sbml_model.getListOfSpecies()
                    if species.getCompartment() == change.target_id
                    and not species.getHasOnlySubstanceUnits()
                ]
                for species_id in conc_species:
                    if species_change := next(
                        (c for c in changes if c.target_id == species_id), None
                    ):
                        # there is an explicit change for this species
                        #  in this period
                        new_conc = species_change.target_value
                    else:
                        # no explicit change, use the pre-event concentration
                        new_conc = sp.Symbol(species_id)

                    _add_assignment(
                        event,
                        species_id,
                        # new_conc * new_volume / old_volume
                        new_conc
                        * change.target_value
                        / sp.Symbol(change.target_id),
                    )
            elif (
                target_type != libsbml.SBML_SPECIES
                or sbml_target.getCompartment() not in changed_compartments
                or sbml_target.getHasOnlySubstanceUnits() is True
            ):
                # Handle any changes other than compartments and
                #  concentration-based species inside resized compartments
                #  that we already handled above.
                # Those translate directly to event assignments.
                _add_assignment(event, change.target_id, change.target_value)

    @staticmethod
    def _add_assignment(
        event: libsbml.Event, target_id: str, target_value: sp.Basic
    ) -> None:
        """Add a single event assignment to the given event
        and apply any necessary changes to the model."""
        sbml_model = event.getModel()
        ea = event.createEventAssignment()
        ea.setVariable(target_id)
        set_math(ea, target_value)

        # target needs const=False, and target may not exist yet
        #  (e.g., in case of output parameters added in the observable
        #  table)
        target = sbml_model.getElementBySId(target_id)
        if target is None:
            add_sbml_parameter(
                sbml_model, id_=target_id, constant=False, value=0
            )
        else:
            # We can safely change the `constant` attribute of the target.
            #  "Constant" does not imply "boundary condition" in SBML.
            target.setConstant(False)

        # the target value may depend on parameters that are only
        #  introduced in the PEtab parameter table - those need
        #  to be added to the model
        for sym in target_value.free_symbols:
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


DEFAULT_EXPERIMENT_ID = "default_experiment"


def assignments_to_experiments(problem: Problem) -> Problem:
    """Replace time-dependent piecewise assignments by experiment periods.

    Fixed parameters may change over the course of a single simulation, e.g.,
    as described by a v2 experiment with multiple periods. As v1 did not have
    experiments, then these changes were
    sometimes directly encoded in an SBML model as assignment rules with
    piecewise functions. This method takes a v2 problem
    (possibly converted from a v1 problem) and replaces those SBML assignment
    rules with v2 experiments.

    .. note::

        Measurements without an experiment ID are assigned to a new experiment
        :data:`DEFAULT_EXPERIMENT_ID`, if needed, which starts at time 0.
        Every experiment keeps the period that starts it, without conditions
        if all of them were replaced, since a period's time overrides the
        model's initial time.

    **Scope:** conversion is supported for SBML assignment rules to
    parameters that depend on time through the conditions of piecewise
    functions, and at most linearly through their values, where

    * every such condition is an inequality that can be re-arranged
      explicitly into `t </<=/>/>= ...`,
    * every piecewise function that depends on time has an explicit
      "otherwise" piece,
    * every switching time evaluates to a real number in every experiment
      (it may depend on parameters that are fixed or set by the condition
      table, but not on estimated ones),
    * and all other symbols are truly constant model parameters.

    Within that, the piecewise functions may be combined: a sum of doses
    given at different times (`d1 * piecewise(0, t < t1, 1) + d2 *
    piecewise(0, t < t2, 1)`), phases one after another as several pieces or
    nested piecewise functions (`piecewise(b0, t < t1, piecewise(b1, t < t2,
    b2))`). Each switch that changes the value becomes a period. A value may
    also change linearly with time within a period (a ramp: `piecewise(r * t
    + a, t <= t_end, 0)`). A condition change is applied once at the start of
    its period, so the target of such an assignment becomes a differential
    entity instead, with the rate rule `d target/dt = <target>_slope`, and
    each period sets both the target and its slope; an assignment that only
    switches between constant values stays a constant parameter. Anything else
    raises a :class:`NotImplementedError`. Initial assignments stay in the
    model, since they are evaluated once, at the start of the simulation.

    Condition changes of the replaced assignments' symbols are dropped, unless
    the model, an observable or another change still refers to them.

    .. warning::

        As SBML assignment rules, the changes would occur during
        pre-equilibration. As v2 experiments, the changes do not occur during
        pre-equilibration. Hence, using this converter may affect the
        correctness of your model's pre-equilibration. In this case, you could
        e.g. keep the SBML assignment rules and/or add indicator variables to
        trigger them correctly in the model for some/all pre-equilibration
        conditions.


    :param problem: The PEtab problem to convert. This will not be modified.
    :returns: The converted PEtab problem.
    :raises NotImplementedError:
        If a time-dependent assignment is outside the scope described above.
    :raises ValueError:
        If the converted problem does not pass linting.
    """
    if not isinstance(problem.model, SbmlModel):
        raise TypeError("Only SBML models are supported.")

    problem = deepcopy(problem)
    sbml_model = problem.model.sbml_model

    # Initial assignments are evaluated once, at the start of the
    #  simulation, so they never switch and stay in the model.
    time_dependent = []
    for rule in sbml_model.getListOfRules():
        if not rule.isAssignment():
            continue
        expr = sbml_math_to_sympy(rule)
        if any(pw.has(TimeSymbol) for pw in expr.atoms(sp.Piecewise)):
            time_dependent.append((rule, expr))

    experiments = list(problem.experiments)
    if any(m.experiment_id is None for m in problem.measurements):
        default_experiment = Experiment(id=DEFAULT_EXPERIMENT_ID)
        experiments.append(default_experiment)
    else:
        default_experiment = None
    droppable = _add_start_periods(experiments) if time_dependent else set()

    # (experiment, period start time, target ID, target value)
    additions = []
    replaced_symbols = set()
    converted_ids = []
    for rule, expr in time_dependent:
        assignment = _PiecewiseAssignment(rule, expr, problem)
        assignment.replace_in_model()
        for experiment in experiments:
            additions += [
                (experiment, *change)
                for change in assignment.changes(experiment)
            ]
        replaced_symbols |= assignment.symbol_ids
        converted_ids.append(assignment.target_id)

    if converted_ids and any(e.has_preequilibration for e in experiments):
        warnings.warn(
            "Switches of time-dependent piecewise assignments during "
            "preequilibration are not supported. The converted "
            f"assignments to {converted_ids} keep their value at time 0 "
            "throughout "
            "preequilibration.",
            stacklevel=2,
        )

    # replace the changes that are covered by the new conditions, unless
    #  something else still refers to their targets
    replaced_symbols -= _referenced_ids(problem, additions)
    _drop_changes(problem, experiments, replaced_symbols)
    _add_conditions(problem, additions)

    for experiment in experiments:
        experiment.sort_periods()
        experiment.periods = [
            p
            for p in experiment.periods
            if p.condition_ids or id(p) not in droppable
        ]

    if default_experiment and any(
        p.condition_ids for p in default_experiment.periods
    ):
        problem += default_experiment
        for measurement in problem.measurements:
            if measurement.experiment_id is None:
                measurement.experiment_id = DEFAULT_EXPERIMENT_ID

    validation_issues = problem.validate()
    if validation_issues.has_errors():
        messages = "\n".join(f"{issue.message}" for issue in validation_issues)
        raise ValueError(
            f"The converted PEtab problem did not pass linting:\n{messages}"
        )

    return problem


class _PiecewiseAssignment:
    """A time-dependent piecewise assignment to a model parameter, and the
    condition changes that replace it, see :func:`assignments_to_experiments`.
    """

    def __init__(
        self,
        rule: libsbml.Rule | libsbml.InitialAssignment,
        expr: sp.Expr,
        problem: Problem,
    ):
        """Check that the assignment ``rule``, with the value ``expr``, is
        supported."""
        self.rule = rule
        self.problem = problem
        self.sbml_model = problem.model.sbml_model
        self.target_id = rule.getId()
        if self.sbml_model.getParameter(self.target_id) is None:
            raise NotImplementedError(
                "Only assignments to parameters are supported, but "
                f"`{self.target_id}` is not a parameter."
            )
        self.expr = expr
        (self.time,) = expr.atoms(TimeSymbol)
        self.symbol_ids = {str(s) for s in expr.free_symbols}
        self.parameter_ids = {p.id for p in problem.parameters}
        self.switches = _supported_time_switches(
            expr,
            self.time,
            self.target_id,
            {p.id for p in problem.parameters if p.estimate},
        )
        if non_constant := sorted(
            str(s)
            for s in expr.free_symbols - {self.time}
            if not (p := self.sbml_model.getParameter(str(s)))
            or not p.getConstant()
        ):
            raise NotImplementedError(
                "Only constant model parameters are supported as symbols in "
                f"the assignment to `{self.target_id}`, but it refers to "
                f"{non_constant}."
            )
        # the value before every switch: a condition whose two sides grow
        #  apart with time holds only after its switch
        self.default = expr.xreplace(
            {
                c: sp.false if slope > 0 else sp.true
                for c, (_, slope) in self.switches.items()
            }
        )
        # unless a condition sets them, the switching times' symbols keep
        #  their value from the parameter table or else from the model
        nominal_values = {
            p.id: p.nominal_value for p in problem.parameters if not p.estimate
        }
        self.fixed_values = {
            s: nominal_values.get(str(s), self._model_value(s))
            for t_switch, _ in self.switches.values()
            for s in t_switch.free_symbols
        }
        # whether the value changes with time within a period
        self.ramps = expr.xreplace({c: sp.Dummy() for c in self.switches}).has(
            self.time
        )
        self.slope_id = None

    def replace_in_model(self) -> None:
        """Replace the assignment in the model by a target that the
        condition table sets.

        A value that changes with time within a period (a ramp) cannot be a
        condition change on a constant target, which is applied once at the
        start of its period. The model only makes the target a differential
        entity, `d target/dt = <target>_slope`; the slope's value, like the
        target's, is set by every period that starts one.
        """
        if self.ramps:
            self.slope_id = f"{self.target_id}_slope"
            while self.sbml_model.getElementBySId(self.slope_id) is not None:
                self.slope_id += "_"
            add_sbml_parameter(
                self.sbml_model, self.slope_id, value=0.0, constant=False
            )
            rate_rule = self.sbml_model.createRateRule()
            rate_rule.setVariable(self.target_id)
            rate_rule.setMath(libsbml.parseL3Formula(self.slope_id))
        model_values = {
            s: self._model_value(s)
            for s in self.default.free_symbols - {self.time}
        }
        target = self.sbml_model.getParameter(self.target_id)
        target.setValue(
            float(self.default.subs(self.time, 0).subs(model_values))
        )
        target.setConstant(False)
        self.rule.removeFromParentAndDelete()

    def changes(
        self, experiment: Experiment
    ) -> list[tuple[float, str, sp.Expr]]:
        """The changes that replace the assignment in ``experiment``, each as
        ``(period start time, target ID, target value)``."""
        changes = []
        periods = experiment.sorted_periods
        overrides = {}
        # the value at the end of the previous period
        held = None
        for i_period, period in enumerate(periods):
            period_changes = self.problem.get_changes_for_period(period)
            overrides |= {c.target_id: c.target_value for c in period_changes}
            subs = self.fixed_values | {
                s: overrides[str(s)]
                for s in self.expr.free_symbols
                if str(s) in overrides
            }
            times = self._switching_times(subs, experiment.id)

            if period.is_preequilibration:
                # the value at time 0, which respects whether a condition
                #  holds for a switch at time 0, held there: a ramp does not
                #  run during preequilibration
                held = self.expr.subs(subs).subs(self.time, 0)
                changes.append(
                    (
                        period.time,
                        self.target_id,
                        self._with_model_values(held),
                    )
                )
                if self.ramps:
                    changes.append((period.time, self.slope_id, sp.S.Zero))
                continue

            start_value = self._value_from(subs, times, period.time)
            # each switch within the period that changes the value
            t_end = (
                periods[i_period + 1].time
                if i_period + 1 < len(periods)
                else sp.oo
            )
            switch_values = []
            previous = start_value
            for t in sorted({t for t in times if period.time < t < t_end}):
                value = self._value_from(subs, times, t)
                if not (value - previous).is_zero:
                    switch_values.append((t, value))
                previous = value

            if i_period == 0:
                # The model already has the pre-switch value at the start of
                #  the first period, which stays even without conditions,
                #  unless that value refers to parameter table symbols, for
                #  which the model only has a number. A ramp's start and
                #  slope are always set by the tables.
                if (
                    self.ramps
                    or not (start_value - self.default).is_zero
                    or {str(s) for s in self.default.free_symbols}
                    & self.parameter_ids
                ):
                    changes += self._start(
                        period.time, self._with_model_values(start_value)
                    )
            elif (
                {c.target_id for c in period_changes} & self.symbol_ids
                # preequilibration held the slope at zero
                or (self.ramps and periods[i_period - 1].is_preequilibration)
                # a switch at the start of the period
                or not (start_value - held).is_zero
            ):
                changes += self._start(period.time, start_value)
            for t, value in switch_values:
                changes += self._start(t, value)
            held = previous
        return changes

    def _model_value(self, symbol: sp.Symbol) -> float:
        """The value of the model parameter ``symbol``."""
        return self.sbml_model.getParameter(str(symbol)).getValue()

    def _with_model_values(self, value: sp.Expr) -> sp.Expr:
        """``value`` for a change at the start of an experiment, which may
        only refer to parameter table symbols: the other symbols are constant
        model parameters, at their values."""
        return value.subs(
            {
                s: self._model_value(s)
                for s in value.free_symbols - {self.time}
                if str(s) not in self.parameter_ids
            }
        )

    def _switching_times(self, subs: dict, experiment_id: str) -> list[float]:
        """The switching times, for the symbol values ``subs``."""
        times = []
        for t_switch, _ in self.switches.values():
            t = t_switch.subs(subs)
            if not (t.is_number and t.is_real and t.is_finite):
                raise NotImplementedError(
                    "Only switching times that evaluate to a real number are "
                    "supported, but a switching time of the assignment to "
                    f"`{self.target_id}` in experiment `{experiment_id}` is "
                    f"`{t}`."
                )
            times.append(float(t))
        return times

    def _value_from(self, subs: dict, times: list[float], t0: float):
        """The value from ``t0`` up to its next switch, for the symbol values
        ``subs``, every condition replaced by whether it holds in between."""
        later = [t for t in times if t > t0]
        probe = (t0 + min(later)) / 2 if later else t0 + 1
        truth = {c: c.subs(subs).subs(self.time, probe) for c in self.switches}
        return self.expr.xreplace(truth).subs(subs)

    def _start(
        self, t: float, value: sp.Expr
    ) -> list[tuple[float, str, sp.Expr]]:
        """The changes that start ``value`` at time ``t``: the target set to
        the value at ``t``, and, for a target that ramps, its slope set to
        the value's rate of change."""
        changes = [(t, self.target_id, value.subs(self.time, t))]
        if self.ramps:
            changes.append((t, self.slope_id, sp.diff(value, self.time)))
        return changes


def _add_start_periods(experiments: list[Experiment]) -> set[int]:
    """Give every experiment a period that starts it, and return the
    :func:`id` of the periods that are dropped if all their conditions are
    replaced.

    The period that starts an experiment stays, even without conditions: its
    time overrides the model's initial time, so the experiment would start at
    its first switch instead.
    """
    droppable = set()
    for experiment in experiments:
        if all(p.is_preequilibration for p in experiment.periods):
            experiment += ExperimentPeriod(time=0)
        start = min(
            (p for p in experiment.periods if not p.is_preequilibration),
            key=lambda p: p.time,
        )
        droppable |= {
            id(p)
            for p in experiment.periods
            if p.condition_ids and p is not start
        }
    return droppable


def _referenced_ids(
    problem: Problem,
    additions: list[tuple[Experiment, float, str, sp.Expr]],
) -> set[str]:
    """The IDs that the math of the model, the observables, or the condition
    changes, including those in ``additions``, refer to."""
    ids = set()
    for element in problem.model.sbml_model.getListOfAllElements():
        if not (hasattr(element, "isSetMath") and element.isSetMath()):
            continue
        nodes = [element.getMath()]
        while nodes:
            node = nodes.pop()
            if node.getType() == libsbml.AST_NAME:
                ids.add(node.getName())
            nodes += [node.getChild(i) for i in range(node.getNumChildren())]
    expressions = [
        *(o.formula for o in problem.observables),
        *(o.noise_formula for o in problem.observables),
        *(value for *_, value in additions),
    ]
    ids |= {
        str(s)
        for expression in expressions
        if expression is not None
        for s in expression.free_symbols
    }
    ids |= {
        str(s)
        for table in problem.condition_tables
        for s in table.free_symbols
    }
    return ids


def _drop_changes(
    problem: Problem, experiments: list[Experiment], target_ids: set[str]
) -> None:
    """Drop the changes of ``target_ids`` from the condition tables, and the
    conditions left without changes from the periods of ``experiments``."""
    condition_ids = {c.id for c in problem.conditions}
    for table in problem.condition_tables:
        for condition in table.conditions:
            condition.changes = [
                c for c in condition.changes if c.target_id not in target_ids
            ]
        table.elements = [c for c in table.conditions if c.changes]
    removed_ids = condition_ids - {c.id for c in problem.conditions}
    for experiment in experiments:
        for period in experiment.periods:
            period.condition_ids = [
                cid for cid in period.condition_ids if cid not in removed_ids
            ]


def _add_conditions(
    problem: Problem,
    additions: list[tuple[Experiment, float, str, sp.Expr]],
) -> None:
    """For each ``(experiment, time, target ID, value)`` in ``additions``,
    add a condition with that change to the period of ``experiment`` that
    starts at ``time``, which is added if needed.

    Equal changes share a condition.
    """
    new_conditions = {}
    for experiment, t, target_id, value in additions:
        if (cond_id := new_conditions.get((target_id, value))) is None:
            cond_id = problem.get_unused_id(f"cond_{len(new_conditions)}")
            new_conditions[target_id, value] = cond_id
            problem += Condition(
                id=cond_id,
                changes=[Change(target_id=target_id, target_value=value)],
            )
        period = next((p for p in experiment.periods if p.time == t), None)
        if period is None:
            experiment += (period := ExperimentPeriod(time=t))
        period.condition_ids.append(cond_id)


def _supported_time_switches(
    expr: sp.Expr, time: sp.Symbol, target_id: str, estimated_ids: set[str]
) -> dict[sp.Rel, tuple[sp.Expr, sp.Expr]]:
    """Check that ``expr`` depends on ``time`` only in the supported way, and
    return the conditions through which it does, each with the time at which
    it switches and the slope of its two sides' difference.

    ``expr`` may combine several piecewise functions, each with several
    pieces and nested in one another (a sum of doses, phases one after
    another), as long as time enters only through their conditions, each a
    single inequality linear in time, each piecewise function has an
    explicit "otherwise" piece, and no switching time depends on the
    parameters ``estimated_ids``. Anything else raises
    :class:`NotImplementedError`.
    """
    unsupported = NotImplementedError(
        "Only time dependence through piecewise conditions, each a single "
        "inequality linear in time, with an explicit otherwise piece, is "
        f"supported, but the assignment to `{target_id}` is `{expr}`."
    )
    switches = {}
    for pw in expr.atoms(sp.Piecewise):
        if not pw.has(time):
            continue
        # an explicit "otherwise" piece
        if pw.args[-1].cond != sp.true:
            raise unsupported
        for piece in pw.args[:-1]:
            condition = piece.cond
            if not condition.has(time):
                continue
            if not (
                # a supported inequality
                isinstance(condition, (sp.Gt, sp.Ge, sp.Lt, sp.Le))
                # that changes value at maximally a single time point
                and (
                    slope := sp.diff(condition.gts - condition.lts, time)
                ).is_number
                # and this time point is identifiable
                and slope.is_nonzero
            ):
                raise unsupported
            t_switch = sp.solve(condition.gts - condition.lts, time)[0]
            switches[condition] = (t_switch, slope)
    # outside the conditions, time enters at most linearly: the value may
    #  ramp, at a rate that does not itself change with time
    rest = expr.xreplace({c: sp.Dummy() for c in switches})
    if rest.has(time) and sp.diff(rest, time).has(time):
        raise NotImplementedError(
            "Only values that change at most linearly with time are "
            f"supported, but the assignment to `{target_id}` is `{expr}`."
        )
    # no period can start at an estimated time
    if estimated := sorted(
        {
            str(s)
            for t_switch, _ in switches.values()
            for s in t_switch.free_symbols
            if str(s) in estimated_ids
        }
    ):
        raise NotImplementedError(
            "Only switching times that are not estimated are supported, but "
            f"the switching times of the assignment to `{target_id}` depend "
            f"on the estimated parameters {estimated}."
        )
    return switches
