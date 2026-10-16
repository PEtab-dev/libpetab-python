"""Conversion of PEtab problems."""

from __future__ import annotations

import warnings
from copy import deepcopy
from itertools import count

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
from .models._sbml_utils import add_sbml_parameter, check
from .models.sbml_model import SbmlModel

__all__ = ["ExperimentsToSbmlConverter", "assignments_to_experiments"]


class ExperimentsToSbmlConverter:
    """Convert PEtab experiments to SBML.

    For an SBML-model-based PEtab problem, this class converts the PEtab
    experiments to initial assignments and events as far as possible.

    If the model already contains events, PEtab events are added with a higher
    priority than the existing events to guarantee that PEtab condition changes
    are applied before any pre-existing assignments.
    This requires that all event priorities in the original model are numeric
    constants.

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
        # The priority that will be used for the PEtab events.
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

                # Create a new initial assignment if necessary, otherwise
                #  overwrite the existing one.
                if ia is None:
                    ia = model.createInitialAssignment()
                    ia.setSymbol(target_id)

                set_math(ia, expr)
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
        :data:`DEFAULT_EXPERIMENT_ID`, if needed.

    **Scope:** conversion is supported for SBML assignment rules that depend
    on time through the conditions of piecewise functions, and at most
    linearly through their values, where

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
    raises a :class:`NotImplementedError`.

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
    parameter_ids = {p.id for p in problem.parameters}
    estimated_ids = {p.id for p in problem.parameters if p.estimate}
    nominal_values = {
        p.id: p.nominal_value for p in problem.parameters if not p.estimate
    }

    experiments = list(problem.experiments)
    if any(m.experiment_id is None for m in problem.measurements):
        default_experiment = Experiment(id=DEFAULT_EXPERIMENT_ID)
        experiments.append(default_experiment)
    else:
        default_experiment = None
    # periods that are dropped if all their conditions were replaced
    droppable = {
        id(p) for e in experiments for p in e.periods if p.condition_ids
    }
    for experiment in experiments:
        if all(p.is_preequilibration for p in experiment.periods):
            experiment += (period := ExperimentPeriod(time=0))
            droppable.add(id(period))

    # (experiment, period start time, target ID, target value)
    additions = []
    replaced_symbols = set()
    converted_ids = []

    for assignment in [
        *(r for r in sbml_model.getListOfRules() if r.isAssignment()),
        *sbml_model.getListOfInitialAssignments(),
    ]:
        expr = sbml_math_to_sympy(assignment)
        if not any(pw.has(TimeSymbol) for pw in expr.atoms(sp.Piecewise)):
            continue
        target_id = assignment.getId()
        (time,) = expr.atoms(TimeSymbol)
        switches = _supported_time_switches(
            expr, time, target_id, estimated_ids
        )
        if non_constant := sorted(
            str(s)
            for s in expr.free_symbols - {time}
            if not (p := sbml_model.getParameter(str(s)))
            or not p.getConstant()
        ):
            raise NotImplementedError(
                "Only constant model parameters are supported as symbols in "
                f"the assignment to `{target_id}`, but it refers to "
                f"{non_constant}."
            )
        # the value before every switch: a condition whose two sides grow
        #  apart with time holds only after its switch
        default = expr.xreplace(
            {
                c: sp.false if slope > 0 else sp.true
                for c, (_, slope) in switches.items()
            }
        )
        model_values = {
            s: sbml_model.getParameter(str(s)).getValue()
            for s in default.free_symbols - {time}
        }
        # unless a condition sets them, the switching times' symbols keep
        #  their value from the parameter table or else from the model
        fixed_values = {
            s: nominal_values.get(
                str(s), sbml_model.getParameter(str(s)).getValue()
            )
            for t_switch, _ in switches.values()
            for s in t_switch.free_symbols
        }
        target = sbml_model.getParameter(target_id)
        # A value that changes with time within a period (a ramp) cannot be a
        #  condition change on a constant target, which is applied once at
        #  the start of its period. The model only makes the target a
        #  differential entity, `d target/dt = <target>_slope`; the slope's
        #  value, like the target's, is set by every period that starts one.
        ramps = expr.xreplace({c: sp.Dummy() for c in switches}).has(time)
        slope_id = None
        if ramps:
            slope_id = f"{target_id}_slope"
            while sbml_model.getElementBySId(slope_id) is not None:
                slope_id += "_"
            slope = sbml_model.createParameter()
            slope.setId(slope_id)
            slope.setConstant(False)
            slope.setValue(0.0)
            rate_rule = sbml_model.createRateRule()
            rate_rule.setVariable(target_id)
            rate_rule.setMath(libsbml.parseL3Formula(slope_id))
        target.setValue(float(default.subs(time, 0).subs(model_values)))
        target.setConstant(False)
        assignment.removeFromParentAndDelete()
        symbol_ids = {str(s) for s in expr.free_symbols}
        replaced_symbols |= symbol_ids
        converted_ids.append(target_id)

        for experiment in experiments:
            periods = experiment.sorted_periods
            overrides = {}
            for i, period in enumerate(periods):
                changes = problem.get_changes_for_period(period)
                overrides |= {c.target_id: c.target_value for c in changes}
                subs = fixed_values | {
                    s: overrides[str(s)]
                    for s in expr.free_symbols
                    if str(s) in overrides
                }
                times = []
                for t_switch, _ in switches.values():
                    t = t_switch.subs(subs)
                    if not (t.is_number and t.is_real and t.is_finite):
                        raise NotImplementedError(
                            "Only switching times that evaluate to a real "
                            "number are supported, but a switching time of "
                            f"the assignment to `{target_id}` in experiment "
                            f"`{experiment.id}` is `{t}`."
                        )
                    times.append(float(t))

                if period.is_preequilibration:
                    # the value at time 0, which respects whether a
                    #  condition holds for a switch at time 0, held there:
                    #  a ramp does not run during preequilibration
                    value = expr.subs(subs).subs(time, 0)
                    additions.append(
                        (experiment, period.time, target_id, value)
                    )
                    if ramps:
                        additions.append(
                            (experiment, period.time, slope_id, sp.S.Zero)
                        )
                    continue

                t_end = periods[i + 1].time if i + 1 < len(periods) else sp.oo
                start_value = _value_from(
                    expr, switches, time, subs, times, period.time
                )
                # each switch within the period that changes the value
                switch_values = []
                previous = start_value
                for t in sorted({t for t in times if period.time < t < t_end}):
                    value = _value_from(expr, switches, time, subs, times, t)
                    if not (value - previous).is_zero:
                        switch_values.append((t, value))
                    previous = value
                # The model already has the pre-switch value at the start of
                #  the first period, unless that value refers to parameter
                #  table symbols, for which the model only has a number. The
                #  switches can become the first periods if they only refer
                #  to parameter table symbols. A ramp's start and slope are
                #  always set by the tables.
                redundant = (
                    not ramps
                    and i == 0
                    and (start_value - default).is_zero
                    and not {str(s) for s in default.free_symbols}
                    & parameter_ids
                    and all(
                        {str(s) for s in value.free_symbols} <= parameter_ids
                        for _, value in switch_values
                    )
                )
                after_preequilibration = (
                    i > 0 and periods[i - 1].is_preequilibration
                )
                if not redundant and (
                    i == 0
                    or {c.target_id for c in changes} & symbol_ids
                    # preequilibration held the slope at zero
                    or (ramps and after_preequilibration)
                ):
                    if i == 0:
                        # at the start of an experiment, a change may only
                        #  refer to parameter table symbols; other symbols
                        #  are constant model parameters, at their values
                        start_value = start_value.subs(
                            {
                                s: sbml_model.getParameter(str(s)).getValue()
                                for s in start_value.free_symbols - {time}
                                if str(s) not in parameter_ids
                            }
                        )
                    additions += _changes(
                        experiment,
                        period.time,
                        target_id,
                        start_value,
                        time,
                        slope_id,
                    )
                for t, value in switch_values:
                    additions += _changes(
                        experiment, t, target_id, value, time, slope_id
                    )

    if converted_ids and any(e.has_preequilibration for e in experiments):
        warnings.warn(
            "Switches of time-dependent piecewise assignments during "
            "preequilibration are not supported. The converted "
            f"assignments to {converted_ids} keep their value at time 0 "
            "throughout "
            "preequilibration.",
            stacklevel=2,
        )

    # drop changes that are covered by the new conditions
    condition_ids = {c.id for c in problem.conditions}
    for table in problem.condition_tables:
        for condition in table.conditions:
            condition.changes = [
                c
                for c in condition.changes
                if c.target_id not in replaced_symbols
            ]
        table.elements = [c for c in table.conditions if c.changes]
    removed_condition_ids = condition_ids - {c.id for c in problem.conditions}
    for experiment in experiments:
        for period in experiment.periods:
            period.condition_ids = [
                cid
                for cid in period.condition_ids
                if cid not in removed_condition_ids
            ]

    # add the new conditions to new or existing periods
    new_conditions = {}
    cond_counter = count()
    for experiment, t, target_id, value in additions:
        if (cond_id := new_conditions.get((target_id, value))) is None:
            while (cond_id := f"cond_{next(cond_counter)}") in condition_ids:
                pass
            new_conditions[target_id, value] = cond_id
            problem += Condition(
                id=cond_id,
                changes=[Change(target_id=target_id, target_value=value)],
            )
        period = next((p for p in experiment.periods if p.time == t), None)
        if period is None:
            experiment += (period := ExperimentPeriod(time=t))
        period.condition_ids.append(cond_id)

    for experiment in experiments:
        experiment.sort_periods()
        experiment.periods = [
            p
            for p in experiment.periods
            if p.condition_ids or id(p) not in droppable
        ] or experiment.periods[:1]

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


def _changes(experiment, t, target_id, value, time, slope_id):
    """The changes that start ``value`` at time ``t``: the target set to the
    value at ``t``, and, for a target that ramps, its slope set to the
    value's rate of change."""
    changes = [(experiment, t, target_id, value.subs(time, t))]
    if slope_id is not None:
        changes.append((experiment, t, slope_id, sp.diff(value, time)))
    return changes


def _value_from(
    expr: sp.Expr,
    switches: dict,
    time: sp.Symbol,
    subs: dict,
    times: list[float],
    t0: float,
) -> sp.Expr:
    """The value of ``expr`` from ``t0`` up to its next switch, every
    condition replaced by whether it holds in between."""
    later = [t for t in times if t > t0]
    probe = (t0 + min(later)) / 2 if later else t0 + 1
    truth = {c: c.subs(subs).subs(time, probe) for c in switches}
    return expr.xreplace(truth).subs(subs)


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
